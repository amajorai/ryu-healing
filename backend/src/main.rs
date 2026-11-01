//! `ryu-healing` — the standalone, out-of-process self-healing sidecar.
//!
//! Runs the extracted `ryu_healing` capability crate (the diagnose→propose engine,
//! the loop-prevention decision logic, the per-source attempt cap in its plugin data dir,
//! and the `/api/healing/*` HTTP surface, defined in `lib.rs` / `api.rs`) as a
//! SEPARATE PROCESS that Core spawns, health-checks, and drives on loopback — like
//! `ryu-quests` / `ryu-mail`. The engine, decision logic, and handlers all live in
//! the crate lib; this binary is only the process shell around them. Core links NO
//! code from this crate — the two processes share only the `ryu-healing-contracts`
//! wire types, which both link and neither owns.
//!
//! ## What runs here vs. in Core
//! Core raises the `core#run.failed` kernel event and its fan-out POSTs it to
//! `POST /api/healing/report-failure` (manifest `subscribes`, grant `runs:failures`).
//! The engine here runs the cap/cooldown/never-heal-a-heal decision + the Gateway
//! diagnosis and then applies the verdict itself ([`ryu_healing::apply_verdict`])
//! through the host's action methods, which call Core's grant-gated kernel
//! capabilities: `approvals.request` (`approvals:request`), `runs.rerunAgent`
//! (`runs:rerun-agent`) and `workflows.rerun` (`workflows:runstate`). The approvals
//! write and the re-runs stay Core-owned; Core decides their shape (the approval
//! kinds, the `healrun_` re-run id) and this sidecar only asks.
//!
//! ## HOST SHIM (the sidecar's [`ryu_healing::HealingHost`] impl)
//! - **preferences** → a JSON file in the plugin data dir (`prefs.json`, durable
//!   across restarts), matching the in-process PreferencesStore-backed behaviour;
//! - **default diagnosis model** → env `RYU_DEFAULT_LLM_MODEL` → [`DEFAULT_DIAGNOSE_MODEL`];
//! - **attempt-cap data dir** → `paths::data_dir` (`RYU_PLUGIN_DATA_DIR`, injected by Core);
//! - **Gateway diagnosis call** → one non-streaming `/v1/chat/completions` on env
//!   `RYU_GATEWAY_URL` / `RYU_GATEWAY_TOKEN` (mirroring `apps/core/src/server/mod.rs::call_side_model`);
//! - **agent/workflow re-run + approvals delivery** → Core kernel capabilities.
//!
//! SECURITY: loopback-only bind (127.0.0.1) + a shared-secret bearer gate
//! (`RYU_EXT_TOKEN`, injected by Core at spawn and presented on the health probe +
//! every proxied hop). EVERY `/api/healing/*` route is protected; the gate is
//! FAIL-CLOSED (no token → reject all). `/health` is the ONE un-gated route so
//! Core's pre-auth health check succeeds.
//!
//! Port: `RYU_HEALING_PORT` env, default `8001` (7990-7999 are taken by the
//! wave-1..3 sidecars). Data: the app's plugin data dir
//! (`RYU_PLUGIN_DATA_DIR`, injected by Core; `<data dir>/plugin-data/@ryu+healing`),
//! holding `prefs.json` and `attempts.json`.

mod paths;

use std::collections::HashMap;
use std::net::{Ipv4Addr, SocketAddr};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use axum::{
    extract::Request,
    http::{header::AUTHORIZATION, StatusCode},
    middleware::{from_fn, Next},
    response::{IntoResponse, Response},
    routing::get,
    Json, Router,
};
use serde_json::{json, Value};

use ryu_healing::{routes, HealEngine, HealingCtx, HealingHost};

/// Default loopback port for the healing sidecar (overridable via `RYU_HEALING_PORT`).
/// 8001 is free (7990 finetune · 7991 quests · 7992 clips · 7993 browser · 7994
/// teams · 7995 research · 7996 mail · 7997 dashboards · 7998 meetings · 7999
/// recipes are taken). Kept identical in `healing.manifest.json`.
const DEFAULT_PORT: u16 = 8001;

/// The bundled local default diagnosis model when no pref/env is set — mirrors
/// Core's `registry::DEFAULT_LOCAL_CHAT_MODEL_ID`. Nothing is hardcoded to a remote
/// provider; a pref/env still overrides this.
const DEFAULT_DIAGNOSE_MODEL: &str = "gemma-4-E2B-it-Q4_K_M";

/// Fallback local gateway URL when `RYU_GATEWAY_URL` is unset — mirrors
/// `apps/core/src/sidecar/gateway.rs`.
const DEFAULT_GATEWAY_URL: &str = "http://127.0.0.1:7981";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let port: u16 = std::env::var("RYU_HEALING_PORT")
        .ok()
        .and_then(|p| p.trim().parse().ok())
        .unwrap_or(DEFAULT_PORT);

    // Shared-secret bearer Core injects via the generic ext-proxy loader
    // (`RYU_EXT_TOKEN`) — presented on every proxied hop + the health probe. The
    // protected `/api/healing/*` routes require it.
    let token = std::env::var("RYU_EXT_TOKEN")
        .ok()
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty());
    if token.is_some() {
        tracing::info!(
            "ryu-healing: protected /api/healing/* routes require the injected shared-secret bearer"
        );
    } else {
        tracing::warn!(
            "ryu-healing: no RYU_EXT_TOKEN set; protected /api/healing/* routes are FAIL-CLOSED (reject all). Core injects this token when it spawns the sidecar."
        );
    }

    let dir = paths::data_dir();
    // ONE host instance, shared by the engine and the router's `/config` handlers.
    // `SidecarHealingHost` caches prefs in an in-memory `Mutex<HashMap>`; a second
    // instance would have its OWN cache, so a `POST /config {enabled:false}` through
    // the router would flip the router's copy while the engine's `evaluate` kept
    // reading the stale copy — the config change would not reach the loop until a
    // restart. Sharing the Arc makes `pref_set` and `evaluate` read one map.
    let host: Arc<dyn HealingHost> = Arc::new(SidecarHealingHost::new(dir.join("prefs.json")));
    let engine = HealEngine::new(host.clone());

    // Publish the process-global engine: `GET /api/healing/status` and
    // `POST /api/healing/report-failure` read it (the crate handlers use
    // `global_engine()`, not the state-baked ctx, for the attempt map + evaluation).
    ryu_healing::set_global_engine(engine.clone());

    // The crate router (paths relative to `/api/healing`) nested under the external
    // prefix, with the shared-secret gate layered over the whole nest — healing has
    // no public route. `from_fn` closes over the resolved token so no extra state
    // field is needed. The state-baked ctx backs `/config` (pref read/write) and
    // shares the engine's host so a live config change reaches `evaluate`.
    let ctx = HealingCtx::new(host);
    let gated_token = token.clone();
    //
    // `/openapi.json` rides INSIDE that same gate, at the SERVER ROOT. Core fetches
    // `http://127.0.0.1:<port>/openapi.json` on this sidecar's first Healthy edge and
    // lowers every operation it finds into searchable LLM tools, so routing this one
    // endpoint is what makes the whole `/api/healing` surface callable by an agent
    // (`ryu_healing::api::openapi()` was dead code until now — only tests read it).
    //
    // Root, not under `/api/healing`: Core tries the root FIRST and only falls back to
    // the mount-prefixed form, and keeping the document off the mount keeps it out of
    // the manifest's declared `http.routes[]` — anything declared there is reachable
    // through the generic ext-proxy, and the schema is Core's to read, not an app
    // surface. Inside the gate, not next to the un-gated `/health`: Core stamps the
    // injected `RYU_EXT_TOKEN` on the fetch (the Python sidecars already require the
    // bearer for everything but `/health`), so the gate costs the fetcher nothing —
    // while un-gated it would disclose this app's entire internal API surface to any
    // other process on loopback.
    let healing = Router::new()
        .nest("/api/healing", routes(ctx))
        .route(
            "/openapi.json",
            get(|| async { Json(ryu_healing::api::openapi()) }),
        )
        .layer(from_fn(move |req: Request, next: Next| {
            let expected = gated_token.clone();
            async move { require_healing_token(req, next, expected.as_deref()).await }
        }));

    // `/health` sits OUTSIDE the gated nest so the loopback health probe succeeds
    // before auth. It returns process liveness only (no heal data).
    let app = Router::new().route("/health", get(health)).merge(healing);

    // LOOPBACK ONLY (belt) + shared-secret bearer (suspenders): Core is the auth
    // front and re-stamps the bearer on the proxied hop.
    let addr = SocketAddr::from((Ipv4Addr::LOCALHOST, port));
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!("ryu-healing sidecar listening on http://{addr}");

    axum::serve(listener, app).await?;
    Ok(())
}

/// Loopback health probe: process liveness. Un-gated and data-free.
async fn health() -> Response {
    (StatusCode::OK, Json(json!({ "ok": true }))).into_response()
}

/// Shared-secret bearer gate for the proxied `/api/healing/*` surface. Core stays
/// the auth front — it runs `require_auth`, then re-stamps `Authorization: Bearer
/// <RYU_EXT_TOKEN>` on the loopback hop — so a request that did NOT come through Core
/// is rejected with 401. **Fail-closed:** `expected == None`/empty rejects every
/// request rather than falling open.
async fn require_healing_token(req: Request, next: Next, expected: Option<&str>) -> Response {
    let provided = req
        .headers()
        .get(AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "));
    if ryu_sidecar_runtime::token_ok(provided, expected) {
        next.run(req).await
    } else {
        (StatusCode::UNAUTHORIZED, "unauthorized").into_response()
    }
}

/// The sidecar's standalone [`HealingHost`]: everything the moved heal engine needs
/// from the host, provided by the process itself rather than by Core.
///
/// - **preferences** → a JSON map persisted in the plugin data dir (durable across restarts);
/// - **default diagnosis model** → env `RYU_DEFAULT_LLM_MODEL` → [`DEFAULT_DIAGNOSE_MODEL`];
/// - **Gateway diagnosis** → env `RYU_GATEWAY_URL` / `RYU_GATEWAY_TOKEN`;
/// - **agent/workflow re-run + approvals delivery** → Core's grant-gated kernel
///   capabilities (`approvals.request`, `runs.rerunAgent`, `workflows.rerun`).
struct SidecarHealingHost {
    prefs_path: PathBuf,
    prefs: Mutex<HashMap<String, String>>,
    http: reqwest::Client,
    core: CoreCapabilities,
}

impl SidecarHealingHost {
    fn new(prefs_path: PathBuf) -> Self {
        let prefs = load_prefs(&prefs_path);
        Self {
            prefs_path,
            prefs: Mutex::new(prefs),
            http: reqwest::Client::new(),
            core: CoreCapabilities::from_env(),
        }
    }
}

/// Read the persisted preference map (empty on missing/corrupt file — a fresh
/// install just falls back to the resolver defaults).
fn load_prefs(path: &PathBuf) -> HashMap<String, String> {
    let Ok(bytes) = std::fs::read(path) else {
        return HashMap::new();
    };
    serde_json::from_slice(&bytes).unwrap_or_default()
}

/// Persist the preference map atomically (write a temp file, then rename) so a
/// crash mid-write cannot corrupt the live config file.
fn save_prefs(path: &PathBuf, map: &HashMap<String, String>) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let bytes = serde_json::to_vec_pretty(map).map_err(|e| e.to_string())?;
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, &bytes).map_err(|e| e.to_string())?;
    std::fs::rename(&tmp, path).map_err(|e| e.to_string())
}

#[async_trait]
impl HealingHost for SidecarHealingHost {
    async fn pref_get(&self, key: &str) -> Option<String> {
        self.prefs.lock().ok()?.get(key).cloned()
    }

    async fn pref_set(&self, key: &str, value: &str) -> Result<(), String> {
        // Hold the lock across the write: concurrent sets otherwise race on the
        // shared `.json.tmp` path (one rename fails, or an older snapshot lands
        // last and silently drops the newer preference).
        let mut guard = self
            .prefs
            .lock()
            .map_err(|_| "preferences lock poisoned".to_string())?;
        guard.insert(key.to_string(), value.to_string());
        save_prefs(&self.prefs_path, &guard)
    }

    fn default_diagnose_model(&self) -> String {
        std::env::var("RYU_DEFAULT_LLM_MODEL")
            .ok()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| DEFAULT_DIAGNOSE_MODEL.to_string())
    }

    fn data_dir(&self) -> PathBuf {
        paths::data_dir()
    }

    async fn call_side_model(
        &self,
        model: &str,
        effort: &str,
        system: &str,
        user: &str,
    ) -> Result<String, String> {
        // One non-streaming Gateway completion — byte-for-byte the request
        // `apps/core/src/server/mod.rs::call_side_model` builds, so a diagnosis is
        // firewalled / DLP'd / budgeted / audited exactly as in-process.
        let base = std::env::var("RYU_GATEWAY_URL")
            .ok()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| DEFAULT_GATEWAY_URL.to_string());
        let base = base.trim_end_matches('/');
        let mut payload = json!({
            "model": model,
            "stream": false,
            "messages": [
                { "role": "system", "content": system },
                { "role": "user", "content": user },
            ],
        });
        let effort = effort.trim();
        if !effort.is_empty() {
            payload["reasoning_effort"] = json!(effort);
        }
        let mut req = self
            .http
            .post(format!("{base}/v1/chat/completions"))
            .timeout(std::time::Duration::from_secs(60))
            .json(&payload);
        if let Some(t) = std::env::var("RYU_GATEWAY_TOKEN")
            .ok()
            .filter(|s| !s.is_empty())
        {
            req = req.bearer_auth(t);
        }
        let resp = req
            .send()
            .await
            .map_err(|e| format!("gateway unreachable: {e}"))?;
        if !resp.status().is_success() {
            return Err(format!("gateway returned HTTP {}", resp.status()));
        }
        let body: Value = resp
            .json()
            .await
            .map_err(|e| format!("response was not valid JSON: {e}"))?;
        let text = body
            .get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(|t| t.as_str())
            .unwrap_or_default();
        Ok(text.to_string())
    }

    async fn rerun_agent(&self, agent_id: Option<String>, run_id: String, prompt: String) {
        // Core mints its own `healrun_` id for the re-run (a caller-chosen id could
        // collide with a real conversation); the engine's id is only a local label.
        tracing::debug!("ryu-healing: requesting agent re-run (local id {run_id})");
        self.core
            .call(
                CAP_RERUN_AGENT,
                json!({ "agent_id": agent_id, "prompt": prompt }),
            )
            .await;
    }

    async fn rerun_workflow(&self, source_id: &str) {
        self.core
            .call(CAP_RERUN_WORKFLOW, json!({ "source_id": source_id }))
            .await;
    }

    async fn queue_heal_fix(
        &self,
        source_id: &str,
        agent_id: Option<String>,
        diagnosis: &str,
        corrected: String,
    ) {
        self.core
            .call(
                CAP_APPROVALS_REQUEST,
                json!({
                    "kind": "heal_fix",
                    "source_id": source_id,
                    "agent_id": agent_id,
                    "diagnosis": diagnosis,
                    "corrected": corrected,
                }),
            )
            .await;
    }

    async fn queue_heal_workflow(&self, source_id: &str, diagnosis: &str) {
        self.core
            .call(
                CAP_APPROVALS_REQUEST,
                json!({ "kind": "heal_workflow", "source_id": source_id, "diagnosis": diagnosis }),
            )
            .await;
    }

    async fn queue_heal_exhausted(&self, source_id: &str, note: &str) {
        self.core
            .call(
                CAP_APPROVALS_REQUEST,
                json!({ "kind": "heal_exhausted", "source_id": source_id, "note": note }),
            )
            .await;
    }
}

/// Queue a heal approval (grant `approvals:request`).
const CAP_APPROVALS_REQUEST: &str = "approvals.request";
/// Re-run a failed agent run under a Core-minted `healrun_` id (grant `runs:rerun-agent`).
const CAP_RERUN_AGENT: &str = "runs.rerunAgent";
/// Re-run a failed workflow run (grant `workflows:runstate`).
const CAP_RERUN_WORKFLOW: &str = "workflows.rerun";

/// The `x-ryu-plugin-id` header Core's `authenticate_sidecar` reads.
const HDR_PLUGIN_ID: &str = "x-ryu-plugin-id";
/// This app's id, presented on every Core callback.
const HEALING_PLUGIN_ID: &str = "@ryu/healing";
/// Core's release loopback port; the injected `RYU_CORE_PORT` wins.
const DEFAULT_CORE_PORT: u16 = 7980;

/// Core's kernel-capability broker, reached as
/// `POST http://127.0.0.1:<RYU_CORE_PORT>/api/host/capability/<cap>` with this
/// sidecar's minted bearer (`RYU_EXT_TOKEN`). Applying a verdict is best-effort:
/// a refusal (missing grant, approvals not running) is logged, never retried, and
/// never wedges the engine.
struct CoreCapabilities {
    base: String,
    token: Option<String>,
    http: reqwest::Client,
}

impl CoreCapabilities {
    fn from_env() -> Self {
        let port: u16 = std::env::var("RYU_CORE_PORT")
            .ok()
            .and_then(|p| p.trim().parse().ok())
            .unwrap_or(DEFAULT_CORE_PORT);
        Self {
            base: format!("http://127.0.0.1:{port}"),
            token: std::env::var("RYU_EXT_TOKEN")
                .ok()
                .map(|s| s.trim().to_owned())
                .filter(|s| !s.is_empty()),
            http: reqwest::Client::new(),
        }
    }

    async fn call(&self, capability: &str, body: Value) {
        let Some(token) = self.token.as_deref() else {
            tracing::warn!("ryu-healing: no RYU_EXT_TOKEN; cannot call Core '{capability}'");
            return;
        };
        let result = self
            .http
            .post(format!("{}/api/host/capability/{capability}", self.base))
            .timeout(std::time::Duration::from_secs(30))
            .bearer_auth(token)
            .header(HDR_PLUGIN_ID, HEALING_PLUGIN_ID)
            .json(&body)
            .send()
            .await;
        match result {
            Ok(response) if response.status().is_success() => {}
            Ok(response) => tracing::warn!(
                "ryu-healing: Core refused '{capability}': HTTP {}",
                response.status()
            ),
            Err(error) => tracing::warn!("ryu-healing: Core '{capability}' unreachable: {error}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{load_prefs, save_prefs};
    use std::collections::HashMap;

    #[test]
    fn prefs_roundtrip_through_file() {
        let dir = std::env::temp_dir().join(format!("ryu-healing-test-{}", std::process::id()));
        let path = dir.join("prefs.json");
        let _ = std::fs::remove_file(&path);

        assert!(load_prefs(&path).is_empty());

        let mut map = HashMap::new();
        map.insert("healing.enabled".to_string(), "false".to_string());
        save_prefs(&path, &map).expect("save prefs");

        let reloaded = load_prefs(&path);
        assert_eq!(
            reloaded.get("healing.enabled").map(String::as_str),
            Some("false")
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
