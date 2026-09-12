//! HTTP API for the self-healing loop (`/api/healing/config` + `/api/healing/status`).
//!
//! Thin handlers over the healing engine: read/write the `healing.*` config
//! (master switch + auto-decide + caps + diagnosis model) and inspect the
//! in-memory per-source attempt map.
//!
//! The router is built with its own state ([`HealingCtx`]) inside this crate so it
//! returns a state-less, mergeable `Router<()>`. The routes are declared relative
//! to `/api/healing` (Core nests this service at that prefix behind the
//! Self-Healing-App gate, alongside the kernel-coupled `/api/healing/simulate-failure`
//! debug endpoint that stays Core-side), while the OpenAPI annotations keep the
//! full external paths.

use std::sync::Arc;

use axum::{
    extract::State,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde_json::{json, Value};

use crate::{
    global_engine, resolve_config, HealSource, HealVerdict, HealingHost, HEALING_AUTO_DECIDE_PREF,
    HEALING_COOLDOWN_SECS_PREF, HEALING_DIAGNOSE_EFFORT_PREF, HEALING_DIAGNOSE_MODEL_PREF,
    HEALING_ENABLED_PREF, HEALING_MAX_ATTEMPTS_PREF,
};

/// Router state for the healing HTTP surface: the inverted [`HealingHost`] (for
/// reading/writing `healing.*` prefs). The per-source attempt map is read from the
/// process-global engine ([`global_engine`]).
#[derive(Clone)]
pub struct HealingCtx {
    pub host: Arc<dyn HealingHost>,
}

impl HealingCtx {
    pub fn new(host: Arc<dyn HealingHost>) -> Self {
        Self { host }
    }
}

/// Build the `/api/healing/*` config+status router with its own state baked in,
/// returning a state-less `Router<()>` the host nests at `/api/healing` behind the
/// App gate.
pub fn routes(ctx: HealingCtx) -> Router<()> {
    Router::new()
        .route("/config", get(config).post(set_config))
        .route("/status", get(status))
        .route("/report-failure", post(report_failure))
        .with_state(ctx)
}

/// The OpenAPI sub-document for the healing config+status surface, merged into
/// Core's spec when the `healing` feature is enabled.
pub fn openapi() -> utoipa::openapi::OpenApi {
    <HealingApiDoc as utoipa::OpenApi>::openapi()
}

/// The document Core imports. `components(schemas(...))` is what turns
/// `request_body = HealingConfigBody` into a resolvable
/// `#/components/schemas/HealingConfigBody` entry: without it the operation still
/// carries a `$ref`, but the target is missing and Core's `resolve_ref` yields
/// nothing — a derived write tool with zero visible arguments. utoipa 5 also
/// auto-collects schemas reachable from the annotated paths, so this row is
/// belt-and-braces; it is listed explicitly anyway so the registration is
/// greppable and cannot be silently lost to an attribute edit.
#[derive(utoipa::OpenApi)]
#[openapi(
    paths(config, set_config, status),
    components(schemas(HealingConfigBody))
)]
struct HealingApiDoc;

/// Request body for `POST /api/healing/config` — a partial patch: send only the
/// keys you want to change.
// Everything below is `//`, not `///`, ON PURPOSE: utoipa lifts a struct's doc
// comment into the schema's own `description`, so internal rationale written as
// `///` ships to the model alongside the arguments.
//
// The FIELD docs below are the opposite — utoipa lifts them into each property's
// `description`, and they are the only prose the model reads when choosing
// arguments. Before this type existed the annotation said
// `request_body = serde_json::Value`, so the tool reached the model with no
// arguments at all: discoverable, callable, and unable to change anything.
//
// This type describes the wire shape; it is deliberately NOT used as the axum
// extractor. `set_config` reads each key through `Value::as_bool`/`as_u64`/
// `as_str`, so a mistyped field is IGNORED and the request still succeeds — a
// contract `set_config_ignores_absent_and_mistyped_fields` locks down on purpose,
// and one a typed extractor would replace with a whole-request 422. The annotation
// is the half Core reads, so typing it buys the arguments without touching the
// handler's tolerance, and
// `documented_fields_are_the_ones_the_handler_writes` keeps the two from drifting.
#[derive(Debug, serde::Deserialize, utoipa::ToSchema)]
pub struct HealingConfigBody {
    /// Master switch for the whole self-healing loop. Off means failed runs are
    /// never diagnosed.
    #[serde(default)]
    pub enabled: Option<bool>,
    /// Whether a diagnosed fix is applied automatically. Off (the default) queues
    /// it in the approvals inbox for a human instead.
    #[serde(default)]
    pub auto_decide: Option<bool>,
    /// How many times one source may be healed before the loop gives up on it.
    #[serde(default)]
    pub max_attempts: Option<u64>,
    /// Minimum seconds between two heals of the same source.
    #[serde(default)]
    pub cooldown_secs: Option<i64>,
    /// Model id used to diagnose a failure, e.g. `anthropic/claude-sonnet-4`.
    /// Empty means the node default.
    #[serde(default)]
    pub diagnose_model: Option<String>,
    /// Reasoning effort for the diagnosis, e.g. `low`, `medium`, or `high`.
    #[serde(default)]
    pub diagnose_effort: Option<String>,
}

/// `GET /api/healing/config` — resolved healing config (switches + caps + model).
#[utoipa::path(
    get,
    path = "/api/healing/config",
    tag = "Healing",
    summary = "resolved healing config (switches + caps + model).",
    responses((status = 200, description = "OK", body = serde_json::Value))
)]
pub async fn config(State(ctx): State<HealingCtx>) -> impl IntoResponse {
    Json(resolve_config(&*ctx.host).await)
}

/// `POST /api/healing/config` — set any provided `healing.*` prefs. Body accepts
/// any of: `enabled`, `auto_decide` (bool), `max_attempts`, `cooldown_secs`
/// (number), `diagnose_model`, `diagnose_effort` (string).
#[utoipa::path(
    post,
    path = "/api/healing/config",
    tag = "Healing",
    summary = "set any provided `healing.*` prefs. Body accepts",
    request_body = HealingConfigBody,
    responses((status = 200, description = "OK", body = serde_json::Value))
)]
pub async fn set_config(State(ctx): State<HealingCtx>, Json(body): Json<Value>) -> Response {
    async fn set_bool(host: &dyn HealingHost, key: &str, v: Option<bool>) {
        if let Some(b) = v {
            let _ = host.pref_set(key, if b { "true" } else { "false" }).await;
        }
    }
    async fn set_str(host: &dyn HealingHost, key: &str, v: Option<&str>) {
        if let Some(s) = v {
            let _ = host.pref_set(key, s).await;
        }
    }
    set_bool(
        &*ctx.host,
        HEALING_ENABLED_PREF,
        body.get("enabled").and_then(Value::as_bool),
    )
    .await;
    set_bool(
        &*ctx.host,
        HEALING_AUTO_DECIDE_PREF,
        body.get("auto_decide").and_then(Value::as_bool),
    )
    .await;
    if let Some(n) = body.get("max_attempts").and_then(Value::as_u64) {
        let _ = ctx
            .host
            .pref_set(HEALING_MAX_ATTEMPTS_PREF, &n.to_string())
            .await;
    }
    if let Some(n) = body.get("cooldown_secs").and_then(Value::as_i64) {
        let _ = ctx
            .host
            .pref_set(HEALING_COOLDOWN_SECS_PREF, &n.to_string())
            .await;
    }
    set_str(
        &*ctx.host,
        HEALING_DIAGNOSE_MODEL_PREF,
        body.get("diagnose_model").and_then(Value::as_str),
    )
    .await;
    set_str(
        &*ctx.host,
        HEALING_DIAGNOSE_EFFORT_PREF,
        body.get("diagnose_effort").and_then(Value::as_str),
    )
    .await;
    Json(resolve_config(&*ctx.host).await).into_response()
}

/// `POST /api/healing/report-failure` — the INTERNAL Core→sidecar ingress. Core's
/// loopback client posts a failed run's context (already extracted host-side: the
/// run-status bus, scheduler, and workflow executor all stay kernel), the sidecar
/// engine runs the cap/cooldown/never-heal-a-heal decision + the Gateway diagnosis,
/// and returns a [`HealVerdict`] JSON for Core to `apply_verdict` (Core owns the
/// welded approvals write + agent/workflow re-run). Deliberately NOT listed in the
/// manifest's public `routes[]` — it is reachable only on loopback with the ext
/// bearer, never through the public ext-proxy mount.
///
/// Body: `{ "source_id": string, "kind": "agent"|"workflow", "agent_id"?: string,
/// "instruction"?: string, "failure"?: string }`.
pub async fn report_failure(State(_ctx): State<HealingCtx>, Json(body): Json<Value>) -> Response {
    let Some(engine) = global_engine() else {
        return Json(HealVerdict::Skip {
            reason: "heal engine unavailable".to_owned(),
        })
        .into_response();
    };
    let source_id = body
        .get("source_id")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    let agent_id = body
        .get("agent_id")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let instruction = body
        .get("instruction")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    let failure = body
        .get("failure")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    // The spelling comes from the shared contract, not a literal: this arm is
    // `workflow`-or-else-agent, so a drift against Core's side would not 400 — it
    // would quietly heal a failed workflow as if it were an agent.
    let source = match body.get("kind").and_then(Value::as_str) {
        Some(ryu_healing_contracts::SOURCE_KIND_WORKFLOW) => HealSource::Workflow,
        _ => HealSource::Agent { agent_id },
    };
    let verdict = engine
        .evaluate(&source_id, source, instruction, failure)
        .await;
    Json(verdict).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{tmp_dir, MockHost};
    use crate::{
        set_global_engine, HealEngine, HEALING_AUTO_DECIDE_PREF, HEALING_ENABLED_PREF,
        HEALING_MAX_ATTEMPTS_PREF,
    };
    use axum::body::to_bytes;
    use axum::extract::State;
    use axum::http::StatusCode;

    async fn body_json(resp: Response) -> Value {
        let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[tokio::test]
    async fn config_get_returns_resolved_view() {
        let host = Arc::new(
            MockHost::new(tmp_dir())
                .with_pref(HEALING_ENABLED_PREF, "false")
                .with_pref(HEALING_MAX_ATTEMPTS_PREF, "7"),
        );
        let ctx = HealingCtx::new(host);
        let resp = config(State(ctx)).await.into_response();
        assert_eq!(resp.status(), StatusCode::OK);
        let v = body_json(resp).await;
        assert_eq!(v.get("enabled").and_then(Value::as_bool), Some(false));
        assert_eq!(v.get("max_attempts").and_then(Value::as_u64), Some(7));
    }

    #[tokio::test]
    async fn set_config_writes_every_provided_pref() {
        let host = Arc::new(MockHost::new(tmp_dir()));
        let ctx = HealingCtx::new(host.clone());
        let body = json!({
            "enabled": false,
            "auto_decide": true,
            "max_attempts": 5,
            "cooldown_secs": 0,
            "diagnose_model": "m-x",
            "diagnose_effort": "high",
        });
        let resp = set_config(State(ctx), Json(body)).await;
        assert_eq!(resp.status(), StatusCode::OK);

        let prefs = host.prefs.lock().unwrap();
        assert_eq!(
            prefs.get(HEALING_ENABLED_PREF).map(String::as_str),
            Some("false")
        );
        assert_eq!(
            prefs.get(HEALING_AUTO_DECIDE_PREF).map(String::as_str),
            Some("true")
        );
        assert_eq!(
            prefs.get(HEALING_MAX_ATTEMPTS_PREF).map(String::as_str),
            Some("5")
        );
        assert_eq!(
            prefs
                .get(crate::HEALING_COOLDOWN_SECS_PREF)
                .map(String::as_str),
            Some("0")
        );
        assert_eq!(
            prefs.get(HEALING_DIAGNOSE_MODEL_PREF).map(String::as_str),
            Some("m-x")
        );
        assert_eq!(
            prefs.get(HEALING_DIAGNOSE_EFFORT_PREF).map(String::as_str),
            Some("high")
        );
    }

    #[tokio::test]
    async fn set_config_ignores_absent_and_mistyped_fields() {
        let host = Arc::new(MockHost::new(tmp_dir()));
        let ctx = HealingCtx::new(host.clone());
        // enabled as a string (not a bool) is ignored; max_attempts as a string too.
        let body = json!({ "enabled": "true", "max_attempts": "9", "unrelated": 1 });
        let resp = set_config(State(ctx), Json(body)).await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert!(host.prefs.lock().unwrap().is_empty(), "no prefs written");
    }

    // This single test owns the process-global engine (a `OnceLock`), so it is the
    // ONLY test that calls `set_global_engine`. It exercises both the
    // `report-failure` ingress and the `status` handler against that engine.
    #[tokio::test]
    async fn report_failure_and_status_handlers_use_global_engine() {
        let host = Arc::new(MockHost::new(tmp_dir()).with_pref(HEALING_AUTO_DECIDE_PREF, "true"));
        host.set_reply(Ok(
            "{\"diagnosis\":\"d\",\"corrected_prompt\":\"cp\"}".to_string()
        ));
        let engine = HealEngine::new(host);
        set_global_engine(engine);

        // report-failure: agent kind, auto-decide ON -> rerun_agent verdict.
        let ctx = HealingCtx::new(Arc::new(MockHost::new(tmp_dir())));
        let body = json!({
            "source_id": "conv-global",
            "kind": "agent",
            "agent_id": "ag",
            "instruction": "orig",
            "failure": "boom",
        });
        let resp = report_failure(State(ctx.clone()), Json(body)).await;
        let v = body_json(resp).await;
        assert_eq!(v.get("action").and_then(Value::as_str), Some("rerun_agent"));
        assert_eq!(v.get("prompt").and_then(Value::as_str), Some("cp"));

        // status: the attempt just recorded is now visible.
        let resp = status(State(ctx)).await;
        let v = body_json(resp).await;
        let attempts = v.get("attempts").expect("attempts");
        assert!(attempts.get("conv-global").is_some(), "attempt recorded");
    }

    // ── OpenAPI document ───────────────────────────────────────────────────────

    /// This app's own manifest, read at compile time. The route contract lives there,
    /// so the invariants below compare the document against the real declaration
    /// rather than against a second list that could drift from it.
    fn openapi_manifest() -> serde_json::Value {
        serde_json::from_str(include_str!("../../manifest.json")).expect("valid JSON")
    }

    /// The manifest sidecar whose HTTP surface this router serves: the one that
    /// declares an `http.mount`. Selected BY mount rather than by index because an app
    /// may declare a second, mountless sidecar (finetune already does), and
    /// `sidecars[0]` would then quietly start asserting against the wrong process.
    fn mounted_sidecar() -> serde_json::Value {
        openapi_manifest()["sidecars"]
            .as_array()
            .expect("sidecars must be an array")
            .iter()
            .find(|s| s["http"]["mount"].is_string())
            .expect("one sidecar must declare an http.mount")
            .clone()
    }

    /// A manifest route (relative to the mount, in axum's `:param` form) rewritten
    /// into the form the OpenAPI document uses (absolute, in `{param}` form).
    ///
    /// The two forms differ ON PURPOSE — the router registers paths relative to the
    /// mount because Core nests it there, while the `#[utoipa::path]` annotations carry
    /// the absolute EXTERNAL path a caller actually hits. Normalise here; do not
    /// "align" either side.
    fn doc_path_for(mount: &str, route: &str) -> String {
        let joined = if route == "/" {
            mount.to_owned()
        } else {
            format!("{mount}{route}")
        };
        joined
            .split('/')
            .map(|seg| match seg.strip_prefix(':') {
                Some(name) => format!("{{{name}}}"),
                None => seg.to_owned(),
            })
            .collect::<Vec<_>>()
            .join("/")
    }

    #[test]
    fn openapi_doc_is_served_and_non_empty() {
        // The doc is no longer dead code: Core fetches it to derive tools.
        assert!(!super::openapi().paths.paths.is_empty());
    }

    #[test]
    fn every_declared_route_appears_in_the_openapi_doc() {
        // The direction that decides tool yield. Core's `ext_api::lower` keeps only the
        // document operations the manifest ALSO declares, so a declared route with no
        // `#[utoipa::path]` annotation is a tool that silently never exists — nothing
        // errors, the agent simply cannot call it. (The other direction is harmless: an
        // annotated path the manifest does not declare is dropped by the same filter.)
        let sidecar = mounted_sidecar();
        let mount = sidecar["http"]["mount"].as_str().expect("an http.mount");
        let doc = super::openapi();
        for route in sidecar["http"]["routes"]
            .as_array()
            .expect("routes must be an array")
        {
            let path = route["path"].as_str().expect("a route path");
            let expected = doc_path_for(mount, path);
            assert!(
                doc.paths.paths.contains_key(&expected),
                "'{path}' is declared in manifest.json but the OpenAPI document has no \
                 '{expected}' operation — Core derives no tool for it"
            );
        }
    }

    // ── Request-body schema ────────────────────────────────────────────────────

    /// The one pointer Core reads to give a derived write tool its arguments.
    fn body_schema(wire: &Value, path: &str, method: &str) -> Value {
        wire.pointer(&format!(
            "/paths/{}/{method}/requestBody/content/application~1json/schema",
            path.replace('/', "~1")
        ))
        .unwrap_or_else(|| panic!("{method} {path} must declare a JSON request body"))
        .clone()
    }

    #[test]
    fn post_routes_document_their_request_body() {
        // The regression this locks down: the annotation used to say
        // `request_body = serde_json::Value`, which serialises to an untyped schema.
        // Core derives a tool per operation and fills `input_schema` from THIS node,
        // so an untyped body produced a tool the model could discover, could call,
        // and could never pass a single argument to — an agent asked to "turn on
        // self-healing" had no way to say which switch it meant.
        //
        // A `$ref` is the CORRECT and expected shape, not a near-miss: Core's
        // `openapi_import::resolve_ref` resolves it against `components.schemas`
        // before reading `properties`. So accept either a ref or inlined properties;
        // asserting "inlined" would fail on a healthy document.
        let wire = serde_json::to_value(super::openapi()).expect("the doc must serialize");
        let schema = body_schema(&wire, "/api/healing/config", "post");
        assert!(
            schema.get("$ref").is_some() || schema.get("properties").is_some(),
            "a derived write tool for POST /api/healing/config would have no arguments: {schema}"
        );
    }

    #[test]
    fn every_request_body_ref_resolves_against_components() {
        // The half of the retrofit that a `$ref`-shaped assertion alone cannot see:
        // a `$ref` pointing at a schema that was never registered in
        // `components(schemas(...))` looks identical in the operation and still
        // yields zero arguments once Core tries to resolve it. Walk every request
        // body in the document and check the target actually exists and carries
        // properties.
        let wire = serde_json::to_value(super::openapi()).expect("the doc must serialize");
        let paths = wire["paths"].as_object().expect("paths must be an object");
        let mut checked = 0usize;
        for (path, item) in paths {
            for (method, op) in item.as_object().expect("a path item is an object") {
                let Some(schema) = op.pointer("/requestBody/content/application~1json/schema")
                else {
                    continue;
                };
                let Some(reference) = schema.get("$ref").and_then(Value::as_str) else {
                    // Inlined schemas are fine as long as they describe something.
                    // The failure this catches in practice is `request_body =
                    // Option<T>`, which utoipa renders as a nullable `oneOf` wrapper:
                    // Core resolves only a TOP-LEVEL `$ref`, so the wrapper reaches the
                    // importer unresolved and contributes no properties at all.
                    assert!(
                        schema.get("properties").is_some(),
                        "{method} {path} has a request-body schema Core cannot read \
                         (a `oneOf` here means `request_body = Option<T>` — use the \
                         plain type): {schema}"
                    );
                    checked += 1;
                    continue;
                };
                let name = reference
                    .strip_prefix("#/components/schemas/")
                    .unwrap_or_else(|| {
                        panic!("unexpected ref form '{reference}' at {method} {path}")
                    });
                let target = wire
                    .pointer(&format!("/components/schemas/{name}"))
                    .unwrap_or_else(|| {
                        panic!(
                            "{method} {path} refs '{name}' but it is missing from \
                             components.schemas — add it to components(schemas(..))"
                        )
                    });
                assert!(
                    target.get("properties").is_some(),
                    "{method} {path} refs '{name}', which has no properties: {target}"
                );
                checked += 1;
            }
        }
        assert_eq!(
            checked, 1,
            "expected the one write route to carry a body schema, saw {checked}"
        );
    }

    #[test]
    fn body_field_docs_reach_the_schema_as_argument_descriptions() {
        // Doc comments on the body-struct fields are the whole payoff of the
        // retrofit: they are the only prose the model reads when choosing arguments.
        // utoipa lifts them into `description`, so a future edit that drops them
        // silently degrades tool-call quality with no compile error.
        let wire = serde_json::to_value(super::openapi()).expect("the doc must serialize");
        let auto = wire
            .pointer("/components/schemas/HealingConfigBody/properties/auto_decide/description")
            .and_then(Value::as_str)
            .unwrap_or_default();
        assert!(
            auto.contains("approvals inbox"),
            "HealingConfigBody::auto_decide lost its doc comment, got {auto:?}"
        );
    }

    #[test]
    fn schema_descriptions_carry_no_internal_rationale() {
        // utoipa lifts a STRUCT's doc comment into the schema's own `description`,
        // exactly as it lifts field docs into property descriptions — so the `///`
        // paragraphs explaining why this type is not the axum extractor would ship to
        // the model as part of the tool. The convention that prevents it: one `///`
        // line naming the body, and every rationale paragraph below it demoted to
        // `//`. Wrapped prose is fine — the tell is VOCABULARY, so this greps for the
        // Rust implementation words that only ever appear in rationale, never in
        // something written for a caller.
        let wire = serde_json::to_value(super::openapi()).expect("the doc must serialize");
        let schemas = wire["components"]["schemas"]
            .as_object()
            .expect("components.schemas must be an object");
        for (name, schema) in schemas {
            let mut descriptions = vec![schema.get("description")];
            if let Some(props) = schema.get("properties").and_then(Value::as_object) {
                descriptions.extend(props.values().map(|p| p.get("description")));
            }
            for description in descriptions.into_iter().flatten().filter_map(Value::as_str) {
                for leak in ["axum", "utoipa", "extractor", "Deserialize", "serde_json"] {
                    assert!(
                        !description.contains(leak),
                        "{name} ships the word '{leak}' to the model in a schema \
                         description — demote that rationale from `///` to `//`: \
                         {description:?}"
                    );
                }
            }
        }
    }

    #[tokio::test]
    async fn documented_fields_are_the_ones_the_handler_writes() {
        // [`HealingConfigBody`] documents the wire shape but is not the extractor, so
        // nothing in the type system ties it to `set_config`. This closes that gap
        // behaviourally: drive the handler with ONE key at a time, typed as the
        // schema says, and require each to land as exactly one written pref. A field
        // added to the struct but not read by the handler fails here (nothing
        // written), as does a documented type the handler's `as_bool`/`as_u64`/
        // `as_str` cannot read.
        let wire = serde_json::to_value(super::openapi()).expect("the doc must serialize");
        let props = wire["components"]["schemas"]["HealingConfigBody"]["properties"]
            .as_object()
            .expect("the body schema must document properties");
        assert_eq!(props.len(), 6, "expected six documented switches");
        for (field, schema) in props {
            // `Option<T>` renders as `type: ["<t>", "null"]`; take the non-null half.
            let ty = match &schema["type"] {
                Value::String(s) => s.clone(),
                Value::Array(a) => a
                    .iter()
                    .filter_map(Value::as_str)
                    .find(|t| *t != "null")
                    .expect("a non-null type")
                    .to_owned(),
                other => panic!("{field} has an unreadable type: {other}"),
            };
            let sample = match ty.as_str() {
                "boolean" => json!(true),
                "integer" | "number" => json!(1),
                "string" => json!("x"),
                other => panic!("{field} documents an unhandled type '{other}'"),
            };
            let mut body = serde_json::Map::new();
            body.insert(field.clone(), sample);
            let host = Arc::new(MockHost::new(tmp_dir()));
            let ctx = HealingCtx::new(host.clone());
            let resp = set_config(State(ctx), Json(Value::Object(body))).await;
            assert_eq!(resp.status(), StatusCode::OK);
            assert_eq!(
                host.prefs.lock().unwrap().len(),
                1,
                "the documented field '{field}' ({ty}) wrote no pref — the schema and \
                 the handler have drifted apart"
            );
        }
    }
}

/// `GET /api/healing/status` — the in-memory per-source attempt map.
#[utoipa::path(
    get,
    path = "/api/healing/status",
    tag = "Healing",
    summary = "the in-memory per-source attempt map.",
    responses((status = 200, description = "OK", body = serde_json::Value))
)]
pub async fn status(State(_ctx): State<HealingCtx>) -> Response {
    let attempts = match global_engine() {
        Some(engine) => engine.attempt_snapshot().await,
        None => Default::default(),
    };
    Json(json!({ "attempts": attempts })).into_response()
}
