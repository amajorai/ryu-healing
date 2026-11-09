# ryu-healing

Self-healing loop for failed runs — an extracted **Core capability crate**. When an
agent or workflow run fails, this engine asks a Gateway-governed side model to
diagnose *why* and propose a corrected instruction, then either auto-applies the fix
(re-runs) or queues it in the approvals inbox for the user.

## Role in the decomposition

A primitive lifted out of `apps/core`. It owns the diagnose→propose engine, the
loop-prevention decision logic, the per-source attempt state, and the
`/api/healing/*` HTTP surface. **The surface is now served OUT-OF-PROCESS** by the
`ryu-healing` bin (`[[bin]]`, `kind:local`, `public_mount`, `RYU_HEALING_BIN`/
`RYU_HEALING_PORT`, default `:8001`); Core proxies to it. Core links no code from
this crate and knows no healing: it raises the `core#run.failed` kernel event, which
this app subscribes to (manifest `subscribes`, grant `runs:failures`), and the
sidecar applies its own verdict by calling Core's grant-gated kernel capabilities:
`approvals.request` (`approvals:request`), `runs.rerunAgent` (`runs:rerun-agent`)
and `workflows.rerun` (`workflows:runstate`). Every kernel coupling it needs —
`healing.*` preference read/write, the Gateway diagnosis call, re-running an
agent/workflow, and delivering a proposed fix into the approvals inbox — is inverted
through the `HealingHost` trait, so the crate has **zero dependency on `apps/core`**.

## Placement (Core vs Gateway)

Orchestrating diagnose→propose and the attempt-cap/cooldown state runs in the
`ryu-healing` sidecar; the approvals write and the re-runs stay Core-owned behind
kernel capabilities. The diagnosis LLM call routes through the Gateway via
`HealingHost::call_side_model`, so it is firewalled / DLP'd / budgeted / audited —
*what is allowed and measured* stays Gateway-side.

## Key API

- `HealEngine` — the engine; the `/report-failure` subscription route calls
  `evaluate(source_id, HealSource, instruction, failure)` and then `apply_verdict`.
- `HealingHost` (trait) — the inversion seam; implemented by the sidecar binary
  (`src/main.rs`), whose action methods call Core's kernel capabilities.
- `decide_heal(...) -> HealDecision` — pure, unit-tested loop-prevention (never-heal-a-
  heal via the `healrun_` prefix, per-source cap, growing cooldown, give-up→escalate).
- `resolve_config` / `HealingConfigView` — client-safe resolved config.
- `routes()` / `HealingCtx` — the `/config` (get/post) + `/status` axum router.
- `truncate_context` — crate-owned length policy Core applies before reporting.

## Consumed as

Compiled-into-core crate (path dependency), merged into Core's axum router.
Attempt caps persist to `~/.ryu/healing-attempts.json` and survive a restart.

## Swap-seam

Defaults live in the `resolve_*` functions (master switch default ON;
`healing.auto-decide` default OFF = propose, dispose; max-attempts 2; 60s cooldown;
diagnose model falls back to `HealingHost::default_diagnose_model`). Nothing is
hardcoded — every knob is a `healing.*` preference, and the diagnosis model routes
through the swappable Gateway.
