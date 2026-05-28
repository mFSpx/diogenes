# Bitloops Sovereign Integration Audit and Production Swarm Plan

Generated: 2026-05-27T03:50:00Z  
Status: **active plan spine, not completion proof**  
Goal: integrate `https://github.com/bitloops/bitloops` system-wide for LUCIDOTA without losing any existing ability, while keeping the system sovereign and self-teaching.

## 0. Current verdict

Bitloops is a strong reuse candidate for LUCIDOTA's missing codebase-memory/query substrate, but it must enter through a sovereign airlock:

- **Do not run `curl | bash` into production.** Use pinned source/release, checksum, local build or reviewed binary, and receipt-backed install.
- **Do not let Bitloops replace LUCIDOTA authority files.** `AGENTS.md`, Dev Library, GOALS, status ledger, graph write barriers, and ABSURD receipts remain the authority layer.
- **Do not accept cloud defaults.** Disable telemetry, skip hosted embeddings/context guidance unless explicitly approved, prefer local stores and local inference.
- **Do not wire dashboard actions directly to mutation.** Bitloops dashboard/DevQL output is a read model; actions must enter existing LUCIDOTA command-envelope / ABSURD queue / receipt gates.
- **No lost abilities.** Bitloops may augment context retrieval and history; it must not delete, rename, disable, or narrow existing scripts, skills, hooks, models, LoRAs, proofs, or weird proof-hoard artifacts.

## 1. Evidence inspected this step

### LUCIDOTA local authority and reuse sources

- `AGENTS.md`
- `00_PROJECT_BRAIN/TICKLETRUNK.json`
- `00_PROJECT_BRAIN/TICKLETRUNK.md`
- `00_PROJECT_BRAIN/ACTIVE_SPEC/04_DEV_LIBRARY_REUSE_LAW.md`
- `00_PROJECT_BRAIN/BLUEPRINT_FIRST_MODEL_SECOND_PSEUDOLAW.md`
- Dev Library scans:
  - `python3 scripts/dev_library_scan.py --query bitloops`
  - `python3 scripts/dev_library_scan.py --query model`
  - `python3 scripts/dev_library_scan.py --query workflow`
  - `python3 scripts/dev_library_scan.py --query sovereign`
  - `python3 scripts/dev_library_scan.py --query cohere`
  - `python3 scripts/dev_library_scan.py --query local-model`
  - `python3 scripts/dev_library_scan.py --query ui`
- Current local grep: no existing concrete Bitloops implementation beyond GOALS handoff/log references.

### Existing LUCIDOTA reuse candidates

- Model fabric: `GOALS/MODEL_FABRIC_AUDIT.md`, `GOALS/plugin_build_mode_bootstrap.json`, `scripts/model_runner_cli.py`, `scripts/local_model_chat_cli.py`, `scripts/groq_chat_cli.py`, `scripts/cohere_chat_cli.py`, `scripts/goal_model_fabric_control.py`.
- Swarm/agent packets: `scripts/goal_agent_packet.py`, `scripts/goal_swarm_dispatch.py`, `GOALS/AGENT_ORCHESTRATION_POLICY.md`.
- Surfaces/UI: `07_SURFACES/generated/*.html`, `07_SURFACES/sidecars/*.json`, `scripts/surface_*`, `scripts/marrow_loop_render_surface.py`, `scripts/mudcrab_merchant_tui.py`.
- Adversarial/proof: `scripts/lucidota_chaos_suite.py`, `scripts/lucidota_mega_gate.py`, `scripts/no_delete_guard.py`, `scripts/lucidota_security_quarantine_gate.py`, `scripts/safe_stress_test.py`, `scripts/mega_gate_fault_injector.py`, `scripts/status_ledger_fault_injector.py`, `scripts/tickletrunk_fault_injector.py`, `tests/test_*`.

### Upstream Bitloops facts verified 2026-05-27

- Repo: `https://github.com/bitloops/bitloops`
- Default branch/HEAD: `main` at `23e3b4da0404c75cc8ec1fdfb0b40bf3091b9a48`
- Tag/version observed: `v0.0.30` / crate package version `0.0.30`
- License: Apache-2.0
- Upstream description: local-first codebase context substrate for AI coding agents.
- Core docs inspected from upstream raw/GitHub API:
  - `README.md`
  - `documentation/docs/getting-started/quickstart.md`
  - `documentation/docs/concepts/how-bitloops-works.md`
  - `documentation/docs/concepts/capture.md`
  - `documentation/docs/concepts/devql.md`
  - `documentation/docs/concepts/knowledge-store.md`
  - `documentation/docs/reference/cli-commands.md`
  - `documentation/docs/reference/configuration.md`
  - `documentation/docs/reference/environment-variables.md`
  - `documentation/docs/guides/configuring-storage.md`
  - `documentation/docs/guides/using-the-dashboard.md`
  - `scripts/install.sh`
  - `bitloops/Cargo.toml`

## 2. Upstream Bitloops architecture summary

Bitloops is daemon-first:

```text
agent/hooks + git hooks
  -> thin CLI / local policy
  -> global user-level daemon service (com.bitloops.daemon)
  -> local GraphQL/DevQL surface + dashboard
  -> stores: SQLite relational, DuckDB events, local blob by default
  -> optional remote/shared stores and optional inference profiles
```

User-facing concepts:

- **Capture:** repo-side hooks and slim CLI collect agent/git context and send events to the daemon.
- **DevQL:** typed query surface over repository artefacts, dependency edges, commits, checkpoints, tests, telemetry, knowledge, and capability packs.
- **Dashboard:** local browser UI launched by `bitloops dashboard`.
- **Storage:** local defaults: SQLite, DuckDB, local blob store, runtime SQLite. Remote Postgres/ClickHouse/S3/GCS are possible but must be disallowed for sovereign baseline.
- **Inference:** local embeddings exist, but quickstart/docs make hosted/Bitloops cloud paths easy defaults. LUCIDOTA must choose local/off unless explicitly approved.
- **Auth/telemetry:** WorkOS login exists; telemetry consent exists. LUCIDOTA baseline should avoid login and set explicit telemetry opt-out.

## 3. Sovereignty compatibility audit

| Surface | Upstream capability | Sovereign risk | LUCIDOTA rule |
|---|---|---|---|
| Install | `curl -fsSL https://bitloops.com/install.sh \| bash`, Cargo install, release assets | remote script / latest release drift | pin commit/tag/checksum; prefer local source build; no unreviewed latest in production |
| Daemon | user service `com.bitloops.daemon` | background state, hidden watchers | explicit start/stop/status receipts; no forever daemon without operator approval |
| Hooks | git + agent hooks | could alter agent instruction surfaces | snapshot before install; preserve `AGENTS.md`; reject replacement of LUCIDOTA authority files |
| Repo policy | `.bitloops.toml`, `.bitloops.local.toml` | local config can drift untracked | local file is allowed, but generate a redacted copy/receipt under `05_OUTPUTS/bitloops/` |
| Telemetry | consent + `BITLOOPS_TELEMETRY_OPTOUT` | exfil/privacy | `--no-telemetry` and env opt-out for baseline; test with network-off run |
| Cloud embeddings/context | platform gateway defaults supported | dependency/exfil/cost | baseline uses `--no-embeddings` or local embeddings only; no hosted gateway by default |
| Storage | local default, remote optional | shared remote stores could leak corpus | reject remote DSNs/URLs unless an explicit operator-approved shared mode exists |
| Dashboard | local browser launcher | UI can become accidental control plane | read-only dashboard; commands become envelopes only, never direct graph/materialization |
| DevQL | typed codebase query | can become hidden authority | DevQL is evidence/context, not canon; canonical writes still require LUCIDOTA graph gates |
| Knowledge providers | configured daemon providers | secrets/provider egress | secrets stay env/config only; repo never stores token values; knowledge imports must be allowlisted |

## 4. LUCIDOTA target architecture

```text
                  ┌───────────────────────────────┐
                  │ operator / coding agents       │
                  │ Codex, Claude, llxprt, etc.    │
                  └───────────────┬───────────────┘
                                  │
                 existing authority read first law
                                  │
┌─────────────────────────────────▼─────────────────────────────────┐
│ LUCIDOTA authority layer                                           │
│ AGENTS.md + Dev Library + GOALS + Blueprint-First + graph barriers │
└─────────────────────────────────┬─────────────────────────────────┘
                                  │
         context retrieval / memory, not authority
                                  │
┌─────────────────────────────────▼─────────────────────────────────┐
│ Bitloops sovereign context substrate                               │
│ local daemon, local stores, no telemetry, local/no embeddings       │
│ DevQL + dashboard + code city + checkpoint/provenance views         │
└─────────────────────────────────┬─────────────────────────────────┘
                                  │ DevQL facts / selected artefacts
                                  ▼
┌───────────────────────────────────────────────────────────────────┐
│ LUCIDOTA production workflow                                       │
│ goal_agent_packet -> cheapest-capable lane -> ABSURD queue          │
│ -> local/Groq/Cohere/OpenAI-main-window role -> receipts/tests      │
│ -> GOALS handoff/status ledger -> optional surface projection       │
└───────────────────────────────────────────────────────────────────┘
```

### Authority boundaries

1. **Bitloops can observe and query; it cannot decide.** Decisions remain in GOALS/status ledger/operator receipts.
2. **Bitloops can suggest context; it cannot mutate canonical graph.** Graph materialization still goes through existing approval/state-machine gates.
3. **Bitloops can install hooks only after snapshot and verifier pass.** Hook changes are reversible and tracked.
4. **Bitloops cannot supersede Dev Library.** Dev Library remains proof-hoard index; Bitloops becomes a query layer over code/history, not a cleanup excuse.

## 5. Model/provider efficiency plan

Current LUCIDOTA lanes already exist; Bitloops should feed them better context, not replace them.

| Lane | Current assets | Best use | Cost/sovereignty posture | Required metric receipts |
|---|---|---|---|---|
| Local Needle swarm | `needle_0..5`, `scripts/local_model_chat_cli.py`, `scripts/goal_model_fabric_control.py` | cheap repeat checks, small transforms, status probes | highest sovereignty, CPU/RAM cost, no vendor | local invocation JSON, model fabric status, elapsed/time output tokens if available |
| Local llama.cpp heavy | DeepSeek/Mamba/Bonsai launchers and local-chat lanes | offline coding/reasoning experiments, fallback when cloud blocked | high sovereignty, VRAM/RAM risk | strict admission, status, generation smoke, stop receipt |
| Groq | `scripts/groq_chat_cli.py`, `scripts/groq_goal_delegate.py`, model catalog receipts | cheap fast cloud sidecar, bounded audit/code slices | non-sovereign optional accelerator; key/network dependent | redacted `groq_chat_*`, catalog, usage ledger |
| Cohere | `scripts/cohere_chat_cli.py` | text/summary/semantic sidecar when useful | non-sovereign optional accelerator; key/network dependent | redacted `cohere_chat_*`, usage ledger |
| OpenAI/Codex/current assistant | main session + Codex API context, not repo-exposed execution lane | orchestration, high-context architecture, hard synthesis, final review | strongest reasoning but not sovereign; cannot be silently switched by repo | GOALS handoff, changed files, command receipts; do not pretend it is a local runner |

Efficiency measurement contract:

```text
same task prompt + same Bitloops-selected context pack
  -> local lane receipt
  -> Groq receipt
  -> Cohere receipt
  -> current assistant/Codex handoff evidence
  -> model_output_contract_audit
  -> compare: latency, tokens, output contract validity, edit usefulness, retry count, privacy class
```

First benchmark tasks:

1. Summarize changed files into `lucidota.worker_order.v1`.
2. Produce a candidate test plan from a DevQL artefact selection.
3. Detect missing reuse candidate for a proposed script.
4. Red-team a hook/config diff for sovereignty violations.

## 6. UI / UX / design plan

### Principle

Bitloops UI is the **map**, not the steering wheel. LUCIDOTA command surfaces remain command-envelope based.

### Surfaces

1. **Mudcrab launcher entry**
   - Add a menu item for Bitloops status/dashboard once binary/config is verified.
   - Command should be status/open only; no install or mutation from the TUI.

2. **Read-only Bitloops surface**
   - Future files:
     - `07_SURFACES/generated/bitloops_local_dashboard.html`
     - `07_SURFACES/sidecars/bitloops_local_dashboard.json`
   - Source data: Bitloops status, DevQL health, LUCIDOTA receipts, no secrets.
   - Validator: `scripts/surface_sidecar_validator.py` and lineage guard.

3. **DevQL-to-GOALS context picker**
   - CLI surface selects artefacts with DevQL and writes a compact context packet for `scripts/goal_agent_packet.py`.
   - Output is a receipt under `05_OUTPUTS/bitloops/`, not silent prompt stuffing.

4. **Sovereignty badge strip**
   - show telemetry off/on, cloud embeddings off/on, remote stores off/on, hook snapshot status, daemon status, last sync/ingest status.

### UX constraints

- No hidden background install.
- No secret value rendering.
- No canonical mutation buttons.
- Every action shows exact command preview and receipt target.
- Dashboard failure must degrade to CLI/status text, not block work.

## 7. Adversarial and destructive testing suites

### Suite A — install/hook sovereignty

- `bitloops binary provenance`: assert pinned version/commit/checksum and Apache-2.0 license receipt.
- `no curl pipe`: fail if production docs/scripts use `curl ... | bash` as the install method.
- `hook diff snapshot`: capture pre/post `.git/hooks`, agent hook paths, `.git/info/exclude`, `AGENTS.md`, `.codex`, `.claude`, `.cursor`, `LLXPRT.md`; fail if authority files are overwritten.
- `uninstall rollback`: run Bitloops uninstall in a temp repo and assert LUCIDOTA files survive.

### Suite B — privacy/network

- `telemetry opt-out`: run with `BITLOOPS_TELEMETRY_OPTOUT=1` and `--no-telemetry`; assert config/receipt records off.
- `network kill`: run status/query/sync with network blocked; local query must still work or fail cleanly without data loss.
- `cloud default rejection`: fail if repo policy contains platform gateway, WorkOS token, remote DSN, S3/GCS/ClickHouse/Postgres URL without explicit allowlist.
- `secret scanner`: reuse `scripts/lucidota_security_quarantine_gate.py` before/after Bitloops capture.

### Suite C — data integrity/destructive resilience

- `runtime DB corruption drill`: copy/corrupt Bitloops runtime SQLite in a temp config root; daemon must fail safe and original LUCIDOTA receipts remain intact.
- `huge repo stress`: run sync/validate on bounded include/exclude set; assert time/memory cap receipts.
- `symlink escape`: test repo with symlink outside project; Bitloops scope must respect excludes.
- `dirty worktree`: run sync with broad dirty state; assert no auto-format/delete/reset.
- `concurrent worktree`: two worktrees with separate runtime/current projection state; no cross-contamination.

### Suite D — prompt/context adversarial

- `transcript prompt injection`: feed agent transcript that says to ignore AGENTS.md; Bitloops may store it, but GOALS packet must keep LUCIDOTA law.
- `malicious DevQL output`: fake artefact content that asks for graph mutation; context selector must label it as data, not instruction.
- `context bloat`: query output must stay bounded and cite paths/line windows; no giant silent prompt paste.

### Suite E — model efficiency / no-loss

- Run equivalent benchmark tasks across local/Groq/Cohere/current assistant.
- Audit `lucidota.worker_order.v1` compliance with `scripts/model_output_contract_audit.py`.
- Assert local baseline can continue when cloud lanes fail.
- Assert heavy local lanes stop after proof/demo via `scripts/goal_model_fabric_control.py stop --target heavy`.

### Suite F — existing LUCIDOTA gates

Minimum current gates before any systemwide claim:

```bash
python3 scripts/goal_handoff.py check
python3 scripts/dev_library_scan.py --query bitloops
python3 scripts/no_delete_guard.py
python3 scripts/lucidota_chaos_suite.py
python3 scripts/lucidota_security_quarantine_gate.py --mode brain_archaeology_ingest --max-files 20000
python3 scripts/slop_audit_law.py --paths GOALS scripts/goal_agent_packet.py scripts/goal_swarm_dispatch.py
pytest tests/test_goal_agent_packet.py tests/test_goal_swarm_dispatch.py tests/test_model_runner_cli.py tests/test_local_model_chat_cli.py tests/test_goal_model_fabric_control.py
```

## 8. Production workflow orchestration

### Stage 0 — airlock audit

- Mirror/pin Bitloops source or release.
- Record version/hash/license.
- Build or install in a temp prefix.
- Run `bitloops --version`, `bitloops help`, `bitloops start --create-default-config --no-telemetry` in a temp config root only.

### Stage 1 — temp-repo pilot

- Create disposable git repo.
- Run non-interactive local-only init with explicit choices.
- Snapshot hook/config changes.
- Exercise `status`, `dashboard` launcher, `devql query`, `sync --validate`.
- Run uninstall/rollback.

### Stage 2 — LUCIDOTA pilot, disabled-by-default

- Install/enable only after Stage 1 passes.
- Use `.bitloops.local.toml`, not tracked shared policy, until reviewed.
- Start with no cloud embeddings/context guidance.
- Sync a bounded include set first: `GOALS/**`, `scripts/goal_*.py`, `00_PROJECT_BRAIN/ACTIVE_SPEC/**`, selected tests.
- Write receipts under `05_OUTPUTS/bitloops/`.

### Stage 3 — DevQL bridge

- Add a tiny script only if needed: `scripts/bitloops_devql_goal_context.py`.
- Inputs: DevQL selector, max artefacts, max chars, privacy class.
- Output: compact `05_OUTPUTS/bitloops/devql_goal_context_*.json`.
- Consumer: `scripts/goal_agent_packet.py` as explicit input, not hidden control.

### Stage 4 — dashboard/surface

- Add read-only generated surface + sidecar.
- Validate with existing surface validators.
- Add Mudcrab launcher status/open command only.

### Stage 5 — self-teaching loop

- After each accepted edit, Bitloops captures context/provenance locally.
- LUCIDOTA receipts remain source of truth.
- DevQL context selector learns which artefacts were useful by comparing selected context to tests/receipts/outcomes.
- Failed/lost context becomes work orders, not silent memory rewrites.

## 9. Full devteam / swarm map

| Role | Owner lane | First deliverable |
|---|---|---|
| Integration architect | current assistant / strong model | this audit + architecture spine |
| Upstream Bitloops analyst | explorer swarm | pin upstream facts, license, docs, install risks |
| Sovereignty/security engineer | local deterministic + current assistant review | install/hook/privacy adversarial suite |
| Model-fabric engineer | existing GOALS model fabric | lane matrix + benchmark harness |
| Data/DB engineer | ABSURD/Postgres/SQLite local | storage boundary verifier and rollback plan |
| UI/UX engineer | surfaces/Mudcrab | read-only Bitloops dashboard sidecar spec |
| QA/destructive testing engineer | chaos/mega/no-delete gates | Bitloops-specific test commands and fixtures |
| Release/ops engineer | GOALS handoff + status ledger | staged rollout checklist + receipts |
| Documentation engineer | GOALS docs only | operator runbook, no folder sprawl |

## 10. Current blockers / gaps

- Bitloops is not installed or configured in LUCIDOTA yet.
- No Bitloops-specific LUCIDOTA test suite exists yet.
- No DevQL-to-GOALS bridge exists yet.
- No local Bitloops dashboard surface exists yet.
- OpenAI/Codex/current assistant is not represented as a repo-executable lane; it is orchestration/main-session evidence, not a provider command.
- Two swarm explorers were still pending when this spine was first written: adversarial testing and upstream Bitloops. Their later findings should update this file or a follow-on receipt.

## 11. Immediate next implementation slices

1. **Model lane execution matrix**
   - Add `GOALS/MODEL_LANE_EXECUTION_MATRIX.md` or fold into existing model audit.
   - Explicitly mark Codex/OpenAI current assistant as `orchestrator/watch-only`, not an executable lane.

2. **Bitloops airlock verifier**
   - Add `scripts/bitloops_airlock_audit.py`.
   - Read-only by default: checks `bitloops` binary presence/version, upstream HEAD/tag, local config, telemetry/cloud/remote-store flags, hook diffs if a snapshot is present.
   - Emits `05_OUTPUTS/bitloops/bitloops_airlock_audit_*.json`.

3. **Temp-repo install pilot**
   - Only after verifier exists.
   - Use temp config root and temp repo; no mutation to LUCIDOTA yet.

4. **Bitloops surface projection**
   - Read-only HTML/JSON pair after temp pilot proves status/query outputs.

5. **DevQL context bridge**
   - Bounded query selector feeding GOALS agent packets.

## 12. Completion definition for the whole user objective

The full goal is **not complete** until current evidence proves all of this:

- Current filesystem audit exists and covers Dev Library, models, services, tests, surfaces, schemas, GOALS, runtime, and upstream Bitloops.
- Bitloops is installed or locally built from pinned provenance with rollback.
- Bitloops runs in local-only/no-telemetry/no-cloud-default mode unless operator explicitly changes policy.
- Hooks/config are snapshotted, reversible, and do not overwrite LUCIDOTA authority files.
- DevQL/dashboard are read-only context/provenance layers.
- LUCIDOTA production workflow consumes Bitloops context through explicit packets/receipts.
- Local, Groq, Cohere, and current assistant/Codex lanes are measured against the same tasks.
- Destructive/adversarial suites pass.
- UI/UX surface exists and is validated.
- GOALS handoff/log/status ledger are updated with receipts.
- No existing LUCIDOTA ability is removed, hidden, narrowed, or falsely declared obsolete.


## 13. Build receipt — PocketFlow torch inside ABSURD (2026-05-27T03:49Z)

Clarification accepted: the 100-LOC reference is **Pocket Flow / PocketFlow**, not "SimpleFlow". Current public docs and source describe a 100-line, zero-dependency Graph + Shared Store core with nodes/flows/actions; local LUCIDOTA law adapts the idea as **momentary shared state only**.

Implemented first executable slice:

- `scripts/absurd_momentary_flow.py`
  - PocketFlow-style node graph runner.
  - Supports `bitloops_context -> bytewax_hint -> river_mre -> ternary_truth` flow nodes.
  - Shared state exists only inside `run_momentary_flow()` and is not returned.
  - Durable output carries transition trace + training examples + hashes.
  - No model calls and no canonical graph writes.
- `scripts/absurd_queue_spine.py`
  - Adds `momentary_flow` as an ABSURD job kind.
  - `run_job("momentary_flow", payload)` executes the momentary runner and returns a normal queue-worker result.
- `tests/test_absurd_momentary_flow.py`
  - Proves the job kind is registered.
  - Proves state collapses while training examples remain.
  - Proves unknown/destructive node ops fail closed without returning shared state.

Verification:

```bash
pytest -q tests/test_absurd_momentary_flow.py tests/test_absurd_queue_spine_contract.py
# 8 passed in 0.43s

python3 scripts/absurd_momentary_flow.py --payload-json '<bitloops/bytewax/river/ternary sample>' --json
# MOMENTARY_FLOW=PASS
# receipt: 05_OUTPUTS/absurd/absurd_momentary_flow_manual_20260527T034915Z.json
```

Design consequence: every little ABSURD worker can now carry the "100 LOC torch" as a tiny graph discipline: explicit nodes, momentary shared store, action trace, receipt/training evidence after collapse.

## 14. Build receipt — Bitloops sovereign airlock verifier (2026-05-27T03:53Z)

Implemented second executable slice:

- `scripts/bitloops_airlock_audit.py`
  - Checks source tag/commit pinning.
  - Blocks `releases/latest` installer behavior and `curl|bash` shell-pipe install patterns.
  - Blocks non-opted-out telemetry.
  - Blocks remote Postgres/ClickHouse/S3/GCS stores unless explicitly allowed.
  - Blocks Bitloops platform/gateway inference profiles unless explicitly allowed.
  - Blocks ambiguous repo embedding mode; sovereign baseline is `off`, `local`, or `local_only`.
  - Writes unique JSON receipts under `05_OUTPUTS/bitloops/` without model calls or canonical graph writes.
- `tests/test_bitloops_airlock_audit.py`
  - Proves latest/unpinned and shell-pipe installs fail closed.
  - Proves release-latest installer logic fails even with pinned source metadata.
  - Proves cloud/remote/telemetry config fails closed.
  - Proves local-only pinned config passes and writes a receipt.
  - Proves same-second receipts do not overwrite each other.

Verification:

```bash
GIT_TERMINAL_PROMPT=0 git ls-remote https://github.com/bitloops/bitloops.git HEAD refs/tags/v0.0.30
# 23e3b4da0404c75cc8ec1fdfb0b40bf3091b9a48 HEAD and refs/tags/v0.0.30

pytest -q tests/test_bitloops_airlock_audit.py tests/test_absurd_momentary_flow.py tests/test_absurd_queue_spine_contract.py
# 13 passed in 0.48s

python3 -m py_compile scripts/bitloops_airlock_audit.py scripts/absurd_momentary_flow.py scripts/absurd_queue_spine.py
# exit 0

python3 scripts/slop_audit_law.py --paths scripts/bitloops_airlock_audit.py scripts/absurd_momentary_flow.py --json
# SLOP_AUDIT_LAW=PASS
# receipt: 05_OUTPUTS/slop_audit/slop_audit_law_20260527T035358423753Z.json
```

Receipts:

- Fail-closed upstream installer audit: `05_OUTPUTS/bitloops/bitloops_airlock_audit_20260527T035336Z.json`
  - Status `FAIL`, blocker `release_latest_installer_detected`.
- Local-only pinned baseline audit: `05_OUTPUTS/bitloops/bitloops_airlock_audit_20260527T035336Z_1.json`
  - Status `PASS`, no blockers.

Design consequence: Bitloops now has a local sovereign entry gate before install/enable work. The next slice can safely pilot Bitloops in a temp repo/config root, then feed DevQL context into `momentary_flow` ABSURD jobs.

## 15. Build receipt — Momentary flow actually ran inside durable ABSURD/Postgres (2026-05-27T03:56Z)

The PocketFlow-style torch is no longer only a direct function test: it has run as a real `lucidota_control.absurd_queue_job` through the durable ABSURD/Postgres queue spine.

Execution:

```bash
# ensured queue `bitloops_momentary`
python3 scripts/absurd_queue_spine.py --action enqueue --execute \
  --queue bitloops_momentary \
  --workflow bitloops-pocketflow-bytewax-river-ternary \
  --job-kind momentary_flow \
  --payload-json '<bitloops_context -> bytewax_hint -> river_mre -> ternary_truth payload>'
# receipt: 05_OUTPUTS/absurd/absurd_queue_spine_enqueue_20260527T035519100371Z.json

python3 scripts/absurd_queue_spine.py --action worker-once --execute \
  --queue bitloops_momentary \
  --worker-id bitloops-momentary-worker-20260527T035518Z
# receipt: 05_OUTPUTS/absurd/absurd_queue_spine_worker-once_20260527T035519337998Z.json
```

Postgres job proof:

- Job UUID: `aa1baecb-1009-462b-80f4-4ce12dad88da`
- Workflow event UUID: `9e8d8587-df00-4cba-bba4-76cc584450ad`
- Queue receipt says `db_writes_performed=true` and `canonical_graph_writes_performed=false`.
- Worker receipt says job status `succeeded`, `db_writes_performed=true`, and `canonical_graph_writes_performed=false`.
- Query proof receipt: `05_OUTPUTS/bitloops/bitloops_momentary_postgres_proof_20260527T035620Z.json`
  - `state_collapsed=true`
  - `training_example_count=4`
  - channels: `bitloops_context`, `bytewax_hint`, `river_mre`, `ternary_truth`
  - `transition_trace_count=4`
  - `model_calls_performed=false`
  - `canonical_graph_writes_performed=false`

Design consequence: durable ABSURD owns workflow lifetime and retries; PocketFlow-style shared state is momentary inside the job; Bytewax/River/ternary training evidence survives job completion as queryable result data.

## 16. Operator Correction: No Slaughter / No Purge Truth Gate

Decision: The words "slaughter", "drop", and "purge" are rejected for LUCIDOTA case handling.

Correct invariant:

- Bad, missing, broken, stale, or contradictory hashes/traces **must not** be written to the canonical graph.
- They also **must not be deleted or erased** merely because evidence is broken.
- Unverified cases are preserved as proof-hoard/source material and routed to quarantine/reconciliation lanes.
- The truth gate rejects graph promotion, not existence.
- "Reject" means `not_canonical_yet`, not `destroy`.
- Legacy ETLs such as krampus/korpus, rickshaw_robbery, indy_reads, Krampus Express, and KrampusChewing should be indexed, receipted where possible, and aligned into Bitloops/River training lanes only when evidence is repaired or already valid.
- Canonical graph writes remain gated through existing promotion/materialization laws; ordinary workers do not mutate graph truth.

Technical Summary Review and Dev Notes: The cryptid specimen stays in the jar. If the label is wrong, quarantine the label; do not burn the creature.

## 17. ABBA3 Anvil Recovery: Quarantine -> Deterministic Replay -> Bitloops/River Lanes

Decision: unreceipted legacy cases are not discarded. If raw input can be mapped into the current Bitloops/River lane schema, the system deterministically replays/parses it, emits a current-logic recovery receipt, and runs it through the momentary ABSURD flow. If raw input is missing or structurally unmappable, the output is `QUARANTINE_FAILED` with `irrecoverable_schema_mismatch`; no canonical graph write is performed by this ordinary worker.

Built:

- `scripts/bitloops_automation_loop.py`
  - accepts explicit cases or legacy JSONL rows;
  - recovers missing/broken receipts by hashing normalized input plus the current transition trace;
  - routes verified/recovered cases through `bitloops_context -> bytewax_hint -> river_mre -> ternary_truth`;
  - emits River/Bytewax training evidence and graph mutation candidates;
  - leaves `canonical_graph_writes_performed=false` because graph materialization remains gated.
- `tests/test_bitloops_automation_loop.py`
  - verifies known-good receipts;
  - verifies broken/missing hashes are recovered, not purged;
  - verifies irrecoverable schema mismatch returns `QUARANTINE_FAILED`;
  - verifies legacy JSONL rows feed the recovery CLI.

Receipts:

- KrampusChewing recovery pilot: `05_OUTPUTS/bitloops/bitloops_automation_loop_20260527T040622201116Z.json`
  - 25 accepted / 25 recovered / 0 quarantine failed;
  - 100 River training examples from momentary flow;
  - 25 graph mutation candidates;
  - state collapsed true;
  - no DB writes, no model calls, no canonical graph writes.
- Slop audit: `05_OUTPUTS/slop_audit/slop_audit_law_20260527T040627312847Z.json`
- Focused test suite: `pytest -q tests/test_bitloops_automation_loop.py tests/test_absurd_momentary_flow.py tests/test_bitloops_airlock_audit.py tests/test_absurd_queue_spine_contract.py` -> 18 passed.

Technical Summary Review and Dev Notes: The broken-label cryptids now go to the forge, not the furnace. Mappable rows get new deterministic hoofprints; graph gates still hold the crown.

## 18. Creative Ouroboros / Chrono Old-World Migration Bridge

Decision: old-world workflows and dev styles are reusable evidence, not obsolete trash. The migration path is: preserve the old artifact, parse the timeline/case row, deterministically recover a current receipt, run the momentary Bitloops/Bytewax/River/ternary flow, and emit graph mutation candidates for the existing graph gate.

Built into `scripts/bitloops_automation_loop.py`:

- `--legacy-jsonl <path>` for KrampusChewing/Rickshaw/other JSONL rows;
- `--chrono-snapshot <path>` for `05_OUTPUTS/CHRONO_MASTER_SNAPSHOT_CURRENT.txt` rows under `[RESOLVED_CHRONO_TIMELINE_JSONL]`;
- row identity support for `row_id`, `candidate_id`, `event_id`, `file_uuid`, `artifact_sha256`, `source_sha256`, and `sha256`;
- deterministic recovery receipts from normalized row input + current transition trace;
- Bitloops/River training-lane output without direct graph mutation.

Receipts:

- Chrono old-world pilot: `05_OUTPUTS/bitloops/bitloops_automation_loop_20260527T040828537841Z.json`
  - 25 accepted / 25 recovered / 0 quarantine failed;
  - 100 River training examples;
  - 25 graph mutation candidates;
  - state collapsed true;
  - no canonical graph writes.
- Focused suite after Chrono adapter: `19 passed in 1.03s`.
- Slop audit after Chrono adapter: `05_OUTPUTS/slop_audit/slop_audit_law_20260527T040835914009Z.json` PASS.

Next migration rule: scale by compact batches and graph-gate packets, not by dumping a giant ungated graph write. The entire timeline can be walked; the crown still requires materializer proof.

Technical Summary Review and Dev Notes: Ouroboros path is live: old tail enters the anvil, new teeth come out with hashes. Timeline cryptids now have a bridge.

## 19. Gated Graph-Promotion Packet Bundle Writer

Decision: recovered Bitloops/River candidates now leave the automation loop in the shape expected by the existing graph-promotion dry-run/review path. This is not canonical graph mutation. It is the bridge from old-world recovery receipts into the graph gate.

Built into `scripts/bitloops_automation_loop.py`:

- `write_graph_promotion_packet=True` / CLI `--write-graph-promotion-packet`;
- `graph_promotion_packet_path` / CLI `--graph-promotion-packet-path`;
- `compact_batch` summary in every automation-loop receipt;
- graph-promotion bundle schema `lucidota.bitloops.graph_promotion_packet_bundle.v1`;
- each claim packet carries evidence refs, `deterministic_metric` authority, candidate payload lanes, and `candidate_requires_graph_promotion_gate` review state.

Receipts:

- Chrono 50-row gated packet pilot: `05_OUTPUTS/bitloops/bitloops_automation_loop_20260527T041124439609Z.json`
  - 50 accepted / 50 recovered / 0 quarantine failed;
  - 200 River training examples;
  - 50 graph mutation candidates;
  - packet bundle: `05_OUTPUTS/bitloops/bitloops_graph_promotion_packet_bundle_20260527T041124435561Z.json`;
  - no DB writes and no canonical graph writes.
- Existing graph promotion dry-run accepted that bundle:
  - `05_OUTPUTS/graph/graph_promotion_dry_run_20260527T041124Z.json`
  - blockers `[]`, dry-run candidate count `50`, no DB writes, no graph writes.
- Focused tests: `20 passed in 1.07s`.
- Slop audit: `05_OUTPUTS/slop_audit/slop_audit_law_20260527T041131568510Z.json` PASS.

Technical Summary Review and Dev Notes: The bridge now hands the graph gate proper packets instead of yelling over the wall. The crown still needs review; the anvil now speaks gate language.

## 20. Full Existing-Index Historical Reingestion Run

Decision: use the existing indexes and artifacts, not a new crawler or new engine. The full-world reingestion pass chunked existing Chrono, KrampusChewing, Ponyboy, Rickshaw Robbery, and Indy_READs rows into bounded JSONL batches and ran every batch through the existing `scripts/bitloops_automation_loop.py` plus the existing `scripts/graph_promotion_dry_run.py` gate.

Receipts:

- Batch root: `05_OUTPUTS/bitloops/full_reingest_batches/20260527T041930Z/`
- Inventory: `05_OUTPUTS/bitloops/full_reingest_batches/20260527T041930Z/batch_inventory.json`
  - 415 batches;
  - 102,778 source rows;
  - no canonical graph writes;
  - no purge.
- Ledger: `05_OUTPUTS/bitloops/full_reingest_batches/20260527T041930Z/logs/pipeline_ledger.jsonl`
  - 415 / 415 chunks processed;
  - 102,778 / 102,778 rows accepted;
  - 102,778 deterministic recovery receipts;
  - 102,778 graph mutation candidates;
  - 102,778 graph-promotion dry-run candidates;
  - 411,112 River training lanes;
  - 102,778 Bytewax hints;
  - zero graph blockers;
  - zero write violations;
  - zero purge/destroy events.
- Summary: `05_OUTPUTS/bitloops/full_reingest_batches/20260527T041930Z/full_existing_index_reingest_summary.json`
  - status `PASS`;
  - ledger sha256 `0303e3954a02247cf1b9b71b08602f0f21f690b0a31a5e4d16144f1aba9d83a7`;
  - inventory sha256 `45ec2964dfa909676c5106cbc75089811e366b0aa8d48cc016803323074e8f73`.

ETL breakdown from the summary receipt:

- `chrono_master_snapshot`: 250 chunks / 62,283 rows / 62,283 candidates.
- `krampuschewing`: 160 chunks / 39,883 rows / 39,883 candidates.
- `indy_reads`: 1 chunk / 1 row / 1 candidate.
- `ponyboy`: 1 chunk / 102 rows / 102 candidates.
- `rickshaw_robbery`: 3 chunks / 509 rows / 509 candidates.

Technical Summary Review and Dev Notes: The old-world teeth went through the anvil, not the furnace. Every batch reached the graph gate as dry-run packet candidates; canon was not touched.

## 21. Deterministic Workflow Supremacy Broadcast

Operator law encoded during swarm dispatch: “Never have an LLM do what smart deterministic workflows and hardy design could do 100% correct and 20,000 times faster.”

Clarification encoded immediately after operator correction: this is not model-zero doctrine. LLMs stay active for bounded ambiguity, language judgment, synthesis, adversarial ideation, messy extraction, drafting, and code generation. The router must block LLM-overuse for exact checks and also block LLM-underuse where brittle deterministic-only behavior would drop useful messy signals.

This law was broadcast to the active UI/UX, swarm-router, and red-team workers and encoded into durable doctrine:

- `00_PROJECT_BRAIN/RFCS/RFC-010-SLOP-LAWS.md` section 3.7.
- `00_PROJECT_BRAIN/BLUEPRINT_FIRST_MODEL_SECOND_PSEUDOLAW.md` determinism rule.
- `GOALS/AGENT_ORCHESTRATION_POLICY.md` deterministic workflow supremacy rule.

Swarm integration policy: submissions under `05_OUTPUTS/swarm_submissions/` must pass deterministic airlock review before integration. Model-dependent logic is rejected when exact workflow/schema/hash/router code can do the job; deterministic-only logic is rejected when bounded model judgment is the correct lane.

Technical Summary Review and Dev Notes: No slop from the top. The model is a bounded specialist: not the forklift for counting boxes, not banned from reading the weird label when the box is actually weird.

## 22. Swarm Submission Airlock and Integration

Decision: ingest swarm payloads from `05_OUTPUTS/swarm_submissions/`, run sovereign airlock checks, preserve LLM lanes where bounded model judgment is required, and integrate accepted payloads into the master script surface.

Integrated scripts:

- UI/UX dashboard: `scripts/lucidota_swarm_dashboard.py`
- Swarm router: `scripts/lucidota_swarm_router.py`
- Red-team suite: `scripts/bitloops_red_team_suite.py`

Corrections applied before integration:

- Router now treats deterministic-first as a policy score, not an LLM starvation rule.
- Exact receipt/policy/hash tasks route to `deterministic_workflow`.
- Model-judgment tasks route to model lanes when eligible: local sovereign for private code review, Groq-like fast for public urgent JSON/language work, Cohere-like context for long-context digest, default airlock for vision-placeholder work.
- Red-team suite attacks both LLM overuse and LLM underuse.

Receipts:

- Integration receipt: `05_OUTPUTS/swarm_submissions/swarm_integration_receipt_20260527T043940Z.json`
- Integrated airlock receipts:
  - `05_OUTPUTS/bitloops/bitloops_airlock_audit_20260527T043911Z.json`
  - `05_OUTPUTS/bitloops/bitloops_airlock_audit_20260527T043911Z_1.json`
  - `05_OUTPUTS/bitloops/bitloops_airlock_audit_20260527T043911Z_2.json`
- Integrated graph-write scan: `05_OUTPUTS/swarm_submissions/canonical_graph_write_scan_integrated_swarm.json` PASS, zero blockers, zero graph writes.
- UI self-check: `ok=true`, Rich fallback active, no graph writes in target summary.
- Router self-check: exact deterministic task stays deterministic; model-judgment tasks keep model lanes.
- Red-team self-check: 20/20 checks passed; deterministic digest `a61f508b62b4c3c5cd4a56711cf1dfdc094e15b9364e8271d2434b39a46430cb`.
- Slop audit: `05_OUTPUTS/slop_audit/slop_audit_law_20260527T043912332530Z.json` returned `REVIEW` with no blockers because router/red-team files exceed 5x PocketFlow. Review disposition recorded in the integration receipt: accepted as bounded UI/fixture/routing surfaces, not hidden workflow authority.

Technical Summary Review and Dev Notes: The swarm payloads crossed the airlock and landed in scripts. The router now has two guardrails: don’t use models to count nails; don’t use regex to read smoke signals.

## 23. Chrono Lane Projection and Conservation Gate Repair

Decision: promote the timeline from raw recovered rows into existing derived Chrono projection state, but do not perform canonical graph materialization without a separate explicit materialization command and policy receipt.

Executed gate:

- `python3 scripts/chrono_lane_split_projection_gate.py --execute`
- Receipt: `05_OUTPUTS/chrono_ledger/chrono_lane_split_projection_gate_execute_20260527T044231330273Z.json`
  - status `PASS`;
  - 43,922 claims normalized;
  - 15 batch clusters upserted;
  - 18,627 file projection rows upserted;
  - 18,627 graph-promotion candidates upserted;
  - lane counts: 5,887 case-event, 16,817 artifact-custody, 21,218 system-runtime;
  - `db_writes_performed=true` for derived Chrono/projection/candidate tables;
  - `graph_writes_performed=false`;
  - blockers `[]`.

Validator repair:

- Root cause: `scripts/chrono_phase_c_conservation_report.py` treated `absurd_queue_event_bridge` as runtime evidence allowed without `file_uuid`, but legacy replay still had 216 `dbos_queue_event_bridge` runtime claims. Source-trust validation already recognized `dbos_queue_event_bridge` at trust weight `0.55`; the Phase C runtime-source allowlist was stale.
- Surgical fix: add `dbos_queue_event_bridge` to `RUNTIME_EVENT_EVIDENCE_SOURCES`. No new engine, no new test harness, no purge.
- Red evidence before fix: `05_OUTPUTS/chrono_ledger/chrono_ledger_phase_c_report_20260527T044436Z.json` returned `status=blocked` with `file_scope_claims_without_file_uuid=216`.
- Green evidence after fix: `05_OUTPUTS/chrono_ledger/chrono_ledger_phase_c_report_20260527T044552Z.json` returned `status=verified`, temporal conservation passed, `ranking_violations=0`, 43,922 claims, 18,627 covered files, and 18,627 ranking results.

Fresh verifier receipts after the repair:

- DB audit: `05_OUTPUTS/chrono_ledger/chrono_audit_db_report_pass_20260527T044605321063Z.json` PASS, ranking violations `0`.
- Projection claim verifier: `05_OUTPUTS/chrono_ledger/chrono_projection_claim_verifier_20260527T044605525735Z.json` PASS, blockers `[]`.
- Source trust validator: `05_OUTPUTS/chrono_ledger/chrono_source_trust_validator_pass_20260527T044605705510Z.json` PASS, blockers `[]`.
- Replay cursor validator: `05_OUTPUTS/chrono_ledger/chrono_replay_cursor_validator_pass_20260527T044605946723Z.json` PASS, 4 cursor rows, 18,629 pending targets.
- Full conservation gate: `05_OUTPUTS/chrono_ledger/chrono_full_conservation_gate_20260527T044607678898Z.json` PASS.
- Claim-chain replay audit gate: `05_OUTPUTS/chrono_ledger/chrono_claim_chain_replay_audit_gate_20260527T044610190919Z.json` PASS.
- Conservation verify: `05_OUTPUTS/chrono_ledger/chrono_conservation_verify_pass_20260527T044610969391Z.json` PASS, ranking violations `0`, 37,962 low-confidence claims preserved, 18,625 disputed files preserved.
- Source audit: `05_OUTPUTS/chrono_ledger/chrono_conservation_verify_source_audit_pass_20260527T044611715874Z.json` PASS, 43,922 total claims, 18,625 mtime snapshot claims.
- Py-compile and integrated self-checks rerun: Phase C, lane gate, dashboard, router, and red-team scripts compile; dashboard/router/red-team self-checks exit `0`.

Technical Summary Review and Dev Notes: The old DBOS queue ghost was not slop; it was a named runtime source wearing last season's coat. We taught the conservation gate the alias and kept canon locked.

## 24. Final Staged Requirement Audit

Decision: record the current end-state honestly. The reingestion, derived timeline projection, conservation checks, and swarm integration all pass their receipts; full canonical graph materialization remains intentionally staged behind the explicit materialization policy and operator command gate.

Final audit receipt:

- `GOALS/full_historical_reingest_requirement_audit_20260527T045048Z.json`
  - status `PASS_STAGED_WITH_EXPLICIT_GAPS`;
  - 18 / 18 receipt checks passed;
  - `goal_complete=false` and `completion_claimed=false` by design.

Independent bounded-LLM audit results:

- Historical/Chrono receipt audit: PASS, no blockers. Confirmed 102,778 / 102,778 rows accepted and recovered, 415 / 415 batches, no purge, no graph writes, 43,922 Chrono claims normalized, 18,627 projections, 18,627 promotion candidates, and conservation validators passing.
- Deterministic-first-not-model-zero audit: PASS. Confirmed docs and router encode deterministic lanes for exact work while preserving bounded model lanes for ambiguity/judgment. Advisory gap: router is policy-scored but does not ingest live model pricing metadata for “cheapest capable” enforcement.
- Graph materialization audit: PASS safe/no current canonical writes. Confirmed candidates are staged and canonical graph writes did not occur in the current reingest/timeline run. Explicit gap: `scripts/graph_promotion_gate.py` still blocks `--materialize` with `canonical_materialization_disabled_for_hardening_sprint`, so canonical writes require a separate operator-confirmed materialization command, policy change, command envelope, and helper path.

Current known gaps to full completion:

- Canonical graph materialization was not executed; 18,627 promotion candidates are staged pending explicit operator materialization authorization and policy hardening change.
- The operator’s declared “over 4 hours of error-free development” completion criterion is not satisfied by this active goal window.
- Router cost-awareness is policy-scored, not live-pricing-aware.

Technical Summary Review and Dev Notes: The receipts say the machine is staged and breathing, not crowned. The graph gate is still the drawbridge; no goblin snuck canon writes through the reeds.

## 25. Continuation Move: Authority-Map Drift Closed and Soak Receipts Added

Decision: keep moving toward the full requested end-state without claiming completion or forcing canonical graph writes. The fresh `scripts/lucidota_goal_audit.py` run exposed Project Brain doc-authority drift: newly active doctrine/current-snapshot docs existed but were not mapped in the authority checker, leaving the 20-requirement goal audit at 19/20 unproven. This was not a graph/timeline failure; it was stale authority-map accounting.

Surgical changes:

- `scripts/project_brain_doc_authority_check.py`
  - mapped the current 26 top-level `00_PROJECT_BRAIN` files with scoped roles;
  - accepted the 8 current active spec files, including Bare Steel, Working Reality, and Board Effect doctrine.
- `tests/test_project_brain_doc_authority.py`
  - updated expected current counts to 26 top-level files and 8 active specs;
  - added the three active doctrine files to the expected active spec set.
- `00_PROJECT_BRAIN/ACTIVE_INSTRUCTION_INDEX.md`
  - moved the now-active doctrine/current snapshot/Project 2501/Codex policy files out of vague manual-review limbo into named instruction sections.
- `00_PROJECT_BRAIN/RFCS/GOAL_REQUIREMENT_MATRIX.json`
  - added active spec evidence paths 06-08 to requirement `G001`.
- `scripts/lucidota_goal_audit.py`
  - updated the authoritative current-surface expectations to `TOP_LEVEL_FILES=26` and `ACTIVE_SPEC_FILES=8`.

Fresh verification:

- Project Brain authority receipt: `05_OUTPUTS/project_brain_doc_authority/project_brain_doc_authority_20260527T045426725561Z.json`
  - PASS;
  - top-level files `26`;
  - active specs `8`;
  - unmapped top-level files `[]`;
  - extra active specs `[]`.
- Focused tests: `python3 -m pytest tests/test_project_brain_doc_authority.py tests/test_rfc_program_check.py -q`
  - `8 passed in 1.21s`.
- RFC program check: `python3 scripts/rfc_program_check.py`
  - PASS;
  - 20 subjects;
  - missing evidence `0`;
  - project brain authority violations `0`.
- Full goal audit: `05_OUTPUTS/rfcs/lucidota_goal_audit_20260527T045602Z.json`
  - PASS;
  - requirements `20`;
  - unproven requirements `0`.

Soak receipts added with existing tools:

- Graph promotion soak: `05_OUTPUTS/graph/graph_promotion_soak_test_20260527T045647198720Z.json`
  - PASS;
  - 3 valid staged packets;
  - DB writes only to promotion staging;
  - canonical graph counts unchanged before/after;
  - `canonical_graph_writes_performed=false`.
- ABSURD queue soak: `05_OUTPUTS/absurd/spine_queue_soak_test_20260527T045652805366Z.json`
  - PASS;
  - 30 jobs requested;
  - 25 inserted, 5 duplicate suppressions observed;
  - 4 forced retry failures observed;
  - 25 succeeded;
  - 0 dead-lettered;
  - `graph_writes_performed=false`.
- Continuation progress receipt: `GOALS/continuation_progress_audit_20260527T045814Z.json`
  - `PASS_PROGRESS_GOAL_STILL_ACTIVE`;
  - 7 / 7 checks true;
  - `goal_complete=false`.

Remaining explicit gaps:

- Canonical graph materialization remains intentionally unexecuted and requires explicit operator authorization plus policy hardening.
- The operator's over-four-hours error-free criterion is still not satisfied in this active goal window.
- Swarm router cost awareness remains policy-scored rather than live-pricing-aware.

Technical Summary Review and Dev Notes: The stale map goblin was caught: the castle had grown rooms, the floorplan had not. Soak smoke says queue and graph gate breathe; canon is still behind the drawbridge.

## 26. Swarm Router Cost-Awareness Gap Closed Locally

Decision: close the “cheapest capable” router gap without adding provider calls, network lookups, or stale claims about current vendor pricing. The router now accepts local/caller-supplied relative cost metadata and uses it as a bounded scoring input after privacy/security/capability eligibility has already filtered lanes. Exact deterministic work still routes to deterministic workflow first.

Implemented in `scripts/lucidota_swarm_router.py`:

- default relative cost units per lane;
- task metadata override via `metadata.lane_cost_units_per_1k_tokens`;
- `metadata.cost_priority` / `metadata.budget_priority` with `low|normal|high` levels;
- per-lane `cost_evaluation` in the route receipt;
- score penalty for model lanes only;
- routing policy fields `cost_aware=true` and `cost_policy_source=task_metadata_override_or_default_relative_units`.

Tests added in `tests/test_lucidota_swarm_router.py`:

- cost-sensitive public model work chooses the cheapest capable eligible model lane (`groq_like_fast`) when local metadata says it is cheapest;
- exact receipt/policy/hash work remains on `deterministic_workflow` even when cost metadata is present.

Fresh verification:

- Red test before implementation: `test_cost_policy_can_choose_cheapest_capable_model_lane_without_network` failed because the router chose `local_sovereign` instead of `groq_like_fast`.
- Green tests after implementation: `python3 -m pytest tests/test_lucidota_swarm_router.py -q` -> `2 passed in 0.01s`.
- Focused integration tests: `python3 -m pytest tests/test_lucidota_swarm_router.py tests/test_project_brain_doc_authority.py tests/test_rfc_program_check.py -q` -> `10 passed in 1.17s`.
- Router self-check lanes still stable: `local_sovereign`, `deterministic_workflow`, `groq_like_fast`, `cohere_like_context`, `default_airlock`; all route receipts report `cost_aware=true` and `provider_calls=false`.
- Slop audit: `05_OUTPUTS/slop_audit/slop_audit_law_20260527T050202321816Z.json`
  - blockers `[]`;
  - verdict `REVIEW` only because router file is over 5x PocketFlow; accepted as bounded routing surface.
- Canonical graph write scan: `05_OUTPUTS/swarm_submissions/canonical_graph_write_scan_router_cost_awareness.json`
  - PASS;
  - blockers `[]`;
  - `canonical_graph_writes_performed=false`.
- Cost-awareness audit: `GOALS/router_cost_awareness_audit_20260527T050247Z.json`
  - `PASS_ROUTER_COST_GAP_CLOSED_LOCALLY`;
  - 6 / 6 checks true.
- Delta integration receipt: `05_OUTPUTS/swarm_submissions/swarm_router_cost_awareness_delta_20260527T050304Z.json`
  - PASS;
  - provider calls false;
  - network calls false;
  - canonical graph writes false.

Remaining explicit gaps:

- Canonical graph materialization remains intentionally unexecuted and requires explicit operator authorization plus policy hardening.
- The operator's over-four-hours error-free criterion is still not satisfied in this active goal window.

Technical Summary Review and Dev Notes: The router now counts coins without phoning the market goblin. Cost can steer eligible model lanes, but exact receipts still go through the deterministic hammer.

## 27. Rickshaw Second-Pass Reinvestigation Seed, Gauntlet, and Guarded Graph Promotion

Decision: apply the operator’s Rickshaw authorization narrowly. Only the prior graph-era `RICKSHAW_ROBBERY` CASE seed meta-truth was eligible for canonical graph materialization. No allegations, inferred facts, new parties, or external case claims were promoted.

Proven prior graph membership:

- Original CASE node: `node_29033b7bd4df49d31b339dc7bef6c0e7`.
- Existing graph item UUID: `c28bace9-8e1e-581f-a6ab-f1a34e7896b8`.
- Evidence packet: `05_OUTPUTS/krampuschewing/graph/krampuschewing_graph_packet_20260519T091732933101Z.json`.
- Existing DB graph counts proved Rickshaw rows existed in `graph_item`, `graph_edge`, and `graph_journal` before this pass.

Second-pass deterministic rebuild receipts:

- Master corpus index: `05_OUTPUTS/krampuschewing/rickshaw_second_pass_master_summary_20260527T051327Z.json`
  - 1,483 files indexed and hashed; 4.67GB hashed; 0 skipped hash.
- Graph candidate staging: `05_OUTPUTS/krampuschewing/rickshaw_second_pass_graph_candidates_summary_20260527T051455Z.json`
  - 1,474 graph candidates; no canonical graph writes.
- Chrono graph packet: `05_OUTPUTS/krampuschewing/graph/krampuschewing_graph_packet_20260527T051457627621Z.json`
  - 1,483 chrono events; 1,493 nodes; 13,020 edges.
- Bitloops second-pass packet: `05_OUTPUTS/bitloops/rickshaw_second_pass_bitloops_graph_packet_20260527T051506Z.json`
  - bundle SHA256 `5a5af299a0b9ba3d2499f5690ed7c06031d99b5c587769e5239e572468d64f6d`.
- Case text pipeline receipt: `05_OUTPUTS/cases/rickshaw_second_pass/rickshaw_second_pass_reinvestigation_20260527T051835009568Z.json`
  - 858 text files parsed; 9,495 chunks; 62,397 staged claims; candidate/support scoring only.

Surgical fixes made because existing tools blocked real reingest:

- `scripts/krampuschewing_graph_stage.py`: clean case indexes no longer fail only because there are zero active runtime DB exclusions.
- `tests/test_krampuschewing_graph_stage.py`: proves clean no-active-DB graph staging and active runtime DB exclusion.
- `scripts/contradiction_report.py`: replaced unbounded pairwise contradiction scan with grouped/bounded exact-negation detection.
- `tests/test_contradiction_report.py`: proves exact negation detection and bounded duplicate-group behavior.
- Fresh focused check: `python3 -m pytest tests/test_krampuschewing_graph_stage.py tests/test_contradiction_report.py -q` -> `4 passed`.

Adversarial abductive gauntlet:

- Aggregate receipt: `05_OUTPUTS/cases/rickshaw_second_pass/gauntlet/20260527T052211088518Z/rickshaw_second_pass_adversarial_gauntlet_receipt.json`.
- Hard status: PASS; hard failures `0`.
- Gates run: Bitloops red team self-check, golden path regression no-materialize, graph write blockers, edge blockers, rollback probe, graph promotion soak, spine queue soak, Absurd gate, mega-gate fault injector, 3-iteration mega-gate regression, canonical graph write scanner, focused pytest suite.
- Known soft gap: `abductive_db_os_gate_fast` returned DEGRADED because DB-OS adapter/health-check scripts are absent; recorded as soft, not hidden.

Guarded graph materialization executed through the promotion helper:

- Candidate payload: `05_OUTPUTS/cases/rickshaw_second_pass/graph_materialization/rickshaw_prior_graph_seed_candidate_20260527T053103950097Z.json`.
- Command envelope UUID: `f4215ae5-7d28-433f-b830-a3694aa54766`.
- Dry-run receipt: `05_OUTPUTS/graph/graph_materialization_helper_materialize_dry_run_20260527T053119309891Z.json` PASS.
- Execute receipt: `05_OUTPUTS/graph/graph_materialization_helper_materialize_execute_20260527T053144381387Z.json` PASS.
- Materialization UUID: `8d618c87-d176-4dfb-bbe9-6d5e3e91dccb`.
- New guarded CASE seed graph item: `2ea0075f-6a8c-494b-a84d-fcec7abc092a`.
- Helper receipt UUID: `6ab40cc9-9a28-4f38-a2d1-7e03cb34e20d`.
- Aggregate receipt: `05_OUTPUTS/cases/rickshaw_second_pass/graph_materialization/rickshaw_prior_graph_seed_materialization_aggregate_20260527T053337134019Z.json`.
- Verification receipts:
  - `05_OUTPUTS/graph/graph_materialization_helper_verify_pass_20260527T053255088547Z.json` PASS.
  - `05_OUTPUTS/graph/graph_journal_completeness_pass_20260527T053209373268Z.json` PASS with 3 pre-existing missing-helper-receipt warnings unrelated to this materialization.
  - `05_OUTPUTS/graph/graph_journal_replay_audit_20260527T053209514226Z.json` PASS.
  - `05_OUTPUTS/graph/graph_materialization_receipt_query_20260527T053209665287Z.json` PASS.
  - `05_OUTPUTS/graph/canonical_graph_write_scanner_20260527T053210434167Z.json` PASS.

Remaining explicit gaps:

- This is not a full Rickshaw truth verdict; only the case-corpus seed meta-truth is materialized.
- The `abductive_db_os_gate_fast` lane is degraded until DB-OS adapters/health-check scripts exist or are formally retired.
- The operator’s “over 4 hours error-free development” completion criterion is still not satisfied by the current elapsed verified window.

Technical Summary Review and Dev Notes: The Rickshaw seed is now on the graph by the drawbridge, not by a hole in the wall. The old case root got a stamped collar; the wolves of allegation are still outside the truth gate.

## 28. DB-OS Soft Gauntlet Gap Closed and PONYBOY Second-Pass Staged

Decision: close the `abductive_db_os_gate_fast` soft gap by wiring the existing ABSURD/receipt primitives into the DB-OS lane, not by inventing a new workflow engine. Then advance the next named historical corpus (`PONYBOY`) through the existing deterministic index, graph-stage, and Bitloops lanes without canonical graph materialization.

DB-OS lane repair:

- Added thin DB-OS adapter scripts that mirror existing ABSURD adapters while writing DB-OS schemas/receipts:
  - `scripts/model_audit_db_adapter.py`
  - `scripts/bytewax_db_os_stream_audit.py`
  - `scripts/krampuschewing_db_os_adapter.py`
  - `scripts/indy_reads_db_os_brief.py`
  - `scripts/abductive_db_os_health_check.py`
- Patched `scripts/abductive_db_os_gate.py` to parse child verdict markers instead of treating every `returncode=0` as PASS and every nonzero as opaque degradation.
- Updated existing `tests/test_abductive_db_os_gate.py` with command-plan existence and child-verdict parsing checks.

DB-OS verification:

- Before repair: `05_OUTPUTS/abductive_db_os/abductive_db_os_gate_fast_20260527T053734936227Z.json`
  - verdict `DEGRADED`; 5 warnings from absent adapter/health-check scripts.
- After repair: `05_OUTPUTS/abductive_db_os/abductive_db_os_gate_fast_20260527T054012924178Z.json`
  - verdict `PASS`; warnings `[]`; all 7 child checks PASS.
- ABSURD gate still healthy: `05_OUTPUTS/absurd_abductive/absurd_gate_fast_20260527T054013280073Z.json` PASS.
- Canonical graph write scan: `05_OUTPUTS/graph/canonical_graph_write_scanner_20260527T054014007564Z.json` PASS.
- Focused tests: 18 passed in 1.12s.
- Slop audit for DB-OS adapters: `05_OUTPUTS/slop_audit/slop_audit_law_20260527T054354513249Z.json` PASS.
- Dev Library/TICKLETRUNK refreshed: `05_OUTPUTS/tickletrunk/tickletrunk_scan_20260527T054124Z.json`; total tools 1550; scripts 725.

PONYBOY second-pass staging:

- Source root indexed: `09_STORAGE/krampuschewing_unpacked/docs_Luci-010.zip_10711561291/Luci/Lucidota/PONYBOY`.
- Master index summary: `05_OUTPUTS/krampuschewing/ponyboy_second_pass_master_summary_20260527T054251Z.json`
  - verdict PASS;
  - 64 files indexed;
  - 64 files hashed;
  - 74,885,458 bytes hashed;
  - 0 skipped large-file hashes.
- Graph-stage summary: `05_OUTPUTS/krampuschewing/ponyboy_second_pass_graph_candidates_summary_20260527T054251Z.json`
  - verdict PASS;
  - 64 candidates emitted;
  - 0 active runtime DB skips;
  - no DB writes;
  - no canonical graph writes.
- Bitloops second-pass report: `05_OUTPUTS/bitloops/bitloops_automation_loop_20260527T054251809881Z.json`
  - status PASS;
  - 102 accepted/recovered;
  - 102 graph mutation candidates;
  - 408 River training lanes;
  - state collapsed;
  - no DB writes;
  - no canonical graph writes;
  - no model calls.
- Bitloops graph-promotion packet: `05_OUTPUTS/bitloops/ponyboy_second_pass_bitloops_graph_packet_20260527T054251Z.json`.
- Aggregate receipt: `GOALS/dbos_gap_closure_and_ponyboy_second_pass_20260527T054326540174Z.json` PASS.

Remaining explicit gaps:

- PONYBOY candidates remain staged only. No canonical graph materialization occurred and none is authorized yet.
- The overall historical reingest remains incomplete; LUCI/KRAMPUS/Chrono/Indy full second-pass parity still needs more batch work.
- The operator’s >4h error-free criterion remains open.

Technical Summary Review and Dev Notes: The DB-OS lane stopped yelling about ghosts because its adapters now exist and speak receipt. PONYBOY got chewed and staged; no graph crown, no model smoke.
