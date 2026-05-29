# Consolidation Notice

Status: SUPPORTING SOURCE MATERIAL. The active spec set is `ACTIVE_SPEC/01_IDENTITY_AND_CLAIM_STATE_LAW.md`, `ACTIVE_SPEC/02_EXECUTION_SPINE.md`, `ACTIVE_SPEC/03_CUSTODY_ETL_PIPELINE.md`, `ACTIVE_SPEC/04_DEV_LIBRARY_REUSE_LAW.md`, and `ACTIVE_SPEC/05_COMPONENT_AUTHORITY_MAP.md`. Use this file as historical/source material unless a section is explicitly re-promoted by the active instruction index.

---

# LUCIDOTA Canonical Manifest & Onboarding Guide

Status: **CANONICAL ONBOARDING / ARCHITECTURE MAP**  
Updated: **2026-05-20**  
Read first: **`AGENTS.md`, `00_PROJECT_BRAIN/TICKLETRUNK.json`, `00_PROJECT_BRAIN/TICKLETRUNK.md`, then this file.**

This document is the plain-English map for any new coding assistant or ephemeral Littleworker entering LUCIDOTA. It explains what the system is, how work moves through it, which files are authoritative, and what must never be bypassed.

---

## 0. Prime Law

LUCIDOTA is a **local proof-hoard operating system**: it preserves raw artifacts, turns observations into receipts, stages candidate claims, and only promotes truth through explicit authority gates.

The operating law is:

1. **Index the jungle; do not pave it.** Strange, disconnected, experimental, and useful-later artifacts are allowed.
2. **Search TICKLETRUNK before building.** Copy/adapt/reuse existing tools before writing new ones.
3. **Do not mutate sovereign toolbox originals unless explicitly ordered.** Promote by copying/adapting into a hardened lane, not by “cleaning up” originals.
4. **Raw evidence is not truth.** Evidence can be stored, hashed, and staged. Truth promotion requires authority, evidence refs, and graph gates.
5. **Littleworkers are ephemeral.** `.lucidota_agents/specialists/` is a static prompt/manual store, not a resident swarm or required runtime.
6. **Indy_READs is not a Littleworker.** Indy_READs is the operator’s named reasoning companion / synthesis identity, not a disposable prompt target or daemon.
7. **Receipts beat vibes.** Completion means machine receipts in `05_OUTPUTS/`, status checks, and backlog closure — not a reassuring Markdown claim.

---

## 1. Resource Budget / “80%, Not 110%” Rule

The machine is constrained. Treat it like an 8 GB RAM / 4 GB VRAM workstation unless telemetry proves otherwise.

Before heavy work, run a quick telemetry check:

```bash
free -h
uptime
python3 scripts/safe_stress_test.py --samples 1 --loop-iters 100 --payload-kb 1 --write-report
```

Known good telemetry receipt from this audit:

- `05_OUTPUTS/safe_stress_test/safe_stress_test_20260520T072810Z.json`
- GPU at audit time: ~52 C, ~2 MB VRAM used, ~3714 MB VRAM free.
- Status ledger check: `CHECK_OK status ledger valid`.
- TICKLETRUNK check: `CHECK_OK TICKLETRUNK valid`.

Stop or downshift if any of these are true:

- available RAM below ~1.5 GiB,
- swap used above ~60%,
- load is climbing and commands are not bounded,
- GPU temp approaches the configured wall (`LUCIDOTA_GPU_TEMP_WALL_C`, default 84 C),
- a command wants to recursively read/hash `KRAMPUSCHEWING`, `03_VAULT`, `09_STORAGE`, `01_REPOS`, or `.venv` without a limit.

Safe command style:

```bash
# Good: bounded / pruned / receipt-oriented
find . \( -path './KRAMPUSCHEWING' -o -path './03_VAULT' -o -path './09_STORAGE' -o -path './01_REPOS' -o -path './.venv' \) -prune -o -type f -name '*.py' -print

# Good: low-impact manifest query
python3 scripts/tickletrunk_scan.py --query kernel

# Risky: do not do this casually on the hoard
rg "anything" KRAMPUSCHEWING 03_VAULT 09_STORAGE
```

### Crash finding from 2026-05-20

The prior crash was a real kernel OOM event:

- Time: **2026-05-20 00:21:16 America/Vancouver**.
- Victim: `python3`, PID `39174`.
- Kernel line: `Out of memory: Killed process 39174 (python3)`.
- `total-vm`: ~10.5 GB.
- `swapents`: `2476896` pages, roughly 9.4 GB swapped.
- Free swap at kill time: **244 kB of ~11.6 GB**.
- Earlier warning: page allocation failure at **2026-05-19 23:42:12**.

The exact Python argv was not present in the kernel log, so the exact command cannot be proven after the fact. The evidence does prove a Python workload overcommitted memory/swap. It was not the tiny quarantine receipt writer itself; the likely class is an unbounded local Python job such as model/embedding/Chroma/large-ingest/extraction work or a recursive workload that pulled too much into memory.

Fix applied in this audit:

- `scripts/safe_stress_test.py` now runs directly from the repo without needing manual `PYTHONPATH` setup.
- This guide makes the resource guardrail explicit.
- Current audit commands avoided unbounded hoard scans and used telemetry checkpoints.

---

## 2. Filesystem Map

### 2.1 Top-level meaning

- `00_PROJECT_BRAIN/` — canonical manifests, registries, onboarding, status summaries.
- `ALGOS/` — pure/reusable algorithm primitives; scripts import these instead of owning math.
- `scripts/` — executable workers, gates, audits, scanners, receipts, and orchestration glue.
- `06_SCHEMA/` — SQL and JSON schemas for control, storage, queues, graph staging, authority, and receipts.
- `core/` — shared runtime helpers: DSN resolution, language membrane, telemetry sanity, TICKLETRUNK helper loops.
- `pypeline/` — reusable math/runtime planning modules used by workers.
- `services/` — service-like local backends such as FairyFuse.
- `01_REPOS/` — external/local codebases and forks; proof-hoard source, not automatically active architecture.
- `03_VAULT/` — vault/CAS/models/runtime assets; large, do not casually scan.
- `04_RUNTIME/` — runtime state, logs, DBs, local mutable state.
- `05_OUTPUTS/` — generated receipts, reports, dashboards, manifests, work orders.
- `07_SURFACES/` — generated/operator-facing surfaces.
- `09_STORAGE/` — cold/storage/proof-kernel and corpus storage; large, do not casually scan.
- `KRAMPUSCHEWING/` — ingestion hoard / archive / evidence jungle. Treat as preserved source material.
- `TOOLS/` — generated TICKLETRUNK access layer / navigation READMEs.
- `.lucidota_agents/specialists/` — static Littleworker instruction manuals only.

### 2.2 Current scale snapshot

Bounded live map from this audit:

| Area | Approx shape |
|---|---:|
| `00_PROJECT_BRAIN` | 14 dirs / 38 files |
| `ALGOS` | 1 dir / 112 files |
| `scripts` | 2 dirs / 535 files |
| `06_SCHEMA` | 1 dir / 139 files |
| `core` | 3 dirs / 14 files |
| `pypeline` | 5 dirs / 56 files |
| `services` | 4 dirs / 8 files |
| `tests` | 9 dirs / 150 files |
| `01_REPOS` | ~16 GB |
| `03_VAULT` | ~34 GB |
| `05_OUTPUTS` | ~23 GB |
| `09_STORAGE` | ~78 GB |
| `KRAMPUSCHEWING` | ~75 GB |
| `.venv` | ~7.8 GB |

The complete component index is TICKLETRUNK, not a hand-maintained file list here.

---

## 3. Epistemic Garbage Collection State

Active architectural Markdown was consolidated into this canonical manifest plus generated/reference manifests. Obsolete/speculative Markdown is being preserved in `KRAMPUSCHEWING/System_Archive_Docs/` for passive ingestion instead of being deleted.

Receipts from this cleanup:

- Existing GC plan: `05_OUTPUTS/epistemic_gc/epistemic_gc_markdown_archive_plan_20260520T070309Z.json`
- Existing ingest list: `05_OUTPUTS/epistemic_gc/archived_markdown_ingest_list_20260520T070552Z.txt`
- Audit repair receipt: `05_OUTPUTS/epistemic_gc/deleted_markdown_archive_receipt_20260520T072539Z.json`
- Audit repair ingest list: `05_OUTPUTS/epistemic_gc/deleted_markdown_ingest_list_20260520T072539Z.txt`

Important audit result:

- Git showed **46 deleted Markdown files** from active areas.
- Only 1 had an obvious archive match at first.
- This audit recovered the other **45 missing deleted Markdown files from git HEAD** into `KRAMPUSCHEWING/System_Archive_Docs/` using collision-safe names like `00_PROJECT_BRAIN__GOALS.md`.
- Verification after repair: **found=46, missing=0**.

Markdown intentionally still active:

- This file: `00_PROJECT_BRAIN/LUCIDOTA_CANONICAL_MANIFEST.md`.
- Generated manifests: `TICKLETRUNK.md`, `STATUS_LEDGER.md`.
- Local workflow/product subproject docs under `00_PROJECT_BRAIN/*/`.
- Access-layer READMEs under `TOOLS/`.
- Runtime/project READMEs needed to explain local code areas.

Do not create new doctrine Markdown by default. Prefer receipts, JSON, SQL, code, and updates to this single manifest.

---

## 4. TICKLETRUNK — Sovereign Toolbox Registry

TICKLETRUNK is the canonical registry and access layer for the proof hoard.

Primary files:

- `00_PROJECT_BRAIN/TICKLETRUNK.json` — machine manifest.
- `00_PROJECT_BRAIN/TICKLETRUNK.md` — human-readable manifest.
- `scripts/tickletrunk_scan.py` — scanner/generator.
- `TOOLS/` — generated navigation/access layer.

Current counts from `TICKLETRUNK.json`:

| Category | Count |
|---|---:|
| ALGOS | 57 |
| SCRIPTS | 472 |
| MODELS | 25 |
| LORAS | 23 |
| SCHEMAS | 143 |
| SKILLS | 35 |
| PLUGINS | 9 |
| SERVICES | 6 |
| BOOKS | 19 |
| SURFACES | 6 |
| SCRAPERS | 8 |
| WORKFLOWS | 39 |
| REPOS | 47 |
| RUNTIME | 21 |
| VAULT | 28 |
| OTHER | 133 |

Useful commands:

```bash
python3 scripts/tickletrunk_scan.py --check
python3 scripts/tickletrunk_scan.py --query krampus
python3 scripts/tickletrunk_scan.py --query kernel
python3 scripts/tickletrunk_scan.py --category ALGOS
python3 scripts/tickletrunk_scan.py --execute
```

TICKLETRUNK scanner law:

- default mode is dry-run;
- it skips sensitive path patterns;
- it never deletes/moves/renames sovereign toolbox artifacts;
- execute mode writes manifests/access/report files only.

---

## 5. ALGOS — Algorithmic Primitive Layer

`ALGOS/` contains reusable primitives. They should be pure or near-pure logic. Runtime scripts own DB scans/writes, process orchestration, and receipts.

Major clusters:

- **Evidence/math hygiene:** `decision_hygiene.py`, `cockpit_metrics.py`, `hard_truth_math.py`, `bayes_update.py`, `counterfactual_effects.py`.
- **Ingestion/corpus helpers:** `korpus_text.py`, `krampus_chrono.py`, `krampus_stickers.py`, `krampus_brainmap.py`.
- **Routing/admission:** `rete_bandit_gate.py`, `bandit_router.py`, `endpoint_circuit_breaker.py`, `regret_engine.py`, `tri_algo_conduit.py`.
- **Dedup/similarity:** `minhash.py`, `semantic_neighbors.py`, `perceptual_dedupe.py`, `ssim.py`, `sketches.py`.
- **Graph/search/network math:** `physarum_network.py`, `voronoi_partition.py`, `minimum_cost_tree.py`, `infotaxis.py`, `distributed_leader_election.py`.
- **Model/extraction support:** `gliner_zero_shot_extractor.py`, `shap_attribution.py`, `xgboost_objective.py`, `ternary_lens_router.py`, `ternary_router.py`.
- **Biology-inspired control primitives:** `serpentina_self_righting.py`, `poikilotherm_schoolfield.py`, `thanatosis.py`, `chelydrid_ambush.py`, `capybara_optimization.py`, `honeybee_store.py`.

Rule for new work:

- If it is reusable math or deterministic extraction logic, put/adapt it in `ALGOS/`.
- If it reads/writes DBs, queues, files, or receipts, make it a script/worker that imports `ALGOS/`.
- Do not treat “missing imports” or “sandbox” status as proof that an algo is useless.

---

## 6. PERSCYPHONAI / Percyphon.ai

The code spelling is **Percyphon.ai** and the implementation is:

- `ALGOS/percyphon.py`

Operator references to **PERSCYPHONAI / perscyphon / percyphon** should map to this component unless a newer manifest says otherwise.

Plain English role:

- Percyphon is a **zero-VRAM procedural entity generator**.
- It does not load model weights.
- It uses SHA-256 deterministic seeds to generate stable procedural slots, aliases, UUID-like identifiers, and a ternary offset.
- It is used as a low-cost identity/entity-mask generator inside ingestion/synthesis, not as a truth oracle.

Current runtime use:

- `scripts/unified_absurd_ingest_worker.py` imports `procedural_entity_generator`.
- During `step_indy_reads_synthesis`, the worker puts Percyphon output into `analysis["percyphon"]`.
- The streamed header reports only small summary fields such as slot count and ternary offset.

Why it exists:

- It gives the pipeline deterministic “entity-ish” scaffolding without spending VRAM.
- It can enrich or stabilize synthesis/route metadata while keeping raw evidence and graph truth separate.

---

## 7. Kernel / Control / Authority Routing

The Kernel is not one monolithic daemon. It is a set of control-packet, ledger, authority, and graph-gate contracts that prevent random scripts from mutating canonical state.

Core files:

- `scripts/kernel_control_packet.py` — creates/verifies domain-separated control packets.
- `scripts/ckdog_kernel_route_plan.py` — verifies packets, rejects replays, emits route plans.
- `scripts/control_packet_ledger.py` — JSONL packet ledger under `05_OUTPUTS/kernel/`.
- `scripts/ckdog_kernel_events.py` — JSONL kernel event log under `05_OUTPUTS/kernel/`.
- `scripts/spine_authority_checker.py` — checks whether an authority class may perform an effect in a lane.
- `00_PROJECT_BRAIN/spine_authority_registry.json` — authority registry.
- `scripts/graph_promotion_gate.py` — evidence + authority + preflight gate for graph-promotion packets.
- `scripts/graph_promotion_approval_worker.py` — candidate/defer/reject/operator-confirmed/materialized state machine.
- `scripts/proof_kernel.py` — byte custody without truth promotion or graph mutation.

Control packet facts:

- Namespace: `ckdog1-control-packet`.
- Valid actions include `add_mandatory`, `add_optional`, and `disable_lane`.
- Packets are hash-verified and can expire.
- Route planning rejects invalid packets and packet replays.
- Authorization-sensitive queue jobs require a valid packet.

Authority facts:

- Registry schema: `lucidota.spine_authority_registry.v1`.
- Authority classes include raw evidence, deterministic metric, model-computed finding, operator-authored assertion, operator-confirmed finding, and external-action authorization.
- Effects are explicit: queue work order, stage graph packet, materialize canonical graph, no canonical mutation, external command, operator override.
- Missing evidence refs block privileged effects.

Graph promotion facts:

- Candidate packets are staged in `lucidota_go.graph_promotion_packet`.
- Decisions are recorded in `lucidota_go.graph_promotion_decision`.
- The runtime gate writes audit receipts and refuses materialization without proper preflight/operator confirmation.
- During this hardening state, canonical materialization is deliberately blocked unless the gate and policy say otherwise.

Proof kernel facts:

- Stores bytes under `09_STORAGE/proof_kernel` by SHA-256.
- Makes stored objects read-only where possible.
- Appends immutable-ish proof index entries.
- Explicitly records: `truth_promotion: False`, `canonical_graph_materialization: False`, `canonical_graph_writes_performed: False`.

Important terminology note:

- Some historical filenames and schema names still contain an old queue-framework label. Treat those as legacy compatibility names. The active architecture is **queue spine / worker harness / daemon supervisors**, not a dependency on permanent resident Littleworkers.

---

## 8. Queue Spine / Worker Harness / Runtime

The active runtime pattern is:

1. **Submit bounded work** with source path, payload, priority, and draft-only defaults.
2. **Claim exactly one pending row** with `FOR UPDATE SKIP LOCKED`.
3. **Lease the work** so stalled jobs can be requeued or dead-lettered.
4. **Do expensive file/parser work outside DB locks.**
5. **Write staging packets, comments, events, receipts.**
6. **Finish as completed/failed/dead-lettered/cancelled.**

Central worker:

- `scripts/unified_absurd_ingest_worker.py`

Important behavior in that worker:

- Uses state DB via `core/runtime_dsns.py` (`lucidota_state` by default).
- Uses storage DB via `core/runtime_dsns.py` (`lucidota_storage` by default).
- Preserves raw local evidence in CAS.
- Stages evidence/comment packets in `lucidota_go`.
- Streams an Amalgamated Text Surface for operator review.
- Runs prompt-injection detection and display neutralization.
- Uses ALGOS for recovery, throttling, route decisions, language membrane, and Percyphon.
- Enforces draft-only outbound behavior.
- Does not send email, publish, or file external records.

Key runtime helpers:

- `core/runtime_dsns.py` — environment-driven DB resolution.
- `core/language_membrane.py` — deterministic text routing and output weaving.
- `pypeline/math/model_vram_scheduler.py` — advisory VRAM/LoRA preemption planner; does not load weights.
- `scripts/lucidota_model_governor.py` — observes GPU state and writes advisory decisions.
- `scripts/lucidota_ingest_watchdog.py` — conservative ingestion observation/dashboard loop.
- `scripts/safe_stress_test.py` — read-only throughput/resource telemetry.

---

## 9. Ingestion Lifecycle in Plain English

### A. A raw artifact enters

A file, text, command output, or archive appears in a known input area such as `KRAMPUSCHEWING/`, `03_VAULT/`, or a controlled drop path.

### B. Security gate decides what is allowed

`scripts/lucidota_security_quarantine_gate.py` scans configured roots for sensitive classes: env files, credentials, browser stores, private keys, token logs, etc.

The latest clean security manifest at this consolidation point is:

- `05_OUTPUTS/security/security_quarantine_manifest_20260520T065123Z.json`

Important rule:

- Included clean files may be processed.
- Excluded/quarantined/deferred artifacts remain blocked from embedding, summarization, extraction, design-atom extraction, and graph promotion.

### C. Custody/proof happens before interpretation

The proof kernel or ingest worker hashes bytes and records custody. Raw data remains raw; it is not promoted to truth.

### D. Text/extraction/synthesis runs as staging

Workers extract text surfaces and deterministic signals:

- filesystem observation certainty,
- parser extraction certainty,
- prompt-injection flags,
- GO term guesses,
- language-membrane route,
- Percyphon procedural mask metadata,
- GLiNER span/claim candidates when invoked.

GLiNER dry-run path:

- `scripts/gliner_claim_packet_dry_run.py`
- Emits claim packets as candidates.
- Writes no canonical graph truth.

### E. Candidate packets are staged

Staging packet examples:

- evidence packet,
- comment/synthesis packet,
- graph-promotion packet,
- claim packet.

These are receipts/candidates, not final truth.

### F. Graph promotion requires authority

`spine_authority_checker.py` and `graph_promotion_gate.py` check:

- authority class,
- requested effect,
- permitted lane,
- evidence refs,
- operator confirmation when required,
- status ledger health for privileged materialization.

If the gate blocks it, the candidate remains staged/deferred.

### G. Completion means receipts

A task is only “done” when the relevant receipt exists under `05_OUTPUTS/`, status checks pass, and the work order/backlog item can be closed with evidence.

---

## 10. Current Backlog / Target-State Vector

The gauntlet is the target-state vector:

- `05_OUTPUTS/work_orders/lucidota_600_work_order_gauntlet_20260517T211943051910Z.jsonl`

Do not replace the gauntlet with speculative planning docs. Use receipts to close work orders.

Known checks from this audit:

```bash
python3 scripts/tickletrunk_scan.py --check
python3 scripts/lucidota_status_ledger.py --check
python3 scripts/safe_stress_test.py --samples 1 --loop-iters 100 --payload-kb 1 --write-report
```

---

## 11. Littleworker Rules

Littleworkers are optional, bounded accelerators.

They are:

- static instruction sets in `.lucidota_agents/specialists/`,
- summoned for one concrete task,
- expected to return evidence/patches/receipts,
- terminated after the task.

They are not:

- permanent daemons,
- an always-on swarm,
- a required runtime dependency,
- a substitute for receipts,
- a license to mutate canonical graph state.

If you dispatch Littleworkers:

1. Give each a bounded scope and clear read/write boundaries.
2. Do not make multiple workers write the same files.
3. Do not let them recursively scan the hoard without limits.
4. Integrate their output through receipts and verification.
5. Close or stop them when no longer needed.

---

## 12. Rules of Engagement for a New Coding Assistant

On entry:

```bash
pwd
sed -n '1,120p' AGENTS.md
python3 scripts/tickletrunk_scan.py --check
python3 scripts/tickletrunk_scan.py --query <your-topic>
python3 scripts/lucidota_status_ledger.py --check
free -h
```

Before writing code:

1. Read `TICKLETRUNK.json` / `TICKLETRUNK.md` enough to find existing tools.
2. Search for relevant scripts, schemas, tests, ALGOS, and receipts.
3. Prefer adapting existing code.
4. Touch only files needed for the task.
5. Keep sovereign originals intact unless the operator names the exact target.

When writing code:

- Put reusable math in `ALGOS/` or `pypeline/math/`.
- Put orchestration/DB/file/receipt behavior in `scripts/`.
- Put durable schemas in `06_SCHEMA/`.
- Put generated reports/receipts in `05_OUTPUTS/`.
- Update TICKLETRUNK after adding/changing tools:

```bash
python3 scripts/tickletrunk_scan.py --execute
python3 scripts/tickletrunk_scan.py --check
```

Before claiming completion:

```bash
python3 scripts/tickletrunk_scan.py --check
python3 scripts/lucidota_status_ledger.py --check
python3 scripts/safe_stress_test.py --samples 1 --loop-iters 100 --payload-kb 1 --write-report
```

Then report:

- files changed,
- receipts produced,
- checks run,
- any blockers/dead letters,
- whether canonical graph writes occurred. Default should be **no**.

---

## 13. Absolute Don’ts

- Do not delete, rename, move, normalize, or “clean up” sovereign proof-hoard artifacts without explicit operator instruction.
- Do not recursively grep/hash/read all of `KRAMPUSCHEWING`, `03_VAULT`, `09_STORAGE`, `01_REPOS`, or `.venv` without a tight reason and resource limits.
- Do not treat raw evidence, model output, or GLiNER spans as truth.
- Do not bypass `spine_authority_registry.json`, `spine_authority_checker.py`, or `graph_promotion_gate.py` for graph-affecting work.
- Do not generate new doctrine Markdown when a receipt, JSON, SQL schema, code patch, or update to this manifest will do.
- Do not treat Indy_READs as a disposable Littleworker.
- Do not create resident Littleworker swarms.

---

## 14. One-Screen Mental Model

LUCIDOTA is:

> A local evidence-preserving proof OS with a giant indexed toolbox, bounded workers, strict authority gates, and receipt-first memory.

The flow is:

> Artifact or command → security/custody → deterministic extraction/synthesis → candidate/staging packet → authority gate → optional operator-confirmed promotion → receipt/backlog closure.

The design philosophy is:

> Preserve everything useful, index it instead of paving it, keep workers ephemeral, keep archives read-only by default, and never let convenience outrank evidence, authority, or resource safety.
