# Consolidation Notice

Status: SUPPORTING SOURCE MATERIAL. The active spec set is `ACTIVE_SPEC/01_IDENTITY_AND_CLAIM_STATE_LAW.md`, `ACTIVE_SPEC/02_EXECUTION_SPINE.md`, `ACTIVE_SPEC/03_CUSTODY_ETL_PIPELINE.md`, `ACTIVE_SPEC/04_DEV_LIBRARY_REUSE_LAW.md`, and `ACTIVE_SPEC/05_COMPONENT_AUTHORITY_MAP.md`. Use this file as operational source material, not as a free-floating manual that can override scoped specs.

---

# LUCIDOTA OPS MANUAL — ABSURD/Postgres Runtime

Status: ACTIVE OPERATOR MANUAL  
Last updated: 2026-05-22  
Authority: `00_PROJECT_BRAIN/CANONICAL_FINISHED_PRODUCT_MAP.md`, `00_PROJECT_BRAIN/TICKLETRUNK.json`, current operator directive.

## 0. What This System Is

Plain English: LUCIDOTA is a **solo-dev intelligence agency** for extending the operator's hyper-systemic thinking at rapid speed on constrained local hardware: **Palantir on a shitty laptop, for good ethics, with operator control**.

The short version:

```text
operator intent
  -> Rust command envelope / Diogenes-style kernel authority
  -> DB-as-OS durable queue + receipts
  -> deterministic algorithms first
  -> bounded model/LoRA helpers when useful
  -> claim/evidence/hypothesis separation
  -> graph staging, never hallucinated truth
  -> operator-facing ops response
```

The old phrase "proof hoard" only means: **a preserved local pile of evidence, receipts, old code, experiments, documents, and tool outputs that must stay findable and hashable instead of being thrown away.** It is not a mystical subsystem. It is the archive that lets the active system get small without losing memory.

So: proof hoard is not the product. It is the evidence vault and digestion stomach. The product is the controlled intelligence pipeline.

The working shape is:

```text
artifact or operator instruction
  -> custody / command envelope
  -> ABSURD/Postgres queued worker
  -> strict worker contract at dequeue
  -> bounded deterministic handler
  -> receipt / event / dead-letter
  -> candidate evidence only
  -> authority mapping
  -> graph promotion gate
  -> guarded materialization helper + graph_journal only when authorized
```

Models advise or extract. They do not steer the workflow. Direct canonical graph writes stay blocked.

Design north star: **Rust owns the core; Python stays wet clay for on-the-fly operating code, adapters, experiments, and model-lane glue until each lane earns a Rust port.**

## 1. ABCD Policy

External policy lookup: the World Governments Summit / Oliver Wyman ABCD playbook defines policy innovation as **Agility, Buy-in, Convergence, Durability**.

LUCIDOTA operationalizes ABCD in executable form through `scripts/updated_abcd_sequence_runner.py`:

| Letter | Runtime meaning | Operator test |
|---|---|---|
| A | Declarative Contract Mapping | Is the target named in TICKLETRUNK, schema, product map, or ledger before work starts? |
| B | Continuous State Validation | Does the command run a real gate/check before claiming progress? |
| C | Immutable Data Modeling | Are raw artifacts preserved and are failures dead-lettered instead of erased? |
| D | Pure Logic Execution | Is control flow in source/schema, not prompt improvisation? |

For every tranche: map first, contract second, execute third, ledger last.

## 2. Start / Interlock

Run once at the start of an operational tranche:

```bash
python3 scripts/lucidota_status_ledger.py --check
python3 scripts/script_survival_coverage.py
python3 scripts/tickletrunk_scan.py --check
```

If any command fails, stop mutation and preserve the failing stdout/stderr under `05_OUTPUTS/dead_letter/`.

## 3. Main Runtime Spine

ABSURD/Postgres is the current durable workflow substrate. DBOS names are legacy provenance only.

Core files:

```text
06_SCHEMA/035_absurd_queue_spine.sql
06_SCHEMA/082_absurd_worker_contract_registry_enforcement.sql
scripts/absurd_queue_spine.py
scripts/absurd_consume_one.py
scripts/absurd_worker_contracts.py
```

Worker law:

1. Select queued job with a lock (`FOR UPDATE SKIP LOCKED` or equivalent claim path).
2. Validate `(queue_name, job_kind, worker_key)` through `scripts/absurd_worker_contracts.py` before handler side effects.
3. If rejected, mark failed/dead-letter; do not run the handler.
4. Write queue events and workflow events.
5. Never mutate canonical graph tables from ordinary workers.

## 3A. Rust-Core / Python-Wet-Clay Law

Rust is the permanent product body. Python remains the fast operating clay.

Local proof:

- `KRAMPUSCHEWING/System_Archive_Docs/core_architecture/root_notes/TODO6969` names the machine shape: `luci CLI -> Command Envelope -> Absurd durable workflow journal -> Bytewax mandatory stream interstate -> hot tiny routers/deterministic algos first -> model residency scheduler -> ... -> signal aggregator -> CLAIM/EVIDENCE/HYPOTHESIS split -> proof/oracle/audit -> graph staging -> operator response`.
- `KRAMPUSCHEWING/System_Archive_Docs/core_architecture/root_notes/TODO6969` also states: core Rust body = `luci CLI`, command envelope, kernel authorization, receipt writer, proof gate runner, inference scheduler client, graph/proof safety scanner.
- `00_PROJECT_BRAIN/rust_port_candidacy_registry.json` already separates NOW/LATER/NO Rust candidates.
- `01_REPOS/lucidota_etl/Cargo.toml` already declares the Rust workspace: `lucidota-core`, `lucidota-db`, `lucidota-intake`, `lucidota-kernel`, `lucidota-launcher`, `lucidota-audit`, `lucidota-workers`, `lucidota-chrono-ledger`.
- `01_REPOS/claudecode/rust/README.md` and `./claw` prove the Claw/CLAUD Code surface is already a safe-Rust local agent shell.
- `01_REPOS/claudecode/rust/crates/claw-cli/src/command_envelope.rs` is the command-envelope seed.

Port rule:

1. Authority, queue, custody, receipts, graph gates, resource policy, and long-running services belong in Rust.
2. Python is allowed for operator-side experiments, scrapers/adapters, notebooks, model glue, River/sklearn/scipy work, and temporary repair lanes.
3. A Python lane that becomes stable must be registered as a Rust-port candidate or explicitly marked "keep Python" with evidence.
4. No Python lane becomes the source of truth merely because it worked once.

## 3B. Safe Ops / Shitty-Laptop Throughput Law

The machine must not thrash itself to death. Throughput comes from bounded streaming, hot deterministic routers, and explicit model residency—not from loading everything.

Wired profile:

```bash
source scripts/lucidota_safe_ops_env.sh
```

That profile is sourced by `./claw`, the DeepSeek/Mamba/Needle launchers, and the Indy_READs watchers. It sets the safe-laptop envelope:

- one default worker lane,
- bounded batch/chunk sizes,
- BLAS/tokenizer thread caps,
- 4GB-VRAM defaults with reserve,
- context defaults for tiny models,
- direct graph writes blocked unless explicit gate variables are set.

Local proof docs:

- `.lucidota_agents/specialists/04_vram_budget_enforcer.txt` — preflight CUDA/VRAM before model/LoRA/embedding/llama.cpp; reject insufficient reserve; no full-file buffering; enforce limits/batch.
- `.lucidota_agents/specialists/11_token_packing_quartermaster.txt` — stream source chunks, respect token and overlap limits, emit chunk manifests with hashes/source offsets.
- `pypeline/math/model_vram_scheduler.py` — 4GB-GPU advisory scheduler for Mamba/DeepSeek/LoRA/embeddings.
- `scripts/gpu_runtime_budget.py` and `scripts/model_runner_config.py` — runtime budget checks.

4GB video-card operating shape:

```text
Needles / deterministic scouts stay cheap and hot.
DeepSeek R1 distill / Mamba / Bonsai / LoRA lanes are scheduled, not blindly resident.
Embeddings flash in, write receipts, and evict.
Sub-200ms routing path never waits for a cold LLM.
```

## 4. Canonical Graph Boundary

Canonical graph truth requires:

1. evidence refs,
2. authority class,
3. graph-promotion preflight,
4. command envelope when materialization is requested,
5. guarded helper path,
6. graph journal append,
7. materialization receipt.

Safety command:

```bash
python3 scripts/canonical_graph_write_scanner.py --output /tmp/canonical_graph_write_scan.json
```

A PASS means direct graph write patterns are either blocked, test fixtures, or explicitly allowed helper paths.

## 4A. Three-Table Mental Model

Do not create endless schemas to explain the same thing.

The practical mental model is three logical tables:

1. **OBJECT** — things: people, books, files, concepts, tools, claims, models, adapters.
2. **EVENT** — what happened: ingest, parse, route, score, operator decision, model call, promotion attempt.
3. **EDGE** — relationships: supports, contradicts, derived-from, owns, used-by, happened-before, part-of.

GO-25 is the active ontology skin over that model. Existing SQL files may provide indexes, queues, materialized views, policies, and lane-specific staging tables, but the operator-facing design should keep collapsing back to OBJECT/EVENT/EDGE unless a new structure proves its right to exist.

Active ontology proof: `OFFICIAL_ONTOLOGY.json`, `BOOKS/ACTIVE_ONTOLOGY.json`, `06_SCHEMA/016_go_graph_core.sql`, `06_SCHEMA/078_operator_ontology_fidelity_runtime.sql`.

## 5. Compaction / Slop Law

PocketFlow baseline: `01_REPOS/PocketFlow/pocketflow/__init__.py` is the simplicity mirror.

```bash
python3 scripts/slop_audit_law.py --paths scripts/absurd_river_worker.py scripts/absurd_worker_contracts.py
```

Thresholds:

- >5x PocketFlow: review now.
- >10x: split, template, or move mode tables into schema.
- Never delete code to reduce size; archive/ingest it through KRAMPUSCHEWING/KORPUS and keep hashes.

## 6. Fast Local Product Path

Use the local file product to prove end-to-end operator value without graph mutation:

```bash
python3 scripts/lucidota_cli.py --base-dir 05_OUTPUTS/cases new-case demo --max-files 100 --redaction strict
python3 scripts/lucidota_cli.py --base-dir 05_OUTPUTS/cases run-pipeline demo --source <drop_dir>
python3 scripts/lucidota_cli.py --base-dir 05_OUTPUTS/cases status demo
python3 scripts/lucidota_cli.py --base-dir 05_OUTPUTS/cases export demo
```

This proves custody, parsing, receipts, resume/export behavior, and operator review packaging. It does not prove canonical graph truth.

Case-file law:

- Every workspace receives a standard case file number (`KE26-#####`) in `case_workspace.json`.
- Every workspace receives a deterministic `case_hash`.
- Every accepted file receives SHA-256 custody in the intake manifest.
- Duplicate content is grouped by SHA-256, not filename vibes.
- Case packets are hash-addressed and include the standard case number when the workspace metadata is present.

Hypertimeline law:

- `scripts/hypertimeline_engine.py` is the bulk communications/timeline ingester.
- It accepts only standard `KE26-#####` or explicit legacy `CASE-YYYYMMDD-SLUG` case keys.
- It links imported sources to `lucidota_investigation.case_file`, `artifact`, and `case_artifact`, then writes timeline messages into `lucidota_commdump`.
- It streams large JSON arrays instead of full-loading by default.
- It dedupes candidate large artifacts by SHA-256/hardlink where safe.

Graph-population law:

- Raw evidence becomes OBJECT/EVENT/EDGE candidates first.
- Population means staged, deduped, receipt-backed graph candidates.
- Approval/materialization still requires the graph promotion gate; no model or scraper gets to write canonical truth directly.

## 7. Hard-Math Arsenal

Use algorithms as bounded tools, not hidden controllers:

- `ALGOS/rete_bandit_gate.py` — deterministic pruning plus bandit routing.
- `ALGOS/regret_engine.py` — regret/EV ranking.
- `ALGOS/shap_attribution.py` — attribution math.
- `ALGOS/xgboost_objective.py` — gradient/tree objective helpers.
- `ALGOS/serpentina_self_righting.py` — recovery priority under failure economics.
- `ALGOS/hard_truth_math.py` — evidence/stylometry/vector helpers.
- `ALGOS/decision_hygiene.py` — decision hygiene scoring.

Operational rule: an algorithm can rank, gate, score, or route. It cannot directly write truth.

## 7A. Scraper Nirvana Ladder

Every scraper is only an adapter. Blessedness is ranked by determinism and operator safety:

1. **Buddha form** — official API, export, local dump, database snapshot, static file, or documented feed.
2. **Clean HTTP adapter** — requests/httpx against stable URLs with bounded retries, hashes, provenance, and no ambient browser.
3. **Structured parser** — parse HTML/PDF/EPUB/TXT deterministically after storing raw source.
4. **Headless browser fallback** — browser used only when the target requires runtime rendering/auth interaction and a safer adapter is unavailable.
5. **Playwright desperation** — least blessed. It must identify itself as fallback/desperation, avoid hidden control flow, emit receipts, and never be treated as canonical truth.

The safe-ops profile exports `LUCIDOTA_SCRAPER_IDEAL=adapter_buddha_form`, `LUCIDOTA_BROWSER_FALLBACK_TIER=playwright_desperation_only`, and `LUCIDOTA_ALLOW_AMBIENT_BROWSER=0`. Browser body-capture reports include the scraper form so desperation does not masquerade as clean evidence.

## 7B. Unique Invention Map

These names are not random vibes; they are operator design handles:

- **Indy_READs** — sidekick book/intake brain: watches books, chunks safely, stages LoRA cartridges, keeps reading memory aligned to GO-25.
- **Krampus Express / KRAMPUSCHEWING** — high-speed preserve-now, digest-later stomach for slop, old code, evidence, and strange artifacts. Cutting active surface means feeding Krampus, not deleting memory.
- **Diogenes Kernel** — authority kernel: command envelope, permission boundary, receipt discipline, graph-promotion guard.
- **PercyphonAI / PercyhpnAI** — operator-named efficient reasoning organ. Current code evidence includes `ALGOS/percyphon.py`; the name is promoted as a first-class concept even where spelling variants exist.
- **Needles** — cheap hot scouts/lanes for fast local routing and evidence pokes before expensive model calls.
- **FairyFuse / ternary / Bonsai / Mamba / DeepSeek / LoRA lanes** — constrained-model fabric: small, scheduled, specialized organs, not one bloated always-on model.
- **GO-25** — current ontology spine for operator concepts; ROOT-414 remains archived provenance.

Rule: if a concept looks weird, first map it as an invention and search for its implementation evidence before judging it as slop.

## 7C. Paused Labs

Paused means preserved, not discarded.

- **AHOY** is paused and kept. Its simulator/training/strategy files remain preserved under `ahoy_sim/`, `06_SCHEMA/ahoy/`, `05_OUTPUTS/ahoy/`, and the root script launchers were moved out of the active launch surface to `scripts/legacy/ahoy/`.
- Safe ops exports `LUCIDOTA_AHOY_PAUSED=1`.
- Launch orchestrators must not start AHOY unless that variable is explicitly disabled by the operator.
- AHOY can come back later as a lab, benchmark, or algorithm mine; it is not part of the smooth-launch critical path.

## 8. Release Gate

Before declaring a tranche stable:

```bash
python3 -m pytest tests/test_absurd_river_worker_contract.py -q
python3 scripts/canonical_graph_write_scanner.py --output /tmp/canonical_graph_write_scan.json
python3 scripts/lucidota_status_ledger.py --check
```

Then update:

- `00_PROJECT_BRAIN/STATUS_LEDGER.md`
- `05_OUTPUTS/status_ledger.json`

The ledger must name exact evidence paths and blockers. No vague completion claims.

## 9. Active-Surface Pruning Ledger

- `scripts/unified_absurd_ingest_worker.py` was audited as a 19.94x PocketFlow monolith and moved out of the active script root to `scripts/legacy/unified_absurd_ingest_worker.py` after copying the exact hashed body into `KRAMPUSCHEWING/Script_Corpses/`. Current active replacement path: `scripts/absurd_queue_spine.py` + `scripts/absurd_consume_one.py` + `scripts/absurd_worker_contracts.py` + bounded lane workers.
