# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Required reading before writing code

LUCIDOTA enforces a startup-law chain. Before editing or adding code, read in order:

1. `AGENTS.md` — Agent Startup Law (Dev Library reuse, Goal Handoff, Model Economy).
2. `00_PROJECT_BRAIN/READTHISFIRST_CURRENT.md` — current operating snapshot.
3. `00_PROJECT_BRAIN/TICKLETRUNK.json` + `00_PROJECT_BRAIN/TICKLETRUNK.md` — Dev Library manifest (the map of every existing tool/script/schema/workflow).
4. `00_PROJECT_BRAIN/ACTIVE_INSTRUCTION_INDEX.md` — pointer to the canonical instruction set.
5. `00_PROJECT_BRAIN/BLUEPRINT_FIRST_MODEL_SECOND_PSEUDOLAW.md` — workflow hygiene law.
6. `GOALS/CURRENT_HANDOFF.md` — current goal state and resume command.

Before inventing anything new, run `python3 scripts/dev_library_scan.py --query <topic>` and prefer copy/adapt/reuse over writing from scratch. **Do not mutate sovereign toolbox originals** unless explicitly ordered.

## Core doctrine (non-obvious, load-bearing)

- **Receipts, not prose.** A claim with no fresh receipt under `05_OUTPUTS/` or no live Postgres fact in `lucidota_control.runtime_status_fact` is theater, not status. "Looks green" is not completion.
- **Blueprint first, model second.** The workflow path lives in source/schema/queues, not in prompts. Models do bounded extraction/summarization at named nodes; they never act as hidden controllers. Slop thresholds against PocketFlow (`01_REPOS/PocketFlow/pocketflow/__init__.py`, ~100 lines): >5x review, >10x split/template, >20x explicit receipted justification.
- **Rust owns the core; Python is wet clay.** Authority, queues, custody, receipts, graph gates, and long-running services belong in Rust (`01_REPOS/claudecode/rust/` and `01_REPOS/lucidota_etl/`). Python is for adapters, scrapers, notebooks, model glue, and temporary repair lanes. A stable Python lane must register as a Rust-port candidate in `00_PROJECT_BRAIN/rust_port_candidacy_registry.json` or be explicitly marked "keep Python".
- **No worker writes canonical graph truth directly.** Canonical graph promotion requires: evidence refs → authority class → graph-promotion preflight → command envelope → guarded helper → graph journal append → materialization receipt. Ordinary workers stage candidates only.
- **Capability preservation.** Do not delete code or remove ability unless the operator explicitly asks or receipts prove it's dead. Archive into `KRAMPUSCHEWING/Script_Corpses/` with hashes instead of deleting.
- **GO-25 ontology.** Operator-facing graph collapses to three logical tables: OBJECT, EVENT, EDGE (skinned by GO-25 in `OFFICIAL_ONTOLOGY.json`). ROOT-414 is archived reference only.

## Repository layout (numbered ontology)

- `00_PROJECT_BRAIN/` — law shelf: specs, instruction index, TICKLETRUNK manifest, RFCs, ACTIVE_SPEC.
- `01_REPOS/` — source trees. `claudecode/rust/` is the LUCIDOTA-owned Rust CLI fork (`claw`). `doggystyle/` is the separately-maintained CKDOG1 kernel — do not modify here. `PocketFlow/` is the simplicity yardstick. Upstream forks (`llama.cpp`, `llxprt-code`, `needle`, etc.) are tooling candidates, not LUCIDOTA-owned.
- `02_RECORDS_OFFICE/` — historical audit/report archives.
- `03_VAULT/`, `09_STORAGE/` — local hoard/storage layers (gitignored except README).
- `04_RUNTIME/` — live runtime state, observation center, LoRA cartridges (gitignored except README).
- `05_OUTPUTS/` — proof shelf: receipts, dead letters, status JSON. **Where receipts go.** (gitignored except README).
- `06_SCHEMA/` — Postgres SQL contract layer (numbered `NNN_*.sql`). The contract layer.
- `07_SURFACES/` — generated/sidecar/promoted/archived surfaces.
- `ALGOS/` — deterministic algorithm arsenal (bandit, RETE, regret, SHAP, etc.). Algorithms can rank/gate/score/route; they cannot write truth.
- `BOOKS/` — ontology schemas (GO-25 graph item/edge envelopes, identity rules).
- `GOALS/` — operator goal handoff state. Always update `CURRENT_HANDOFF.md` and append to `GOAL_LOG.md` on goal steps; use the exact prefix `"Save This Prompt, Pass on this Handoff:"`.
- `KRAMPUSCHEWING/` — preserve-now/digest-later stomach for slop and old code. Script corpses, evidence, paused labs (AHOY).
- `core/`, `pypeline/`, `services/`, `src/`, `TOOLS/` — Python runtime modules.
- `scripts/` — worker/gate/receipt layer (~459 scripts). The active workforce.
- `tests/` — pytest suite (`tests/test_*.py`). pytest.ini excludes `01_REPOS`, `03_VAULT`, `04_RUNTIME`, `05_OUTPUTS`, `KRAMPUSCHEWING`.

## Execution spine

```
operator intent or raw artifact
  -> custody / command envelope
  -> ABSURD/Postgres work order
  -> bounded worker (contract-checked at dequeue)
  -> receipt / event / dead letter
  -> evidence / claim / hypothesis candidate
  -> authority mapping
  -> graph promotion gate
  -> optional operator-confirmed materialization
  -> review/export/status surface
```

Two Postgres databases (see `OFFICIAL_ONTOLOGY.json`):
- `lucidota_state` (env `LUCIDOTA_GO_STATE_DSN`): workflow/runtime/control state. No graph truth here.
- `lucidota_storage` (env `LUCIDOTA_GO_STORAGE_DSN`): graph items, edges, layers, staging packets, journals.

Durable queue is **ABSURD/Postgres**. DBOS naming in old receipts is legacy provenance, not current architecture. Worker law: select with `FOR UPDATE SKIP LOCKED` → validate `(queue_name, job_kind, worker_key)` against `scripts/absurd_worker_contracts.py` → handler → events/receipts/dead letters. Never mutate canonical graph tables from an ordinary worker.

## Common commands

Always source the safe-ops profile before running heavy workers, model servers, scrapers, or the CLI. It caps CPU threads, sets a 4GB-VRAM envelope (the dev box is a GTX 1650), and blocks ambient browser/graph-write paths. `./claw` and the watcher launchers source it for you.

```bash
source scripts/lucidota_safe_ops_env.sh
```

Status / facts / Dev Library:

```bash
python3 scripts/system_runtime_facts_refresh.py --execute
python3 scripts/dev_library_scan.py --query <topic>
python3 scripts/slop_audit_law.py --paths <file-or-dir>
python3 scripts/lucidota_status_ledger.py --check
python3 scripts/canonical_graph_write_scanner.py --output /tmp/canonical_graph_write_scan.json
psql postgresql:///lucidota_state -Atc "SELECT fact_key,fact_value FROM lucidota_control.runtime_status_fact WHERE subsystem='system' ORDER BY fact_key;"
```

Local file product (proves the spine end-to-end without graph mutation):

```bash
python3 scripts/lucidota_cli.py --base-dir 05_OUTPUTS/cases new-case demo --max-files 100 --redaction strict
python3 scripts/lucidota_cli.py --base-dir 05_OUTPUTS/cases run-pipeline demo --source <drop_dir>
python3 scripts/lucidota_cli.py --base-dir 05_OUTPUTS/cases status demo
python3 scripts/lucidota_cli.py --base-dir 05_OUTPUTS/cases export demo
```

ABSURD queue + scenario batch / swarm:

```bash
python3 scripts/absurd_queue_spine.py --help
python3 scripts/absurd_consume_one.py --help
python3 scripts/goal_scenario_batch.py --scenario-count 12 --batch-size 4 --json
python3 scripts/goal_scenario_batch.py --scenario-count 18 --batch-size 6 --holdout-stride 3 --json
python3 scripts/goal_scenario_compare.py --current <holdout.json> --baseline <batch.json> --json
python3 scripts/goal_swarm_brief.py --json
python3 scripts/goal_swarm_dispatch.py --target local --task <task> --jobs 2 --json
```

Tests (pytest, restricted by `pytest.ini`):

```bash
python3 -m pytest -q                                  # full suite
python3 -m pytest -q tests/test_absurd_*_contract.py  # worker-contract gates
python3 -m pytest -q tests/test_absurd_queue_spine_contract.py   # single test file
python3 -m pytest tests/test_absurd_queue_spine_contract.py::test_name  # single test
```

Release gate before declaring a tranche stable:

```bash
python3 -m pytest tests/test_absurd_river_worker_contract.py -q
python3 scripts/canonical_graph_write_scanner.py --output /tmp/canonical_graph_write_scan.json
python3 scripts/lucidota_status_ledger.py --check
```

Then update `00_PROJECT_BRAIN/STATUS_LEDGER.md` and `05_OUTPUTS/status_ledger.json` with exact evidence paths.

## The Rust CLI (`./claw`)

`./claw` is the operator shell — a LUCIDOTA-owned fork of claudecode at `01_REPOS/claudecode/rust/`. The launcher (`./claw`) sources `scripts/lucidota_safe_ops_env.sh`, starts the strict model stack and Indy_READs watchers, loads secrets from `~/.config/lucidota/secrets.env` and `/tmp/lucidota_groq_key`, then execs the release binary with `--permission-mode danger-full-access`.

Rust workspace (`01_REPOS/claudecode/rust/Cargo.toml`) crates: `api`, `claw-cli`, `commands`, `compat-harness`, `lsp`, `plugins`, `runtime`, `server`, `tools`. Workspace lints forbid `unsafe_code` and warn on clippy `pedantic`. Build with `cargo build --release` from `01_REPOS/claudecode/rust/`.

Provider lanes (from `./claw` env): local llama.cpp at `http://127.0.0.1:8080/v1`, Mamba RAM at `:8081`, Mamba GPU at `:8083`, Groq at `https://api.groq.com/openai/v1`. The orchestrator agent (`LLXPRT.md`, `00_PROJECT_BRAIN/PROJECT_2501_ADMIN_PROMPT.md`) is Groq `openai/gpt-oss-120b`; local does the heavy work.

## Operating conventions

- **Goal handoff.** At every persistent goal step: refresh `GOALS/CURRENT_HANDOFF.md` with `X/N` progress and append the same entry to `GOALS/GOAL_LOG.md`. Use `scripts/goal_handoff.py` and `GOALS/GOAL_HANDOFF_PROMPT.md`. Keep goal notes in `GOALS/`; do not create folder sprawl.
- **Receipt scope must be named.** Every acceptance receipt says which mode it proves: `LOCAL_FILE_PRODUCT`, `ABSURD_POSTGRES_RUNTIME`, `GRAPH_PROMOTION_RUNTIME`, `CHRONO_EVIDENCE_LEDGER`, `KRAMPUS_KORPUS_INGESTION`, or `MODEL_RUNTIME`. See `00_PROJECT_BRAIN/CANONICAL_FINISHED_PRODUCT_MAP.md` for the mode table.
- **Mutation class declaration.** Every component is one of: `read_only`, `receipt_only`, `custody_writer`, `queue_writer`, `candidate_writer`, `authority_gate`, `materializer`, `external_effect`. If a component can't name its class, it cannot mutate.
- **AHOY is paused, not deleted.** Safe-ops exports `LUCIDOTA_AHOY_PAUSED=1`. Do not start AHOY orchestrators unless the operator explicitly unpauses.
- **Scraper Nirvana Ladder.** Prefer official API/export/dump > clean HTTP adapter > structured parser > headless fallback > Playwright desperation. Browser body-capture reports must label the scraper form; desperation does not masquerade as clean evidence.
- **Language router.** Stays GO-25 strict inside workflows; only flips to final-print behavior at explicit workflow-end markers.
- **Batching.** Shuttling intent to another worker/model uses 2–7 primitives per batch. Smallest batch that preserves meaning.

## What not to do

- Don't write code without first searching `dev_library_scan.py` for an existing implementation.
- Don't bypass the graph-promotion gate or write directly to canonical graph tables from workers.
- Don't delete scripts, docs, or experiments to "clean up." Archive into `KRAMPUSCHEWING/Script_Corpses/` with hashes.
- Don't trust stale handoffs, old prompts, or uncited prose as current status.
- Don't add slop audit/proof gates to low-risk reversible work. Heavy gates go at graph materialization, destructive actions, external effects, and slow-lane review.
- Don't claim completion without a fresh receipt path or live DB fact.
