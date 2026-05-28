# LUCIDOTA READ THIS FIRST — CURRENT

Generated: `2026-05-26T23:39:40.181461Z`

## Start here

1. Read `AGENTS.md`.
2. Read `00_PROJECT_BRAIN/TICKLETRUNK.md` and `00_PROJECT_BRAIN/ACTIVE_INSTRUCTION_INDEX.md`.
3. Read `GOALS/CURRENT_HANDOFF.md`.
4. Use `python3 scripts/dev_library_scan.py --query <topic>` before inventing anything.
5. Use receipts and runtime facts, not prose memory, as the authority layer.

## What is ledgered vs theater

- **Ledgered:** `lucidota_control.runtime_status_fact`, `05_OUTPUTS/*` receipts, `*_latest` audit docs, Postgres tables, and generated JSONL/MD receipts.
- **Theater:** stale handoffs, old prompts, uncited prose, guessed status, and any "looks green" claim without a fresh receipt.

## Ontology rule

LUCIDOTA is a highly advanced ternary/binary ontological truth machine: truth means the current status as understood, no more and no less. The active math is base 35 plus `1`, `0`, and `-1` lane logic, and the fast path stays deterministic.
- The language router stays GO-25 strict inside workflows and only flips to final-print behavior at explicit workflow-end markers.

## Slop laws in effect

- Dev Library reuse first; do not pave sovereign originals.
- Blueprint first, model second.
- Proportional proof only; don't freeze low-risk work with useless audit bloat.
- Temporal truth: current means freshly verified.
- Capability preservation: do not remove ability unless the operator explicitly asks or receipts prove it is dead.

## Orchestration lane

- This agent is orchestration only.
- Groq and the local system do the heavy work; this agent coordinates, checks, and keeps the loop cheap.
- Prefer bounded scripts and current Postgres facts over long prose.
- Keep the loop cheap: one instruction, one receipt, one verification.
- Scenario batches and worker packets run through `scripts/goal_scenario_batch.py` and `scripts/goal_agent_packet.py` using GO-25 JSON packets.
- Holdout evaluation for scenario batches runs through `scripts/goal_scenario_batch.py`'s holdout report path, which scores held-out routes, emits promoted rules, and keeps the decision tree compact.
- The scenario batch CLI accepts `--holdout-stride N` to emit the holdout evaluator receipt directly when the operator wants a compact train/holdout comparison instead of a plain training batch.
- The scenario batch CLI accepts `--compare-report <path>` to bias family ordering from the latest compare report before it generates the next batch or holdout receipt.
- Receipt-to-receipt comparison for scenario batches runs through `scripts/goal_scenario_compare.py`, which compares rule conditions/actions across reports and proposes the next GO-25 seed plus scenario focus.
- The scenario matrix now explicitly includes evidence-ingest and queue-integrity families so the batch lane tests byte-perfect evidence ingestion, embeddings, and no-loss JSON queue handoff alongside the normal/adversarial/noisy/fast-slow cases.
- `GOALS/SWARM_CURRENT_BRIEF.md` plus `scripts/goal_swarm_brief.py` turn the compare-driven loop into compact per-script worker packets for new sessions.
- Worker packets now carry an explicit output contract: required fields are `status`, `result`, `next_action`, and `receipt_path`, with minimum decision-pair evidence requested from the worker.
- Exact JSON envelopes should be given breathing room: the worker packet contract now recommends a 256-token floor and 512-token ceiling for compact, parseable responses.
- Confidence-weighted symbol condensation runs through `scripts/graph_symbol_condensation.py` against graph/Postgres evidence, with Bayes-style confidence and morphing claims.
- Temporal comparison runs through `scripts/graph_symbol_compare.py` to diff condensation receipts and propose the next GO-25 seed.
- Symbol dispatch runs through `scripts/graph_symbol_dispatch.py` to fan that next seed out to Groq/local worker packets.
- Model receipt normalization and drift audit runs through `scripts/model_output_contract_audit.py`, which canonicalizes `decisions` into `decision_pairs`, parses fenced JSON text when present, and recovers scalar fields from truncated envelopes when possible.
- Durable queue handoff runs through `scripts/goal_swarm_dispatch.py` and `scripts/absurd_queue_spine.py` when the packet should actually be consumed asynchronously.

## Current batching rule

- When shuttling intent to another worker/model, batch ontology in small chunks, usually 2-7 primitives.
- Use the smallest batch that preserves meaning.
- If the receiver's answer makes the next instruction obvious, send only the next small instruction.

## Current operating snapshot

- Active DB state: `lucidota_state`
- Storage DB: `lucidota_storage`
- Active ontology: `GO-25 / OBJECT-EVENT-EDGE`
- Current queue health: succeeded `2465`, failed `0`, queued `0`, total `2545`
- Baseline queue reconciliation fact: `absurd_queue_health.dead_letter_rows=45` and unresolved dead letters `0`
- Active queues: `21`
- Groq wiring: `groq / llama-3.1-8b-instant`
- Hitman loop: `scheduled`

## Operator commands that matter

```bash
python3 scripts/system_runtime_facts_refresh.py --execute
python3 scripts/dev_library_scan.py --query <topic>
python3 scripts/slop_audit_law.py --query <topic>
python3 scripts/goal_scenario_batch.py --scenario-count 12 --batch-size 4 --json
python3 scripts/goal_scenario_batch.py --scenario-count 18 --batch-size 6 --holdout-stride 3 --json
python3 scripts/goal_scenario_batch.py --scenario-count 18 --batch-size 6 --holdout-stride 3 --compare-report 05_OUTPUTS/goals/<compare>.json --json
python3 scripts/goal_scenario_compare.py --current 05_OUTPUTS/goals/<holdout>.json --baseline 05_OUTPUTS/goals/<batch>.json --json
python3 scripts/goal_swarm_brief.py --json
python3 - <<'PY'
from scripts.goal_scenario_batch import build_holdout_report, write_report
report = write_report(build_holdout_report(objective='...', target='groq|cohere|local', scenario_count=18, batch_size=6, holdout_stride=3, packet={'schema':'lucidota.worker_order.v1','target':'local','intent':'scenario_batch'}))
print(report['report_path'])
PY
python3 scripts/graph_symbol_condensation.py --symbols OBJECT,EVENT,EDGE --json
python3 scripts/graph_symbol_compare.py --current <current-report> --baseline <baseline-report> --json
python3 scripts/graph_symbol_dispatch.py --compare <compare-report> --lanes groq,local --json
python3 scripts/goal_swarm_dispatch.py --target local --task <task> --jobs 2 --json
pytest -q tests/test_absurd_*_contract.py
psql postgresql:///lucidota_state -Atc "SELECT fact_key,fact_value FROM lucidota_control.runtime_status_fact WHERE subsystem='system' ORDER BY fact_key;"
```

## Where the rest of the map lives

- Exhaustive Dev Library: `00_PROJECT_BRAIN/TICKLETRUNK.md` and `00_PROJECT_BRAIN/TICKLETRUNK.json`
- Tree index: `00_PROJECT_BRAIN/FILESYSTEM_TREE_INDEX_CURRENT.md`
- Postgres audit: `00_PROJECT_BRAIN/POSTGRES_AUDIT_CURRENT.md`
- Live receipts: `05_OUTPUTS/`
- Queue/goal continuity: `GOALS/CURRENT_HANDOFF.md` and `GOALS/GOAL_LOG.md`

## Fast truth anchors

- ABSURD queue jobs: `{'succeeded': 2465, 'dead_lettered': 78, 'cancelled': 2}`
- State schemas: `lucidota_control=60, lucidota_learning=14, lucidota_runtime=10, lucidota_go=1, lucidota_indy=3, lucidota_pivot=4, lucidota_ternary=3, lucidota_vault=5, lucidota_bus=2`
- Storage schemas: `lucidota_archaeology=20, lucidota_chatdump=4, lucidota_commdump=3, lucidota_etl=5, lucidota_go=23, lucidota_hardmath=5, lucidota_indy=6, lucidota_investigation=10, lucidota_korpus=43, lucidota_learning=8, lucidota_runtime=5, lucidota_semantic=1, lucidota_spine=12, lucidota_vault=5, lucidota_workflow_foundry=1`

## Read this if you only read one thing

This repo runs on receipts, schema facts, and gated workers. If a claim does not have a fresh receipt or live DB fact, it is not current.
