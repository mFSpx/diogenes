# DoltHub + PromptFlow Integration — 40-Item Shortlist
_Generated 2026-05-31 via 5 ADHD-framed DoltHub agents + 1 PromptFlow sweep agent_

## DoltHub Technical Capabilities (1–10)

1. `dolt_diff_summary('main~3','main')` returns `schema_change` and `data_change` booleans per table — gate graph-promotion on `data_change=TRUE AND schema_change=FALSE`; route schema-change commits into the `06_SCHEMA/` review lane automatically.
2. `DOLT_SCHEMA_DIFF('commit_a','commit_b','graph_item')` returns full `from_create_statement`/`to_create_statement` DDL — wire as canonical diff input for every `NNN_*.sql` migration; versioned DDL delta receipts become automatic.
3. `dolt_commits` sees ALL branches; `dolt_log` only sees current branch HEAD ancestry — use `dolt_commits` for multi-arm bandit cross-arm queries; `dolt_log` silently drops arm histories.
4. `CALL DOLT_COMMIT('--skip-empty','-a','-m','tick')` no-ops when nothing changed — without this flag a 15-second heartbeat generates 5,760 empty commits/day.
5. `to_commit='WORKING'` in `dolt_diff_$TABLE` exposes uncommitted working-set as CDC stream — live feature vector before any commit lands.
6. Dolt Prolly tree shares physical chunks across branches — branching a 10M-row `graph_item` table costs near-zero storage; speciation is free.
7. `dolt_diff()` is O(changed rows) not O(total rows) — diff latency flat as tables grow into tens of millions.
8. `dolt_schemas` tracks views/triggers/events but NOT `CREATE TABLE` — table structure requires `DOLT_SCHEMA_DIFF()`, not `dolt_schemas`.
9. **DoltgreSQL (Postgres-wire Dolt) is Beta** — `dolt_add/commit/log/status` confirmed; `DOLT_DIFF/SCHEMA_DIFF/blame` parity NOT confirmed. Audit before substituting MySQL-Dolt API.
10. `DOLT_COMMIT()` implicitly closes SQL transaction AND creates commit atomically — safe at 15s intervals; races at sub-second cadence.

## Wild/Unexpected Use Cases (11–20)

11. **Database Insurance** — Dolt as MySQL replica; bad agent write → `dolt_reset('--hard','pre-agent-run-commit')` instead of full restore.
12. **Branches as speciation events** — `dolt_checkout('-b','schema-migration-experiment')` = complete isolated copy; merge back as schema-promotion ceremony.
13. **Structured metadata in commit messages** — embed `job_kind=river_score outcome=1 latency_ms=342`; parse via `REGEXP_SUBSTR` on `dolt_log` — real training signal, no separate event table. 🔥
14. **Per-arm bandit branches** — each arm owns a Dolt branch; cross-arm training via `dolt_commits WHERE message LIKE '%arm=a%'`.
15. `dolt_diff_$TABLE WHERE diff_type='modified'` IS a CDC stream — no Debezium, no WAL parsing.
16. **Agentic kill switch** — agent writes attributed to named commits; watchdog resets bad agent run in 15 seconds, surgical not nuclear.
17. **Dead code corpus loop via Dolt** — commit corpse with message `archive=true hash=<sha>`; reingestion pipeline queries `dolt_log WHERE message LIKE '%archive=true%'`. 🔥
18. `dolt_log` = fossil record; `DOLT_SCHEMA_DIFF()` = morphological diff; `dolt_diff_summary()` = vitals check — three questions, three SQL surfaces.
19. **Time-travel `AS OF`** — `SELECT * FROM graph_item AS OF 'commit_abc123'` = canonical replay lane for chrono-evidence ledger mode.
20. Prolly tree storage bounded by diff surface not write volume — organism grows genome, not fat.

## DoltHub × LUCIDOTA Integration (21–30)

21. Wire `dolt_diff_summary()` into `runtime_status_fact` as `fact_key='dolt_last_schema_change_commit'`.
22. Replace `canonical_graph_write_scanner.py` with `SELECT table_name FROM dolt_diff_summary(...) WHERE schema_change=TRUE`.
23. Dolt MySQL replica of `lucidota_storage` — every materialization receipt = named commit; receipt shelf and DB history = same artifact.
24. Auto-generate before/after DDL receipt via `DOLT_SCHEMA_DIFF('HEAD','WORKING',<table>)` before every `NNN_*.sql` is applied.
25. Dolt branches as A/B containers for `ALGOS/` bandit changes — merge only on improved regret.
26. `dolt_commits` cross-branch as River ML training corpus — train on structured commit messages.
27. Add `dolt_commit_class` to `absurd_worker_contracts.py` — `receipt_only` workers auto-commit; `read_only` workers blocked from `DOLT_COMMIT()`.
28. `scripts/dolt_schema_sentinel.py` — poll `dolt_diff_summary()` every 60s; on `schema_change=TRUE` enqueue `schema_review` ABSURD job.
29. `dolt_reset('--hard',<commit>)` as graph rollback primitive — one SQL call, no compensating-transaction chain.
30. **Branch per goal step** — `goal/<step-id>` branch at session start; branch history IS the goal handoff receipt.

## PromptFlow Integration (31–35)

31. `@tool` decorator wraps Python functions as named PromptFlow nodes — wrap `absurd_queue_spine.py`; ABSURD queue becomes drag-drop primitive.
32. `activate_config when=` conditional branching — map graph-promotion gate logic to PromptFlow DAG condition node; gate becomes visible.
33. Batch run mode executes flow against JSONL → `outputs.jsonl` — wire against `05_OUTPUTS/` receipts as eval harness.
34. PromptFlow traces every node I/O — redirect to `05_OUTPUTS/promptflow_traces/` as `MODEL_RUNTIME` receipt class; automatic receipt coverage.
35. PromptFlow `Connection` abstraction for provider credentials — replace `secrets.env` manual loading; provider switching = DAG config change.

## Combined Synthesis Moves (36–40)

36. DoltHub + PromptFlow branch-per-flow — `flow/corpus-ingest-v3` branch; schema delta between flow versions as structured artifact.
37. PromptFlow batch run → Dolt commit cycle — `outputs.jsonl` auto-committed with `flow=X pass_rate=0.87`; commit log = evaluation history queryable by River ML.
38. **Two-layer agentic undo** — PromptFlow `activate_config when=false` disables node; Dolt `dolt_reset` rolls back data; combined no-bad-write primitive.
39. **`claw` as Dolt+PromptFlow orchestration surface** — `claw dolt commit --job-kind X` wraps structured message embedding; `claw flow run <dag.yaml>` invokes PromptFlow; operator shell = single entry point. 🔥
40. **The self-evolution loop fully wired** — drift detected → ABSURD job → Dolt experiment branch → SQL applied → diff receipt → PromptFlow batch validates → if pass: `dolt_merge` + version increment + Dolt records promotion. Full audit trail, one-command rollback. 🔥

---
_Architecture law (from the friend): Dolt versions the fossils. Postgres runs the organism. KRAMPUSCHEWING eats the dead. PromptFlow breeds variants. Worker contracts stop mutations from killing the body._
