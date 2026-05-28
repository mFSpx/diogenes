# CURRENT SWARM BRIEF — compare-driven, byte-tight, no-loss

"Save This Prompt, Pass on this Handoff:"

Objective: keep the compare-driven scenario lane moving with the smallest useful actions, while preserving byte-perfect evidence ingestion and no-loss queue handoff.

## Minimal actionable steps

### 1) `scripts/goal_scenario_batch.py`
- Input: latest `05_OUTPUTS/goals/goal_scenario_compare_*.json`
- Action: run `--compare-report <latest>` plus `--holdout-stride 3`
- Output: one new holdout receipt
- Success: `GOAL_SCENARIO_HOLDOUT=PASS` and receipt path on disk

### 2) `scripts/goal_scenario_compare.py`
- Input: new holdout receipt + previous holdout receipt
- Action: compare rule conditions/actions and identify stable / new / lost / morphing
- Output: one compare receipt plus `scenario_focus`
- Success: `GOAL_SCENARIO_COMPARE=PASS`

### 3) `scripts/language_router.py`
- Input: evidence / queue / ops text
- Action: keep GO-25 strict and prefer deterministic routing
- Output: `route_text(...)` receipt with `decision_hygiene`, `lane`, and `model_route`
- Success: `GO25_STRICT`, `model_calls_performed=False`, `canonical_graph_writes_performed=False`

### 4) `scripts/goal_agent_packet.py`
- Input: current objective/task
- Action: emit a single exact top-level JSON worker packet
- Output: `lucidota.worker_order.v1`
- Success: required keys present, token floor/ceiling preserved

### 5) `scripts/model_output_contract_audit.py`
- Input: live Groq/local receipts
- Action: normalize `decisions -> decision_pairs`, parse fenced JSON, recover truncated fields
- Output: compact audit receipt
- Success: missing fields surfaced explicitly, not hidden

### 6) `scripts/absurd_queue_spine.py` + `scripts/absurd_consume_one.py`
- Input: queued worker payloads
- Action: enforce queue integrity and kernel authorization before any execution
- Output: queue/event/dead-letter receipts only
- Success: no canonical graph writes, no hidden mutation, no loss of queue semantics

### 7) `scripts/korpus_embedding_worker.sh` + Indy library ingest surfaces
- Input: books / evidence / chunks
- Action: preserve byte-perfect ingestion and embeddings, chunk deterministically, stage without drift
- Output: embeddings and ingestion receipts
- Success: no silent truncation, no field loss, no hidden rewrite

## Current steering rule

- Latest compare focus: `normal`, `ops`, `queue_integrity`
- If the next compare shifts toward `evidence_ingest`, keep it only if the receipt delta stays stable and actionable; otherwise broaden back to `queue_integrity` / `normal`.

## Current proof rule

- No claim without a receipt path.
- No progress without focused tests.
- No canonical graph writes unless explicitly allowed.
- Side lanes may think; shared surfaces must stay exact.

