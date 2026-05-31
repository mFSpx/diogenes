# LUCIDOTA Evolution Spine — Phantom Integration Design
## Source: Phantom agent runtime logic extracted 2026-05-31

## What Phantom does (the 7 steps we're stealing)
1. GATE: After each session, lightweight LLM judge decides: "does this have learning signal?"
2. QUEUE: Sessions with signal → absurd_queue (queue_name=learning_evolution)
3. DRAIN: Cron or demand trigger (queue>=5) drains batch
4. REFLECT: LLM subprocess reads batch + current config, proposes mutations
5. INVARIANT: 9 deterministic rules block bad mutations before they land
6. COMMIT: Changes versioned with parent pointer + append-only evolution log
7. ROLLBACK: Pre-snapshot restored on any invariant fail

## LUCIDOTA-native mapping (NO TypeScript)
| Phantom | LUCIDOTA equivalent |
|---|---|
| SQLite session store | lucidota_control.workflow_event |
| evolution_queue | lucidota_control.absurd_queue (queue_name=learning_evolution) |
| Haiku gate judge | lucidota_evolution_gate.py (dead_letter_rate / success_rate threshold) |
| Drain subprocess (LLM) | lucidota_evolution_drain.py (Groq llama-3.3-70b) |
| phantom-config/ mutation | lucidota_control.runtime_status_fact UPDATE |
| version.json + evolution-log.jsonl | lucidota_learning.evolution_version + evolution_drain_run |
| 9 invariant rules | 5-rule check in lucidota_evolution_drain.py |
| Constitution (immutable) | absurd_worker_contract table (never mutate) |
| Metrics snapshot | river_score.success_rate at time of version commit |
| Rollback | Postgres transaction rollback (no file snapshot needed) |

## What was already wired but never connected
- fn_train_operator_feedback_batch() — defined 2024, never called
- bandit_router.py (LinUCB, Thompson, epsilon-greedy) — defined, never called
- river_prediction — computed, never used for routing
- operator_feedback_signal.consumed_at — always NULL
- jepa_energy.py — 444 lines, never wired
- pypeline/math/regret_engine.py — defined, never called

## Files being built (this session)
- scripts/lucidota_feedback_drain.py — closes operator_feedback_signal loop
- scripts/lucidota_evolution_gate.py — Phantom-equivalent gate
- scripts/lucidota_evolution_drain.py — Phantom-equivalent reflect subprocess
- scripts/lucidota_bandit_update.py — wires bandit_router.py to actual routing
- scripts/bytewax_abductive_blender.py — modified to weight by river_score
- 06_SCHEMA/136_evolution_spine.sql — gate_run, drain_run, version tables
