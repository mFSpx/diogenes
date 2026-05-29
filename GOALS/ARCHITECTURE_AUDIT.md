# GOALS Crash-Recovery Architecture Audit

Generated during active GOALS goal, 2026-05-26.

## Verdict

Use the local GOALS handoff layer. Do **not** install a larger task system right now.

Why: LUCIDOTA already has heavyweight runtime recovery in Postgres/ABSURD/ETL, and what was missing was a tiny agent-facing crash-resume card. The right wire-in is startup instructions + active instruction index + status ledger/recovery matrix checks, not a new daemon, DB table, hook farm, or agent framework.

## Local scan findings

Evidence receipt: `05_OUTPUTS/goals/local_recovery_architecture_scan_20260526T011932Z.txt`.

Existing recovery systems:

- Runtime queue recovery: `lucidota_control.absurd_queue_job`, `absurd_queue_dead_letter`, `queue_transition_audit`, `workflow_event`, `dbos.workflow_status`.
- ETL crash recovery: `lucidota_etl.step_checkpoint` and `lucidota_etl.dead_letter` in `lucidota_storage`.
- Recovery doctrine: `00_PROJECT_BRAIN/ETL_PIPELINE/DBOS_CONTRACT.md` says expensive Python work is split into steps and each step writes a Postgres checkpoint.
- Recovery catalog: `scripts/recovery_matrix.py` and `06_SCHEMA/105_recovery_receipt.sql`.
- Session-ish recovery: `.claw/sessions/*.json`, Codex session JSONL archives in `03_VAULT/.../.codex/sessions`, and old `SESSION_RESUME_2026-05-14.md`.
- Goal-ish docs: old ingested `00_PROJECT_BRAIN/GOALS.md`, current `lucidota_goal_audit.py`, RFC goal audit matrix.

Gap found:

- Nothing existing gave a **cheap per-goal, per-step, LLM-readable current handoff** with `X/N`, next command, and brief technical/dev notes.
- DB recovery solves jobs and data. It does not tell a fresh LLM exactly where the previous LLM crashed inside an operator goal.

## External scan findings

- Maestro has a full local-first agent harness with specs, tasks, evidence, contracts, handoffs, and principles; it is useful-looking but much larger than this need and requires another runtime/toolchain.
- Partio captures AI sessions into git checkpoints and supports Codex, but it installs hooks and checkpoint branches; good for commit/session history, too much for a zero-overhead step card.
- mdtask is a Markdown task CLI with no DB/server/GUI, but it is task tracking, not handoff recovery, and its license is not plain FOSS for our purposes.
- A small public handoff gist matches the useful shape: concise markdown sections, exact files/commands/tests/blockers/next steps. This is pattern inspiration, not a dependency.
- Copilot CLI has local session resume and rollback snapshots, but that is CLI-specific and not a repo-native LUCIDOTA handoff law.

## Wiring decision

Keep GOALS as a lazy file protocol:

1. `AGENTS.md` startup law points agents at GOALS.
2. `00_PROJECT_BRAIN/ACTIVE_INSTRUCTION_INDEX.md` lists GOALS files as active instruction sources.
3. `GOALS/CURRENT_HANDOFF.md` is the only current crash card.
4. `GOALS/GOAL_LOG.md` is a concise step history, not a transcript dump.
5. `GOALS/GOAL_HANDOFF_PROMPT.md` is the copy/paste rule.
6. `scripts/goal_handoff.py` is optional convenience only; currently under 100 meaningful code lines.
7. `scripts/recovery_matrix.py` includes `goal_handoff_check` so recovery audits know this layer exists.
8. `00_PROJECT_BRAIN/STATUS_LEDGER.md` tracks the GOALS layer as verified when checks pass.

## Anti-yap law

GOALS is the **Yap Trap**: it captures the LLM urge to narrate and compresses it into the smallest useful handoff.

Budget:

- `CURRENT_HANDOFF.md`: one screen, ideally under 250 words.
- Dev notes: max two short sentences.
- Per-step log entry: facts, evidence, next action. No therapy essays.
- Heavy analysis goes only into a named audit/receipt when it changes the engineering decision.

## Current decision

No new external dependency. Local solution wins because it is repo-native, inspectable, grep-able, test-covered, under the local 100-LOC smell threshold, and imposes zero background CPU/DB/token cost unless an agent explicitly updates a handoff.

## Chain + telemetry addendum

The GOALS layer now has three more lazy, bounded tools: `goal_chain.py` for next-goal packets, `goal_system_index.py` for system/function inventory, and `goal_telemetry.py` for explicit snapshot/monitor receipts. They are not daemons; they run only when invoked or for an operator-requested bounded window.

Recovery matrix coverage includes `goal_chain_check`, `goal_system_index`, `goal_telemetry_snapshot`, and `goal_model_fabric_stop_heavy`, so the crash-recovery surface can prove both pass-on intent and resource state without inventing a new scheduler.
