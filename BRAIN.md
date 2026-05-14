# LUCIDOTA Brain Snapshot

Generated: 2026-05-13 America/Vancouver

## Truth Sources

- `00_PROJECT_BRAIN/STATUS.md` — current build status and verified edge.
- `00_PROJECT_BRAIN/BUILD_PLAN_AUDIT.md` — lifecycle bars and checklist truth.
- `05_OUTPUTS/big_board.json` — regenerated dashboard JSON from local docs + Postgres counters.
- `06_SCHEMA/014_indy_runtime.sql` — Indy_Reads memory, quiet queue, and redacted auth inventory.

## Current Runtime Spine

Operator → Clawd → Indy_Reads brief/queues → CKDOG1 kernel / DBOS / Postgres / CAS / Survey / Body_Capture.

## Hardening Notes

- Bars are not forced to 100%; current `big_board.json` min bar is 52 and max is 100.
- Survey now exposes an optional Tree-sitter structural slot with explicit `unavailable` reporting when grammars are absent.
- Bytewax live cursor now uses a transaction-scoped advisory lock to avoid duplicate live cursor workers.
- Wake Bus CTE batch shape is audited by `scripts/lucidota_wake_bus_audit.py`.
- EQ-001..EQ-100 recovered validator IDs are covered by `scripts/lucidota_validator_stress.py` synthetic slop corpus.
