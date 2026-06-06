# TESTS AGENT STARTUP LAW

1. Read root `CLAUDE.md` + `AGENTS.md` first.
2. Route tests through `test_receipt_gate.py` for persistent changes.
3. Follow T0 → T1 → T2 → T3 escalation. Don't jump to T3.
4. If DB gate is unavailable, report `DB_BLOCKED` — don't invent file-plane truth.
5. New tests go in `tests/test_<name>.py` matching the module under test.
