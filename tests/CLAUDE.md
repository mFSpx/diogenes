# TESTS — Pytest Test Suite

Test files at `tests/test_*.py`. Restricted by pytest.ini (excludes 01_REPOS, 03_VAULT, 04_RUNTIME, 05_OUTPUTS, KRAMPUSCHEWING).

## Key Rules

- **Test gating law:** Route through `scripts/test_receipt_gate.py run --scope <scope> -- <command>` for persistent changes.
- **T0:** Syntax checks for touched scripts.
- **T1:** Unit tests for touched subsystem only.
- **T2:** Live smoke when DB/API/systemd surfaces changed.
- **T3:** Full suite only for cross-subsystem or pre-merge.
- **Postgres is source of truth for test receipts.** Don't invent file-plane fallbacks.
- Run: `python3 -m pytest -q` or specific `python3 -m pytest tests/test_*.py -q`.
