# CORE AGENT STARTUP LAW

1. Read root `CLAUDE.md` + `AGENTS.md` first.
2. `core/models.py` is pure data — no side effects. Keep it that way.
3. Session persistence goes through `SessionManager`, not direct file/DB writes.
4. LLM streaming is the hot path — keep it fast.
5. auth/security changes need receipts.
