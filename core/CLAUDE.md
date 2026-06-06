# CORE — Python Runtime Module

Core chat infrastructure: streaming LLM, session management, model routing, authentication.

## Key Rules

- **No database logic here.** `core/models.py` are pure data containers. Persistence is in `SessionManager`.
- **No direct graph writes.** Ever.
- **Streaming is the default.** Blocking calls need justification.
- Used by: `scripts/`, `src/`, `routes/` — don't break import chains.
