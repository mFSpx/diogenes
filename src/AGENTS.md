# SRC AGENT STARTUP LAW

1. Read root `CLAUDE.md` + `AGENTS.md` first.
2. No direct graph writes. Ever. Use the promotion gate.
3. Memory goes through `MemoryManager` — no parallel storage.
4. Embeddings through the provider stack — no direct model imports.
5. Config through `src/config.py` — no hardcoded values.
6. MCP lifecycle through `MCPManager` — no manual server starts.
