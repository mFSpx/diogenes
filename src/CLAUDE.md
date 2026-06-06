# SRC — Python Source Modules

Application source code: memory, chat processing, MCP, embeddings, document processing, research, tool schemas, config.

## Key Rules

- **No direct graph writes.** Use the promotion gate.
- **MemoryManager** is the canonical memory layer — don't invent parallel storage.
- **MCP Manager** handles MCP server lifecycle — don't start MCP servers manually.
- **Embedding provider** at `src/embeddings.py` routes through Groq or local — don't bypass.
- **Config** at `src/config.py` is the canonical settings source.
- **No hardcoded secrets.** All credentials through env or secrets.env.
