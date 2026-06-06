# SERVICES AGENT STARTUP LAW

1. Read root `CLAUDE.md` + `AGENTS.md` first.
2. Services are independent — no cross-service imports.
3. Each service must handle its backend being down gracefully.
4. Memory is the canonical persistence layer — don't invent parallel storage.
5. Research uses the IterResearch DeepResearcher as primary engine.
