# SERVICES — Backend Service Modules

Research, memory, search, TTS/STT, docs, faces, shell, YouTube.

## Key Rules

- **Each service is independent.** Services don't import from each other.
- **Service configuration comes from settings/env, not hardcoded.**
- **Graceful degradation.** If a service's backend is down (SearXNG, ChromaDB, etc.), return helpful errors, not crashes.
- **Memory service** is the canonical persistence layer for user facts.
- **Research service** uses DeepResearcher (IterResearch) with fallbacks.
- **Search service** prefers SearXNG, falls back to DuckDuckGo.
