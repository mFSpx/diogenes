# ODYSSEUS RUNTIME MANUAL — SERVICES & CORE



Generated 2026-06-06T08:28:59Z


## 3.0 SERVICES — RUNTIME MODULES

| # | Service | Module | Purpose |
|---|---------|--------|---------|
| 1 | `research` | `services/Research Handler` | Iterative deep-research agent (LLM-in-the-loop). Uses DeepResearcher with fallback to legacy orchestrator then basic web search. |
| 2 | `memory` | `services/Memory Service` | Persistent fact storage with vector embedding. Uses ChromaDB + fastembed. Categories: fact, preference, skill, session. |
| 3 | `search` | `services/Search Service` | Meta-search engine. Primary: SearXNG. Fallback: DuckDuckGo. Caching, analytics, ranking, content extraction. |
| 4 | `docs` | `services/Document Service` | Document text extraction. Supports PDF, EPUB, Office formats via markitdown. Chunking + embedding for RAG. |
| 5 | `hwfit` | `services/Hardware Fit` | "What Fits?" hardware detection + quant-aware model fit scoring. GPU VRAM, RAM, CPU capability detection. |
| 6 | `faces` | `services/Face Service` | Face detection/tagging for gallery photos. |
| 7 | `tts` | `services/TTS Service` | Text-to-speech generation. |
| 8 | `stt` | `services/STT Service` | Speech-to-text transcription. |
| 9 | `shell` | `services/Shell Service` | Remote terminal command execution via SSH. |
| 10 | `youtube` | `services/YouTube Service` | YouTube transcript extraction for research/document processing. |


## 4.0 CORE — APPLICATION FOUNDATION

### 4.1 Core Package (`core/`)

| # | Module | Purpose |
|---|--------|---------|
| 1 | `models.py` | Pure data containers: ChatMessage, Session |
| 2 | `session_manager.py` | Session persistence, message history |
| 3 | `auth.py` | AuthManager — password hashing, JWT, 2FA |
| 4 | `middleware.py` | Security headers, CORS, rate limiting |
| 5 | `database.py` | SQLAlchemy session factory |
| 6 | `constants.py` | App-wide constants |
| 7 | `exceptions.py` | Exception classes: SessionNotFound, LLMServiceError |

### 4.2 Source Modules (`src/`)

| # | Module | Purpose |
|---|--------|---------|
| 1 | `llm_core.py` | LLM call/stream/discovery — OpenAI-compatible API |
| 2 | `chat_handler.py` | Chat orchestration, tool execution, MCP routing |
| 3 | `chat_processor.py` | Message processing, context building |
| 4 | `agent_loop.py` | Autonomous agent loop with tool execution |
| 5 | `memory.py` | MemoryManager — persistent fact storage with RAG |
| 6 | `mcp_manager.py` | MCP server lifecycle management |
| 7 | `embeddings.py` | Embedding generation (ChromaDB + fastembed) |
| 8 | `document_processor.py` | Document upload/conversion pipeline |
| 9 | `cookbook_serve_lifecycle.py` | Model server lifecycle (download/serve/stop) |
| 10 | `research_handler.py` | Deep research orchestration |
| 11 | `config.py` | Settings management |
| 12 | `tool_schemas.py` | Tool definition schemas |

> **Receipt:** 105 source files in `core/` + `src/`


## 5.0 RIVER ML STREAMING PIPELINE

### 5.1 Stream Architecture

Every code chunk extracted from odysseus flows through a RiverML online
streaming feature extractor. The pipeline produces 46-dimensional feature
vectors per chunk in a single online pass.

| # | Feature Group | Features | Purpose |
|---|--------------|----------|---------|
| 1 | Size | byte_len, line_count, avg_line_len | Code scale metrics |
| 2 | Structure | blank_lines, comment_lines, import_lines | Code organization |
| 3 | Complexity | function_defs, branch_points, return_points, nest_depth | Cyclomatic signals |
| 4 | Entropy | unique_tokens, type_token_ratio | Lexical diversity |
| 5 | Language | upper_ratio, symbol_ratio | Code vs prose detection |
| 6 | Subsystem | subsys_* (one-hot) | Routing classification |
| 7 | Type | type_code, type_doc | Media type detection |
| 8 | TF-IDF | tfidf_* (top 20) | Content fingerprinting |

### 5.2 Stream Stats

| Metric | Value |
|--------|-------|
| Stream Position | 10041 |
| Churn Count | 10041 |
| Feature Count | 46 |
| RiverML Version | 0.25.0 |
| Model | BagOfWords (ngram 1-2) + MultinomialNB |
| Scaler | StandardScaler |

### 5.3 Pipeline Stages

```
Raw Doc ──▶ BagOfWords ──▶ StandardScaler ──▶ Feature Vector ──▶ GO-25 Tag
              │                                       │
              ▼                                       ▼
        TF-IDF Weights                          Classifier (NB)
                                                    │
                                                    ▼
                                              Subsystem Route
```

> **Receipt:** Generated from 20082 chunks across 857 files
