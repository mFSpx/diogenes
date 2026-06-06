╔══════════════════════════════════════════════════════════════╗
║              ODYSSEUS — OPERATOR'S MANUAL                   ║
║                    LAW OF ROOT EDITION                       ║
║                                                              ║
║  "Receipts, not prose. Evidence, not claims."                ║
╚══════════════════════════════════════════════════════════════╝


## META

| Field | Value |
|---|---|
| Manual Version | 1.0.0 |
| Schema | `lucidota.odysseus.root_manual.v1` |
| Generated | 2026-06-06T08:28:59Z |
| Source | `01_REPOS/odysseus` (dev) |
| Origin | https://github.com/pewdiepie-archdaemon/odysseus |
| Files Extracted | 857 |
| Total Chars | 16,137,970 |
| GO-25 Chunks | 3591 |
| O-75 Chunks | 6450 |
| ROOT-414 Hashes | 10041 |
| Total Shapes | 20082 |
| RiverML Available | True |
| Stream Features | 46 |
| Extraction Time | 11.75s |
| Snapshot Cron | Every Friday 06:00 UTC |

> **Receipt:** `05_OUTPUTS/brag/riverml_extract_receipt_20260606T082654.json`


## 1.0 INTRODUCTION

Odysseus is a full-stack AI personal assistant web UI. It provides chat,
email, calendar, contacts, notes, memory, document management, research,
model serving (Cookbook), gallery, web search, and more — all through a
self-hosted web interface.

### 1.1 What Odysseus Is

- **Self-hosted AI companion** — runs on your hardware, connects to any
  OpenAI-compatible LLM endpoint (local or remote)
- **Full-stack web app** — FastAPI backend, vanilla JS frontend, SQLite persistent storage
- **MCP-native** — Built-in MCP server support for tool-using agent workflows
- **Open source** — MIT-licensed, community-developed

### 1.2 What Odysseus Is NOT

- NOT a model server — it calls out to Ollama/vLLM/OpenAI/etc.
- NOT a replacement for infrastructure — it needs a working SearXNG,
  ChromaDB, and optionally ntfy, Radicale, Dovecot
- NOT a multi-tenant platform — single-user design with auth

### 1.3 Architecture Overview

```
┌──────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Browser    │────▶│  FastAPI Backend  │────▶│  LLM Endpoint   │
│  (VanillaJS) │     │  (core + routes)  │     │  (Ollama/vLLM)  │
└──────────────┘     └──────┬───────────┘     └─────────────────┘
                            │
                    ┌───────┴────────┐
                    │  Services      │
                    │  Research      │
                    │  Memory/RAG    │
                    │  Search        │
                    │  TTS/STT       │
                    │  Docs          │
                    └───────┬────────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
         ┌────────┐  ┌──────────┐  ┌──────────┐
         │SQLite  │  │ChromaDB  │  │SearXNG   │
         │(data)  │  │(vectors) │  │(search)  │
         └────────┘  └──────────┘  └──────────┘
```

> **Receipt:** 49 route handlers in `routes/`, 105 core files in `core/` + `src/`,
> 39 service modules in `services/`, 147 UI modules in `static/js/`


## SUBSYSTEM BREAKDOWN

| # | Subsystem | Files | Description |
|---|-----------|-------|-------------|
| 1 | `test` | 444 | Pytest test suite — unit and integration tests |
| 2 | `ui` | 147 | Frontend JS — editor, email, calendar, research, compare, model UI |
| 3 | `core` | 105 | Core data models, session management, auth, middleware, platform compat |
| 4 | `api` | 49 | FastAPI route handlers — chat, auth, email, docs, search, calendar, etc. |
| 5 | `service` | 39 | Backend services — research, memory, search, TTS/STT, docs, faces, shell |
| 6 | `other` | 37 | Root-level configs (pyproject.toml, Dockerfile, etc.) |
| 7 | `script` | 16 | CLI tools — odysseus-* command suite |
| 8 | `integration` | 7 | Claude Code skill, Codex plugin, scope-gated agent API |
| 9 | `mcp` | 5 | MCP servers — memory, email, RAG, image gen |
| 10 | `companion` | 4 | Companion services (email, calendar integration) |
| 11 | `infra` | 3 | Docker, CI/CD, deployment config |
| 12 | `config` | 1 | Application configuration (SearXNG, etc.) |


## 2.0 API ROUTES — COMMAND SURFACE

| # | Route Module | Purpose |
|---|--------------|---------|
| 1 | `auth_routes.py` | Authentication — login, logout, 2FA, API tokens |
| 2 | `chat_routes.py` | Chat — streaming LLM responses, session management |
| 3 | `email_routes.py` | Email — IMAP read/send, drafts, folders |
| 4 | `calendar_routes.py` | Calendar — CalDAV events, reminders, recurrence |
| 5 | `contacts_routes.py` | Contacts — CardDAV address book |
| 6 | `note_routes.py` | Notes — CRUD, tags, search |
| 7 | `memory_routes.py` | Memory — persistent user facts, categories |
| 8 | `document_routes.py` | Documents — upload, convert, search, PDF forms |
| 9 | `search_routes.py` | Search — web search via SearXNG/DuckDuckGo |
| 10 | `research_routes.py` | Research — deep multi-step research agent |
| 11 | `cookbook_routes.py` | Cookbook — model serve/download/manage |
| 12 | `gallery_routes.py` | Gallery — image viewing/management |
| 13 | `preset_routes.py` | Presets — saved chat model presets |
| 14 | `task_routes.py` | Tasks — todo/checklist management |
| 15 | `session_routes.py` | Sessions — chat session CRUD |
| 16 | `model_routes.py` | Models — model discovery/configuration |
| 17 | `mcp_routes.py` | MCP — Model Context Protocol server management |
| 18 | `skills_routes.py` | Skills — skill importer/manager |
| 19 | `shell_routes.py` | Shell — terminal command execution |
| 20 | `backup_routes.py` | Backup — data export/import |
| 21 | `webhook_routes.py` | Webhook — generic webhook receiver |
| 22 | `tts_routes.py` | TTS — text-to-speech |
| 23 | `stt_routes.py` | STT — speech-to-text |
| 24 | `upload_routes.py` | Upload — file upload handling |
| 25 | `codex_routes.py` | Codex — scope-gated agent API (Claude/Codex) |

> **Receipt:** 49 route files in `01_REPOS/odysseus/routes/`


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


## 6.0 INTEGRATIONS — EXTERNAL SURFACES

### 6.1 Claude Code Integration

| # | Component | Path |
|---|-----------|------|
| 1 | Skill Definition | `integrations/claude/skills/odysseus/SKILL.md` |
| 2 | API Helper | `integrations/claude/skills/odysseus/scripts/odysseus_api.py` |
| 3 | Setup Guide | `integrations/claude/README.md` |

Scope-gated API surface (`/api/codex/*`):
- `GET /api/codex/capabilities` — discover enabled tools
- `GET/POST /api/codex/todos` — task management
- `GET/POST/DELETE /api/codex/memory` — persistent memory
- `GET/POST/DELETE /api/codex/calendar/events` — calendar management
- `GET/POST/DELETE /api/codex/documents` — document library
- `GET /api/codex/emails` — email reading (scope-gated)
- `POST /api/codex/emails/draft` — email drafting
- `POST /api/codex/emails/send` — email sending (requires `email:send`)
- `GET/POST /api/codex/cookbook/*` — model serve lifecycle

### 6.2 Codex (OpenAI) Integration

| # | Component | Path |
|---|-----------|------|
| 1 | Plugin Config | `integrations/codex/.codex-plugin/plugin.json` |
| 2 | API Helper | `integrations/codex/scripts/odysseus_api.py` |
| 3 | Skill Definition | `integrations/codex/skills/odysseus/SKILL.md` |

### 6.3 MCP Servers

| # | Server | Purpose |
|---|--------|---------|
| 1 | `memory_server.py` | Memory read/write MCP surface |
| 2 | `email_server.py` | Email MCP tools |
| 3 | `rag_server.py` | RAG/document retrieval MCP |
| 4 | `image_gen_server.py` | Image generation via diffusion |

> **Receipt:** 5 MCP servers, 2 integration bundles, 2 skill definitions


## 7.0 SCHEDULED OPERATIONS

| # | Schedule | Operation | Script | Receipt |
|---|----------|-----------|--------|---------|
| 1 | Every Friday 06:00 UTC | Full Odysseus code extraction + DB snapshot | `scripts/odysseus_friday_snapshot.py` | `05_OUTPUTS/odysseus_snapshot/*.json` |
| 2 | Daily 02:00 UTC | LUCIDOTA daily backup | `scripts/lucidota_daily_backup.sh` | system crontab |

### 7.1 ABSURD Queue Contracts

Registered external commands:
- `scripts/odysseus_riverml_extract.py` — full RiverML extraction
- `scripts/odysseus_friday_snapshot.py` — Friday snapshot workflow

> **Receipt:** Registered in `scripts/absurd_queue_spine.py` ALLOWED_EXTERNAL_COMMANDS


## 8.0 EVIDENCE LEDGER

| # | Claim | Receipt | Status |
|---|-------|---------|--------|
| 1 | 857 files extracted from odysseus | `05_OUTPUTS/brag/riverml_extract_receipt_*.json` | PASS |
| 2 | 20,082 shapes generated (GO-25 + O-75 + ROOT-414) | same | PASS |
| 3 | 46 RiverML stream features extracted | same | PASS |
| 4 | Friday cron installed | `crontab -l` entry | PASS |
| 5 | ABSURD queue registered | `scripts/absurd_queue_spine.py` | PASS |
| 6 | Allowed external commands | `scripts/absurd_queue_spine.py:59-77` | PASS |
| 7 | Snapshot wrapper script | `scripts/odysseus_friday_snapshot.py` | PASS |
| 8 | This manual generated | `20260606T082859Z` | PASS |

### 8.1 Known Gaps / Future Work

| # | Gap | Priority | Notes |
|---|-----|----------|-------|
| 1 | GO-25 ingestion to Postgres | High | Shapes JSONL exists but not yet in `lucidota_korpus.brag_cell` |
| 2 | Embedding generation | Medium | `--embed` flag available but not run for full corpus |
| 3 | BRAG ABSURD worker (RETE/LTC) | Medium | Worker exists but not wired to odysseus queue |
| 4 | ByteWax installation | Low | Requested but not yet installed for stream processing |



---
*Generated by scripts/odysseus_manual_builder.py*
*Law of ROOT Edition — Receipts, not prose*
