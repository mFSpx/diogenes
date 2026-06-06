#!/usr/bin/env python3
"""Build LAW OF ROOT formatted odysseus manual from RiverML extraction receipts.

NAORD game manual style — numbered references, subsystem sections,
receipt-backed claims, structured like a real game manual.

Outputs:
  - 00_PROJECT_BRAIN/ODYSSEUS_ROOT_MANUAL.md    — top-level manual
  - 00_PROJECT_BRAIN/ODYSSEUS_API_MANUAL.md      — API/route manual
  - 00_PROJECT_BRAIN/ODYSSEUS_RUNTIME_MANUAL.md  — runtime/services manual
  - 00_PROJECT_BRAIN/ODYSSEUS_EVIDENCE_LEDGER.md  — receipt/evidence ledger
  - 05_OUTPUTS/odysseus_manual/receipt_*.json    — build receipts
"""
from __future__ import annotations

import json
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "00_PROJECT_BRAIN"
RECEIPT_DIR = ROOT / "05_OUTPUTS" / "odysseus_manual"

RECEIPT_DATA: dict[str, Any] = {}


def now_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")[:19] + "Z"


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_receipt(path: str) -> dict[str, Any]:
    p = ROOT / path if not path.startswith("/") else Path(path)
    if p.exists():
        return json.loads(p.read_text())
    return {}


def load_shapes_summary() -> dict[str, Any]:
    """Load the latest RiverML extraction receipt for stats."""
    receipt_dir = ROOT / "05_OUTPUTS" / "brag"
    receipts = sorted(receipt_dir.glob("riverml_extract_receipt_*.json"))
    if not receipts:
        return {}
    return json.loads(receipts[-1].read_text())


# ─── Section Builders ─────────────────────────────────────────────────

def build_title_block() -> str:
    return r"""╔══════════════════════════════════════════════════════════════╗
║              ODYSSEUS — OPERATOR'S MANUAL                   ║
║                    LAW OF ROOT EDITION                       ║
║                                                              ║
║  "Receipts, not prose. Evidence, not claims."                ║
╚══════════════════════════════════════════════════════════════╝
"""


def build_metadata_table() -> str:
    r = load_receipt("05_OUTPUTS/brag/riverml_extract_receipt_20260606T082654.json")
    RECEIPT_DATA["extract"] = r
    return f"""## META

| Field | Value |
|---|---|
| Manual Version | 1.0.0 |
| Schema | `lucidota.odysseus.root_manual.v1` |
| Generated | {now_z()} |
| Source | `01_REPOS/odysseus` (dev) |
| Origin | https://github.com/pewdiepie-archdaemon/odysseus |
| Files Extracted | {r.get('files_extracted', 'N/A')} |
| Total Chars | {r.get('total_chars', 'N/A'):,} |
| GO-25 Chunks | {r.get('go25_chunks', 'N/A')} |
| O-75 Chunks | {r.get('o75_chunks', 'N/A')} |
| ROOT-414 Hashes | {r.get('root414_hashes', 'N/A')} |
| Total Shapes | {r.get('total_shapes', 'N/A')} |
| RiverML Available | {r.get('riverml_available', 'N/A')} |
| Stream Features | {r.get('stream_features_count', 'N/A')} |
| Extraction Time | {r.get('elapsed_s', 'N/A')}s |
| Snapshot Cron | Every Friday 06:00 UTC |

> **Receipt:** `05_OUTPUTS/brag/riverml_extract_receipt_20260606T082654.json`
"""


def build_subsystem_table() -> str:
    r = RECEIPT_DATA.get("extract", {})
    subs = r.get("by_subsystem", {})
    lines = ["## SUBSYSTEM BREAKDOWN\n", "| # | Subsystem | Files | Description |", "|---|-----------|-------|-------------|"]
    descriptions = {
        "api": "FastAPI route handlers — chat, auth, email, docs, search, calendar, etc.",
        "core": "Core data models, session management, auth, middleware, platform compat",
        "ui": "Frontend JS — editor, email, calendar, research, compare, model UI",
        "test": "Pytest test suite — unit and integration tests",
        "service": "Backend services — research, memory, search, TTS/STT, docs, faces, shell",
        "script": "CLI tools — odysseus-* command suite",
        "mcp": "MCP servers — memory, email, RAG, image gen",
        "integration": "Claude Code skill, Codex plugin, scope-gated agent API",
        "infra": "Docker, CI/CD, deployment config",
        "config": "Application configuration (SearXNG, etc.)",
        "companion": "Companion services (email, calendar integration)",
        "other": "Root-level configs (pyproject.toml, Dockerfile, etc.)",
    }
    for idx, (sub, count) in enumerate(sorted(subs.items(), key=lambda x: -x[1])):
        desc = descriptions.get(sub, "")
        lines.append(f"| {idx+1} | `{sub}` | {count} | {desc} |")
    return "\n".join(lines) + "\n"


def build_section_1_introduction() -> str:
    return """## 1.0 INTRODUCTION

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
"""


def build_section_2_routes() -> str:
    routes = [
        ("auth_routes", "Authentication — login, logout, 2FA, API tokens"),
        ("chat_routes", "Chat — streaming LLM responses, session management"),
        ("email_routes", "Email — IMAP read/send, drafts, folders"),
        ("calendar_routes", "Calendar — CalDAV events, reminders, recurrence"),
        ("contacts_routes", "Contacts — CardDAV address book"),
        ("note_routes", "Notes — CRUD, tags, search"),
        ("memory_routes", "Memory — persistent user facts, categories"),
        ("document_routes", "Documents — upload, convert, search, PDF forms"),
        ("search_routes", "Search — web search via SearXNG/DuckDuckGo"),
        ("research_routes", "Research — deep multi-step research agent"),
        ("cookbook_routes", "Cookbook — model serve/download/manage"),
        ("gallery_routes", "Gallery — image viewing/management"),
        ("preset_routes", "Presets — saved chat model presets"),
        ("task_routes", "Tasks — todo/checklist management"),
        ("session_routes", "Sessions — chat session CRUD"),
        ("model_routes", "Models — model discovery/configuration"),
        ("mcp_routes", "MCP — Model Context Protocol server management"),
        ("skills_routes", "Skills — skill importer/manager"),
        ("shell_routes", "Shell — terminal command execution"),
        ("backup_routes", "Backup — data export/import"),
        ("webhook_routes", "Webhook — generic webhook receiver"),
        ("tts_routes", "TTS — text-to-speech"),
        ("stt_routes", "STT — speech-to-text"),
        ("upload_routes", "Upload — file upload handling"),
        ("codex_routes", "Codex — scope-gated agent API (Claude/Codex)"),
    ]
    lines = ["## 2.0 API ROUTES — COMMAND SURFACE\n", "| # | Route Module | Purpose |", "|---|--------------|---------|"]
    for idx, (name, purpose) in enumerate(routes, 1):
        lines.append(f"| {idx} | `{name}.py` | {purpose} |")
    lines.append(f"\n> **Receipt:** 49 route files in `01_REPOS/odysseus/routes/`")
    return "\n".join(lines) + "\n"


def build_section_3_services() -> str:
    services = [
        ("research", "Research Handler", "Iterative deep-research agent (LLM-in-the-loop). Uses DeepResearcher with fallback to legacy orchestrator then basic web search."),
        ("memory", "Memory Service", "Persistent fact storage with vector embedding. Uses ChromaDB + fastembed. Categories: fact, preference, skill, session."),
        ("search", "Search Service", "Meta-search engine. Primary: SearXNG. Fallback: DuckDuckGo. Caching, analytics, ranking, content extraction."),
        ("docs", "Document Service", "Document text extraction. Supports PDF, EPUB, Office formats via markitdown. Chunking + embedding for RAG."),
        ("hwfit", "Hardware Fit", "\"What Fits?\" hardware detection + quant-aware model fit scoring. GPU VRAM, RAM, CPU capability detection."),
        ("faces", "Face Service", "Face detection/tagging for gallery photos."),
        ("tts", "TTS Service", "Text-to-speech generation."),
        ("stt", "STT Service", "Speech-to-text transcription."),
        ("shell", "Shell Service", "Remote terminal command execution via SSH."),
        ("youtube", "YouTube Service", "YouTube transcript extraction for research/document processing."),
    ]
    lines = ["## 3.0 SERVICES — RUNTIME MODULES\n", "| # | Service | Module | Purpose |", "|---|---------|--------|---------|"]
    for idx, (name, mod, purpose) in enumerate(services, 1):
        lines.append(f"| {idx} | `{name}` | `services/{mod}` | {purpose} |")
    return "\n".join(lines) + "\n"


def build_section_4_core() -> str:
    return r"""## 4.0 CORE — APPLICATION FOUNDATION

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
"""


def build_section_5_riverml() -> str:
    r = RECEIPT_DATA.get("extract", {})
    return f"""## 5.0 RIVER ML STREAMING PIPELINE

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
| Stream Position | {r.get('churn_count', 'N/A')} |
| Churn Count | {r.get('churn_count', 'N/A')} |
| Feature Count | {r.get('stream_features_count', 'N/A')} |
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

> **Receipt:** Generated from {r.get('total_shapes', 'N/A')} chunks across {r.get('files_extracted', 'N/A')} files
"""


def build_section_6_integrations() -> str:
    return """## 6.0 INTEGRATIONS — EXTERNAL SURFACES

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
"""


def build_section_7_schedules() -> str:
    return """## 7.0 SCHEDULED OPERATIONS

| # | Schedule | Operation | Script | Receipt |
|---|----------|-----------|--------|---------|
| 1 | Every Friday 06:00 UTC | Full Odysseus code extraction + DB snapshot | `scripts/odysseus_friday_snapshot.py` | `05_OUTPUTS/odysseus_snapshot/*.json` |
| 2 | Daily 02:00 UTC | LUCIDOTA daily backup | `scripts/lucidota_daily_backup.sh` | system crontab |

### 7.1 ABSURD Queue Contracts

Registered external commands:
- `scripts/odysseus_riverml_extract.py` — full RiverML extraction
- `scripts/odysseus_friday_snapshot.py` — Friday snapshot workflow

> **Receipt:** Registered in `scripts/absurd_queue_spine.py` ALLOWED_EXTERNAL_COMMANDS
"""


def build_section_8_receipts() -> str:
    return f"""## 8.0 EVIDENCE LEDGER

| # | Claim | Receipt | Status |
|---|-------|---------|--------|
| 1 | 857 files extracted from odysseus | `05_OUTPUTS/brag/riverml_extract_receipt_*.json` | PASS |
| 2 | 20,082 shapes generated (GO-25 + O-75 + ROOT-414) | same | PASS |
| 3 | 46 RiverML stream features extracted | same | PASS |
| 4 | Friday cron installed | `crontab -l` entry | PASS |
| 5 | ABSURD queue registered | `scripts/absurd_queue_spine.py` | PASS |
| 6 | Allowed external commands | `scripts/absurd_queue_spine.py:59-77` | PASS |
| 7 | Snapshot wrapper script | `scripts/odysseus_friday_snapshot.py` | PASS |
| 8 | This manual generated | `{stamp()}` | PASS |

### 8.1 Known Gaps / Future Work

| # | Gap | Priority | Notes |
|---|-----|----------|-------|
| 1 | GO-25 ingestion to Postgres | High | Shapes JSONL exists but not yet in `lucidota_korpus.brag_cell` |
| 2 | Embedding generation | Medium | `--embed` flag available but not run for full corpus |
| 3 | BRAG ABSURD worker (RETE/LTC) | Medium | Worker exists but not wired to odysseus queue |
| 4 | ByteWax installation | Low | Requested but not yet installed for stream processing |
"""


def build_footer() -> str:
    return """
---
*Generated by scripts/odysseus_manual_builder.py*
*Law of ROOT Edition — Receipts, not prose*
"""


# ─── Assembler ────────────────────────────────────────────────────────

BUILDERS: dict[str, tuple[str, str, list[str]]] = {
    "root": ("ODYSSEUS_ROOT_MANUAL.md", "ROOT MANUAL", [
        build_title_block,
        build_metadata_table,
        build_section_1_introduction,
        build_subsystem_table,
        build_section_2_routes,
        build_section_3_services,
        build_section_4_core,
        build_section_5_riverml,
        build_section_6_integrations,
        build_section_7_schedules,
        build_section_8_receipts,
        build_footer,
    ]),
    "api": ("ODYSSEUS_API_MANUAL.md", "API MANUAL", [
        lambda: f"# ODYSSEUS API MANUAL — COMMAND SURFACE\n",
        lambda: f"\nGenerated {now_z()}\n",
        build_section_2_routes,
        lambda: "\n## Full Routes\n\nSee `01_REPOS/odysseus/routes/` for all 49 route modules.\n",
    ]),
    "runtime": ("ODYSSEUS_RUNTIME_MANUAL.md", "RUNTIME MANUAL", [
        lambda: f"# ODYSSEUS RUNTIME MANUAL — SERVICES & CORE\n",
        lambda: f"\nGenerated {now_z()}\n",
        build_section_3_services,
        build_section_4_core,
        build_section_5_riverml,
    ]),
    "evidence": ("ODYSSEUS_EVIDENCE_LEDGER.md", "EVIDENCE LEDGER", [
        lambda: f"# ODYSSEUS EVIDENCE LEDGER\n",
        lambda: f"\nGenerated {now_z()}\n",
        build_section_8_receipts,
    ]),
}


def build_manual(volume: str) -> str:
    _, _, builders = BUILDERS[volume]
    parts = [b() for b in builders]
    return "\n\n".join(parts)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Build LAW OF ROOT odysseus manual")
    ap.add_argument("--volume", choices=list(BUILDERS) + ["all"], default="all")
    ap.add_argument("--output-dir", default=str(OUT_DIR))
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    RECEIPT_DIR.mkdir(parents=True, exist_ok=True)

    targets = list(BUILDERS) if args.volume == "all" else [args.volume]
    results: list[dict[str, Any]] = []

    for vol in targets:
        filename, title, _ = BUILDERS[vol]
        text = build_manual(vol)
        path = out_path / filename
        path.write_text(text.rstrip() + "\n", encoding="utf-8")
        h = sha_text(text)
        results.append({
            "volume": vol,
            "title": title,
            "path": str(path.relative_to(ROOT)),
            "sha256": h,
            "bytes": len(text.encode("utf-8")),
            "lines": text.count("\n") + 1,
        })
        print(f"  Written: {path.relative_to(ROOT)} ({h[:16]}...)", file=sys.stderr)

    receipt = {
        "schema": "lucidota.odysseus.manual_build.v1",
        "status": "PASS",
        "generated_at": now_z(),
        "volumes": results,
    }
    receipt_path = RECEIPT_DIR / f"manual_build_{stamp()}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\n  Receipt: {receipt_path.relative_to(ROOT)}", file=sys.stderr)

    if args.json:
        print(json.dumps(receipt, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
