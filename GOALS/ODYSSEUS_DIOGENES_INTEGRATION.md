# Diogenes x Odysseus: The Symbiote Strategy

**Tag:** `Northern.Strike_xINdyREADs`
**Launch Date:** Tuesday, June 16, 2026 — 09:00 AM PDT
**Status:** PREPPING

---

## Durability Audit (2026-06-06)

**Odysseus DURABLE: NO** (partial)

| Component | Durable? | Gap |
|-----------|----------|-----|
| Sessions/Messages | YES | SQLite persistence |
| Vector memory | YES | ChromaDB w/ disk persistence |
| Task definitions | YES | SQLite `scheduled_tasks` table |
| Task execution | **NO** | `asyncio.Semaphore(1)` — in-memory only |
| Deep research | **NO** | In-flight state lost on crash |
| MCP servers | **NO** | Ephemeral subprocesses, no supervisor |
| Task queue | **NO** | No queue system whatsoever |
| Dead letter queue | **NO** | Failed tasks marked `error` but no DLQ |
| Retry policy | PARTIAL | HTTP retry only for LLM calls, not tasks |
| Workflow engine | **NO** | No state machine, no saga pattern |

### What Diogenes Brings

| Diogenes Component | Fixes Odysseus | Mechanism |
|-------------------|----------------|-----------|
| ABSURD queue | Task execution durability | Postgres `FOR UPDATE SKIP LOCKED` — tasks survive crash/restart |
| RETE bandit gate | Smart model routing | Routes tasks to right model based on complexity/cost |
| elastic.rs | Process isolation | Rust-based hot-path for queue management |
| BitVLA vision | Image processing | SigLIP vision encoder on port 7845 |
| System resource watchdog | OOM prevention | nvidia-smi + dmesg monitoring, auto-throttle |
| Receipt system | Audit trail | Every task produces a receipt in `05_OUTPUTS/receipts/` |

---

## 4-Agent Groq Configuration

```bash
# Agent 1: Deep Research Agent
GROQ_MODEL="openai/gpt-oss-120b"
GROQ_TEMP=0.3
# Agent 2: Code Generation Agent
GROQ_MODEL="meta-llama/llama-4-scout-17b-16e-instruct"
GROQ_TEMP=0.7
# Agent 3: Vision/Image Agent
GROQ_MODEL="meta-llama/llama-4-scout-17b-16e-instruct"
GROQ_TEMP=0.5
# Agent 4: Orchestrator/Router Agent
GROQ_MODEL="openai/gpt-oss-120b"
GROQ_TEMP=0.8
```

---

## Installation Plan

1. Clone Odysseus to `01_REPOS/odysseus/` ✓ DONE
2. Install via Docker: `docker-compose up --build`
3. Configure `.env` with:
   - Ollama endpoint: `http://host.docker.internal:11434`
   - Groq API key from secrets
   - LUCIDOTA MCP endpoints
4. Deploy Diogenes backplane as an Odysseus MCP server
5. Test task durability (kill process, verify resume)

---

## Integration Points

- **MCP Server**: `scripts/mcp_lucidota_models.py` registers as an Odysseus MCP tool
- **Vision**: BitVLA vision server at port 7845
- **Queue**: ABSURD queue replaces in-memory task scheduler
- **Routing**: RETE bandit gate intercepts model selection
- **Receipts**: All task executions produce LUCIDOTA-compatible receipts

---

## Official Tagline

> "Odysseus handles the UI and the hype. Diogenes handles the bare-steel execution underneath."
