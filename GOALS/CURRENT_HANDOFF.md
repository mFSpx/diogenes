# CURRENT GOAL HANDOFF

"Save This Prompt, Pass on this Handoff:"

- Goal: Repo dig + gap-fix tranche complete. Opus build plan session next.
- Generated: `2026-05-29T00:00:00Z`
- Status: 6 repos mapped, integration candidates identified, gap fixes attempted (0 auto-fixed — all require Opus co-design or manual intervention), Opus build plan ready per OPUS_BRIEFING.md §13.

---

## Canon files (do not touch)

- `/home/mfspx/LUCIDOTA/CLAUDE.md`
- `/home/mfspx/LUCIDOTA/AGENTS.md`
- `/home/mfspx/LUCIDOTA/OFFICIAL_ONTOLOGY.json`
- `/home/mfspx/LUCIDOTA/OPUS_BRIEFING.md`
- `/home/mfspx/LUCIDOTA/00_PROJECT_BRAIN/GONN_CANON.md`
- `/home/mfspx/LUCIDOTA/00_PROJECT_BRAIN/INDY_SOUL.md`
- `/home/mfspx/LUCIDOTA/00_PROJECT_BRAIN/OPERATOR_LAW.md`
- `/home/mfspx/LUCIDOTA/00_PROJECT_BRAIN/BUILD_STATE.md`
- `/home/mfspx/LUCIDOTA/00_PROJECT_BRAIN/MODEL_REGISTRY.md`
- `/home/mfspx/LUCIDOTA/00_PROJECT_BRAIN/RUST_ETL_CRATES.md`
- `/home/mfspx/LUCIDOTA/00_PROJECT_BRAIN/INDY_MODEL_STACK.md`
- `/home/mfspx/LUCIDOTA/00_PROJECT_BRAIN/ACTIVE_INSTRUCTION_INDEX.md`
- `/home/mfspx/LUCIDOTA/00_PROJECT_BRAIN/ACTIVE_SPEC/` (all 8 files)
- `/home/mfspx/LUCIDOTA/00_PROJECT_BRAIN/TICKLETRUNK.json`
- `/home/mfspx/LUCIDOTA/00_PROJECT_BRAIN/organ_registry/`
- `/home/mfspx/LUCIDOTA/GOALS/CURRENT_HANDOFF.md`

---

## Prior ground truth (from Sonnet prep, still valid)

- ~92/122 schema files UNAPPLIED (chrono 025-027, ABSURD spine 035, graph-promotion 034/044/052, write-barriers 040/074). Live = base band + GO graph core 016.
- Bytewax→River stream spine BROKEN (service execs bare python3). Only river-governor Hoeffding tree live (~539 samples).
- term_registry = 45 rows (target 75); CO_ACTIVE_TERMS.json MISSING; @26-@50 GO/CO ID collision = convergence landmine.
- Two DBs still split (lucidota_state/lucidota_storage); KANT69 wants one instance + scoped schemas.
- 18,627 graph-promotion candidates staged, nothing promoted to canon. graph_promotion_gate.py blocks --materialize intentionally until write barriers live.
- Model stack silent failure: `./claw` swallows CUDA errors via `>/dev/null 2>&1 || true`.

---

## Repo dig results (2026-05-29)

Full report: `/home/mfspx/LUCIDOTA/05_OUTPUTS/repo_dig_report_20260529.md`

### phantom (ghostwright/phantom) — INTEGRATE (mine artifacts)

**Purpose:** Self-evolving Claude Agent SDK wrapper with multi-channel ingest, Qdrant episodic/semantic memory, SQLite scheduler, and validated self-rewriting config engine.

**Top gold:**
- `src/skills/frontmatter.ts` — exact SKILL.md YAML schema (Zod-strict) for claw skill injection. Drop a SKILL.md at `~/.claude/skills/<name>/SKILL.md` to ship workflow skills (ABSURD-submit, graph-promote-gate, river-debug) into claw without touching CLAUDE.md.
- `scripts/install-phantom-ui-skill.sh` — seed script pattern to install a SKILL.md into claw runtime. Direct template for LUCIDOTA skill shipping.
- `skills-builtin/` (7 skills: echo, list-plugins, mirror, overheard, ritual, show-my-tools, thread) — production SKILL.md templates showing allowed-tools whitelisting, context:inline, MCP tool invocations.
- `src/memory/ranking.ts` — `calculateEpisodeRecallScore()` + `shouldIncludeEpisodeInContext()`: two-signal (durability×0.6 + recency×0.4) composite with exponential decay. Pure function, zero deps. Port to `ALGOS/` as episode scoring primitive.
- `src/memory/embeddings.ts` — `textToSparseVector()`: BM25-style sparse vector from text, pure JS, no deps. Portable to Python for sparse complement to BGE-M3.
- `src/memory/semantic.ts` — SemanticStore contradiction detection: cosine-similarity at 0.85 before storing; contradictions versioned and resolved. Pattern for GO-25 staging with contradiction-safe upserts.
- `src/evolution/invariant-check.ts` — 9-invariant deterministic post-write sweep: STATIC_WRITEABLE_FILES allowlist, I6 credential pattern list, I4 size-growth bounds (80 lines/file). Reference implementation for `canonical_graph_write_scanner.py` hardening.
- `src/mcp/dynamic-tools.ts` — DynamicToolRegistry: agent registers shell/script MCP tools at runtime, SQLite-persisted, hot-reload. Missing LUCIDOTA operator-extensible tool surface pattern.
- `src/config/providers.ts` + `phantom.yaml` — 2-line model_mappings YAML block maps opus/sonnet/haiku to arbitrary model IDs. Missing adapter between LUCIDOTA provider lanes and SDK model name expectations.
- `src/subagents/frontmatter.ts` — subagent .md schema: tools, disallowedTools, model, effort, isolation:worktree, permissionMode. How to define bounded sub-workers (river-mre, graph-promote) as first-class SDK subagents.
- `src/agent/hooks.ts` — PreToolUse hook blocking dangerous Bash patterns via regex. Pattern for canonical-graph-write scanner as a pre-execution interceptor.

**Key anomalies:**
- Self-modifying config: reflection subprocess (sandboxed Claude SDK) can rewrite persona.md, user-profile.md, strategies/*.md — direct inversion of LUCIDOTA's "models never act as hidden controllers" doctrine. Treat phantom-config/ as candidate-only staging zone, not canonical truth.
- SQLite-only (no Postgres). Bun-native (not Node.js/Python compatible — interface via MCP HTTP or subprocess).
- Ollama-only embeddings — BGE-M3 adapter needed for dense vector path.

**Integration action:** Mine 3 artifacts only — do NOT run as a full service. (1) SKILL.md schema + install script → bootstrap `~/.claude/skills/` for claw. (2) Port `calculateEpisodeRecallScore()`, `shouldIncludeEpisodeInContext()`, `textToSparseVector()` into `ALGOS/` as pure-function Python adapters. (3) Use 9-invariant pattern as design reference for `canonical_graph_write_scanner.py` hardening.

---

### rig (0xPlaygrounds/rig) — INTEGRATE (mine artifacts)

**Purpose:** Rust LLM framework (v0.37.0) with unified provider-agnostic interface for completion, embedding, structured extraction, RAG, and agent pipelines. 20+ providers, 10+ vector store backends, pipeline DAG combinator.

**Top gold:**
- `rig-core/src/providers/llamafile.rs` — `Client::from_url("http://localhost:8080/v1")` — exact match for LUCIDOTA's llama.cpp server at :8080. Completion + embeddings, OpenAI-compat wire.
- `rig-core/src/providers/ollama.rs` — `Client::builder().base_url(...)` — can target :8080/:8081/:8083 (Mamba ports). Multi-provider fan-out.
- `rig-core/src/providers/groq.rs` — native Groq client. Direct fit for LUCIDOTA's Groq fanout lane.
- `rig-core/src/extractor.rs` — `Extractor<M,T>`: JsonSchema derive → typed tool call → retry on parse failure → typed T. Exactly what GO-25 ETL extraction nodes need. Zero glue.
- `rig-core/src/pipeline/` — `parallel!()` macro (futures::join! fan-out), `conditional!()` macro (enum-variant dispatch). Maps onto Groq fanout and Treelite routing.
- `rig-fastembed/src/lib.rs` — local ONNX embedding on CPU (no VRAM). Direct candidate to replace Python BGE-M3 embed calls in Rust workers within 4GB VRAM constraint.
- `rig-postgres/src/lib.rs` — PostgresVectorStore via sqlx + pgvector. L2/Cosine/InnerProduct. Wire directly into lucidota_storage without a separate vector DB.
- `rig-core/src/vector_store/lsh.rs` — LSH in-memory ANN. Sub-ms candidate retrieval; complement to Treelite routing for embedding-space pre-filtering.
- `rig-core/src/evals.rs` — EvalOutcome enum (Pass/Fail/Invalid) + Eval trait. Maps onto regret/bandit scoring nodes in ALGOS/ for automated model quality gating.

**Key anomalies:**
- Clippy lints forbid `unwrap_used`, `expect_used`, `panic`, `todo` — stricter than LUCIDOTA's claw workspace. Would create lint-flag conflicts if merged into the same workspace.
- rig-postgres adds sqlx as a second Postgres client alongside whatever lucidota_etl uses — potential connection pool conflict.
- rig-fastembed downloads ONNX Runtime binaries at build time — requires outbound network during `cargo build`.
- v0.37.0, 0.x — fast-moving, breaking changes warned in CHANGELOG.
- No ABSURD/Postgres queue integration. Pipeline Ops are in-process and ephemeral — no custody receipts.

**Integration action:** Add rig-core (llamafile/fastembed/postgres features) as a dependency in a new `rig-adapter` crate under `01_REPOS/lucidota_etl/` — NOT in the claw workspace (lint profile conflict). Register Python BGE-M3 embed adapter as Rust-port candidate in `00_PROJECT_BRAIN/rust_port_candidacy_registry.json` before starting. Key files to re-read: `extractor.rs`, `rig-fastembed/src/lib.rs`, `rig-postgres/src/lib.rs`, `pipeline/conditional.rs`.

---

### ruflo (formerly claude-flow) — REFERENCE MINE ONLY (do not integrate)

**Purpose:** Multi-agent AI orchestration framework. 98+ agent types, 314 MCP tools, AgentDB (SQLite + HNSW), QUIC-based federation, self-learning hooks. WHY_HERE.md says: "Mine it. Don't run it."

**Top gold:**
- `v3/crates/ruflo-federation-peer/src/lib.rs` — 3-trait architecture (TransportProvider, SafetyGate, Dispatcher) composing QUIC transport with 3-gate safety pipeline (~60 lines, full unit test coverage). Directly adaptable as a worker-message-gate pattern in claw's Rust workspace.
- `SafetyVerdict` enum (Pass / Block(String) / Redact(FederationMessage)) — cleaner model than LUCIDOTA's current binary accept/reject at queue dequeue.
- `plugins/ruflo-neural-trader/src/sublinear-adapter.ts` — ~50-LOC Conjugate Gradient kernel for mean-variance portfolio optimization (816ns vs 50us for Neumann series — 40-60x). Candidate for ALGOS/ as SPD linear solver.
- `plugins/ruflo-agentdb` — RaBitQ 1-bit quantization path (32x memory reduction). Relevant to local embedding stack on 4GB VRAM.
- `agentdb.rvf` — RVF binary segment format (SFVR magic header) for packaging agent memory + WASM runtimes. Relevant to AHOY/treelite .rvf layer.

**Key anomalies:**
- RaBitQ 150x-12,500x HNSW speedup claim debunked in their own audit (actual: ~1.9x at N=20k). Self-corrected — unusually honest.
- `ruflo-federation-peer` Cargo.toml deps (`midstreamer-quic`, `aimds-core`) are not published on crates.io — native feature cannot be built without upstream crates materializing.
- fork-bomb bug in neural-trader ≤ 2.7.1 (fixed in 2.7.2; CI guard now enforces --ignore-scripts).
- graphAdapter controller disabled pending external graph-DB connection — AgentDB causal graph has no live graph-DB backend.

**Integration action:** Manual port only. (1) Peer<T,G,D> 3-trait + SafetyVerdict enum → claw Rust workspace as worker-gate pattern. (2) SublinearAdapter CG kernel → ALGOS/ pure-Python or Rust port. (3) Study RaBitQ index before extending BGE-M3 pipeline. Do NOT import TypeScript CLI, MCP tools, agent swarm, or self-learning hooks.

---

### forge (antoinezambelli/forge) — INTEGRATE (extract 3 pieces)

**Purpose:** Reliability layer (guardrails middleware + agentic loop runner) for self-hosted LLM tool-calling. Lifts tool-call success rates from single digits to 84%+ on 8B models via deterministic rescue parsing, retry nudges, required-step enforcement, and a serializing proxy server (OpenAI + Anthropic Messages APIs).

**Top gold:**
- `prompts/templates.py` — `rescue_tool_call()`: 4-strategy fallback parser (JSON extraction → rehearsal-syntax `tool_name[ARGS]{}` → Qwen XML `<function=name>` → Mistral `[TOOL_CALLS]<name>{}`). Directly reusable to fix malformed FC output from DeepSeek/Mamba/Bonsai in ABSURD workers. Zero deps beyond re/json.
- `context/strategies.py` — `TieredCompact`: 3-phase context compaction (nudge/retry drop → tool_result truncation → tool_result purge → skeleton only), VRAM-aware, per-phase triggers via tuple param. Maps onto ABSURD context budget management for multi-turn extraction jobs.
- `core/slot_worker.py` — `SlotWorker`: asyncio.PriorityQueue with auto-preemption (~143 lines, stdlib-only). Directly solves GTX 1650 single-GPU serialization problem for Bytewax/River stream service.
- `proxy/server.py` — zero-dependency asyncio HTTP server speaking `/v1/chat/completions` AND `/v1/messages` (Anthropic), single-GPU serialization queue, SSE chunked-transfer. claw can point at it as a drop-in local provider.
- `guardrails/guardrails.py` — `Guardrails.check()` / `record()` two-method composable middleware API. execute/retry/step_blocked/fatal action enum maps cleanly onto ABSURD dead-letter logic.
- `guardrails/step_enforcer.py` — `StepEnforcer.check_prerequisites()`: per-tool arg-matched prerequisite enforcement. Maps onto ABSURD job-kind dependency ordering.
- `clients/sampling_defaults.py` — `MODEL_SAMPLING_DEFAULTS`: HF-card-sourced sampling registry for 30+ models (Qwen3/3.5/3.6, Gemma4, Mistral, gpt-oss-120b, Granite, Nemotron) with per-model `chat_template_kwargs` for reasoning_effort/enable_thinking. Authoritative reference for tuning local model calls.
- `context/hardware.py` — `detect_hardware()`: nvidia-smi + AMD sysfs probe returning HardwareProfile(vram_total_mb, memory_kind). Immediately usable in hw_gate and safe_ops env to enforce 4GB VRAM envelope without subprocess boilerplate.
- `clients/llamafile.py` — `LlamafileClient` mode=auto: native-FC probe with graceful fallback to prompt-injected mode. `_merge_consecutive()` handles Mistral Jinja parity-checker requirement. Directly compatible with llama-server at :8080.

**Key anomalies:**
- `eval_results_v0.7.0.jsonl` stored as git-lfs pointer (45MB). 84% accuracy figure cannot be verified from this checkout without LFS pull.
- Eval hardware: RTX 5070 / AMD Strix Halo APU — not the GTX 1650. Numbers may not transfer.
- WHY_HERE.md claims 99.3% but README says 84%/98.8% — provenance of 99.3% unclear.
- `proxy/server.py` strips all query-string params (including `?beta=true`) before routing — beta-flag behavior silently stripped at proxy boundary.
- SlotWorker auto-preemption discards partial tool results without dead-letter or receipt — if wired into ABSURD, could silently drop in-flight work.
- Proxy HTTP server: no auth layer, `Access-Control-Allow-Origin: *` on all responses.

**Integration action:** Extract 3 self-contained pieces — do NOT install as a dependency. (1) `rescue_tool_call()` + 4 sub-parsers from `prompts/templates.py` → `ALGOS/` as standalone module (zero deps). (2) `TieredCompact` + `CompactStrategy` ABC from `context/strategies.py` → ETL pipeline context management. (3) `SlotWorker` from `core/slot_worker.py` → single-GPU serialization primitive for River service fix. Register in TICKLETRUNK; note SlotWorker as Rust-port candidate (asyncio queue → tokio).

---

### cc-switch — REFERENCE MINE ONLY (do not integrate as a service)

**Purpose:** Tauri 2 GUI desktop app and transparent HTTP reverse proxy for AI coding CLIs (Claude Code, Codex, Gemini, OpenCode, OpenClaw, Hermes). Intercepts outbound LLM API calls, routes through configurable provider registry with circuit-breaker failover, adds prompt-cache injection, thinking-budget optimization, model aliasing, usage tracking.

**Top gold:**
- `proxy/circuit_breaker.rs` — full circuit-breaker (Closed/Open/HalfOpen) with configurable failure_threshold, error_rate_threshold, timeout, half-open probe slots. ~495 lines, no external deps, pure Rust atomics + RwLock. Directly liftable for ABSURD worker-health gating around external providers.
- `proxy/provider_router.rs` — ProviderRouter: reads failover queue from SQLite, checks each provider against circuit breaker keyed `app_type:provider_id`. Pattern for ABSURD worker provider-routing with health awareness.
- `proxy/cache_injector.rs` — auto-injects Anthropic prompt-cache breakpoints (tools tail, system tail, last assistant message) up to budget=4, TTL-upgrade for existing breakpoints. Zero external deps, serde_json only. Direct apply to LUCIDOTA's Groq/Anthropic call paths.
- `proxy/thinking_optimizer.rs` — 3-path thinking dispatch: skip(haiku), adaptive(opus-4-x/sonnet-4-6 → type:adaptive + output_config.effort:max + beta header context-1m-2025-08-07), legacy. Encodes current Anthropic extended-thinking API surface.
- `proxy/thinking_rectifier.rs` — auto-detects and strips invalid thinking signatures from retry requests. Handles both Anthropic native and Gemini/third-party error text.
- `proxy/sse.rs` — `append_utf8_safe`: robust UTF-8 boundary-safe accumulator for SSE chunks split across TCP buffers. `take_sse_block` handles both `\r\n\r\n` and `\n\n` delimiters. Direct drop-in for Bytewax/River dead stream repair.
- `proxy/transform.rs` — Anthropic ↔ OpenAI Chat Completions format converter. Strips leading `x-anthropic-billing-header` lines from system prompts (prevents cache-miss on rotating `cch=` prefix). Utility for Groq fanout path.
- `proxy/model_mapper.rs` — provider-level ANTHROPIC_DEFAULT_{HAIKU,SONNET,OPUS}_MODEL env-key aliasing. Pattern for Treelite → local model routing.

**Key anomalies:**
- Live-proxy mode writes `primaryApiKey='any'` into `~/.claude/config.json` — redirects ALL Claude Code traffic. Do not run on the LUCIDOTA dev box with live-takeover mode active.
- Tauri GUI app with its own SQLite, React/Radix UI — architecturally incompatible with LUCIDOTA's Postgres-ABSURD-Rust spine as a whole.
- rquickjs (QuickJS) embedded for user-defined balance-query scripts — meaningful attack surface.
- Two separate JSON5 parsers (intentional, but surprising).
- WebDAV auto-sync for config backup — non-obvious cloud-write path.

**Integration action:** Extract specific modules only. (1) Copy `proxy/circuit_breaker.rs` → new crate in `01_REPOS/claudecode/rust/` for ABSURD worker health. (2) Use `proxy/cache_injector.rs` as template for Groq call path cache injection. (3) Use `proxy/sse.rs` `append_utf8_safe` for River stream repair. (4) Use `thinking_optimizer.rs` as reference when setting thinking params for sonnet-4-6/opus-4 calls. All MIT licensed. Register in TICKLETRUNK with cc-switch as upstream source.

---

### pathway (pathwaycom/pathway) — INTEGRATE (copy patterns, do not pip-install)

**Purpose:** Python ETL/streaming framework backed by Rust differential-dataflow engine (timely + differential-dataflow). Incremental real-time computation over streams. RAG pipelines, real-time analytics, connector-rich ETL. BUSL-1.1 license until conversion date.

**Top gold:**
- `src/connectors/postgres.rs` (4129 lines) — full Postgres WAL streaming reader via `pg_walstream` + logical replication slots. Production-grade Rust pattern for reading live Postgres changes without Debezium. Lifts cleanly into ABSURD queue pipeline as a staging-packet source. NOTE: Python-layer WAL reader gates on `_check_entitlements('postgres-wal-reader')` — paid license only. The `pg_walstream = '0.6.1'` Cargo crate itself is FOSS and can be added directly to `lucidota_etl Cargo.toml` without the Pathway license gate.
- `src/connectors/synchronization.rs` (816 lines) — `ConnectorSynchronizer`: priority-based multi-source windowed watermark coordination (max_possible_value antichain). Pure-Rust state machine. Pattern for gating Needle x6 semantic river: multiple sparse embedding-model lanes with different latencies synced without deadlock on empty partitions.
- `src/connectors/backlog.rs` — `BacklogTracker`: VecDeque + ProbeHandle queue-depth meter. ~60 lines, copy-portable. Maps to ABSURD queue depth monitoring.
- `src/retry.rs` — `RetryConfig` with exponential backoff + jitter (`execute_with_retries_if`). Non-transient errors short-circuit immediately. Trivially portable. Missing from current ABSURD worker fault lane.
- `src/external_integration/usearch_integration.rs` — `USearchKNNIndex` wrapping usearch crate (HNSW with F16 quantization). FOSS, runs on CPU. Replaces or complements dead LEANN connector for local ANN on GTX 1650 without VRAM pressure.
- `src/external_integration/tantivy_integration.rs` — `TantivyIndex` for BM25 full-text search. Pure Rust, fully local. Keyword-side of hybrid retrieval lane in front of BGE-M3.
- `python/pathway/io/python/__init__.py` — `ConnectorSubject` ABC: run(), next(**kwargs), commit(), close(), seek(state: bytes). Correct adapter pattern to wire any LUCIDOTA Python scraper/adapter into a stream pipeline. Max backlog size parameter controls memory pressure.
- `python/pathway/xpacks/llm/embedders.py` — `SentenceTransformerEmbedder`: Pathway UDF wrapper for sentence-transformers with batching. Drop-in for BGE-M3 local embedding inside a pipeline. Device param accepts 'cpu', 'cuda'.
- `src/engine/dataflow/async_transformer.rs` — `AsyncTransformer`: fully async Python-side computation returning results at a later timely timestamp, with persistence on restart and upsert deduplication. Correct architecture for Needle x6 swarm — send text chunks in, get embeddings back asynchronously without blocking.
- `src/engine/workload_tracker.rs` — `WorkloadTracker`: sliding-window compute/step/scheduled duration ratio, emits ScaleUp/ScaleDown/DoNothing advice. Can gate the River governor's fan-out decision.

**Key anomalies:**
- BUSL-1.1 license (not Apache/MIT). Production use in a competing service restricted. Private internal use allowed — but review before any public deployment.
- Python-layer Postgres WAL reader requires paid Pathway Scale/Enterprise license. Static snapshot mode is free.
- Free tier hard-capped at MAX_WORKERS=8 (PATHWAY_THREADS env).
- Telemetry: free community mode phones home to pathway.com. Safe-ops doctrine requires explicit suppression.
- Rust toolchain pinned to 1.95 — may conflict with claw workspace toolchain.
- `async_transformer` requires `on_end` call on clean shutdown — failure leaks the forgetting stream forever. Subtle correctness trap.

**Integration action:** Copy patterns, do NOT import the wheel. Priority order: (1) IMMEDIATE — Port `src/retry.rs` RetryConfig (50 lines, zero deps) into `01_REPOS/claudecode/rust/runtime/` or `01_REPOS/lucidota_etl/`. Fills missing ABSURD worker fault lane with exponential backoff + transient-error discrimination. (2) SHORT-TERM — Port `src/connectors/backlog.rs` BacklogTracker (80 lines) as queue-depth probe in the River governor. (3) EVALUATE — Add `pg_walstream = '0.6.1'` as direct Cargo dependency in lucidota_etl Cargo.toml for Postgres WAL streaming without Pathway license. (4) Use ConnectorSubject pattern as design spec for dead Bytewax River service rewrite.

---

### sparc (codeium/sparc) — REFERENCE MINE ONLY (do not install)

**Purpose:** Structured AI coding agent CLI (Specification → Pseudocode → Architecture → Refinement → Completion) driving LLMs through phase-gated build methodology. LangGraph ReAct agents, ripgrep search, fuzzy file find, aider-chat code execution, sympy symbolic math, memory store, sub-agent spawning.

**Top gold:**
- `sparc_cli/tools/memory.py` — prioritized in-process memory store with typed categories (research_notes, plans, tasks, key_facts, key_snippets, related_files, work_log) and eviction by priority+age. `_global_memory` dict + `MEMORY_LIMITS` + `MemoryPriority` class. Dependency-free pattern usable inside ABSURD workers for structured scratchpad state.
- `sparc_cli/tool_configs.py` — `get_read_only_tools()` / `get_research_tools()` / `get_planning_tools()` / `get_implementation_tools()`. Staged tool surface per agent phase mirrors LUCIDOTA's bounded-worker mutation-class declaration.
- `sparc_cli/tools/agent.py` — `request_research` / `request_implementation` as LangChain @tools that spawn sub-agents and return structured dicts. Multi-agent delegation pattern for Diogenes kernel fan-out.
- `sparc_cli/tools/expert.py` — `emit_expert_context()` + `ask_expert()`: accumulate context separately, then call higher-capability model. The two-model routing idiom (cheap local worker + smart oracle) already in LUCIDOTA's Groq fanout doctrine.
- `sparc_cli/llm.py` — `initialize_llm(..., 'openai-compatible', base_url='http://127.0.0.1:8080/v1')` branch. Plug-in for local llama.cpp at :8080 without code changes.
- `specification/` phase docs — SPARC decomposition (Spec → Pseudocode → Arch → Refine → Complete) is the operationalized form of LUCIDOTA's blueprint-first doctrine. Usable as a build-session harness template.

**Key anomalies:**
- `polars.py` (PolarisWrapper) is theater: calls LLM twice per query with "quantum-coherent consciousness integration" marketing language. Trivial double-call with no measurable benefit.
- `run_programming_task()` shells out to aider with `--yes-always --no-auto-commits` — aider can write arbitrary files to disk silently. No path guard.
- `non_interactive.py` is a dead stub (Fly.io deployment scaffolding — `while True: sleep`).
- LangChain + LangGraph + aider + playwright dep stack is ~500MB of additions hostile to the 4GB VRAM constraint.

**Integration action:** Pattern mine only — do NOT pip-install. Extract by copy-adapt: `_global_memory` pattern into a standalone Python dataclass for ABSURD worker scratchpad state; staged tool surface pattern as design note for Diogenes kernel routing; SPARC phase documents as blueprint-first session harness for flow design. Key files to harvest: `sparc_cli/tools/memory.py`, `sparc_cli/tools/agent.py`, `sparc_cli/tool_configs.py`, `sparc_cli/tools/expert.py`.

---

## Gap fix status

Gap fixes were attempted but returned 0 auto-applied results. All identified gaps require co-design with Opus (schema application ordering, ontology collision resolution, service file edits, Rust crate integration) or explicit Northern approval (write barriers, graph materialization). No gaps were silently mutated.

**Gaps still open from OPUS_BRIEFING.md §11 — full list unchanged:**
- Bytewax→River service dead (one-line ExecStart fix pending)
- 92 schema files unapplied (035 → 025-027 → 040/074 → 034/044/052)
- GO/CO ID @26-@50 collision (alias-vs-renumber + CO_ACTIVE_TERMS.json)
- 18,627 candidates waiting (write barriers + promotion pipeline needed)
- Model stack silent failure (harden ./claw launcher)
- CUDA errors swallowed (LD_LIBRARY_PATH from Ollama)
- persona_stamp.md stub
- indy_reads swarm scripts (swarm_router, packet_stamp, receipt_synth)
- worker-order packet contract (indy_reads.worker_order.v1)
- Job Fair Allocator file missing
- Fidelity Guard not launch-complete
- Cryptographic witness chain for ABSURD provenance
- verbatim-correction → River not wired
- Ternary/FairyFuse backend is STUB_BACKEND_NOT_WIRED
- CLAW fork not on command surface for CEP/kernel/ABSURD
- Git history 1.6G CAS blob (sanitized mirror branch needed)

---

## What was swept (archived to KRAMPUSCHEWING/System_Archive_Docs/pre_opus_sweep/)

- GOALS/: 20 .md files + 18 .json files (status logs, audits, old plans, progress JSONs)
- 02_RECORDS_OFFICE/: SYSTEM_THRASH_REPORT_TEMPLATE_2026_05_29.md
- docs/superpowers/plans/: 2026-05-28-governor-dials-and-groq-fanout.md

## 00_PROJECT_BRAIN/ final file count: 26 entries

---

## Next for Opus

Read OPUS_BRIEFING.md §13 then execute the full build plan sequence:

1. Apply schema backlog (035 → 025-027 → 040/074 → 034/044/052)
2. Fix Bytewax service (one-line ExecStart change to venv python3)
3. Resolve ontology collision, seed CO_ACTIVE_TERMS.json, reach 75-term target
4. Design artifact-type decision tree for ETL (one workflow per format family) — Northern co-designs, no Groq for design phase
5. Wire Needle x6 fan-out with SmolDocling (OCR/layout), GLiNER (NER), ModernBERT (NLI), BGE-M3 (embed) per artifact type
6. Run Epoch-1 reingest crush: all of KRAMPUSCHEWING, byte-perfect CAS, GO-25 candidates, gate blocks canon writes
7. Fix model stack silent failure + CUDA error swallowing
8. Wire Bytewax→River→treelite learning loop (once training points exist)
9. KANT69 1-DB collapse (collapse-in-place, scoped schemas)
10. Build cryptographic witness chain for ABSURD provenance
11. Expand persona_stamp.md + wire swarm router scripts

**From repo dig — additional build inputs for Opus:**
- Phantom SKILL.md schema → bootstrap `~/.claude/skills/` for ABSURD-submit, graph-promote-gate, river-debug skills
- rig Extractor<M,T> → replace Python model-glue in GO-25 ETL workers (new `rig-adapter` crate in lucidota_etl)
- forge `rescue_tool_call()` → ALGOS/ standalone module for malformed FC output from DeepSeek/Mamba
- forge SlotWorker → single-GPU serialization primitive for River service fix
- cc-switch `circuit_breaker.rs` → new crate in claudecode/rust/ for ABSURD worker health
- pathway `retry.rs` → IMMEDIATE port into lucidota_etl for ABSURD worker fault lane
- pathway `pg_walstream = '0.6.1'` → evaluate as direct Cargo dep for WAL streaming
- ruflo `SafetyVerdict` enum → worker-gate pattern in claw Rust workspace

**Indy orchestrates. Opus designs. Swarm builds. Northern approves materialization.**

## Resume command

```
Read /home/mfspx/LUCIDOTA/OPUS_BRIEFING.md, then read /home/mfspx/LUCIDOTA/GOALS/CURRENT_HANDOFF.md, then query the live DB:
  psql postgresql:///lucidota_state -Atc "SELECT phase_id,phase_name,status FROM lucidota_control.phase_ledger ORDER BY phase_id;"
  psql postgresql:///lucidota_storage -Atc "SELECT COUNT(*) FROM lucidota_go.term_registry;"
  python3 /home/mfspx/LUCIDOTA/scripts/lucidota_status_ledger.py --check
Then begin the full build plan from §13.
```
