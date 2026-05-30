# CURRENT GOAL HANDOFF

"Save This Prompt, Pass on this Handoff:"

- Mode: **OPUS_ARCHITECT_ORCHESTRATOR**. Opus plans/orders/audits receipts; **Groq (unlimited) + Cerebras ($10) + local models + scripts = the hands.** Claude budget is SCARCE (~$134 cap, heavily spent this session) → do NOT spawn large Claude/Sonnet agent fan-outs; route bulk to Groq/Cerebras.
- Generated: `2026-05-29` (crash-recovery + DIOGENES completion session).

## LIVE NOW (running autonomously, box-safe)
- **Archive text crush** → `lucidota_korpus.corpus_chunk` (the 76GB KRAMPUSCHEWING bulk; recursive, streaming, capped). `scripts/krampus_archive_crush.py --root KRAMPUSCHEWING` under `scripts/lucidota_capped_run.sh`.
- **corpus→graph drain loop** (Groq hands) → `lucidota_go.staging_packet` hypothesis field. `scripts/corpus_to_graph_loop.sh` (calls `scripts/corpus_to_graph.py`).
- **BGE embed fleet** = 4 servers :8101-8104, `--ubatch 1024` (lean; 2048 was a RAM hog). `scripts/lucidota_bge_fleet.sh`.
- **RAM watchdog** (suspend-on-pressure, never-cull) `scripts/lucidota_ram_watchdog.sh`. Box held ~2-3G avail throughout; never OOM'd.

## RECEIPTS (live DB facts, not prose)
- `corpus_chunk` ~1372 rows and climbing. `staging_packet` corpus candidates (parser_name='corpus_to_graph') climbing from 0.
- **Ingest→graph break CLOSED**: `scripts/corpus_to_graph.py` proven (staging_packet 130→142 in receipt run). Receipt: `05_OUTPUTS/corpus_to_graph_receipt.json`. Graph ontology validation is LIVE (off-ontology terms flagged `needs_repair`).
- OOM root cause + guardrail: `scripts/lucidota_capped_run.sh` (hard cgroup MemoryMax, no sudo) — proven kills a runaway job in-cgroup without touching the box. Secrets moved OUT of /tmp → `~/.config/lucidota/secrets.env` (0600); `groq_env.py` no longer reads /tmp.

## CORRECTED INTEL (decomposition workflow, live-DB verified — OPUS_BRIEFING §8/§17 STALE)
- ABSURD spine, graph promotion gate, `staging_packet`(130), `term_registry`(**75**, target hit), chrono `temporal_claim` = **ALL APPLIED**. "92 unapplied / 45 terms / 18,627 staged" is a **phantom** — do not redo.
- Real narrow gaps: `graph_staging_candidates` (phantom table — the break), `governor_action` table MISSING, NO write-barrier triggers attached, schemas **045/046/050/094** genuinely unapplied, `corpus_chunk` was a dead-end (now bridged), `claw` starts neither model stack nor governor.

## CANON DOCS written this session
- `00_PROJECT_BRAIN/DIOGENES_OPERATIONAL_SPEC.md` (operator north-star), `00_PROJECT_BRAIN/DIOGENES_MASTER_ROADMAP.md` (8-subsystem grounded roadmap + %-ready + stubs register).

## WORK ORDERS (for hands — Groq/Cerebras/local; Opus audits receipts)
- **WO-1** Fix `scripts/krampus_stage_worker.py`: repoint `lucidota_korpus.graph_staging_candidates` → `lucidota_go.staging_packet` (same target the bridge uses); add `validate_worker_contract()` at dequeue (3 krampus workers skip it). Receipt: watcher-fed file → staging_packet row.
- **WO-2** Term-mapping (deterministic alias map, Groq/local): Groq generic terms (PERSON/ORG/DATE/...) → the 75 `term_registry` terms so candidates land `pending` not `needs_repair`. Receipt: corpus staging_packet rows status=pending ≥ X%.
- **WO-3** 100% disk: wire the `--walk` branch in `krampus_archive_crush.py` (arg added, handler TODO) + run on loose corpus dirs (`luci/`, `Documents/`, `Downloads/`, `Pictures/`). Receipt: corpus_chunk covers loose files.
- **WO-4** Governor Phase-0 `scripts/lucidota_guvna.py` (reuse capped_run/bge_fleet/watchdog/river_governor; PSI+cgroup+NVML read → never-cull ladder → fleet-width/MemoryHigh actuators). Receipt: start 6-fleet+chat-stack, governor drains fleet 6→2, `memory.events.oom_kill==0`.
- **WO-5** `claw`/rig.rs: wire Groq/Cohere/Cerebras/Claude + local endpoints via rig.rs; claw starts model stack + governor. Receipt: claw round-trips each provider + governor live.
- **WO-6** `governor_action` table + write-barrier triggers on canonical tables + apply schemas 045/046/050/094.
- **WO-7** Percyphon: fix `ALGOS/percyphon.py` truncation FIRST; build 5000×128 generative village table + signal-derived relevance + comms/identity/proxy/burn filter.
- **WO-8** Capability suite revival per `OPUS_BRIEFING.md` §14: ORNAMENT/DRE/SNOWBALL/MISTLETOE/SANTA + 22 doc-gen templates + chrono/hypertimeline fill + claims-require-evidence enforcement.
- **WO-9** Adversarial audit suites + daily schedules; audit all scripts + schemas (reuse `groq_hammer_audit`, `slop_audit_law.py`, `lucidota_status_ledger.py`). No stubs/gaps live.

## RESUME COMMAND
```
Read 00_PROJECT_BRAIN/DIOGENES_MASTER_ROADMAP.md + DIOGENES_OPERATIONAL_SPEC.md + this handoff.
Verify live: psql lucidota_storage -Atc "SELECT (SELECT count(*) FROM lucidota_korpus.corpus_chunk), (SELECT count(*) FROM lucidota_go.staging_packet WHERE parser_name='corpus_to_graph');"
Check crush/bridge/watchdog logs in 04_RUNTIME/. Then execute WO-1..WO-9 via Groq/Cerebras/local hands; Opus audits receipts. Budget: spare Claude.
```
