# CURRENT GOAL HANDOFF

"Save This Prompt, Pass on this Handoff:"

- Goal: KANT69 migration — PREP everything (done), then co-design the ingestion/ETL flow suite with Northern (Sonnet), built by Groq/locals.
- Generated: `2026-05-28T22:30:00Z`
- Current step: Opus prep complete; awaiting Opus→Sonnet flip + context clear, then workflow co-design.
- Status: active

## What this Opus session produced (the prep tranche)
1. **Disk reclaim:** 36G→97G free (~61GB). Killed trash 17.9GB audit-text, stray duplicate vault `scripts/03_VAULT` (15,994 CAS dupes; 1,095 unique files KEPT), redundant unpacked copies. More available (9.5GB regenerable repo snapshot; 35GB unique stray-vault to migrate to canonical CAS).
2. **ORGAN_REGISTRY / CAPABILITY_LEDGER** (the gate before flow design): `00_PROJECT_BRAIN/organ_registry/` — `ORGAN_REGISTRY.json` (745 organs) + `.md` summary + 5 shards (scripts/schema/streaming_ml/percyphon_math/ingestion).
3. **System canon mapped to me:** `BOOKS/.indy_reads/INDY_SYSTEM_UNDERSTANDING.md`.
4. **Cache pack:** `GOALS/SONNET_CACHE_PACK.md` — exact 60-min cache (Tier A spine ~21.4K tok + ingestion working set + lazy-load + live-query tiers).
5. Memories written: organ-registry gate, percyphon/villagers/AHOY, production ethos creed, sudo+disk authority, ouroboros ingest sequence, KANT69 migration sequence, Spencer's algo source.

## Ground truth (verify, don't trust blindly)
- ~92/122 schema files UNAPPLIED (chrono 025-027, ABSURD spine 035, graph-promotion 034/044/052, write-barriers 040/074). Live = base band + GO graph core `016` (`lucidota_storage.lucidota_go`).
- Bytewax→River stream spine BROKEN (service execs bare python3). Only river-governor Hoeffding tree live.
- `term_registry` = 45 rows (target 75); `CO_ACTIVE_TERMS.json` MISSING; @26-@50 GO/CO ID collision = convergence landmine.
- Two DBs still split (`lucidota_state`/`lucidota_storage`); KANT69 wants one instance + scoped schemas.

## Next action on resume (Sonnet)
1. Load the Tier A spine from `GOALS/SONNET_CACHE_PACK.md`.
2. Query live state (phase_ledger, runtime_status_fact, term count, df, /proc/pressure).
3. Riff ingestion/ETL workflows conceptually WITH Northern (no Groq for design).
4. Preconditions before heavy run: install missing ingestion deps (`50_ingestion.json` pip line), fix stream-river service, disk/resource budget receipt.
5. Design law: ABSURD=durable queue+custody, PocketFlow=intra-job microflow (idempotent), no "ingested=true", only 4 materializers write graph truth.
6. Ouroboros bar: INGEST-1 backlog → INGEST-2 repo history → INGEST-3 GDrive diff (needs Northern OAuth).
