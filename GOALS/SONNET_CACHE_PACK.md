# SONNET CACHE PACK — 60-min orchestration context (exact map + token counts)

> Built by Indy_READs on Opus, 2026-05-28, as the last act before the Opus→Sonnet flip. This is the EXACT context to hold so Sonnet stays crystal while orchestrating the ingestion/ETL workflow build. Token counts are byte/4 estimates (no tiktoken installed; ±10%). Load discipline: hold Tier A all 60 min; pull Tier B only when a design step touches that organ; never cache Tier C — query it live.

## Load discipline
The spine is tiny on purpose (~21K tok). State is read from Postgres one-liners, NOT by caching big files. The 745-organ registry JSON is **119K tokens — NEVER load whole**; grep it or read the 2.7K summary. We START with ingestion + ETL, so promote the two ingestion shards into the working set.

---

## TIER A — ALWAYS IN CACHE (the spine, hold all 60 min) — **~21,357 tok**
| tok | file | why |
|---|---|---|
| 656 | `AGENTS.md` | agent startup law (reuse / handoff / model economy) |
| 2867 | `CLAUDE.md` | repo law, execution spine, conventions |
| 2957 | `00_PROJECT_BRAIN/KANT69_ARCHITECTURE_CONVERGENCE_CANON.md` | architecture authority #1 |
| 1699 | `BOOKS/.indy_reads/INDY_ORCHESTRATOR_PROMPT.md` | the soul + doctrine + toolbelt |
| 2986 | `BOOKS/.indy_reads/INDY_SYSTEM_UNDERSTANDING.md` | **my synthesized model of the whole system (canon mapped to me)** |
| 2767 | `00_PROJECT_BRAIN/organ_registry/ORGAN_REGISTRY.md` | the 745-organ map SUMMARY (not the json) |
| 4020 | `00_PROJECT_BRAIN/ALGO_MANIFEST.md` | the deterministic arsenal (200+ algos) |
| 1422 | `OFFICIAL_ONTOLOGY.json` | ontology authority (converging 48→75) |
| 1141 | `BOOKS/IO_ACTIVE_TERMS.json` | IO-25 shadow grammar (@69=DAEMON) |
| 842 | `GOALS/CURRENT_HANDOFF.md` | current state + resume |

Plus `MEMORY.md` (~515 tok) auto-loads. **Tier A total incl. memory: ~21.9K tok.**

## INGESTION WORKING SET — promote for the ETL session (+~6.4K → ~28K total)
| tok | file | why |
|---|---|---|
| 1077 | `00_PROJECT_BRAIN/organ_registry/50_ingestion.json` | embedding models on disk + missing tools + artifact-route TODO |
| 5358 | `00_PROJECT_BRAIN/organ_registry/30_streaming_ml.json` | Bytewax/River/Treelite operational topology (what's live vs broken) |

## TIER B — LAZY LOAD (pull only when a design step touches it)
| tok | file | load when |
|---|---|---|
| 8607 | `GOALS/GONN_MASTER_BUILD_PLAN.md` | need build doctrine / WO template / lane table |
| 2700 | `06_SCHEMA/016_go_graph_core.sql` | designing a graph-staging step (the live core) |
| 2449 | `BOOKS/GO_ACTIVE_TERMS.json` | designing ontology-tagging steps |
| 20254 | `00_PROJECT_BRAIN/organ_registry/20_schema.json` | designing schema-touching steps (which SQL is applied vs not) |
| 16014 | `00_PROJECT_BRAIN/organ_registry/40_percyphon_math.json` | designing mask/Villager or validator/math steps |
| 68230 | `00_PROJECT_BRAIN/organ_registry/10_scripts.json` | **grep by subsystem, never load whole** — find an existing script before building |
| 119482 | `00_PROJECT_BRAIN/organ_registry/ORGAN_REGISTRY.json` | **never load whole** — `jq`/grep for a specific organ row |

## TIER C — QUERY LIVE, NEVER CACHE (state comes from the DB, not files)
```bash
psql postgresql:///lucidota_state -Atc "SELECT phase_id,status FROM lucidota_control.phase_ledger ORDER BY phase_id;"
psql postgresql:///lucidota_state -Atc "SELECT subsystem,fact_key,derived_at FROM lucidota_control.runtime_status_fact ORDER BY derived_at DESC LIMIT 20;"
psql postgresql:///lucidota_storage -Atc "SELECT count(*) FROM lucidota_go.term_registry;"   # 45, target 75
df -h /            # disk/resource gate before any heavy ingest
cat /proc/pressure/cpu /proc/pressure/memory /proc/pressure/io   # governor posture
```

---

## SESSION GOAL AFTER THE FLIP
Northern + Sonnet riff the **ingestion + ETL workflows conceptually** (the fun part), then Sonnet dispatches **Groq + locals** to build them (no Groq for the *design* itself). Hard preconditions before any heavy run: (1) organ registry exists ✓, (2) missing ingestion deps installed (disk freed ✓ — `pip` line in `50_ingestion.json`), (3) disk/RAM/GPU/process budget receipt. Design law: ABSURD/Postgres = durable queue+custody; **PocketFlow = intra-job microflow** (idempotent nodes); no "ingested=true" — raw→evidence-candidate→claim→graph-candidate→audited→maybe-canonical; only the 4 materializers write graph truth.

## KNOWN BROKEN / FIX-FIRST
- `lucidota-stream-river` service runs bare `python3` → fix ExecStart to the `.venv` interpreter so Bytewax/River actually trains.
- ~92/122 schema files UNAPPLIED — decide what to apply for the ETL spine (likely ABSURD `035` + chrono `025-027` + write-barriers `040/074`).
- Model stack CUDA path is borrowed from Ollama via `LD_LIBRARY_PATH` and `./claw` swallows the error — harden + unmute.
- Ontology @26-@50 GO/CO ID collision — resolve before any 75-term seed.
