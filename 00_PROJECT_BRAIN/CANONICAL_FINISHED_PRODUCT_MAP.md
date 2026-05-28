# Consolidation Notice

Status: SUPPORTING SOURCE MATERIAL. Product-shape facts from this file are consolidated into `ACTIVE_SPEC/02_EXECUTION_SPINE.md` and `ACTIVE_SPEC/05_COMPONENT_AUTHORITY_MAP.md`. Keep this as source material unless re-promoted by the active instruction index.

---

# LUCIDOTA Canonical Finished Product Map

Status: ACTIVE CANONICAL PRODUCT MAP  
Last reconciled: 2026-05-22  
Authority role: resolves product-mode ambiguity between design docs, local fixtures, ABSURD/Postgres runtime, graph promotion, and model/ML staging.

## Product Identity

LUCIDOTA is a local evidence-preserving proof-hoard operating system.

It is not a generic chatbot, not a hidden model controller, not a direct graph-writing ingest script, and not a single monolithic daemon.

Finished-product shape:

```text
raw artifact or operator instruction
  -> custody / command envelope
  -> bounded worker or local product harness
  -> receipt / event / dead-letter
  -> extracted evidence candidate / claim packet / graph candidate
  -> authority mapping
  -> graph promotion gate
  -> operator-confirmed materialization only through guarded helper + journal
  -> status ledger / export / review surface
```

## Canonical Product Modes

| Mode | Status | What It Is | What It Proves | What It Does Not Prove |
|---|---|---|---|---|
| `LOCAL_FILE_PRODUCT` | active fixture/product shell | `scripts/lucidota_cli.py`, local case workspaces, file-backed jobs, export bundle | operator can create/run/resume/export a case with receipts and no canonical graph mutation | does not prove live Postgres ABSURD, canonical graph materialization, or production semantic quality |
| `ABSURD_POSTGRES_RUNTIME` | partial active target | `06_SCHEMA/035_absurd_queue_spine.sql`, queue jobs/events/dead letters, worker contracts | durable workflow substrate exists and is current target | not yet one single spine; worker registry enforcement and contract coverage are still being hardened |
| `GRAPH_PROMOTION_RUNTIME` | gated partial | graph promotion schemas, gate, barriers, materialization helper | direct graph writes are blocked; packets/decisions can be staged with evidence/authority | default full canonical materialization is not broadly enabled until helper/journal/command-envelope chain is verified |
| `CHRONO_EVIDENCE_LEDGER` | active partial | Chrono temporal claim schemas, projections, ranking/conservation scripts | temporal evidence can be preserved and projected | ABSURD queue-event bridge must use `absurd_queue_event`, not legacy DBOS event tables |
| `KRAMPUS_KORPUS_INGESTION` | active safe-storage target | Korpus/Krampus custody/componentization/indexing | artifacts can be hashed, stored, componentized, queued, and preserved | direct graph materialization from legacy ingest is not canonical and must stay blocked |
| `MODEL_RUNTIME` | staged/advisory | GGUF inventory, runner config, governor receipts, VRAM notes | local model artifacts/backends are tracked with provenance expectations | does not prove a full always-resident 4GB VRAM stack |
| `AHOY_STRATEGY_LAB` | externalized paused lab | `/home/mfspx/BOARD_GAMES/AHOY/` | board-game/simulation work is preserved outside active repo | not production truth, not an active LUCIDOTA runtime dependency |
| `SCRIPT_SURVIVAL_AUDIT` | active partial | script survival manifest, corpse manifest, Krampuschewing corpse queue, slop audit | scripts can be classified/repaired/corpse-logged without deletion | not complete until coverage is measured against all TICKLETRUNK script entries |

## Durable Workflow Decision

Current durable workflow substrate: **ABSURD/Postgres**.

DBOS status: **legacy compatibility only**.

Rules:

1. New work targets ABSURD/Postgres.
2. DBOS wording in old receipts remains provenance, not current architecture.
3. DBOS-era scripts/docs may remain only when labelled legacy, archived, or used to validate old receipts.
4. Active ledger rows should use ABSURD names unless specifically describing legacy provenance.

## Truth Promotion Boundary

Raw evidence, document spans, model output, GLiNER output, local graph candidates, Ahoy strategy labels, and Krampus/Korpus components are **not truth**.

Canonical truth requires:

1. evidence refs,
2. authority class,
3. graph-promotion preflight,
4. command envelope when operator confirmation/materialization is required,
5. guarded helper path,
6. graph journal entry,
7. materialization receipt.

## Plain-English Map

- TICKLETRUNK is the map of all tools.
- `00_PROJECT_BRAIN` is the law shelf.
- `06_SCHEMA` is the contract layer.
- `scripts` is the worker/gate/receipt layer.
- `05_OUTPUTS` is the proof shelf.
- `03_VAULT` and `09_STORAGE` are hoard/storage layers.
- `KRAMPUSCHEWING` and Korpus ingest/index artifacts.
- Chrono preserves time claims.
- ABSURD moves work.
- The graph gate decides whether candidates may approach canonical graph truth.
- Models advise/extract; they do not command.
- Ahoy is a contained strategy lab.

## Current Non-Negotiable Repair Targets

1. Keep status ledger aligned to ABSURD/Postgres.
2. Enforce worker contracts at dequeue.
3. Keep graph materialization gated behind command envelope + helper + journal.
4. Keep Korpus default storage-only until graph promotion replay is implemented.
5. Make acceptance receipts name their scope: local-file, DB integration, graph materialization, or model runtime.
6. Measure script audit coverage against TICKLETRUNK.
