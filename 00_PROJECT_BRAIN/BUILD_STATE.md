# BUILD STATE — WHERE THE BUILD ACTUALLY IS (2026-05-29)

> Do not trust this file without live verification. Query the DB first.
> Verify: `psql postgresql:///lucidota_state -Atc "SELECT phase_id,status FROM lucidota_control.phase_ledger ORDER BY phase_id;"`

---

## WHAT IS ACTUALLY RUNNING

| Component | Status | Notes |
|---|---|---|
| `lucidota_state` Postgres | LIVE | ~60 schemas/tables, workflow/control |
| `lucidota_storage` Postgres | LIVE | graph_item/edge/journal/staging_packet (schema `016`) |
| ABSURD queue | LIVE | 2465 succeeded, 45 dead-lettered, 0 queued |
| `lucidota_river_governor` | LIVE | Hoeffding tree, ~539 samples |
| treelite_router_v0.tl | LIVE | routing gate, scored 0.90 |
| Mamba-7B Q2_K (`:8081`) | LIVE on RAM | CPU, always-on |
| Bonsai-4B Q2 (`:8082`) | LIVE on RAM | always-on generalist |
| DeepSeek R1 1.5B Q4 (`:8080`) | LIVE on GPU | VRAM slot |
| Needle ×6 (`:8090-95`) | LIVE | 26M JSON routers |
| BGE-M3 Q8 | ON DISK | not loaded into inference server |
| SmolDocling 256M | ON DISK | model + ONNX q4 confirmed at `04_RUNTIME/models/smoldocling-256m-preview/` |
| ModernBERT-base | ON DISK | not loaded |
| GLiNER small v2.1 | ON DISK | adapter at `ALGOS/gliner_zero_shot_extractor.py` |
| `lucidota-stream-river` systemd | BROKEN | execs bare `python3`, not venv — Bytewax lane dead |

---

## WHAT IS ONE CONNECTION AWAY

- **Bytewax→River stream spine:** one-line ExecStart fix to venv interpreter activates the full online-learning loop. Command: edit `/etc/systemd/system/lucidota-stream-river.service`, change `ExecStart=python3` to venv path, `systemctl daemon-reload && systemctl restart lucidota-stream-river`.
- **ABSURD queue spine schema `035`:** unapplied. Apply to get durable queue with full chrono audit columns. `psql lucidota_state < 06_SCHEMA/035_*.sql`
- **Chrono three-clock `025-027`:** unapplied. Apply to get temporal evidence ledger.
- **Write-barrier triggers `040/074`:** unapplied. Without these, `canonical_graph_write_scanner.py` is the ONLY guard on canonical writes.
- **Graph-promotion pipeline `034/044/052`:** unapplied. Gates for 18,627 staged candidates.

---

## CRITICAL BLOCKERS (must resolve before heavy ETL)

1. **Schema backlog:** ~92/122 schema files unapplied. Decide which to apply in order. Start with: `035` (ABSURD spine), `025-027` (chrono), `040/074` (write barriers), `034/044/052` (promotion pipeline).
2. **Ontology collision:** `term_registry` has 45 rows (target 75). `CO_ACTIVE_TERMS.json` MISSING. GO/CO IDs @26-@50 collision — resolve alias-vs-renumber + migrate `graph_item.term` refs before any 75-seed.
3. **Bytewax service ExecStart fix** (one line, unblocks all streaming learning).
4. **18,627 staged candidates:** `graph_promotion_gate.py` blocks `--materialize` intentionally. Once write barriers + promotion pipeline applied, these can be reviewed and promoted.
5. **KANT69 1-DB migration:** two DBs still split. Target: one Postgres instance, scoped schemas (`lucidota_canon/ops/stage/corpse`) + REVOKE. Collapse-in-place, never greenfield.

---

## WHAT NORTHERN WANTS NEXT (ordered)

1. Load `GOALS/SONNET_CACHE_PACK.md` Tier A spine into context.
2. Query live state: `phase_ledger`, `runtime_status_fact`, term count, disk, `/proc/pressure`.
3. Co-design ingestion/ETL workflows WITH Northern (no Groq for the design phase).
4. Pre-conditions before heavy run: install missing ingestion deps (`00_PROJECT_BRAIN/organ_registry/50_ingestion.json` pip line), fix stream-river service, disk/resource budget receipt.
5. Apply the blocked schema files in order.
6. Run Epoch-1 reingest crush: full KRAMPUSCHEWING, byte-perfect CAS, GO-25 candidates, no canonical writes without gate.

---

## NEXT EXACT COMMANDS (issue in sequence, verify receipts)

```bash
# 1. Live state check
psql postgresql:///lucidota_state -Atc "SELECT phase_id,phase_name,status FROM lucidota_control.phase_ledger ORDER BY phase_id;"
psql postgresql:///lucidota_storage -Atc "SELECT COUNT(*) FROM lucidota_go.term_registry;"
psql postgresql:///lucidota_storage -Atc "SELECT COUNT(*) FROM lucidota_go.graph_item WHERE staging_status='candidate';"
python3 /home/mfspx/LUCIDOTA/scripts/lucidota_status_ledger.py --check

# 2. Fix stream-river (unblocks streaming learning)
# Find venv path:
find /home/mfspx -name "activate" -path "*/river*" 2>/dev/null | head -3
# Edit ExecStart in service file to use venv python, then:
sudo systemctl daemon-reload && sudo systemctl restart lucidota-stream-river
systemctl status lucidota-stream-river

# 3. Schema application (in order, test first)
# ABSURD spine:
psql postgresql:///lucidota_state < /home/mfspx/LUCIDOTA/06_SCHEMA/035_*.sql
# Write barriers:
psql postgresql:///lucidota_storage < /home/mfspx/LUCIDOTA/06_SCHEMA/040_*.sql
psql postgresql:///lucidota_storage < /home/mfspx/LUCIDOTA/06_SCHEMA/074_*.sql
```

---

## PHASES (from ledger — use live query above, not this)

- PHASE_00: COMPLETE (Indy home, memory, master plan)
- PHASE_01: IN_PROGRESS (LLMwiki, journal cadence)
- PHASE_02: IN_PROGRESS (governor v1)
- PHASE_03: IN_PROGRESS (Epoch-1 reingest crush)
- PHASE_04: PENDING (swarm plumb hardening)
- PHASE_05: IN_PROGRESS (governor v2 + telemetry)
- PHASE_06: IN_PROGRESS (Epoch-2 total legacy reingest)
- PHASE_07: IN_PROGRESS (scrapers → adapters)
- PHASE_08-09: PENDING

---

## DISK STATE (last verified 2026-05-28)

~97GB free after Opus session reclaimed ~61GB. More available: 9.5GB regenerable repo snapshot, 35GB unique stray-vault to migrate to canonical CAS. Git history has 1.6G tracked CAS blob under `scripts/03_VAULT/cas` — full history unsafe to push; use sanitized mirror branch.
