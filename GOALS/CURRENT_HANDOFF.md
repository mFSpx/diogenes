# CURRENT GOAL HANDOFF

"Save This Prompt, Pass on this Handoff:"

- Mode: **SONNET_BUILD**. Sonnet designs + builds; Groq = bulk hands; Opus = emergency only.
- Generated: `2026-05-30` (wave-2 complete + wave-3 major progress).

## WAVE-2 COMPLETE (all receipted)
- **WO-5 ✓** `percyphon_village` = 5000 rows × 128 slots. Seeder extended with synthetic seed generator.
- **WO-6 ✓** MISTLETOE: `ALGOS/mistletoe_graph.py` PageRank/centrality, read-only.
- **WO-7 ✓** Cerebras patch: `openai_compat.rs` provider wired (`gpt-oss-120b` alias).
- **WO-8 ✓** `krampus_stage_worker`: repointed to `lucidota_go.staging_packet`.

## WAVE-3 COMPLETE THIS SESSION
- **Schemas ✓** 045/046/050/094 applied. 094 had bad `authority_class` type → patched to text+CHECK.
- **governor_action table ✓** `lucidota_control.governor_action` in lucidota_state.
- **Governor daemon ✓** `lucidota_guvna.py` Phase-1: daemon loop (30s), DB logging, rung→ABSURD advisory via `runtime_status_fact`. Running. Log: `04_RUNTIME/guvna_daemon.log`.
- **Write-barrier triggers ✓** `graph_item_write_barrier_tg` (blocks approved-status without command_envelope_uuid) + `graph_edge_write_barrier_tg` (blocks NULL operator_uuid). Tested + live.
- **LTC ✓** `ALGOS/ltc.py` — Liquid Time-Constant Networks, Euler sub-stepping, feature_hash, evidence_intensity. LUCIDOTA role: temporal evidence stream processor.
- **JEPA/diffusion_forcing roles ✓** LUCIDOTA role declarations added to both docstrings.
- **Koopman signal refined ✓** `scripts/diffusion_pack_wire.py` v2 — primary signal is 5-D governor PSI telemetry (cpu/mem/io/mem_avail/fleet), not count-rate. Falls back to count-rate if governor series too short.
- **ORNAMENT revival ✓** 75 packs copied to `BOOKS/ontology_packs/ornament/`. Loader: `scripts/ornament_pack_loader.py` — 766 staging candidates upserted (signals→ENTITY/GRIP/SNARE, clusters→EVENT, hypotheses→CLAIM).
- **DIOGENES chat ✓** `scripts/diogenes_chat.py` — SANTA port: Groq HTTP, LUCIDOTA manifest (live DB facts), 04_RUNTIME/diogenes_sessions/ log. Tested with Groq.

## LIVE NOW (running autonomously)
- **Archive text crush** → `lucidota_korpus.corpus_chunk` (3216+, growing).
- **corpus→graph drain loop** → `lucidota_go.staging_packet` (10519+, growing).
- **BGE embed fleet** = 4 servers :8101-8104.
- **RAM watchdog** (suspend-on-pressure, never-cull). RAM avail ~2950MB.
- **Governor daemon** 30s interval, rung=1 (IO/CPU pressure from crushes), advisory=3 workers.

## DB RECEIPTS (live)
```sql
-- Quick resume check:
psql lucidota_storage -Atc "SELECT count(*) FROM lucidota_korpus.corpus_chunk; SELECT count(*) FROM lucidota_go.staging_packet; SELECT count(*) FROM lucidota_go.percyphon_village;"
psql lucidota_state -Atc "SELECT fact_key,fact_value FROM lucidota_control.runtime_status_fact WHERE subsystem='governor';"
```

## REMAINING WAVE-3 (continue here)
- **(a) claw/rig FULL integration**: Rust `cargo build --release` — WAIT for crushes to complete (check with `pgrep -fa "krampus_archive"`). When quiet: `cargo build --release` from `01_REPOS/claudecode/rust/`. Governor + Groq/Cerebras/local provider wiring into claw startup.
- **(c) Percyphon comms filter**: identity/VPN/proxy/burn detection — operator co-design needed before building.
- **(f) Promote→Materialize**: operator-confirmed path through graph_promotion_gate — needs operator.
- **(g) Adversarial audit suites + daily schedules**: run `python3 scripts/slop_audit_law.py --paths scripts/`, wire `groq_hammer_audit` for daily script regression, schedule via `lucidota_control.workflow_schedule`.

## RESUME COMMAND
```
Read 00_PROJECT_BRAIN/SONNET_HANDOFF.md and GOALS/CURRENT_HANDOFF.md.
Verify live: pgrep -fa "[k]rampus_archive_crush"; psql lucidota_storage -Atc "SELECT count(*) FROM lucidota_korpus.corpus_chunk; SELECT count(*) FROM lucidota_go.staging_packet;"
If crushes done → cargo build --release from 01_REPOS/claudecode/rust/.
If crushes still running → continue with (g) adversarial audit + daily schedule wiring.
```
