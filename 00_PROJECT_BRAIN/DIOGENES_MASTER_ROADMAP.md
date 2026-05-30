All load-bearing claims confirmed against the live DB. Key facts now verified:
- ABSURD spine, promotion packets, staging_packet, term_registry=75, temporal_claim table all **applied** (briefing's "92 unapplied / 45 terms" is stale).
- The break is real: `graph_staging_candidates` exists in no schema; `event_envelope`=0 rows, `script_registry`=0 rows, `temporal_claim`=0 rows (built-but-unfed).
- `governor_action` table = **missing** (null). Write-barrier triggers on canonical tables = **none attached**. The 4 genuine unapplied schemas (045/046/050/094) confirmed absent. `corpus_chunk`=1185 (a growing dead-end library). `claw` does not start the model stack or any governor.

I have everything needed. Producing the master roadmap.

---

# DIOGENES MASTER ROADMAP — Co-Design Draft for Operator Approval
*Chief architect synthesis of 8 grounded sub-plans + live-DB verification (read-only, 2026-05-29). Embed crush + Groq OCR in flight — every action below is schema-apply / file-edit / deterministic-test only, no heavy jobs, no model loads.*

**Trust correction (load-bearing):** OPUS_BRIEFING §8/§17 are STALE. Live DB: state=143 tables/15 schemas, storage=154 tables/14 schemas. ABSURD spine, graph promotion, chrono ledger (`lucidota_korpus.temporal_claim`), `staging_packet`=130, `term_registry`=75 are **all applied**. The "92 unapplied / 18,627 staged / 45 terms" backlog is a **phantom** — do not redo applied work. The real gaps are narrow and listed below.

---

## 1. ONE-SCREEN STATUS TABLE

| Subsystem | %-ready | Rollup (built/partial/stub/missing) | Single biggest blocker |
|---|---|---|---|
| **etl-ingest-graph** | ~55% | promotion spine BUILT, ingest front BUILT, but the wire between them is broken | `krampus_stage_worker.py:172` writes to phantom `lucidota_korpus.graph_staging_candidates` (in **no schema**, **not in DB**) → always JSONL fallback → staged claims never reach `graph_promotion_packet`. **The single highest-leverage fix in DIOGENES.** |
| **claw-rig-providers** | ~70% | api crate (the de-facto "rig.rs") + 6 lanes BUILT; Cerebras/Bonsai/Needle/embeddings MISSING; governor routing STUB | Cerebras has a key in secrets but **zero Rust code** (spec-required); governor never feeds `ProviderClient::from_model` (static `--model` only). |
| **governor-guvna** | ~30% | doctrine canon BUILT; parts scattered (hw_gate/feedback_governor/ram_watchdog/capped_run PARTIAL); daemon + slices + `governor_action` table + PSI + claw-wiring MISSING | **No single daemon ties telemetry→decision→enforcement**; embed-fleet-vs-chat RAM collision avoided only by not running both. `governor_action` table absent (verified null). |
| **ux-input-router-logging** | ~50% | router/lane-gate/membrane/percyphon BUILT; DB-logging spine BUILT but **0 live rows**; keystroke logging MISSING; claw→DB wiring MISSING | `event_envelope`=**0 rows** (verified): the UX lane has never logged a real operator turn. Two unmerged parallel routers. |
| **output-multiplexer** | ~45% | weave_output + template_contract + doc-gen template corpus BUILT; book-quote/LLM-gen lanes fed EMPTY; stylography/Indy-commentary MISSING | Live router calls `weave_output(rag_quotes=[], deepseek_synthesis='')` — 2 of 6 lanes structurally present but **fed nothing**; no book-quote corpus exists. |
| **percyphon-village** | ~25% | 90-line scaffold + kernel bridge BUILT; 5000×128 matrix / xxhash / relevance-confidence / concept-expansion / village curator / comms-filter MISSING | The 5000×128 generative engine doesn't exist (today = 12 fixed + N fluid slots, one scalar, SHA-256 not xxhash). Two FALSE doctrine claims to correct first. |
| **capability-suite** | ~20% | KRAMPUS ingest spine + image/adversarial ALGOS BUILT; ORNAMENT/DRE/MISTLETOE/CHIMNEY/LUMP/STOCKING/REGULATOR/TINSEL all ARCHIVED-not-live; asymmetric-game ENTIRELY MISSING | Least-built subsystem. Every named capability lives only under `KRAMPUSCHEWING/.../KRAMPUS_EXPRESS/` on a forked SQLite core; **no ABSURD/GO-25 shim**. Blotto/min-cut/Nash = zero hits anywhere. |
| **hygiene-autonomic** | ~60% | rich audit arsenal + mega_gate + status_ledger + slop_audit BUILT; but run AD HOC, no scheduler, no reconciliation auditor | **No cron/systemd timer drives any audit** (verified: no lucidota timers); no script reconciles 125 DDL files vs live DB; `script_registry`=0 rows; write-barrier triggers **not attached** (verified empty). |

**System-wide:** the *spine* (queue → contract → promotion gate → materialize, plus chrono/promotion schemas) is far more built than the briefing claims. What's missing is **wiring, scheduling, and the net-new generative/analytic capabilities** — not the foundation.

---

## 2. DEPENDENCY ORDER — Critical Path in Phases

```
P0 truth+safety   ─┐
                   ├─► P1 governor-floor ─► P2 close-the-break ─► P3 on-the-graph ─┐
P0 schema-reconcile┘                                                              ├─► P5 capabilities
                                                                                  │
            P1 claw-providers ─► P2 claw-UX-logging ─► P4 output-multiplex ───────┘
                                                                                  │
                                          P4 percyphon-engine ───────────────────►┘
                                                                                  │
                                                       P6 autonomic-scheduler (wraps all)
```

**Hard ordering rules (why):**
1. **Governor Phase-0 precedes running any lane hot** — the embed-fleet-vs-chat RAM collision has no arbiter today. Build the suspend/never-cull daemon *before* turning on chat+embed simultaneously.
2. **Schemas-online + write-barrier attached + graph-promotion gate** must precede anything claiming "on the graph." The gate works (5 packets) but isn't fed and the DB-level barrier trigger isn't attached.
3. **The stage→packet wire (P2)** is the gateway: ingest, corpus_chunk backfill, and every capability that stages candidates all depend on it.
4. **The api crate ("rig.rs") + provider lanes precede claw-finish** — Cerebras/Bonsai must be in the binary before governor routing or UX logging can route to them.
5. **temporal_claim/promotion schemas precede hypertimeline + claim-vs-hypothesis enforcement** (P5 capabilities).
6. **Autonomic scheduler is last** — you don't schedule audits until the things being audited exist and pass once by hand.

---

## 3. PER-PHASE BUILD PLAN

### **P0 — TRUTH + SAFETY FLOOR (this session; pure read-only + schema-apply + doc-fix)**
*Make the map match the territory; close the 4 genuine schema gaps; attach the write-barrier.*

| Build | Reuse | Builder | Acceptance receipt |
|---|---|---|---|
| `scripts/schema_apply_state_audit.py` (read_only): enumerate all 125 `06_SCHEMA/*.sql`, probe both DSNs via `to_regclass`/`pg_proc`/`pg_trigger` | DDL idempotency (`IF NOT EXISTS` universal); `run_instruction_hygiene.py` receipt pattern | orchestrator-designs | `05_OUTPUTS/schema_audit/schema_apply_state_<ts>.json` lists exactly the 4 real gaps (045/046/050/094) and confirms 035/025-027/034 applied |
| Apply the 4 genuine unapplied schemas (045 document_ingestion, 046 catchme, 050 document_claim_packet_bridge, 094 workflow_foundry) | `scripts/chronological_migrator.sh`, idempotent DDL | orchestrator-designs | re-run auditor → 0 unapplied; `to_regclass('lucidota_korpus.document_parse_run')` non-null |
| Attach write-barrier: apply `040`+`074` so `enforce_graph_promotion_path()` triggers are ON `graph_item/edge/journal` | existing `graph_promotion_preflight()` fn | orchestrator-designs | `pg_trigger` query returns the 3 barrier triggers (verified absent today) |
| Correct stale docs: OPUS_BRIEFING §8/§17 (92→4 unapplied, 45→75 terms); `percyphon_doctrine_20260529.md` (remove false "truncated" + false "route_decision.percyphon field"); `schema_audit_20260529.md`→pointer to live receipt | `git show HEAD:ALGOS/percyphon.py` + `ast.parse` as proof | orchestrator-designs | diff committed; doctrine cites live DB facts |

### **P1 — GOVERNOR FLOOR + PROVIDER LANES (parallel; safe to build during crush)**
*The arbiter that lets lanes run hot, and the missing model hands.*

| Build | Reuse | Builder | Acceptance receipt |
|---|---|---|---|
| `scripts/lucidota_guvna.py` Phase-0 daemon (telemetry→decision→never-cull suspend) | promote `lucidota_feedback_governor.py:46-96`; `hw_gate.read_hw_state()`; `ram_watchdog.sh` SIGSTOP/CONT; `bge_fleet.sh` COUNT knob | orchestrator-designs (Groq hands for boilerplate) | `tests/test_guvna_never_cull.py` green (synthetic RED → suspend in slice order, never `luci-custody`); live `guvna_collision_receipt_<ts>.json` |
| 6 systemd `.slice` units (luci-custody/ops/reingest/llm/scrape/archive) | KANT69 canon L111-135 (declarative, no code) | orchestrator-designs | BGE fleet launched into `luci-llm.slice`, crush into `luci-archive.slice`; `systemd-cgls` shows placement |
| `06_SCHEMA/126_governor_action.sql` (`lucidota_control.governor_action`) | runtime_status_fact upsert pattern (039/082) | orchestrator-designs | table applied; daemon appends suspend+resume pairs; `SELECT action,slice,count(*)` returns recovery pairs |
| PSI ingestion (`/proc/pressure/*`) into hw_gate | ~15 LOC into `hw_gate.read_hw_state()` | orchestrator-designs | hw_gate JSON carries `psi.memory.avg10` |
| Rust: Cerebras + Bonsai(:8082) lanes | `openai_compat.rs` `OpenAiCompatConfig` (verbatim); MODEL_REGISTRY rows; `client.rs:113` local-url switch | Groq+Cerebras-hands (Rust edits), orchestrator commits | `cargo test -p api`: `from_model('cerebras').provider_kind()==Cerebras`; `from_model('bonsai-ram')` resolves `:8082` |

### **P2 — CLOSE THE BREAK + UX LOGGING (depends P0 schemas, P1 governor live)**
*The single highest-leverage wire in the whole system.*

| Build | Reuse | Builder | Acceptance receipt |
|---|---|---|---|
| **FIX THE BREAK:** repoint `krampus_stage_worker.py:172` from phantom table to existing `lucidota_go.staging_packet` (016) | `staging_packet` (already applied, 130 rows) | orchestrator-designs | `staged.json` shows `lane='db'` (NOT `jsonl_fallback`), `staged_count>0` |
| **BUILD THE MISSING WIRE:** stage → enqueue `graph_promotion` job → `absurd_graph_promotion_worker` → `graph_promotion_gate.py` → packet | `graph_promotion_gate.py` (BUILT, 5 packets), `absurd_graph_promotion_worker.py` (BUILT) — **both ends exist** | orchestrator-designs | **END-TO-END OUROBOROS RECEIPT:** `graph_promotion_packet` count increases from 5, new packet carries the ingested file's `evidence_ref` |
| Enforce contracts on krampus_intake/route_extract/stage at dequeue | `absurd_worker_contracts.validate_worker_contract` | Groq-hands | contract-violation test rejects unregistered `(queue,kind,worker_key)` |
| Wire `language_router.py` → `project2501_board_move.py persist_bundle` so every operator turn writes `event_envelope` + `route_decision`; add claw→DB hook in `conversation.rs:155` | `project2501_board_move.py` (BUILT, schema 112 applied), `event_envelope` (BUILT, 0 rows) | orchestrator-designs + Rust-hands | `event_envelope WHERE source='claw_chat'` ≥ 1 (verified 0 today); `verbatim_hash` == sha256(input) |
| Archive divergent `corpus_ingest_worker.py` (phantom creds) | `KRAMPUSCHEWING/Script_Corpses/` + hash log | orchestrator-designs | entry in `KRAMPUSCHEWING_SCRIPT_CORPSES.jsonl` |

### **P3 — ON-THE-GRAPH BACKFILL + GOVERNOR WIRED INTO CLAW (depends P2)**
| Build | Reuse | Builder | Acceptance receipt |
|---|---|---|---|
| `scripts/corpus_chunk_promotion_bridge.py` (candidate_writer): 1185 corpus_chunk rows → grouped → term_registry match → `graph_promotion` jobs | `graph_promotion_gate.py`, `term_registry` (75) | orchestrator-designs + local-models-after-embedding (term match) | packet count climbs from corpus provenance; `canonical_graph_write_scanner` = 0 violations |
| Operator-confirmed materialize (one node): lift `canonical_materialization_disabled_for_hardening_sprint` block, run `graph_promotion_materialize.py --execute` with command-envelope | `graph_promotion_materialize.py` (BUILT, only canonical materializer) | orchestrator + **operator confirm** | `graph_promotion_materialization` 0→1; `graph_item` +1; `graph_journal` row links packet→decision→envelope |
| Wire governor into `claw` launcher (guarded bg launch, CLAW_GUVNA_PID) | one launcher line | orchestrator-designs | `pgrep -af lucidota_guvna` returns PID after `./claw` |
| Governor→provider routing hook (`CLAW_GOVERNOR_ROUTING=1`) consults `lucidota_model_governor.py` | `model_vram_scheduler.py`, `lucidota_river_governor.py` | Rust-hands | `route_decision` receipt logs governor-chosen model |

### **P4 — OUTPUT MULTIPLEXER + PERCYPHON ENGINE (depends P1 providers, P3 graph)**
| Build | Reuse | Builder | Acceptance receipt |
|---|---|---|---|
| Feed empty lanes: book-quote index (`scripts/book_quote_index.py`, BGE-M3 over BOOKS/), LLM-gen via model fabric; add stylography + indy_commentary lanes to `weave_output` | `language_membrane.weave_output` (extend, don't rewrite); `template_contract.py`; 43 KRAMPUS `.j2` templates; `stylometry_features`; expand `persona_stamp.md` from INDY_SOUL.md | orchestrator-designs + local-models-after-embedding (quotes/synth) | `tests/test_output_multiplexer_lanes.py` green: all 6 lanes non-empty, quotes carry doc_id+score, `outbound_state:draft_only` |
| Percyphon 5000×128 engine: identity slots 1-28, procedural 29-128 (concept-expansion), per-slot relevance-confidence, **xxhash** coords; `06_SCHEMA/NNN_percyphon_village.sql`; `percyphon_village_curator.py` | extend `ALGOS/percyphon.py` (back-compat fluid_slots=4); `soul_registry` (1-5000); `diogenes_memory_gate.py` probe; FOSS python-xxhash | orchestrator-designs | `tests/test_percyphon_village.py` green; `percyphon_village` up to 640k rows; `diogenes_memory_gate` `zero_vram:true` at 128 slots |

### **P5 — CAPABILITY SUITE (depends P0 chrono schemas, P2 wire, P3 graph)**
| Build | Reuse | Builder | Acceptance receipt |
|---|---|---|---|
| TINSEL→ABSURD scrape shim (`tinsel_dispatch_worker.py` + `tinsel_register_contracts.py` + contract row) | `spine_krampus_worker.py` template; `pathway_venturi_intake.py`; TINSEL `scraper_common.py`/`scraper_registry.py` | Groq-hands | drop TIN-001 BCFSA job → `ScraperResult` in `raw_artifact` + tinsel contract row |
| MISTLETOE → ALGOS (betweenness/PageRank/bridges, read_only over GO-25 EDGE) | 10 pure-stdlib MISTLETOE modules; psycopg params | orchestrator-designs | ported tests green; betweenness ranking JSON over live EDGE rows |
| **NET-NEW** asymmetric-game ALGOS: `max_flow_min_cut.py` (stdlib Edmonds-Karp), `colonel_blotto.py`, `stackelberg_security_game.py` | seed kinematics from `chelydrid_ambush.py`; zero-dep ethos | orchestrator-designs | self-test reproduces known min-cut to machine epsilon (ALGOS smoke-test convention) |
| CHIMNEY emitters → ABSURD candidate_writers; promote `hypertimeline_engine_core.py` out of Script_Corpses (now that temporal_claim applied) | CHIMNEY `extraction/` emitters; evidence_strength 6-tier | orchestrator + local-models | hypertimeline run → `temporal_claim` rows (0→N); claim-vs-hypothesis enforced at write |
| Outbound (gated external_effect): aliased email (Percyphon alias + Rust SMTP), static-site gen (minijinja), scrape-anywhere (TINSEL above) | Percyphon alias provider; Rust core | orchestrator + **operator confirm per effect** | each external effect emits `external_effect` receipt behind operator gate |

### **P6 — AUTONOMIC SCHEDULER (wraps everything; last)**
| Build | Reuse | Builder | Acceptance receipt |
|---|---|---|---|
| `scripts/schema_reconcile_apply.py` + `hygiene_nightly.py` orchestrator | `lucidota_mega_gate.py` (~20 gates), `status_ledger`, `slop_audit_law`, `subsystem_quality_audit`, `diogenes_mirror.py` | orchestrator-designs | `hygiene_nightly_<ts>.json` verdict=PASS with all sub-receipts |
| systemd user timers (`lucidota-hygiene.timer` daily, `lucidota-deep-audit.timer` weekly), **using `.venv/bin/python3`** | declarative units; harden thin fault injectors | orchestrator-designs | `systemctl --user list-timers` shows `lucidota-hygiene.timer` with future NEXT + non-empty LAST |
| Populate `script_registry` (run `diogenes_mirror.py --execute`); fix pytest-outside-venv trap | existing scripts | orchestrator-designs | `script_registry` count > 0 (verified 0 today) |

---

## 4. IMMEDIATE NEXT 5 ACTIONS (this session — crush-safe, no model loads, no heavy jobs)

1. **Run the schema-reconciliation auditor (build + run).** Write `scripts/schema_apply_state_audit.py` (read_only, just `to_regclass`/`pg_proc`/`pg_trigger` probes). **Receipt:** `05_OUTPUTS/schema_audit/schema_apply_state_<ts>.json` showing exactly 4 genuine gaps. Kills the phantom-backlog narrative permanently.

2. **Apply the 4 real schemas + attach the write-barrier.** `psql ON_ERROR_STOP=1` apply 045, 046, 050, 094, then 040+074. **Receipt:** re-run auditor → 0 unapplied; `pg_trigger` query returns the 3 `enforce_graph_promotion_path` triggers on `graph_item/edge/journal` (verified absent now). Schema-apply only — no workers run.

3. **Fix THE BREAK (one-file edit).** Repoint `krampus_stage_worker.py:172` `STAGING_TABLE` from phantom `lucidota_korpus.graph_staging_candidates` → `lucidota_go.staging_packet`; align columns. **Receipt:** `scripts/krampus_stage_worker.py --once` produces `staged.json` with `lane='db'` (not `jsonl_fallback`) — proves the central break is closed. Single small file, no crush impact.

4. **Build the stage→packet wire + run one packet end-to-end.** Have stage enqueue a `graph_promotion` job; let the already-live `absurd_graph_promotion_worker`→`graph_promotion_gate.py` create the packet. **Receipt:** `psql ... -Atc "SELECT count(*) FROM lucidota_go.graph_promotion_packet"` increases from **5**, new row carries the ingest `evidence_ref`. This is the OUROBOROS proof — one corpus item becomes a graph candidate without bypassing the gate.

5. **Correct the stale doctrine docs (commit the truth).** Edit OPUS_BRIEFING §8/§17 (92→4 unapplied, 45→75 terms, ABSURD/promotion/chrono APPLIED), fix the two false Percyphon-doctrine claims (cite `ast.parse` + `git show HEAD`), repoint `schema_audit_20260529.md` to the new live receipt. **Receipt:** committed diff; no future agent chases a phantom backlog.

*Deferred to P1+ (NOT this session, respects crush): governor daemon, Cerebras Rust lane, corpus_chunk backfill, any materialize, anything that loads a model or runs the BGE fleet + chat simultaneously.*

---

## 5. DEFINITION OF OPERATIONAL (the green checklist)

DIOGENES is operational per spec when **all** are true with a fresh receipt or live DB fact:

- [ ] **Ouroboros closes:** a dropped file flows watcher→intake→route_extract→stage→`staging_packet`→`graph_promotion` job→`graph_promotion_packet`→ operator-confirmed `graph_promotion_materialization` (count >0) with a `graph_journal` chain. `corpus_chunk` is no longer a dead-end (bridge live).
- [ ] **Write integrity:** `enforce_graph_promotion_path` triggers attached to `graph_item/edge/journal`; `canonical_graph_write_scanner --output` = 0 unallowlisted violations.
- [ ] **Governor alive & arbitrating:** `pgrep lucidota_guvna` returns PID after `./claw`; `governor_action` shows suspend+resume pairs under load; chat-model load while `MemAvailable<800MB` emits `refuse`/`throttle` and no kernel OOM (the embed-fleet-vs-chat collision is automatically arbitrated, not avoided by hand).
- [ ] **Operator turns logged:** `event_envelope WHERE source='claw_chat'` > 0; `route_decision` carries deterministic lane + (P4) percyphon scaffold; `board_stream_run` > 0.
- [ ] **Model hands present:** `cargo test -p api` green for cerebras + bonsai-ram; per-lane `/chat/completions` round-trip receipts under `05_OUTPUTS/model_runtime/`.
- [ ] **Output is multiplexed:** `test_output_multiplexer_lanes.py` green — all 6 lanes (template, book-quote w/ source refs, LLM-gen, stylography, indy-commentary, smoothing) non-empty, `draft_only`, provenance per lane.
- [ ] **Capabilities live (not archived):** TINSEL scrape job → `raw_artifact`; MISTLETOE betweenness over live EDGE; asymmetric-game ALGOS self-test passes; hypertimeline writes `temporal_claim` rows.
- [ ] **Claim vs hypothesis enforced:** promotion gate blocks evidence-less claims, admits `evidence_strength>=documentary`.
- [ ] **Autonomic clock running:** `systemctl --user list-timers` shows `lucidota-hygiene.timer` with future NEXT + non-empty LAST; nightly `hygiene_nightly_<ts>.json` verdict=PASS within 24h; `script_registry` count>0.
- [ ] **Tranche-stable gate:** `lucidota_status_ledger.py --check` = CHECK_OK; `STATUS_LEDGER.md` cites exact receipt paths.

---

## 6. STUBS / GAPS REGISTER (consolidated — "functionally awesome only"; fakes called out)

**HARD BREAKS / FAKES (fix first):**
| # | Stub/Gap (fake or broken) | Owner | Fix |
|---|---|---|---|
| 1 | `krampus_stage_worker.py:172` → phantom `lucidota_korpus.graph_staging_candidates` (no schema, not in DB, **verified**); always JSONL fallback; receipt always says `canonical_graph_writes:false` | etl-ingest | Repoint to `lucidota_go.staging_packet` (P2) |
| 2 | No stage→packet wire; promotion gate (5 packets) fed only by hand/health-check, not ingest | etl-ingest | Build the wire (P2) — both ends already exist |
| 3 | No `corpus_chunk`→packet bridge; 1185 rows are a dead-end library | etl-ingest | `corpus_chunk_promotion_bridge.py` (P3) |
| 4 | `event_envelope`=**0 rows** (verified): UX-logging lane built/tested but never fed a real turn | ux-router | Wire router + claw hook (P2) |
| 5 | `governor_action` table **missing** (verified null); no daemon ties telemetry→decision→enforcement | governor | `126_governor_action.sql` + `lucidota_guvna.py` (P1) |
| 6 | Write-barrier triggers **NOT attached** to canonical tables (verified empty `pg_trigger`); 040/074 unenforced — canon guarded only by static scan | hygiene/graph | Apply 040+074 (P0) |
| 7 | `tickletrunk_fault_injector.py` + `status_ledger_fault_injector.py` emit PASS **unconditionally** — theater, not adversarial proof | hygiene | Make injectors assert rejection (rc≠0) (P6) |
| 8 | `absurd_flows.py` + `capability_pack_registry.py` set `psycopg=None` and emit success receipts with no DB write; `case_packet_compiler.py` non-idempotent packet_id; `bitloops_river_worker.py` hard-codes `labels=True`; `abductive_db_os_health_check.py` hard-codes `canonical_graph_writes=False` w/o scanning | hygiene | Remediate per groq_hammer_audit (P6) |
| 9 | `corpus_ingest_worker.py` phantom `lucidota`/`lucidota` creds + wrong Groq API shape | etl-ingest | Archive to Script_Corpses (P2) |
| 10 | `script_registry`=**0 rows** (verified) despite "882 upserted" claim | hygiene | Run `diogenes_mirror.py --execute` (P6) |

**MISSING (net-new build):**
| # | Gap | Owner | Fix |
|---|---|---|---|
| 11 | Cerebras Rust lane (key present, **zero code**, spec-required) | claw-providers | `OpenAiCompatConfig::cerebras()` (P1) |
| 12 | Bonsai :8082 + Needle :8090-95 not in MODEL_REGISTRY | claw-providers | Registry rows / mark keep-Python (P1) |
| 13 | Governor never feeds `from_model` (static `--model` only) | claw/governor | `CLAW_GOVERNOR_ROUTING` hook (P3) |
| 14 | Keystroke logging does not exist anywhere | ux-router | event_envelope.detail telemetry sink (P2) |
| 15 | Book-quote corpus/index absent; router passes `rag_quotes=[]`; stylography + indy-commentary lanes absent | output-mux | BGE-M3 index + 2 lanes (P4) |
| 16 | Percyphon 5000×128 engine, xxhash, relevance-confidence, concept-expansion, village curator, comms-filter all missing | percyphon | Extend engine + schema + curator (P4) |
| 17 | Asymmetric/Blotto/min-cut/Nash/Stackelberg — **zero hits** anywhere | capability | NET-NEW ALGOS (P5) |
| 18 | `tinsel_dispatch_worker.py` + `tinsel_register_contracts.py` **missing** (verified); no scrape entry point beyond Playwright desperation | capability | Build shim (P5) |
| 19 | Outbound action (websites, aliased email, code-while-operating) — entirely greenfield | capability | P5 gated external_effect |
| 20 | No schema-reconciliation auditor; no autonomic scheduler (verified: no lucidota timers) | hygiene | P0 auditor + P6 timers |
| 21 | The 6 KANT69 systemd slices + PSI ingestion missing | governor | P1 |

**PARTIAL / DEFERRED:**
| # | Gap | Owner | Fix |
|---|---|---|---|
| 22 | No real PDF/docx/xlsx/audio/video parser in route_extract (regex/Groq-vision only); archives not recursively unpacked there; 7z/rar/zst declared but unimplemented in archive_crush; slow-lane JSONL has no consumer | etl-ingest | CHIMNEY emitters + Rust intake Phase-2 (P5) |
| 23 | `lucidota-intake` Rust Phase-2 Postgres writer unbuilt (Cargo.toml has no sqlx) | claw/rust | Build Phase-2 (P5) |
| 24 | 4 genuine unapplied schemas: 045, 046, 050, 094 (verified null) | hygiene | Apply (P0) |
| 25 | Two parallel routers (`language_router.py` receipt-only vs `project2501_board_move.py` DB-persisting) unmerged; treelite `.tl` artifact unwired (inline tree used) | ux-router | Merge/delegate + wire `.tl` (P2/P4) |
| 26 | `hypertimeline_engine.py` = 11-line wrapper importing from Script_Corpses; FairyFuse native ternary = `STUB_BACKEND_NOT_WIRED`; `ternary_lens_router.py` = `STUB_BACKEND_NOT_WIRED`; GLiNER not in live router path; `persona_stamp.md` 64-byte stub | capability/output/percyphon | Promote core / expand stamp (P4/P5) |
| 27 | pytest only in `.venv` (system python3 has none) — CLAUDE.md commands silently fail | hygiene | venv guard in release gate (P6) |
| 28 | Mutation-class declarations missing on most of 446 scripts; 5 dead + 9 stub scripts unarchived | hygiene | Automated check + archive (P6) |
| 29 | `treelite_gate` falls open (score=0.5/slow) on any failure — masks dead router, no alarm | etl-ingest | Add alarm (P2) |

**DOC ERRORS (not code — correct in P0):** OPUS_BRIEFING §8/§17 stale (92→4 unapplied, 45→75 terms, spine/promotion/chrono actually APPLIED); `percyphon_doctrine_20260529.md` two FALSE claims (source "truncated" — it parses clean at HEAD; `route_decision.percyphon` field — no such column); `schema_audit_20260529.md` stale. Percyphon `ALGOS/percyphon.py` is intact (correct OPUS_BRIEFING §23/§15).

---

**Bottom line for approval:** the foundation (queue, contracts, promotion gate, chrono/promotion schemas, term registry, api crate) is *real and applied* — the briefing undersold it. DIOGENES is gated by **wiring** (one broken table reference + one missing enqueue = the whole ingest→graph loop), **one safety daemon** (the governor floor), and a tier of **net-new generative/analytic capability** (Percyphon engine, asymmetric games, capability-suite revival). P0+P2 actions 1-4 alone close the Ouroboros and turn "looks green" into a live packet-count delta. Awaiting operator approval to execute the Immediate Next 5.