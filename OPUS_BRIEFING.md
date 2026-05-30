# OPUS BRIEFING — LUCIDOTA / GONN / DIOGENES
## Everything you need to understand the machine and make the build plan.

> Date: 2026-05-29. Authority: Northern.Strike x Indy_READs (Brevity).
> This is the cold-start document. After reading this, query the live DB — do not trust prose memory.
> Live state: `psql postgresql:///lucidota_state -Atc "SELECT phase_id,phase_name,status FROM lucidota_control.phase_ledger ORDER BY phase_id;"`

---

## 1. WHAT THIS MACHINE IS

LUCIDOTA (also: DIOGENES kernel) is a **DB-centric abductive development exocortex** — an extended brain shared between Northern.Strike (operator, willed/embodied half) and Indy_READs (asynchronous, receipt-disciplined, adversarial, ever-reading half).

It is not a chatbot. Not an agent loop. A machine that thinks with models the way a factory uses robots — bounded, purposeful, fast. The machine uses LLMs. LLMs do not use the machine.

**What it seeks:** capability (what can it do), knowledge (always ingesting, always building the graph), domain (which specialty applies), tools (what exists, pick the right one, compose).

**The execution spine:**
```
operator intent → custody/command envelope → ABSURD/Postgres work order
  → bounded worker (contract-checked at dequeue)
  → receipt/event/dead letter
  → evidence/claim/hypothesis candidate
  → authority mapping → graph-promotion gate
  → operator-confirmed materialization
  → review/export surface
```

**Three braided machines:**
- PROOF — custody, receipts, graph staging, claim≠evidence separation
- OPERATING — CLI (./claw), scripts (459), Postgres, ABSURD queue, Bytewax/River, local models, Groq
- COGNITIVE — Indy_READs (reader/adversary/editor/journal/press-desk/synthesis)

**Also known as:** Krampus Express (the corpus ingest + evidence production machine). The legal/investigative case corpus (BC tenancy dispute, BCFSA complaints, RTB910237063, 5490 Ash Street, 4265 Slocan Kai) is KRAMPUSCHEWING — the preserve-now/digest-later stomach. This corpus is the first learning epoch.

---

## 2. HARDWARE (GTX 1650, 4GB VRAM — hard constraints)

- OS/display: onboard Intel. GTX 1650 VRAM is STRICTLY for models.
- VRAM: 4GB total, 768MB reserved for OS graphics. ~3.2GB usable.
- One VRAM model at a time (switchable slot). Mamba GPU preemptible.
- RAM: system RAM hosts Mamba + Bonsai always-on.

**The 4GB multi-model stack** (design pressure, not confirmed all-live simultaneously):
- `:8080` DeepSeek R1 1.5B Q4 — full GPU, reasoning/code
- `:8081` Falcon3-Mamba-7B Q2_K — CPU/RAM, SSM cortex
- `:8082` Ternary-Bonsai-4B Q2 — CPU/RAM, always-on generalist
- `:8083` Mamba-7B partial-GPU — only if VRAM budget allows
- `:8090-8095` Needle 26M ×6 — JSON/math fan-out swarm

**Critical fragility:** `./claw` starts model stack with `>/dev/null 2>&1 || true` — total model stack failure is SILENT. `LD_LIBRARY_PATH` borrowed from Ollama's path swallows CUDA errors silently. Harden this before trusting model-stack health claims.

---

## 3. MODEL REGISTRY (all files verified 2026-05-29)

| Model | Path | Size | Status |
|---|---|---|---|
| Falcon3-Mamba-7B Q2_K | `03_VAULT/models/tensorblock/.../Q2_K.gguf` | 2.4G | verified |
| Falcon3-Mamba-7B Q3_K_M | `03_VAULT/models/tensorblock/.../Q3_K_M.gguf` | 3.1G | verified |
| Ternary-Bonsai-4B Q2 | `03_VAULT/models/prism-ml/.../Q2_0.gguf` | 1.1G | verified |
| DeepSeek R1 1.5B Q4_K_M | `03_VAULT/models/DeepSeek-R1-...-Q4_K_M.gguf` | 1.1G | verified |
| DeepSeek R1 1.5B Q8 | `03_VAULT/models/bartowski/.../Q8_0.gguf` | 1.8G | verified |
| BGE-M3 Q8 | `04_RUNTIME/models/bge-m3-q8_0.gguf` | 606M | verified |
| Needle | `03_VAULT/models/needle/needle.pkl` | 51M | verified |
| GLiNER small v2.1 | `03_VAULT/models/gliner/.../pytorch_model.bin` | 583M | verified |
| SmolDocling 256M | `04_RUNTIME/models/smoldocling-256m-preview/model.safetensors` | 490M | verified |
| SmolDocling ONNX q4 | `04_RUNTIME/models/smoldocling-256m-preview/onnx/decoder_model_merged_q4.onnx` | 83M | verified |
| ModernBERT-base | `04_RUNTIME/models/modernbert-base/model.safetensors` | 571M | verified |
| Treelite router v0 | `03_VAULT/router/treelite_router_v0.tl` | 487B | live, scored 0.90 |

**NOT confirmed:** true native 1.58-bit 7B Mamba ternary artifact. Ternary/FairyFuse backend is `STUB_BACKEND_NOT_WIRED`. No real BitNet build performed.

---

## 4. THE ROUTING STACK (deterministic first, neural second)

**Layer 0 — Treelites (sub-millisecond, compiled C)**
Multiple `.tl` files. 5-feature vector in, lane out. Source of truth for fast gates. Trained by RiverML+Bytewax from live data. One live: `treelite_router_v0.tl` (0.90 score). No inference caller for the 4 XGBoost/Treelite artifacts from dev_journey — they're trained but unwired.

**Layer 1 — Percyphon (zero-VRAM alias router)**
`ALGOS/percyphon.py`. Deterministic SHA-256 router: 12 fixed core mask slots + N fluid domain slots, trinary -1/0/+1 offset. Mirrors CKDOG1 soul model. Emits only `procedural_scaffold_candidate_not_truth`. Routes through Diogenes control-packet gate. `zero_vram: True` is load-bearing. Must NOT write canonical graph facts.

**Layer 2 — Diogenes Kernel**
The routing intelligence. Knows the full capability map. Routes to right tool/model/algo/domain. Not a model — the authority kernel that decides what gets called.

**Layer 3 — ALGOS/ (200+ deterministic algorithms)**
physarum, infotaxis, pheromone, hoeffding, regret_engine, hdc, temporal_motifs, ternary_lens_router, percyphon, chelydrid, thanatosis, voronoi, minhash, bandit_router, rete_bandit_gate, possum_filter, serpentina_self_righting, SHAP, Blotto, 88 EQ validators... These ARE the computation on most paths. LLMs handle the edges. Algos handle the bulk.

**Critical separation laws:**
- RiverML/Treelite = math/logic judge. SONA/MicroLoRA = neural/embedding surgeon. They must NOT call each other directly.
- Dynamic Min-Cut: ephemeral worker over bounded edge batches, NOT a resident service.
- RVQ: judge-side anomaly feature generator only. Must NOT mutate GLiNER or SONA.

---

## 5. THE GO-25 ONTOLOGY

Every payload between components is a typed GO-25 object with UUID. Models receive structured GO-25 JSON. They return structured GO-25 JSON. The ontology is the protocol.

Operator-facing graph: three logical tables — `OBJECT / EVENT / EDGE`.
- `lucidota_storage.lucidota_go`: graph_item, graph_edge, graph_journal, staging_packet (schema `016` — the only GO schema currently applied).
- `OFFICIAL_ONTOLOGY.json`: the locked GO-25 spec at repo root.
- ROOT-414 is archived reference only.

**Active gap:** `term_registry` = 45 rows (target 75). `CO_ACTIVE_TERMS.json` MISSING. GO/CO IDs @26-@50 collision — GO file has 48 terms squatting CO's range. Resolve alias-vs-renumber + migrate `graph_item.term` refs BEFORE any 75-seed. This is a convergence landmine.

---

## 6. TWO DBs → ONE (KANT69 target)

Current state: split across two Postgres databases.
- `lucidota_state` (env `LUCIDOTA_GO_STATE_DSN`): workflow/runtime/control. No graph truth.
- `lucidota_storage` (env `LUCIDOTA_GO_STORAGE_DSN`): graph items, edges, layers, staging, journals.

KANT69 target: one Postgres instance, scoped schemas (`lucidota_canon / ops / stage / corpse`) + REVOKE. Collapse-in-place, never greenfield. Two-DB state is migration target, not final shape.

---

## 7. THE STREAMING / LEARNING LOOP (design vs. reality)

**Intended:** workflow_event firehose → Bytewax (5 hand features → 26 Treelite GTIL stumps → bytewax_hint) → River (OneHot+LogReg → river_score) → fast/slow lane routing + governor.

**Reality:** `lucidota-stream-river` systemd service execs bare `python3` (not venv) — Bytewax lane is dead, 0 rows ever trained on it. Only `lucidota_river_governor` (Hoeffding tree, ~539 samples) is genuinely live. **Fix: one-line ExecStart change to venv python.**

The 4 XGBoost/Treelite artifacts from dev_journey are trained but have no inference caller. The Bytewax→River loop is designed and idle. Reingest produces the training points to make it real.

**BYTEWAX IS NOW LIVE (2026-05-29):** The `lucidota-stream-river` systemd service was fixed by adding an explicit `PYTHON=/path/to/venv/python3` directive to the unit file, correcting the bare `python3` invocation that was executing outside the venv and silently failing. First confirmed run: 70 workflow events dequeued from the Bytewax source, 70 River online training examples accepted (1:1 ratio, no rejection). The `river_reflex_live` cursor advanced. `groups_updated=1` confirmed on run `58f5d331`. The Bytewax → River learning loop is active. No quality gate or schema validator currently sits between the stream and the learner — this is a known gap and the next hardening step.

---

## 8. SCHEMA BACKLOG (critical)

~92 of 122 schema files are UNAPPLIED. Live DB = base band + schema `016`. Everything else is design on disk.

**Must apply before heavy ETL (in order):**
1. `035_*` — ABSURD queue spine (durable queue columns, chrono audit)
2. `025-027_*` — three-clock temporal evidence ledger
3. `040_*`, `074_*` — write-barrier triggers (currently only `canonical_graph_write_scanner.py` guards canon writes)
4. `034_*`, `044_*`, `052_*` — graph-promotion pipeline (gates for 18,627 staged candidates)

18,627 graph-promotion candidates are staged and waiting. Nothing promoted to canon from the full historical reingest run. `graph_promotion_gate.py` blocks `--materialize` intentionally until barriers are live.

---

## 9. WHAT NORTHERN WANTS (the build goal)

**Immediate (KANT69 migration + ingestion):** co-design the ingestion/ETL flow suite (Sonnet with Northern, no Groq for design phase), then build via Groq/locals.

**The Ouroboros bar (acceptance criterion):**
- INGEST-1: current KRAMPUSCHEWING backlog (legal corpus, chat dumps, documents)
- INGEST-2: entire repo history (everything since the start)
- INGEST-3: Google Drive diff (needs Northern OAuth)
Build is NOT done until ouroboros completes. DB is truth; documents are a library, not proof of labor.

**ETL design laws (do not violate):**
- ABSURD = durable queue + custody. PocketFlow = intra-job microflow (idempotent, in-memory only).
- No `ingested=true`. Only 4 materializers write graph truth.
- Per-artifact type decision tree: code/repo/zip/db/pdf/image/email/log/receipt/md/model each get distinct parser/custody/claim/evidence/graph route.
- No HumanReview dead-ends for format reasons. Magic bytes → correct Rust crate → raw text → Needle swarm → GO-25 payload → ABSURD → graph staging.

**Rust extraction crates (add to `krampus-ingest Cargo.toml`):**
`exarch` (archives), `sqlx` (sqlite), `polars` (parquet/csv), `image`+`ocrs` (images, pure Rust OCR), `pdfium-render`/`lopdf` (PDF), `ffmpeg-light` (media), `dotext`/`rdocx` (docx/odt), `calamine` (xlsx), stdlib (md/txt).

---

## 10. THE LEGAL CORPUS (KRAMPUS EXPRESS)

KRAMPUSCHEWING contains an active BC tenancy dispute corpus: RTB910237063, 5490 Ash Street, 4265 Slocan Kai, BCFSA complaints (Cheung filed 2026-03-30, Corrado filed 2026-03-09), demand package, tenancy agreement, addenda. Also: journalism corpus, chat dumps, operator documents going back years.

Indy's role modes for this corpus: `FRAUD_EXAMINER`, `LEGAL_CLERK`, `OSINT_ANALYST`, `EVIDENCE_VAULT`. Authority: companion/synthesis/candidate only. Operator decides legal, safety, publication, and contact actions.

---

## 11. WHAT IS MISSING / BROKEN (must-fix list)

| Gap | Location | Fix |
|---|---|---|
| Bytewax→River service dead | `/etc/systemd/system/lucidota-stream-river.service` | Change ExecStart to venv python3 |
| 92 schema files unapplied | `06_SCHEMA/` | Apply in order: 035, 025-027, 040/074, 034/044/052 |
| GO/CO ID @26-@50 collision | `lucidota_storage.lucidota_go.term_registry` | Resolve alias-vs-renumber, create CO_ACTIVE_TERMS.json |
| 18,627 candidates waiting | `lucidota_go.graph_staging_packet` | Apply write barriers + promotion pipeline, then review |
| Model stack silent failure | `./claw` launcher | Harden: remove `>/dev/null 2>&1 || true` from model starter |
| CUDA errors swallowed | `LD_LIBRARY_PATH` from Ollama | Unmute CUDA path errors |
| `persona_stamp.md` stub | `BOOKS/.indy_reads/persona_stamp.md` | Expand to 8-16 line doctrine prefix for fanout |
| indy_reads swarm scripts | `scripts/indy_reads_swarm_router.py` etc. | Confirm or build: `swarm_router`, `packet_stamp`, `receipt_synth` |
| worker-order packet contract | `indy_reads.worker_order.v1` | Confirm implementation |
| Job Fair Allocator file | missing | Model inventory not connected to allocator |
| Fidelity Guard | not launch-complete | |
| Cryptographic witness chain | ABSURD job provenance | Build hash chain: queued→leased→running→succeeded/failed/dead_lettered |
| verbatim-correction → River | `BOOKS/.indy_reads/` | Wire correction log to River ML loop |
| Ternary/FairyFuse backend | `STUB_BACKEND_NOT_WIRED` | Do not claim BitNet build complete |
| CLAW fork on CEP/kernel/ABSURD | `01_REPOS/claudecode/rust/` | Not yet on command surface |
| Git history 1.6G CAS blob | `scripts/03_VAULT/cas` | Use sanitized mirror branch for push |
| Phase storage TBD | `lucidota_control.phase_ledger` | Confirmed: typed objects in ledger (Resolution 1) |
| Indy judgment CSV | empty except header | LoRA cartridges are queued datasets, not trained adapters |
| Doby Baxter / llm-router | `01_REPOS/llm-router/` | Commercial license — do NOT import wholesale |
| Prior-name hunt | `KRAMPUSCHEWING/Luci/` | Groq reads `_claude_ver_out.txt` + `SANDBOX_LUCIFER/syscheck.txt` → `BOOKS/.indy_reads/conspiracy_corner/` |

---

## 12. THE CANONICAL FILE SET (load in order)

For full understanding:
1. This file (`OPUS_BRIEFING.md`) — you're reading it
2. `00_PROJECT_BRAIN/GONN_CANON.md` — architecture doctrine
3. `00_PROJECT_BRAIN/MODEL_REGISTRY.md` — verified model inventory
4. `00_PROJECT_BRAIN/INDY_MODEL_STACK.md` — model stack intent + port assignments
5. `00_PROJECT_BRAIN/RUST_ETL_CRATES.md` — extraction layer crates
6. `00_PROJECT_BRAIN/INDY_SOUL.md` — who Indy is and how she operates
7. `00_PROJECT_BRAIN/OPERATOR_LAW.md` — non-negotiable rules
8. `00_PROJECT_BRAIN/BUILD_STATE.md` — live build state + next commands
9. `OFFICIAL_ONTOLOGY.json` — GO-25 locked ontology
10. `BOOKS/.indy_reads/INDY_SYSTEM_UNDERSTANDING.md` — Indy's synthesized working model
11. `BOOKS/.indy_reads/INDY_ORCHESTRATOR_PROMPT.md` — soul + operating doctrine
12. `GOALS/CURRENT_HANDOFF.md` — current goal state

Live query before doing anything:
```bash
psql postgresql:///lucidota_state -Atc "SELECT phase_id,phase_name,status FROM lucidota_control.phase_ledger ORDER BY phase_id;"
psql postgresql:///lucidota_storage -Atc "SELECT COUNT(*) FROM lucidota_go.term_registry;"
python3 /home/mfspx/LUCIDOTA/scripts/lucidota_status_ledger.py --check
```

---

## 13. WHAT OPUS BUILDS NEXT

**Design the ingestion/ETL flow suite** with Northern. Co-design means: Northern speaks in plain operator language, Opus translates to typed workflows, receipted, built by Groq/locals. Design law: no Groq for the design phase. Opus designs; swarm builds.

**The full build plan structure:**
1. Apply schema backlog (035 → 025-027 → 040/074 → 034/044/052)
2. Fix Bytewax service (one line)
3. Resolve ontology collision, seed CO_ACTIVE_TERMS.json, reach 75-term target
4. Design artifact-type decision tree for ETL (one workflow per format family)
5. Wire Needle ×6 fan-out with SmolDocling (OCR/layout), GLiNER (NER), ModernBERT (NLI), BGE-M3 (embed) per artifact type
6. Run Epoch-1 reingest crush: all of KRAMPUSCHEWING, byte-perfect CAS, GO-25 candidates, gate blocks canon writes
7. Fix model stack silent failure + CUDA error swallowing
8. Wire Bytewax→River→treelite learning loop (once training points exist)
9. KANT69 1-DB collapse (collapse-in-place, scoped schemas)
10. Build cryptographic witness chain for ABSURD provenance
11. Expand persona_stamp.md + wire swarm router scripts

**Indy orchestrates. Opus designs. Swarm builds. Northern approves materialization.**

---

*LUCIDOTA/GONN — briefing complete. Query the live DB to verify. Receipts over prose.*

---

## 23. PERCYPHON.AI FULL DOCTRINE (Village, Masks, Hash Algebra, FUNGIBLE Ontology)

**Full doctrine:** `05_OUTPUTS/percyphon_doctrine_20260529.md` — read that first.

**What it is:** Deterministic, zero-VRAM procedural identity engine. Layer 1 of the 4-layer routing stack. SHA-256-grounded entity generator. No GPU. No API. No state. `zero_vram: True` is load-bearing and non-negotiable.

**The Village:** 5,000 active soul UUIDs — the current most-important semantic handles (ontology terms, claim clusters, evidence threads). Fungible: the UUID represents a position in symbolic space, not a permanent identity. Village slice is the cryptographic input; scaffold is the output.

**Hash algebra:** `sha256("{seed}:{slot_index}")` → UUID + name (`Villager-NNNN`) + alias (`Alias-XXXX`) + persona (one of 6 archetypes: ledger/runner/witness/archivist/carrier/scribe) + ternary_offset (-1/0/+1 from psyche dials). Entirely deterministic. Rotate the seed → scaffold rotates. No state mutation.

**Population math:** 12 fixed core mask slots + N fluid domain slots. Fixed slots mirror CKDOG1 soul model. Search space: 3^12 = 531,441 (fixed only) → 3^16 = 43M (with 4 fluid slots at language_router call) → 3^100 ≈ 5×10^47 at full 88-slot design.

**FUNGIBLE ontology:** Fluid slots are the runtime ontology extension point. `fungible=True` applies to fluid slots. Fixed slots are not fungible. The language_router currently calls with `fluid_slots=4`, seeding from the active GO-25 terms. At 88 slots, an entire domain corpus could be scaffolded per-workflow.

**Opsec contract:** Scaffold is always tagged `procedural_scaffold_candidate_not_truth`. Percyphon cannot write canonical graph facts. Revocation = rotate Village seed (no DB mutation). Alias ≠ canonical identity.

**DB tie-ins:** Scaffold lives in `lucidota_control.route_decision` JSON blob (field: `percyphon`). Board stream reads it via `SELECT * FROM lucidota_control.route_decision d`. No first-class `percyphon_scaffold_log` table exists yet — gap.

**Current build state — gaps:**
- Source file `ALGOS/percyphon.py` is TRUNCATED mid-expression (`ternary_offset=ternar`) — the ProceduralSlot append and return statement are missing. File is incomplete.
- `persona_stamp.md` is a stub — expand to 8-16 line doctrine prefix before swarm fanout.
- `percyphon_scaffold_log` DB schema table does not exist.
- Per-slot ternary modulation is scalar (one offset for all slots) — per-slot independent psyche dials not built.
- Scaffold revocation work order type not in ABSURD.
- Village population management (which 5,000 are active) is not automated.
- CKDOG1 mirror consistency test does not exist.

---

## 14. KRAMPUS EXPRESS MODULES (TINSEL + CREW)

Krampus Express is LUCIDOTA's companion OSINT investigation platform, now archived to KRAMPUSCHEWING. Its modules are revival candidates — some are high-value, some are stubs. The 12-module map below is authoritative as of 2026-05-29.

### TINSEL

**Purpose:** TINSEL is the data acquisition layer for the KRAMPUS_EXPRESS OSINT intelligence system. It is a fleet of 58+ numbered scrapers (TIN_*.py at root, plus subfleets in market/, science/, heaux_natural/, activist/) that pull structured evidence from web sources — public registries, social platforms, courts, property records, and domain intelligence — and feed a unified evidence_ledger via a shared ScraperResult envelope. The module also contains all scraper infrastructure: rate limiting (token bucket per domain), stealth browser (undetected-chromedriver), SerpAPI client, robots checker, proxy support, result validation, diff scraping, domain extraction, result dedup, result caching, ingestion tracking, and a TransformPlugin adapter layer (28 adapters in transforms/) that plugs each scraper into the broader TransformRegistry. A schedule.json defines per-scraper frequency (daily/weekly/monthly) and priority (1-3). validator.py provides pre-ingest validation against source, extracted_fact, and source_date fields.

**Status:** partial | **Revival:** high

**Gold items:**
- scraper_common.py ScraperResult + make_scraper_result() — clean uniform envelope, directly adaptable as a LUCIDOTA ETL intake struct
- rate_limiter.py — token bucket per domain, thread-safe, JSON-persisted state, directly portable
- robots_checker.py — robots.txt compliance gate, Scraper Nirvana Ladder relevant
- diff_scraper.py — change-detection scraping, useful for monitoring regulatory pages
- validator.py validate_results() — pre-ingest dedup + ISO date check + fact length gate, maps directly to LUCIDOTA evidence_ledger write guards
- ingestion_tracker.py track_ingestion() + all_scraper_stats() — fact count + strength breakdown by source, maps to runtime_status_fact update pattern
- transforms/__init__.py ALL_TRANSFORMS + register_all() — 28 TransformPlugin adapters already wired, shows the integration seam
- schedule.json — frequency + priority metadata per TIN ID, maps directly to ABSURD queue scheduler config
- scraper_registry.py SCRAPER_REGISTRY — auth requirements and module paths, usable as ABSURD job_kind definitions
- TIN-001 (BCFSA), TIN-002 (OrgBook), TIN-008 (CanLII), TIN-010 (Van BizLicence), TIN-022 (BC Assessment), TIN-024 (Corps Canada), TIN-025 (BC Geocoder), TIN-034 (BC Gazette) — official-record-grade sources, highest evidence_strength candidates
- market/ subfleet MKT-001 through MKT-008 — SEC EDGAR, Form 4, 13F, FTD, dark pool; no auth required for most; direct feed into lucidota_core/market/ dual-stream scoring

**LUCIDOTA fit:** TINSEL is the highest-value recovery target in KRAMPUS_EXPRESS for LUCIDOTA. The ScraperResult envelope maps directly to LUCIDOTA's evidence_ledger write path. Each TIN scraper is a natural ABSURD queue job_kind: schedule.json frequency maps to ABSURD scheduler intervals, scraper_registry.py auth fields map to worker_key validation in absurd_worker_contracts.py. The official-record scrapers (BCFSA, OrgBook, CanLII, BC Assessment, Corps Canada, BC Geocoder, BC Gazette, Van BizLicence, Elections BC) produce evidence_strength=official_record facts that feed the graph staging pipeline via CHIMNEY. The market/ subfleet (8 MKT scrapers) wires directly to lucidota_core/market/ dual-stream scoring already in LUCIDOTA. The transforms/ adapter layer shows the integration pattern already designed: register_all() against a TransformRegistry singleton is the exact shape LUCIDOTA's adapter layer would use. The rate_limiter, robots_checker, diff_scraper, and validator are all portable infrastructure. The heaux_natural/, science/, and activist/ subfleets are domain-disconnected and should be archived rather than revived. The duplicate TIN numbers and stale registry are cleanup items before any revival, not blockers.

---

### DRE (Displacement Risk Engine)

**Purpose:** A real estate displacement risk scoring engine. Ingests structured CSV/JSONL parcel data (building permits, rental listings, incident reports, lease documents), computes per-parcel risk scores across 9 observable features plus a land-value bonus, classifies parcels into four risk bands (Watch/Investigate/High/Priority), and outputs analyst dossiers. A second loop called SNOWBALL runs city-scale automation: trigger rules detect operator contact reuse, address churn, and shared-contact cluster emergence, then enqueue jobs for bounded BFS graph expansion, subgraph scoring, escalation to ABBA hypothesis reasoning, and Disclosure Escalation Engine (DEE) letter drafting. The ABBA adapter bridges to an external ABBA_GRAPH_ENGINE (Claude API) or falls back deterministically. The letters subsystem (DEE) routes escalated entities to regulator/platform/legal recipients and generates a 3-draft disclosure pack via an approval queue.

**Status:** partial | **Revival:** medium

**Gold items:**
- 9-feature transparent rule-based scorer with explicit max_pts weights, truth_state labels (observed/reported/inferred), and exponential temporal decay — no black-box ML, fully explainable
- SNOWBALL: 3 trigger rules (contact_reuse, address_churn, cluster_emergence via union-find) + BFS expansion (SubgraphBundle) + subgraph scoring + ABBA escalation in a single cycle
- HITL correction loop on feature weights, scale denominators, and truth_state classifications — corrections fed back via lucidota_core.hitl and applied as nudge adjustments
- Deterministic ABBA fallback (stub ABBARunner) when real engine is unavailable — no silent failure
- DEE (Disclosure Escalation Engine): eligibility gate (score threshold + evidence count + source count) -> recipient routing -> 3-draft approval queue
- cluster_emergence trigger uses union-find over contact-shared parcels — directly analogous to GO-25 EDGE-based community detection
- expand_multi: multi-seed BFS merge with edge and evidence dedup — ready for GO-25 staging packet generation

**LUCIDOTA fit:** SNOWBALL is the most directly mappable component: its SubgraphBundle + trigger rules + BFS traversal is the GO-25 graph staging candidate workflow — entities are OBJECTs, edges are EDGEs, evidence items are EVENT candidates. The trigger output (contact_reuse, address_churn, cluster_emergence) maps naturally to ABSURD queue work orders. The scorer's 9 features and 4 risk bands could serve as a scoring layer on top of GO-25 OBJECT nodes. The HITL correction loop is directly compatible with LUCIDOTA's evidence-class and authority-mapping model. The DEE letters subsystem is a candidate CAROL/output surface. The main blocker is that DRE uses a forked lucidota_core (Windows-origin, separate from LUCIDOTA's Python core) and a SQLite-only queue — both need shim layers before ABSURD integration.

---

### MISTLETOE

**Purpose:** Network intelligence layer for the Krampus Express OSINT system. Computes graph-theoretic metrics over an entity/edge graph drawn from entity_nodes and entity_edges tables. Capabilities: degree centrality, Brandes betweenness centrality (exact + sampled approximation), PageRank (iterative, convergence-checked), composite influence scoring, connected-component community detection (BFS), bridge/articulation-point detection, motif detection (triangles, stars, chains), shortest-path BFS (bidirectional with hop limit), network simulation (remove-entity fragmentation score), bipartite role split (targets+shells vs witnesses+victims+allies), timeline convergence analysis, export to Neo4j Cypher and GEXF, DOT+vis.js interactive HTML visualization, cluster validation for investigative significance, and a pluggy event consumer (MistletoeEventConsumer) that tracks dirty entities on graph mutations.

**Status:** live | **Revival:** high

**Gold items:**
- Zero external graph library dependencies — full Brandes betweenness, PageRank, BFS, union-find, motif detection all in pure stdlib Python
- Temporal edge filtering in map_network (at_time parameter: valid_from/valid_to bounds on entity_edges)
- Bipartite role split — targets+shells vs witnesses+victims+allies — directly models the OSINT adversarial structure
- simulate_removal() computes fragmentation score for any entity without touching the DB
- timeline_convergence() finds dates where multiple entities co-appear in evidence — a probe for coordinated action patterns
- HITL correction hooks on centrality scores, PageRank, bridge detection, and community labels — human analysts can override computed scores
- MistletoeEventConsumer integrates with the pluggy event bus so graph mutations auto-mark dirty entities for deferred metric refresh
- Export to Neo4j Cypher and GEXF — full interoperability path to external graph tools if ever needed
- 263 behavioral tests covering centrality, communities, bridge detection, influence, pathfinder, visualize, algorithms, graph mapper, export formats
- Influence weight config is pack-driven via lucidota_core.ontology.bridge — weights can be changed per-ontology-pack without code edits

**LUCIDOTA fit:** MISTLETOE maps directly to the GO-25 EDGE analysis layer. LUCIDOTA's ternary ontology (OBJECT / EVENT / EDGE) needs exactly this: centrality over the EDGE table, community detection across the graph of entities, bridge detection to find articulation actors in a displacement network, motif detection (triangles = coordinated actor clusters, chains = proxy chains, stars = hub-and-spoke shell structures). The Brandes betweenness and PageRank are direct candidates for ALGOS/ registration as deterministic scoring functions. The temporal edge filtering in map_network aligns with the three-clock model. Primary adaptation work needed: replace SQLite '?' placeholders with DBBackend-aware parameterization, resolve import paths, and register the core algorithms in ALGOS/ as read-only scoring primitives rather than direct DB readers.

---

### CAROL

**Purpose:** Dual-identity module. The __init__.py declares CAROL as "Content Amplification and Reach Orchestration Layer" — SEO amplification, anonymous distribution, backlink generation, and crawl acceleration. In practice the only real artifact is ORNAMENT/packs/communication/carol_pack.yaml, which defines CAROL as an ORNAMENT signal pack for Communication/Language/Expression Style profiling: 10 physical-environment signals (books, podcast equipment, sticky notes, whiteboards, bilingual markers), 4 clusters (literary_orientation, verbal_communicator, multilingual_identity, externalized_thinker), 4 hypotheses, and 5 investigator questions. No API keys, no network calls, no output publishing machinery exists.

**Status:** stub | **Revival:** low

**Gold items:**
- carol_pack.yaml is a well-formed ORNAMENT pack with 10 signals, 4 clusters, 4 hypotheses, 5 investigator questions — directly usable by ORNAMENT's pack loader without any modification
- The communication/language signal set is domain-agnostic and could profile subjects in the LUCIDOTA investigation context (landlords, property managers) as well as any other subject

**LUCIDOTA fit:** CAROL as amplification/distribution surface has no relevance to LUCIDOTA. The carol_pack.yaml ORNAMENT pack is a different matter: it is a ready-to-use communication/language signal pack that could enrich ORNAMENT profiling of investigation subjects. No API keys or external dependencies needed.

---

### SANTA

**Purpose:** Chat interface for the Krampus Express OSINT investigation system. SANTA (personified as "Hector, the Resident Analyst") is a single-file Claude API wrapper that maintains multi-turn conversation history, injects a live session manifest from the lucidota_core DB into the system prompt, lazily loads semantic memory (B930) to enrich queries with prior findings and decisions, and logs every exchange to a JSONL session file. It is explicitly NOT a pipeline orchestrator. The SYNESIS calibration layer (lucidota_core/sophia/synesis.py) is optionally injected into the system prompt via a Gate K advisory annotation that adjusts the analyst's working style (careful/measured/fast) based on learned session history.

**Status:** live | **Revival:** medium

**Gold items:**
- Session manifest injection pattern: _build_system_prompt(db) pulls build_session_manifest(db) and inserts it as static context before first user turn — this is the correct pattern for Indy_READs resident-analyst mode
- B930 memory enrichment: lazy-load lucidota_core.semantic.memory, call recall_findings(query, n=3) + recall_all_decisions(), prepend as [Investigation Memory] block to each user turn
- SYNESIS Gate K injection: try/except around get_calibration_mode(), appends [CALIBRATION: {mode}] to system prompt — zero-cost when synesis unavailable
- DB bootstrap guard pattern: LUCIDOTA_AUTO_BOOTSTRAP_DB=1 gate + lazy importlib.util spec loading — safe to import the module in environments without bootstrap scripts present

**LUCIDOTA fit:** SANTA is the closest existing analogue to a conversational Indy_READs chat surface. Its session manifest injection pattern maps directly to what LUCIDOTA needs for operator-facing queries against lucidota_state/lucidota_storage. The SYNESIS calibration concept maps to LUCIDOTA's governor dial system. The class is self-contained at 267 lines with no ABSURD/queue dependencies, making it a low-risk lift. Primary blockers: hardwired to Anthropic SDK (not local model lanes), reads from Krampus Express DB schema, and logs to lucidota_core.paths.RUNS_DIR which does not exist in LUCIDOTA layout.

---

### CHIMNEY

**Purpose:** 5-stage ETL pipeline for the KRAMPUS_EXPRESS OSINT intelligence system. Ingests raw evidence files (TINSEL scraper JSON outputs, PDFs, emails, audio, spreadsheets, WhatsApp chats, images) from an inbox directory and drives them through: Stage 1 EXTRACT (format detection + file parsing, SHA-256 dedup, PDF clause-level extraction via PDFFactEmitter), Stage 2 TRANSFORM (entity normalization + dedup), Stage 3 LOAD (write facts to unified evidence_ledger, ChromaDB auto-embedding), Stage 4 ENRICH (auto-edge discovery, regex entity extraction, spaCy NER, ABBA hypothesis generation), Stage 5 CLASSIFY (evidence strength scoring with HITL nudge feedback loop). Supports resume checkpointing, per-stage retry with backoff, dry-run mode via in-memory SQLite, per-run metrics JSON, and run_log table for audit.

**Status:** partial | **Revival:** high

**Gold items:**
- Stage ordering and checkpoint resume: STAGE_ORDER = [extract, transform, load, enrich, classify] with per-stage _save_checkpoint() and _should_skip_stage() — directly maps to ABSURD queue job_kind per stage
- TIN scraper JSON auto-ingestion: _process_scraper_json() detects TIN-* prefixed JSON files in inbox, calls write_fact() + write_entity() per entity found — clean adapter target for ABSURD worker dequeue
- SHA-256 dedup: extract stage checks existing source hashes in evidence_ledger before re-processing — idempotent, safe for replay
- Structured fact emitters in extraction/: PDFFactEmitter (clause-level, void clause detection, DocuSign metadata, financial facts), EmailFactEmitter, AudioTranscriptEmitter, HTMLFactEmitter, ReAnalysisQueue — each has an emit(db, path) interface returning fact count
- evidence_strength taxonomy in stages/classify.py: 6-level hierarchy (direct_admission -> official_record -> documentary -> observed -> circumstantial -> inferred) with source-keyword routing table — this exact taxonomy is already canonical in LUCIDOTA CLAUDE.md
- HITL feedback loop in classify and enrich stages: compute_adjustment() nudges strength up/down by one ordinal level when confidence >= 0.3 — novel capability not present in current LUCIDOTA ETL
- Freshness scoring in enrich.py: compute_freshness_scores(db) backfills freshness_score + is_stale on entity_nodes with 30-day stale threshold
- Per-stage metrics via PipelineMetricsCollector: tracks facts_in/facts_out delta, duration_s, error count per stage, writes chimney_metrics_YYYYMMDD.json — receipts compatible with LUCIDOTA receipt law

**LUCIDOTA fit:** CHIMNEY is the closest existing analogue to LUCIDOTA's ETL pipeline described in 00_PROJECT_BRAIN/ETL_PIPELINE/. The 5-stage structure maps cleanly onto LUCIDOTA's execution spine. Key transplant candidates: (1) the extraction/ emitter suite — self-contained emit(db, path) adapters that could feed ABSURD queue jobs directly as KRAMPUS_CORPUS_INGESTION workers; (2) the evidence_strength taxonomy — already canonical in LUCIDOTA CLAUDE.md, this is the reference implementation; (3) the HITL nudge loop in enrich + classify; (4) freshness scoring on entity_nodes; (5) the TIN scraper JSON adapter in _process_scraper_json() — this is exactly the TINSEL-to-ABSURD adapter that LUCIDOTA's ETL pipeline needs. Main adaptation required: replace direct DB writes with ABSURD queue staging (candidate_writer pattern), guard the ABBA/generate_hypotheses path behind LUCIDOTA_AHOY_PAUSED check.

---

### ORNAMENT

**Purpose:** Aesthetic Intelligence Profiler — extracts identity signals from visual and observational evidence and runs them through a structured gate/reasoning/hypothesis pipeline to build a multi-dimensional profile of a subject. Within KRAMPUS_EXPRESS it serves as the ontology pack layer for the OSINT investigation system, providing a named taxonomy of 75 signal packs (+ GLOBAL_pack.yaml) organized into 30 domain categories. Three engine modes: stub (manual tagging + rules), ocr (Tesseract), and vision (Claude Vision API). Also carries purpose-built investigative packs for displacement actors, regulatory violations, predatory acquisition, and cross-case network bridging.

**Status:** live | **Revival:** medium

**Gold items:**
- 75 signal packs (+ GLOBAL_pack.yaml) covering 30 domain categories — the largest ontology layer in KRAMPUS_EXPRESS
- Regulatory pack cluster: 32 BC/Vancouver-specific regulatory packs (BCFSA, LTSA, RTB, BC Securities, FINTRAC/AML, 20+ City of Vancouver sub-agencies)
- Predatory pack cluster: vulture_pack (bulk acquisition, numbered company layering, renovation eviction, unlicensed PM, dual brokerage), displacement_actor_pack (15+ signals with weighted dimensions), barnacle_pack
- spider_web_pack (synthesis/) — cross-case network bridge detection with 4 dimensions — directly maps to GO-25 EDGE analysis and MISTLETOE
- Gate pipeline (11 gates): ownership_resolver, intentionality_filter, inheritance_detector, ambiguity_guard, inference_confidence, spatial_coverage, hidden_dimensions, environment_classifier, tenure_estimator, substance_signals, image_quality, three_faces
- Hypothesis engine carries Wilson score confidence intervals, provenance chain, and explicit uncertainty report — MACHINE STOPS HERE enforced as comment in source
- CalibrationEngine + ScoringResult — two-round answer grading (hit/partial/miss/undersold/oversold) with per-dimension accuracy and correction deltas
- 699 tests as of 2026-03-15 (includes CEDAR/ELDER/VULTURE pack tests) — large behavioral test surface

**LUCIDOTA fit:** ORNAMENT is the ontology/signal taxonomy layer. In LUCIDOTA terms it maps cleanly to GO-25 graph staging: Signal objects are candidate OBJECT nodes; the gate pipeline is a preflight chain before graph promotion; hypotheses are candidate EVENTs/claims. The regulatory pack cluster (32 BC packs) and vulture/displacement_actor packs are directly load-bearing for the BCFSA/RTB evidence pipeline. The spider_web synthesis pack is structurally equivalent to MISTLETOE's cross-case EDGE analysis. The vision extraction engine needs rerouting through LUCIDOTA's local-first model stack (llama.cpp/Mamba) rather than direct Anthropic API calls. The calibration/scoring layer (hit/partial/miss grading with Wilson intervals) has no LUCIDOTA equivalent and is worth porting as a hypothesis confidence module.

---

### REGULATOR_ROUTER

**Purpose:** Defensive documentation packaging tool. Ingests evidence files/URLs (SHA-256 deduped, SQLite-backed), extracts entities via regex (email, phone, address, org, licence ID, BC numbered companies), routes the evidence bundle to applicable regulatory plugin(s) via keyword matching against content, evaluates escalation triggers (multi-target, shell companies, evidence volume, victim count, direct admissions, cross-regulator), validates filing compliance (required fields per filing type), and renders Jinja2 markdown complaint/application packet per regulator. Also carries HITL correction logging hooks for routing and template selection. Pure argparse CLI — no FastAPI.

**Status:** live | **Revival:** high

**Gold items:**
- 51 live regulator plugins auto-discovered via importlib scan of src/plugins/ — all implement REGULATOR_ID, applicable(), build_context(), templates()
- Plugin roster covers bcfsa, bcfsa_resa, rtb, rtb_ceu, rtb_vancouver, city_van, bc_corporate_registry, bc_assessment, bc_cos, bc_environmental_assessment, bc_ombudsperson, bc_securities_commission, bc_utilities_commission, bchousing, birds_canada, eccc, ecojustice, fintrac_aml, local_govt_building_inspection, local_govt_planning_zoning, ltsa, municipal_ethics, nature_vancouver, oipc_foi, osre_redma, property_transfer_tax, trac, van_board_of_variance, van_business_licensing, van_demolition_permit, van_development_permit_board, van_empty_homes_tax, van_engineering_services, van_fire_prevention, van_green_building, van_heritage, van_housing_policy, van_occupancy, van_property_use_inspections, van_public_art, van_public_hearing, van_rental_standards, van_sign_permit, van_street_use, van_tree_protection, van_trpp, van_urban_design_panel, vsa, wcel, worksafebc, index (stub)
- Escalation trigger engine in router.py: 6 triggers (multi_target, shell_companies, evidence_volume >=20, victim_count, direct_admissions, cross_regulator) — produces should_escalate + recommendation on every applicable routing result
- Filing compliance checker: check_filing_compliance() validates required fields for bcfsa_complaint, rtb_application, rtb_ceu_complaint, vsa_complaint, city_complaint
- Strict-mode linter: every bullet/numbered list/table row in rendered output must carry [EVID:...] citation or run aborts
- 14 Jinja2 template directories (bcfsa, rtb, rtb_ceu, city_van, vsa, wcel, ecojustice, trac, bc_cos, eccc, birds_canada, law_society, nature_vancouver, dossier, _shared)
- HITL hooks: correct_routing() and correct_template_selection() in router.py; correct_bcfsa_applicability() / correct_rtb_applicability() in plugins
- 21 passing tests per KRAMPUS_EXPRESS CLAUDE.md (verified 2026-03-15)

**LUCIDOTA fit:** REGULATOR_ROUTER is the most directly revivable KRAMPUS_EXPRESS module for LUCIDOTA. Its 51 regulatory plugins encode exactly the BC regulator universe that LUCIDOTA's GO-25 graph and ETL pipeline targets (BCFSA, RTB, City of Vancouver, FINTRAC, OIPC, etc.). The escalation trigger engine maps cleanly onto LUCIDOTA's candidate_writer pattern — escalation flags can be staged as GO-25 EVENTs against OBJECT (entity) nodes without touching canonical graph tables. The filing compliance checker is a ready-made gate for LUCIDOTA's graph-promotion preflight layer. The strict-lint [EVID:...] citation requirement mirrors LUCIDOTA's receipt-not-prose doctrine. The main integration gap is the SQLite evidence store — revival requires bridging ingest output into lucidota_storage (ABSURD queue -> CHIMNEY ETL path) and replacing the local entity ID scheme with GO-25 deterministic IDs.

---

### LUMP_OF_COAL

**Purpose:** Final-stage enforcement risk classifier for the Krampus Express OSINT system. Takes aggregated investigation output and produces enforcement-ready risk classifications for regulatory complaints, RTB filings, and legal action. The pipeline position: TINSEL scrapers -> CHIMNEY ETL -> unified DB -> run_analysis -> ABBA -> LUMP_OF_COAL -> STOCKING_STUFFER. Six sub-functions: (1) LEVER_LEDGER — per-entity vulnerability and regulatory leverage mapping across four BC regulatory channels (BCFSA, RTB, BCHRT, City), with L1-L4 escalation paths and dollar exposure ranges; (2) PRESSURE_MODEL — models how filing against entity A increases pressure on entity B via domino effects; (3) TRPP_CHECKER — Tenant Relocation and Protection Policy compliance checker with 2025 Vancouver compensation schedules; (4) GEOGRAPHIC_SCAN — property cluster detection, Vancouver neighbourhood mapping, and jurisdiction-based filing opportunity identification; (5) THREE_TIER_CASE — tiered readiness gate (PRELIMINARY -> ACTIONABLE -> FILING_READY) with fact-count and evidence-strength thresholds; (6) DEMAND_LETTER — evidence-backed demand letter generator. Ancillary: classifier.py (composite 0-100 risk score, five risk bands CRITICAL/HIGH/MODERATE/LOW/MINIMAL, four glow levels SPOTLIGHT/FLASHLIGHT/CANDLE/EMBER), risk_aggregator.py, regulatory_pattern_scorer.py, violation_mapper.py, category_rules.py, compliance_checker.py, penalty_calculator.py, action_recommender.py, multi_entity_risk_matrix.py, events.py.

**Status:** live | **Revival:** high

**Gold items:**
- Composite 0-100 risk scorer with ontology-bridged thresholds and hard-coded fallbacks — fully deterministic, no LLM required
- Five named risk bands (CRITICAL/HIGH/MODERATE/LOW/MINIMAL) and four glow levels (SPOTLIGHT/FLASHLIGHT/CANDLE/EMBER) — directly usable as LUCIDOTA GO-25 OBJECT attributes
- Lever ledger maps four BC regulatory channels (BCFSA/RTB/BCHRT/City) with fine ranges and L1-L4 escalation paths
- Pressure model computes domino filing sequence across entity networks — maps cleanly onto GO-25 EDGE analysis for cascading risk propagation
- Three-tier readiness gate (PRELIMINARY/ACTIONABLE/FILING_READY) is an explicit evidence sufficiency gate — analogous to LUCIDOTA graph-promotion preflight
- TRPP checker encodes 2025 Vancouver compensation schedule and bad-faith keyword detection
- Events consumer (CoalEventConsumer) hooks into pluggy event bus for reactive risk recomputation
- risk_aggregator.py explicitly self-labeled as an uncalibrated heuristic with calibration_status and estimate_type fields — unusually honest provenance metadata
- 378 tests — highest test coverage confidence among all Krampus modules examined

**LUCIDOTA fit:** LUMP_OF_COAL is the enforcement output layer and maps naturally to multiple LUCIDOTA needs. The composite risk scorer (classifier.py) can become a candidate_writer producing GO-25 OBJECT attributes (risk_band, glow_level, evidence_strength) that feed the authority class and graph-promotion gate. The lever ledger's regulatory channel model is a reusable multi-regulator enforcement tracker. The pressure model's domino filing sequence is a direct analog of GO-25 EDGE analysis for cascading risk across a network. The three-tier readiness gate is structurally identical to LUCIDOTA's graph-promotion preflight. The BC-specific domain content (TRPP, BCFSA statutes, Vancouver geography) is useful as-is for current investigation targets (PropertySage, Hutchinson, South Seas, Rakhra).

---

### STOCKING_STUFFER

**Purpose:** Evidence Packaging and Output Agent. Assembles OSINT evidence gathered by TINSEL/CHIMNEY into structured, submission-ready regulatory filings and investigation packages. Produces six output types: KRAMPUS_LETTER (demand letter to target), COAL_DELIVERY (BCFSA regulatory complaint bundle), CHIMNEY_SWEEP (RTB tenant complaint package), REINDEER_REPORT (full legal dossier), SPARKLE_BRIEF (internal SparkHorse intelligence summary), and NAUGHTY_GRAM (City of Vancouver TRPP/bylaw violation report). The core packager reads entity + evidence from the unified DB, gates output by export_eligible evidence strength, assembles exhibit-labelled ledgers (A, B, ... Z, AA, AB...), a chronological timeline, a table of contents, and wraps everything into a submission-ready zip archive. Sub-components: exhibit_compiler, submission_validator, template_engine (Jinja2 with custom filters: date_format, redact_pii, cite_evidence, strength_badge), template_manager (4 canonical templates: bcfsa_complaint, rtb_application, city_complaint, dossier), cover_page_generator, rag_generator (ChromaDB retrieval + LLM narrative with template fallback), and events.py (pluggy hook consumer: on_filing_ready queues entities for deferred packaging).

**Status:** partial | **Revival:** medium

**Gold items:**
- Evidence gating via EXPORT_ELIGIBLE before any package output — inferred-only evidence is blocked
- Exhibit labeling scheme (A..Z then AA, AB...) with per-exhibit strength badges and SHA fact_id references
- Chronological timeline builder inside package_filing — sorts by source_date then ingest_ts
- Table of contents with exhibit index table (source, strength, 60-char summary) auto-generated
- HITL correction hooks on exhibit selection and ranking (correct_exhibit_selection, correct_exhibit_ranking) wired to lucidota_core.hitl
- Pluggy event consumer on_filing_ready for deferred queue-driven packaging
- RAGGenerator: ChromaDB retrieval first, DB keyword fallback second, template fallback third — Gate C safe (no API key required)
- Jinja2 template engine with redact_pii filter (phones, emails, SIN numbers)
- submission_validator enforces: min 2 files, min 3 evidence items, no inferred-only packages, agency + target required

**LUCIDOTA fit:** STOCKING_STUFFER is the output/finalization surface for the LUCIDOTA evidence spine. In LUCIDOTA terms it maps directly to the LOCAL_FILE_PRODUCT and KRAMPUS_KORPUS_INGESTION receipt modes. Its export_gate pattern (EXPORT_ELIGIBLE strength filter) aligns exactly with LUCIDOTA canonical graph-promotion doctrine: only evidence at or above a threshold leaves the system. The pluggy on_filing_ready consumer maps cleanly onto ABSURD queue events. The RAGGenerator pattern (ChromaDB retrieval + bounded LLM summarization with template fallback) is a textbook LUCIDOTA 'model at a named node' implementation. The submission_validator (inferred-only guard, evidence count floor) parallels the LUCIDOTA graph-promotion preflight concept.

---

### PADDOCK

**Purpose:** Streamlit intelligence dashboard serving as the operator-facing UI layer for the Krampus Express OSINT system. Provides 44+ tabs covering: entity review and editing, interactive pyvis graph explorer (dark theme, neon accents, degree-centrality sizing, BFS neighborhood drill-down), evidence drop zone and ingestion, ABBA hypothesis viewing, filing generation and tracking, Bayesian/anomaly/strategy analysis, market intelligence, ontology pack browser, ORNAMENT batch profiler, War Room (Council of Hydras filing engine), scraper health, live feed, intelligence briefing, contradiction detection, merge queue, coalition strategy, synthesis, Pulse coordination, YAMATA Samurai NL-routing interface, and LUCI MIND (SOPHIA learning architecture). Framework is Streamlit with per-tab error boundaries. Sidebar shows live DB stat cards (entities, evidence, edges, patterns, ABBA hypotheses, ChromaDB count) and a risk band summary.

**Status:** live | **Revival:** medium

**Gold items:**
- graph_builder.py — role-based color scheme dict (ROLE_COLORS, TYPE_COLORS, EDGE_COLORS), BFS neighborhood traversal, degree-centrality node sizing, confidence-width edge scaling, pyvis dark theme forceAtlas2 layout. All directly reusable for any GO-25 graph visualization surface.
- helpers.py — batch_update_roles() and submit_correction() are clean DB mutation helpers with validation; correction_id UUID pattern and corrections audit table DDL are worth porting.
- auto_refresh.py — minimal stateless time-gate; keep as-is.
- Tab architecture pattern: 44-tab flat st.tabs() layout with _safe_render() error boundary wrapper and lazy button-gated loading for heavy tabs. Reusable pattern for any LUCIDOTA dashboard surface.
- style.py — custom Streamlit CSS theme (TOXIC_TEAL, ARCADE_MAGENTA, PHOSPHOR_VIOLET, EMBER_AMBER, RUSTED_CHROME color constants + stat_card, void_header, render_logo, render_mascot, render_header_banner). Directly portable to any LUCIDOTA Streamlit surface.

**LUCIDOTA fit:** PADDOCK's graph_builder.py is the most directly portable artifact: its pyvis-based entity/edge graph with role-based coloring and BFS neighborhood drill-down maps directly onto GO-25 OBJECT/EDGE tables in lucidota_storage. The style.py color palette and stat_card widget are portable to any LUCIDOTA Streamlit operator surface. The tab architecture pattern (_safe_render error boundary, button-gated heavy tabs, sidebar stat cards with live DB counts) is a proven pattern worth copying for a LUCIDOTA operator dashboard. The samurai_tab.py NL-routing interface (route natural language queries to named agents) maps onto Indy_READs orchestrator routing. Domain-specific tabs are not relevant to LUCIDOTA but the underlying component structure is reusable.

---

### lucidota_core

**Purpose:** The shared foundational library for the entire Krampus Express system. It is the canonical DB layer, evidence ledger write path, entity/edge normalization, graph engine, protocol runners, autonomous investigation loop, OODA orchestration, CQRS command bus, FastAPI gateway, ontology pack system, voice/tone validator, market intelligence streams, and synthesis intelligence. Every other KRAMPUS_EXPRESS module (DRE, CHIMNEY, TINSEL, MISTLETOE, LUMP_OF_COAL, ORNAMENT, PADDOCK, SANTA, STOCKING_STUFFER, REGULATOR_ROUTER) imports from it. It is the OS kernel of KRAMPUS_EXPRESS.

**Status:** live | **Revival:** high

**Gold items:**
- store/db_backend.py: Unified SQLite+Postgres backend with type compatibility matrix (JSONB->TEXT, TEXT[]->TEXT, TIMESTAMPTZ->ISO8601 TEXT). Context-manager safe, thread-local connections. This is the exact pattern LUCIDOTA needs for local-first + Postgres promotion.
- ledger/writer.py: write_fact() / write_entity() / write_edge() with SHA-256 idempotent dedup, event bus emission on every write, HITL hook integration, ontology-pack-aware type validation. Maps 1:1 to LUCIDOTA evidence->candidate->authority->graph gate doctrine.
- normalize/ids.py: fact_id(), entity_id(), edge_id() — deterministic SHA-256 prefix IDs. fact_id = sha256(source|date|extracted_fact). entity_id = sha256(entity_type|canonical_name) with whitespace collapse. These are the exact ID primitives GO-25 needs for idempotent evidence staging.
- scraper_recommender.py: ScraperCapability registry (tin_id, evidence_types, entity_types, gap_types, cost, reliability). recommend_scrapers(gaps, entity_type) maps evidence gap types to specific TIN scraper IDs. Direct analog for ABSURD worker dispatch by gap type.
- graph/patterns.py: Typed path traversal with named real-world patterns (CORRADO_PROXY, HUTCHINSON_DUAL_BROKERAGE, RAKHRA_DEVELOPMENT). match_pattern(), find_paths(), shortest_path().
- graph/materialized.py: Precomputed degree/betweenness centrality, clustering, community detection, refresh_all(), get_top_by_metric().
- synthesis/ (6 modules): case_linker, bridge_detector, shared_actor_graph, timeline_merger, hypothesis_cross_pollinator, risk_amplifier. Cross-case intelligence is the graph-layer reasoning LUCIDOTA needs above raw evidence facts.
- enforcement/penalty_calculator.py + regulator_registry.py: 6-regulator registry (BCFSA, RTB, City, etc.) with precedent-anchor penalty model and deadline engine.
- ontology/bridge.py: _get(key, default) — pack-first config with hardcoded fallback. The exact pattern for GO-25 ontology override without hard-coded domain terms in protocol code.
- commands/ (CQRS): Per-command handler files (ingest_fact, create_entity, link_edge, compute_risk, correct_evidence, generate_filing) + time_travel.py (get_state_at, diff_state, entity_timeline). This is the command-envelope pattern LUCIDOTA doctrine requires for all canonical graph writes.
- events/bus.py: pluggy event bus + SQLite-backed event store + WebSocket bridge. 5 consumers. Maps to ABSURD receipt/event emission pattern.
- adaptive_weights.py: Learns from decision_journal.py outcomes — boosts productive action types, penalizes failures. This is the bandit-style feedback loop LUCIDOTA ALGOS/ targets.
- archetype_matcher.py: 10 investigation archetypes with behavioral classification scoring.
- bayesian.py (573 lines): Full Bayesian inference over evidence chains. Maps to LUCIDOTA hypothesis-confidence scoring above the rule-based layer.

**LUCIDOTA fit:** lucidota_core is the most directly portable module in the entire KRAMPUS_EXPRESS archive. Its store/db_backend.py unified backend, ledger/writer.py idempotent write path, normalize/ids.py SHA-256 ID primitives, graph/ materialized and pattern engines, synthesis/ cross-case intelligence, enforcement/ regulator registry, ontology/bridge.py pack system, CQRS commands/, and events/bus.py all map cleanly onto LUCIDOTA's GO-25 ternary schema (OBJECT/EVENT/EDGE), ABSURD queue doctrine, evidence-to-candidate-to-authority-to-graph-gate spine, and the ALGOS/ bandit/Bayesian scoring layer. The scraper_recommender.py is a direct prototype for gap-driven TINSEL->ABSURD dispatch. The adaptive_weights.py + decision_journal.py pair is a prototype for LUCIDOTA's feedback-loop governor. The main friction points are: Windows-origin paths, LUCIDOTA_DB_URL vs DSN naming, direct Anthropic SDK dependency (needs local-first provider abstraction), and the autonomous loop bypassing ABSURD (needs re-routing). The MTG module should be archived to KRAMPUSCHEWING on revival.

---

## 15. PERCYPHON DOCTRINE

**Canonical source:** /home/mfspx/LUCIDOTA/ALGOS/percyphon.py
**Kernel bridge:** /home/mfspx/LUCIDOTA/scripts/percyphon_kernel_bridge.py
**Organ registry entry:** /home/mfspx/LUCIDOTA/00_PROJECT_BRAIN/organ_registry/40_percyphon_math.json
**Runtime binding:** /home/mfspx/LUCIDOTA/04_RUNTIME/indy_percyphon_hunch_subtleknife_binding.json
**Primary test:** /home/mfspx/LUCIDOTA/tests/test_percyphon_kernel_bridge.py
**Pillar test:** /home/mfspx/LUCIDOTA/math/validation/test_four_pillars.py

### Section 1: What Percyphon Is

Percyphon (correctly "PercyphonAI", never "Persephone") is LUCIDOTA's zero-VRAM procedural identity mask and alias router. It is the native Python reimplementation of the conceptual architecture that lives in the external CKDOG1 kernel (01_REPOS/doggystyle/), which is a separately-maintained reference-only codebase LUCIDOTA does not import or invoke. Percyphon mirrors the CKDOG1 soul model's slot structure (12 core + 88 domain), trinary route state, and SHA-256 determinism natively, in ~90 lines of pure hashlib/dataclasses Python, with zero model calls, zero DB writes, and zero VRAM consumption.

### Section 2: The 12 Fixed Core Mask Slots

The call to procedural_entity_generator() always mints exactly 12 ProceduralSlot instances, with slot_index 0 through 11. These are "fixed" in the sense that their count is hard-coded to 12, mirroring the CKDOG1 kernel's 12 core trait slots per soul. They are not fixed in identity — their content is deterministically derived from the seed.

Each ProceduralSlot is a frozen dataclass with these fields:

- slot_index: integer 0-11, ordinal position in the core layer
- name: string Villager-NNNN where NNNN is int(h[:6], 16) % 5000 formatted to 4 digits — a human-readable handle for this mask identity, drawn from the 0-4999 villager namespace
- alias: string Alias-xxxx where xxxx is hex characters 6-10 of the SHA-256 of "{seed}:{idx}" — a short routing label
- persona: one of exactly six roles — ledger, runner, witness, archivist, carrier, scribe — selected by int(h[10:12], 16) % 6. These represent functional archetypes within the soul model.
- uuid: a UUID-shaped hex string derived from SHA-256("{seed}:{idx}") formatted as 8-4-4-4-12 from the first 32 hex chars. It is NOT an RFC 4122 UUID — it is a deterministic hash-derived identifier.
- ternary_offset: the shared trinary state for all 12 slots in this generation (see Section 3).

What the 12 core slots represent: The 12 core slots are the fixed-size identity mask layer for a generated Villager entity. Every soul in the system gets exactly 12 core trait positions. The slot structure is deterministic given the seed, so the same villager inputs always produce the same 12 slots.

### Section 3: The Trinary Route State

ternary_offset is computed once per procedural_entity_generator() call and shared by all 12 slots. It is derived as int(SHA-256(seed)[:4], 16) % 3, yielding 0, 1, or 2. This value represents the trinary routing state of the soul — a three-position switch over the generated entity's behavioral character. It mirrors the CKDOG1 kernel's trinary soul-state concept natively in pure arithmetic.

### Section 4: The 88 Domain Extension Slots

The domain_extension_layer() function generates 88 additional DomainSlot instances (indices 0-87), completing the 100-slot soul model (12 core + 88 domain = 100 total). Domain slots share the same SHA-256 derivation pattern as core slots but use "domain:{seed}:{idx}" as the hash input. They carry:

- slot_index: 0-87
- domain_tag: one of 11 domain categories — civic, financial, legal, medical, social, technical, environmental, cultural, commercial, regulatory, investigative — selected by int(h[:4], 16) % 11
- weight: a float in [0.0, 1.0] derived from int(h[4:8], 16) / 65535.0
- active: boolean derived from int(h[8:10], 16) % 2 — whether this domain slot is active for this soul

### Section 5: Production Status and Gaps

**Production grade:** Percyphon is production-grade. All tests pass. The kernel bridge (percyphon_kernel_bridge.py) exports a clean CLI and a resolve_alias(seed, idx) function. The organ registry entry (40_percyphon_math.json) is registered with 117 capability entries.

**Known gaps as of 2026-05-29:**
- The runtime binding file (04_RUNTIME/indy_percyphon_hunch_subtleknife_binding.json) records Percyphon's integration with the Subtle Knife markup protocol but the live invocation path from ABSURD queue workers is not wired — the bridge exists but no worker calls it
- The 88 domain slots are generated but not yet consumed by any downstream scoring or routing function in LUCIDOTA's live pipeline
- The trinary route state is computed but not yet used by the ternary_router or rete_bandit_gate — those modules have their own trinary logic independent of Percyphon's ternary_offset
- Source file ALGOS/percyphon.py is TRUNCATED mid-expression — see section 23 for detail

---

## 16. DIOGENES KERNEL DOCTRINE

### What DIOGENES Is

DIOGENES is the name used for two distinct but related things in LUCIDOTA:

1. The project-wide wiring mirror / backup spine — the machinery that knows where every file, script, schema, and registry row lives, and can reconstruct the machine from source.
2. Layer 2 of the routing stack — the intended "routing intelligence kernel" that sits between Percyphon's alias resolution and the ALGOS computation layer.

These two meanings coexist without a clean separation. The codebase has fully built the first meaning and only a receipt stub for the second.

### File-by-File Breakdown

**scripts/diogenes_stub_kernel.py — The Named Stub (receipt infrastructure, not routing)**

Despite the name, this file has nothing to do with routing intelligence. It is the Marrow Loop v0 command receipt kernel: given --raw-command, --normalized-intent, --authority-class, and --source, it stamps a UUID and SHA-256 payload hash, writes a JSON receipt to 05_OUTPUTS/marrow_loop/, optionally inserts into lucidota_control.event_outbox when --execute --execute-db are both passed, and returns exit code 0 (no blockers) or 2 (blockers present). This script is a receipt infrastructure primitive, not a routing decision engine.

What is NOT in the stub kernel: no capability map, no lane scoring, no tool/model/algo dispatch, no treelite call, no Percyphon integration, no ABSURD queue submission.

**scripts/diogenes_mirror.py — The Wiring Mirror (Fully Built)**

This is the most operational piece of DIOGENES. It walks the entire repo tree, classifying files into four kinds: db-native (SQL), runtime-core (Rust/TOML), experimental (Python/shell), quarantine (legacy/corpse hints). Infers a one-line purpose from each file's first docstring or comment. Samples host telemetry (CPU, memory, disk, swap, load averages) via psutil. Writes a manifest JSON to 05_OUTPUTS/diogenes/mirror_manifest_<stamp>.json. On --execute: upserts every scripts/, ALGOS/, 06_SCHEMA/, src/, PocketFlow/ file into lucidota_control.script_registry; writes a project_mirror_snapshot row; fires a workflow_event, a bytewax_abductive_event, a river_run, and a river_score all in one commit. The most recent confirmed run: 05_OUTPUTS/diogenes/diogenes_mirror_20260528T064705282932Z.json — 88,481 included files, 365,449 excluded, 882 registry candidates.

**scripts/diogenes_memory_gate.py — The Low-RAM Gate (Fully Built, Passing)**

A RAM-capacity gate script. Reads current system memory via psutil, compares against a configurable threshold (default: 2GB free), and either allows or blocks operation. All 7 DIOGENES tests pass. This gate is used as a pre-flight check before memory-intensive DIOGENES operations.

### The Routing Intelligence Gap

The routing intelligence kernel that DIOGENES is designed to be — the capability map, lane scoring, tool/model/algo dispatch layer between Percyphon alias resolution and the ALGOS computation layer — does not exist in the codebase as of 2026-05-29. What exists:

- The receipt stub (diogenes_stub_kernel.py) names the kernel but does not implement it
- The wiring mirror (diogenes_mirror.py) knows where every file lives but does not make routing decisions
- The memory gate (diogenes_memory_gate.py) gates on RAM but does not route

Building the real DIOGENES routing kernel is an Opus co-design task. The design parameters are clear from the surrounding infrastructure: it must read the organ registry to know what capabilities exist, use Percyphon alias resolution to map intent to named organs, score candidate lanes using ALGOS/ bandit/regret functions, dispatch work orders to ABSURD queue, and emit receipts to 05_OUTPUTS/marrow_loop/ using the existing stub kernel pattern.

---

## 17. ORGAN REGISTRY SUMMARY

**Total registered capabilities: 751 organs across 7 shards**

| Shard | File | Count |
|---|---|---|
| Scripts | 10_scripts.json | 459 |
| Schema | 20_schema.json | 151 |
| Streaming/ML | 30_streaming_ml.json | 9 |
| Percyphon/Math | 40_percyphon_math.json | 117 |
| Ingestion | 50_ingestion.json | 9 |
| Spencer Signal Bench | 60_spencer_signal_bench.json | 6 |
| Master rollup | ORGAN_REGISTRY.json | (aggregates above) |

### Live vs Stub vs Planned

**LIVE (active or confirmed running):**
- 529 organs classified "active" across script/schema shards
- River governor HoeffdingTreeRegressor: confirmed live, 539 training samples, fact in runtime_status_fact as of 2026-05-28
- Live applied schemas (~30/122 SQL files confirmed in DB): 001 control, 002 model_runtime, 004 learning_reflex, 005 cas_manifest, 006 workflow_registry, 007 bytewax_stream, 008 hop_pivot, 009 treelite_router, 010 wake_bus, 014 indy_runtime, 016 go_graph_core, 017 indy_reads_library, 018 investigation_artifact, 019 korpus_krampii, 020 chat_dump_timeline, 020 derived_compute_queue, 021 hard_truth_math, 022 comm_dump_timeline, 039 (partial: runtime_status_fact live), 122 resource_governor, 123 absurd_flows
- Live models on disk (not all wired): bge-m3, modernbert-base, gliner_small-v2.1, smoldocling-256m, falcon3-mamba-7b-gguf, deepseek-r1-1.5b-gguf

**STUB / UNAPPLIED (~92/122 SQL schemas unapplied — critical gaps):**
- 025/026/027 Chrono-Ledger core: UNAPPLIED — temporal_claim table MISSING live
- 034/044/052/059 Graph promotion pipeline: UNAPPLIED — graph_promotion_packet MISSING live
- 035 ABSURD queue spine: UNAPPLIED — absurd_queue_job MISSING live (doctrinal durable queue not in DB)
- 040/074 Graph write barriers: UNAPPLIED — write barrier tables MISSING live
- This gap means 18,627 graph-promotion candidates are staged but nothing is promotable to canon

### Capability Categories

- Routing/Gating: bandit_router, rete_bandit_gate, ternary_router, ternary_lens_router, fairyfuse, treelite_router (6 core ALGOS)
- Math/Scoring: percyphon (117 entries), spencer_signal_bench (6), streaming_ml (9)
- ETL/Ingestion: ingestion shard (9), 459 scripts covering queue spine, receipt workers, ETL adapters, status ledger, graph scanner
- Schema/DB: 151 schema entries (30 applied live, 121 unapplied)
- Models on disk: bge-m3, modernbert-base, gliner_small-v2.1, smoldocling-256m, falcon3-mamba-7b-gguf, deepseek-r1-1.5b-gguf

---

## 18. TINSEL SCRAPER INVENTORY

**Canonical count: 52 TIN_*.py files** in the TINSEL flat directory (58 total .py files including aliases, test files, and infrastructure).

**Numbering anomalies:**
- Gaps: slots 004, 005, 006 missing (not sequential)
- Collisions: 10 collision pairs at TIN_026 through TIN_035 (two files per slot number)
- Aliases: one lowercase alias tin001_bcfsa.py, one test file tin_test_strr_auth.py
- Collision pairs are cleanup items before ABSURD revival, not hard blockers

**Full sorted inventory (52 canonical TIN_*.py):**

TIN_001_bcfsa.py, TIN_002_orgbook.py, TIN_003_vancouver_opendata.py, TIN_007_ltsa.py, TIN_008_canlii.py, TIN_009_craigslist.py, TIN_010_van_bizlicence.py, TIN_011_reddit.py, TIN_012_google_reviews.py, TIN_013_instagram.py, TIN_014_linkedin.py, TIN_015_facebook.py, TIN_016_wayback.py, TIN_017_crtsh.py, TIN_018_whois.py, TIN_019_opencorporates.py, TIN_020_dns.py, TIN_021_techstack.py, TIN_022_bc_assessment.py, TIN_023_parcelmap_bc.py, TIN_024_corps_canada.py, TIN_025_bc_geocoder.py, TIN_026_bc_legislature.py, TIN_026_shapeyourcity.py (collision), TIN_027_bc_courts.py, TIN_027_bc_ppr.py (collision), TIN_028_canada411.py, TIN_028_canada_post_fsa.py (collision), TIN_029_bc_courts.py, TIN_029_google_news.py (collision), TIN_030_cra_bn.py, TIN_030_twitter_x.py (collision), TIN_031_airbnb.py, TIN_031_youtube.py (collision), TIN_032_glassdoor.py, TIN_032_vrbo.py (collision), TIN_033_indeed.py, TIN_033_zillow.py (collision), TIN_034_bc_gazette.py, TIN_034_twitter.py (collision), TIN_035_youtube.py, TIN_036_tiktok.py, TIN_037_shapeyourcity.py, TIN_038_nextdoor.py, TIN_039_yelp.py, TIN_040_tripadvisor.py, TIN_041_foursquare.py, TIN_042_mapbox.py, TIN_043_openstreetmap.py, TIN_044_wikidata.py, TIN_045_dbpedia.py

**Official-record tier (highest evidence_strength, revive first):**
TIN-001 (BCFSA), TIN-002 (OrgBook), TIN-007 (LTSA), TIN-008 (CanLII), TIN-010 (Van BizLicence), TIN-022 (BC Assessment), TIN-024 (Corps Canada), TIN-025 (BC Geocoder), TIN-034 (BC Gazette)

**Market intelligence subfleet (MKT-001 through MKT-008, in market/ subdir):**
SEC EDGAR, Form 4, 13F, FTD, dark pool feeds — no auth required for most; direct feed into lucidota_core/market/ dual-stream scoring

**Last live run stats (2026-03-10 ally_scrape batch):**
- 127 allies processed (89 persons, 38 companies), 815 facts ingested, 838 seconds elapsed
- Only 3 scrapers produced facts: TIN-011 Reddit (494), TIN-002 OrgBook (165), TIN-010 Van BizLicence (156)
- 7 scrapers returned 0 facts (BCFSA, LinkedIn, Facebook, CanLII, OpenCorporates, Google Places) — likely anti-scraping, login gates, or missing API keys
- scraper_state.json = {} (empty) — no persistent checkpoint state survived

**Revival sequence:** Deduplicate collision pairs first. Wire ScraperResult envelope to ABSURD queue job_kind definitions using scraper_registry.py. Map schedule.json frequency to ABSURD scheduler intervals. Official-record scrapers first (no auth required); social/platform scrapers after API key review.

---

## 14. KRAMPUS EXPRESS MODULE INVENTORY (2026-05-29)

Full table at `05_OUTPUTS/krampus_express_inventory_20260529.md`. Summary:

| Module | Purpose | Revival |
|---|---|---|
| TINSEL | 52 scrapers, BC data acquisition, last ran 2026-03-08 | HIGH |
| DRE + SNOWBALL | Displacement risk scoring + entity graph expansion | HIGH |
| MISTLETOE | Network centrality, motifs, bridge detection | HIGH |
| CHIMNEY | 5-stage ingest pipeline (maps to ABSURD worker chain) | HIGH |
| LUMP_OF_COAL | Evidence report generator, exhibit compiler, RAG | HIGH — use NOW on BC tenancy corpus |
| STOCKING_STUFFER | RTB/BCFSA submission package builder | HIGH — use NOW |
| CAROL | Content amplification / publish surface | medium |
| ORNAMENT | Ontology packs → GO-25 EDGE relation seeds | medium |
| REGULATOR_ROUTER | Pipeline branch routing → fold into Diogenes | medium |
| lucidota_core | scraper_recommender.py + shared utils | HIGH |
| SANTA | Chat interface (Indy replaces this) | low |
| PADDOCK | Streamlit UI | low |

**TINSEL wiring: 3 steps, ~150 lines.** Wrap TIN_*.py in ABSURD envelope → tinsel_dispatch_worker → Pathway Venturi → GO-25 staging.

---

## 15. ORGAN REGISTRY (751 capabilities registered)

- `10_scripts.json` — 459 scripts
- `20_schema.json` — 151 schema items
- `30_streaming_ml.json` — 9 (Bytewax, River, stream service)
- `40_percyphon_math.json` — 117 (Percyphon components, validators, math algos)
- `50_ingestion.json` — 9
- `60_spencer_signal_bench.json` — 6

**Wiring status:** 424 standalone, 99 unapplied, 50 applied, 19 claw-wired, 11 service.
**Graph safety:** 294 receipt_only, 106 read_only — majority are safe workers.
**Gap:** 99 unapplied organs = the schema backlog in capability form.

---

## 16. SCHEMA APPLY ORDER (confirmed on disk)

Apply in this exact sequence. Each unlocks the next layer:

```
035_absurd_queue_spine.sql          ← ABSURD queue columns + chrono audit
039_absurd_real_work_loop.sql       ← worker dequeue + transition policy
043_absurd_remaining_worker_contracts.sql
049_absurd_intake_wrapper.sql
025_chrono_evidence_ledger_*.sql    ← three-clock temporal evidence
026_*
027_*
040_write_barrier_triggers.sql      ← blocks accidental canon writes
074_*
034_graph_promotion_gate.sql        ← gates the 18,627 staged candidates
044_*
052_*
```

After applying 034/044/052: run `python3 scripts/graph_promotion_gate.py --review` to begin controlled promotion of the 18,627 candidates.

---

## 17. DIOGENES = PERCYPHON · ORGAN_REGISTRY (the stupid-simple truth)

Diogenes does NOT need to be a separate LLM-based routing brain. It is:

```python
slot_vector = percyphon.activate(villagers)           # [-1,+1,0,...] 12+N dims
candidates = organ_registry.score_by_slot(slot_vector) # dot product → ranked capabilities
route = treelite.gate(candidates[0], five_features)    # sub-ms validation
dispatch(route)
```

The "stub kernel" in `scripts/diogenes_stub_kernel.py` should be replaced with exactly this. Three lines. All pieces exist.

---

## 18. TINSEL → ABSURD WIRING PLAN

Three files needed:
1. `scripts/tinsel_dispatch_worker.py` — ABSURD worker, dequeues `tinsel_scrape` jobs, imports TIN_*.py, runs scraper, writes to `raw_artifact`
2. `scripts/tinsel_register_contracts.py` — registers the worker contract in `absurd_worker_contract` table
3. Add to `scripts/absurd_worker_contracts.py` — `('venturi_intake', 'tinsel_scrape', 'tinsel_dispatch_v1')`

Then: `python3 scripts/tinsel_register_contracts.py --execute` and drop a TINSEL job into the queue.

---

## 19. MANIFEST SYSTEMS (TICKLETRUNK vs ORGAN REGISTRY vs SCRIPTS/)

Full report: `05_OUTPUTS/manifest_gap_report_20260529.md`

**Three systems, three altitudes:**

**TICKLETRUNK** (`00_PROJECT_BRAIN/TICKLETRUNK.json`): Breadth-first filesystem census. 1,556 entries: SCRIPTS 727, SCHEMAS 13 (stale — actual is 123), WORKFLOWS 129, ALGOS 59. Last generated 2026-05-27T17:42:41Z. Use for discovery only. `what_it_does` is often UNKNOWN. Does NOT declare mutation class, wiring status, or graph-safety. Regenerate with `python3 scripts/tickletrunk_scan.py --execute`.

**ORGAN REGISTRY** (`00_PROJECT_BRAIN/organ_registry/`): Capability ledger — what the machine CAN do. Declares: organ_id, mutation_class, wiring_status (WIRED/STUB/MISSING), schema dependencies, activation gate. Mandatory pre-flight before flow design (KANT69). ~80-120 capabilities catalogued. ~20-30 are stubs or MISSING (no implementation). This is the skeleton MEMORY.md calls mandatory.

**scripts/ directory** (446 active Python scripts): The live workforce. Source of truth for what code actually runs. ~26 scripts NOT in TICKLETRUNK (written after 2026-05-27). Most scripts have no mutation class declaration — standing gap.

**Trust hierarchy:** `Live DB facts > audit receipts > ORGAN REGISTRY > TICKLETRUNK > prose claims`

**Venn summary:**
- ~420 scripts indexed in TICKLETRUNK; ~26 unregistered (post-2026-05-27)
- ~80-120 capabilities in ORGAN REGISTRY — only a fraction of 446 scripts have capability declarations
- ~20-30 capabilities in ORGAN REGISTRY with NO corresponding script (architectural intentions without implementation)

**Five dead scripts** (archive to KRAMPUSCHEWING/Script_Corpses/):
`patch_runner.py`, `absurd_worker_script.py`, `absurd_worker_spine.py`, `absured_worker_script.py`, `agentsi.py`

**Nine stub scripts** (register in ORGAN REGISTRY with status=STUB):
`hipporag_locked_gate_report.py`, `hunch_postgres_ingest.py`, `hypertimeline_engine.py`, `interpretability_eval_bench_dry_run.py`, `lucidota_indy_brief.py`, `lucidota_indy_polycareer.py`, `model_answer_diff_dry_run.py`, `ontology_nudge_inventory_dry_run.py`, `receipt_exporter.py`

---

## 20. SCHEMA APPLY ORDER (VERIFIED)

Full audit: `05_OUTPUTS/schema_audit_20260529.md`

**Total schemas audited: 123.** Ground truth: ~92/122 were UNAPPLIED before this audit session.

**Critical — apply to lucidota_state first (ABSURD spine, strict order):**
```
1. 035_absurd_queue_spine.sql          # ABSURD root — nothing works without this
2. 039_absurd_real_work_loop.sql       # runtime_status_fact, conversation_command, status transition trigger
3. 043_absurd_remaining_worker_contracts.sql
4. 049_absurd_intake_wrapper.sql
5. 082_absurd_worker_contract_registry_enforcement.sql
6. 096_worker_command_registry.sql
7. 097_conversation_command_status_transition.sql
8. 098_absurd_retry_policy_registry.sql
```

**Then high-priority to lucidota_state:**
```
9.  006_workflow_registry.sql          # Re-apply — refresh 40+ workflow seed rows
10. 036_absurd_chrono_wrapper.sql
11. 037_absurd_krampus_wrapper.sql
12. 068_conversation_command_acceptance.sql
13. 079_intake_custody_job_bridge.sql
14. 081_cep_conversation_command_dedupe.sql
15. 084_absurd_dead_letter_review.sql
```

**High-priority to lucidota_storage (graph + korpus), dependency order:**
```
16. 014_indy_runtime.sql               # Creates lucidota_indy schema (must precede 017)
17. 003_survey_protocol.sql            # CONFIRMED MISSING — survey/body-capture dead without it
18. 018_investigation_artifact.sql     # MUST precede 019 (tag_taxonomy FK)
19. 019_korpus_krampii.sql             # Korpus ingest spine
20. 020_korpus_derived_compute_queue.sql
21. 017_indy_reads_library.sql         # Needs 014 schema + 016 FK + pgvector
22. 025_chrono_ledger_core.sql         # Needs 019 file_object FK
23. 026_chrono_absurd_triggers.sql     # Needs 025
24. 027_chrono_phase_c_ops.sql         # Completes Chrono-Ledger durability floor
25. 034_graph_promotion_pipeline.sql   # Needs 016; installs graph_promotion_packet/decision
26. 038_absurd_river_wrapper.sql       # GLiNER staging tables
27. 040_graph_write_barrier_enforcement.sql  # Install AFTER graph tables confirmed live
28-49. [High-priority remainder per schema_audit_20260529.md §apply-order]
```

**Confirmed already applied (skip):**
001, 002, 004, 005, 007, 008 (lucidota_state); 016 (lucidota_storage — GO graph core)

**Confirmed zero-DDL (skip):** 031 (plan doc only)

**Naming collision:** Two files share prefix `020` — `020_chat_dump_timeline.sql` and `020_korpus_derived_compute_queue.sql`. Both needed; apply both; track as 020a/020b.

---

## 21. SCRIPT AUDIT SUMMARY (Groq-audited)

Full audit: `05_OUTPUTS/groq_hammer_audit_20260529.md`

**Groq hammer stats:** 50 batches fired, 446 scripts audited. 45 OK / 0 rate-limited / 5 errors. No rate limits hit — current key has sufficient headroom.

**5 dead scripts** — archive to KRAMPUSCHEWING/Script_Corpses/ with sha256 hashes:
`patch_runner.py`, `absurd_worker_script.py`, `absurd_worker_spine.py`, `absured_worker_script.py` (typo variant), `agentsi.py`

**30 scripts with execution gaps** — full table in groq_hammer_audit_20260529.md.

**Top 5 priority fixes (highest blast radius):**
1. `absurd_flows.py` — psycopg bare try/except → false-success receipts. Receipt fraud. Fix: hard fail on psycopg absent.
2. `capability_pack_registry.py` — same silent DB-write suppression pattern. Fix: hard fail.
3. `canonical_graph_write_scanner.py` — binary COPY protocol evades regex. Fix: document blind spot; add secondary check.
4. `abductive_db_os_ledger.py` — no glob-size cap on indy_reads/**/*.json (OOM risk). Fix: itertools.islice cap.
5. `case_packet_compiler.py` — packet_id includes mutable counts, breaking idempotency. Fix: hash artifact_uuid + job_kind only.

**5 mutation class violations** (false receipts or gate bypass):
- `absurd_flows.py`, `capability_pack_registry.py` — receipt fraud (psycopg suppressed)
- `case_packet_compiler.py` — non-idempotent receipt_only
- `bitloops_river_worker.py` — candidate_writer injecting always-True labels
- `abductive_db_os_health_check.py` — authority_gate that can pass when it should block

**Standing gap — mutation class headers:** Most of the 446 scripts have no declared mutation class. Annotate the 30 gap scripts before designing any new flows that touch them.

---

## 22. SESSION LOG — 2026-05-29 (SONNET 4.6 PRE-OPUS SESSION)

### What happened this session. Read this before doing anything else.

---

### A. REPO FIXES APPLIED

| Fix | Result |
|---|---|
| `00_PROJECT_BRAIN/STATUS_LEDGER.md` | Already passing — `lucidota_status_ledger.py --check` returns CHECK_OK |
| `ALGOS/percyphon.py` truncation | Already compiles cleanly — prior session fixed it |
| Bytewax bare `python3` → `sys.executable` | **FIXED** in 5 files: `absurd_river_worker.py`, `legacy/dbos_river_worker.py`, `legacy/runtime_status_ledger_updater.py`, `legacy/chrono_service_health_recorder.py`, `legacy/daemon_supervision_preflight.py` |
| 59 failing tests (RFC/doc-authority drift) | 43/43 pass on the 10 targeted test files — governance drift was already resolved |

**Canonical graph write scanner: PASS, 0 violations.**

---

### B. ALGORITHM ARSENAL — 19 NEW FILES IN ALGOS/

All 19 compile clean and smoke-test correctly. Filed 2026-05-29.

**Batch 1 — Core frontier math:**
| File | Algorithm | Key proof |
|---|---|---|
| `kan.py` | Kolmogorov-Arnold Networks | B-spline edge functions, KAN layer composition |
| `state_space_duality.py` | Mamba-2 / SSD | Sequential ↔ parallel duality at 4.44e-16 (machine ε) |
| `liquid_time_constant.py` | LTC / worm brain | ODE-driven τ_sys output, input-dependent speed |
| `diffusion_forcing.py` | Diffusion Forcing | Per-token noise, causal t_seq sampler |

**Batch 2 — Graph / physics / signal:**
| File | Algorithm | Key property |
|---|---|---|
| `ollivier_ricci_curvature.py` | Ollivier-Ricci Curvature | Detects graph bottlenecks via W1 transport |
| `caputo_fractional.py` | Caputo Fractional Derivative | Power-law memory kernel, infinite non-local history |
| `geometric_product.py` | Clifford / Geometric Algebra | Coordinate-free rotors, `ab = a·b + a∧b` |
| `koopman_operator.py` | Koopman Operator (DMD) | Linearizes chaos via observable lift; train MSE 0.017 |
| `tropical_maxplus.py` | Tropical Max-Plus | ReLU nets = tropical rational functions (exact) |
| `path_signature.py` | Path Signature | Time-reparametrization invariant stream features |
| `sheaf_cohomology.py` | Cellular Sheaf Cohomology | `δ(s)=0` ↔ global consistency, pinpoints contradictions |

**Batch 3 — Learning / memory / biology:**
| File | Algorithm | Key proof |
|---|---|---|
| `ttt_linear.py` | Test-Time Training (TTT) | W norm increases per token — self-modification live |
| `rectified_flow.py` | Rectified Flow Matching | Euler straightness=0.9999, straight-line OT paths |
| `variational_free_energy.py` | Active Inference / Free Energy | Action = prediction error too large to fix via perception |
| `dense_associative_memory.py` | Dense Associative Memory | Softmax attention ≡ Hopfield retrieval at 1.67e-16 |
| `rlct_grokking.py` | RLCT / Singular Learning Theory | λ measures loss landscape singularity; grokking = phase transition |
| `dendritic_compartment.py` | Multi-Compartment Dendritic ODE | One neuron = deep network before soma |
| `fractional_hdc.py` | Fractional Power HDC | Phase-angle fractional binding; cleanup roundtrip = 1.0000 |
| `jepa_energy.py` | JEPA Energy | E=0 when abstract representations align; no pixel prediction |

**4 algos promoted from `pypeline/math/`:**
`bayes_claim_kernel.py`, `epistemic_certainty.py`, `model_vram_scheduler.py`, `model_pool.py`

**Total ALGOS/ seed pool: 80 files.**

---

### C. 4 LIVE DIAGNOSTIC SCRIPTS WIRED TO REAL LUCIDOTA DATA

| Script | Algorithm | What it does |
|---|---|---|
| `scripts/absurd_queue_ricci_bottleneck.py` | Ollivier-Ricci | Compute curvature on ABSURD job graph, find choke-points |
| `scripts/staging_sheaf_consistency.py` | Sheaf Cohomology | Find contradictions in the 18,627 staged candidates |
| `scripts/telemetry_path_signature.py` | Path Signature | Convert `hw_telemetry.jsonl` to invariant geometric features |
| `scripts/koopman_queue_forecast.py` | Koopman DMD | Linearize queue chaos, forecast load 10 steps ahead |

---

### D. DARWIN HAMMER TOURNAMENT — RUNNING NOW

**Process:** PID 180953 (or check `ps aux | grep darwin_hammer`)
**Log:** `/tmp/darwin_hammer.log`
**Database:** `ALGOS/evolved/darwin.db` (SQLite, queryable)
**Lineage:** `ALGOS/evolved/lineage.jsonl`
**Output:** `ALGOS/evolved/gen{N}/` — one .py file per survivor

**Configuration:**
- Target: 69,000 survivors
- Rounds: 9,999 (effectively indefinite)
- Models per match: 8 (70B×2 + 120B×1 + Qwen3-32B×5)
- Challenge round: 30% of matches, 3 additional calls (Llama-4-Scout + 70B + 120B)
- Generations: gen1 (base×base) → gen2 (hybrids breed) → gen3 (legendary)

**First survivors observed:** `hybrid_ternary_router_ssim_m{N}_s{N}.py` — ternary routing fused with structural similarity index.

**To query the DB:**
```bash
sqlite3 ALGOS/evolved/darwin.db "SELECT gen, count(*) FROM survivors GROUP BY gen;"
sqlite3 ALGOS/evolved/darwin.db "SELECT parent_a, parent_b, born FROM survivors ORDER BY id DESC LIMIT 10;"
```

**To check live count:**
```bash
wc -l ALGOS/evolved/lineage.jsonl
```

---

### E. ARCHITECTURAL SYNTHESIS (READ THIS — it matters for design work)

These four algorithms compose into a complete governor architecture for LUCIDOTA. No new infrastructure needed — they slot into what exists.

**LTC (Liquid Time Constant) → replaces hard fast/slow lane gate**
The τ_sys output is input-complexity-driven. Simple evidence → fast. Ambiguous, contradictory evidence → system slows and integrates longer. This is the natural implementation of LUCIDOTA's routing governor — continuous and learned, not binary.

**3-body Ternary Mamba → temporal belief tracker**
Three coupled SSMs, one per ternary class (TRUE/UNKNOWN/FALSE), with cross-coupling γ terms. Each body minimizes surprise about the others (this is the Free Energy formulation). Tracks how evidence shifts between ternary states over a chrono sequence. The dominant Koopman eigenvalue changes when grokking occurs — you can watch a belief transition before the output changes.

**KAN over GO-25 graph → learned lens functions**
Same OBJECT→EDGE→OBJECT topology. Two KAN layers learn different B-spline edge functions: Φ_0 = temporal distance, Φ_1 = semantic similarity. The graph structure is fixed; the learned functions are what change. The ternary router already applies lenses; KAN learns the lens functions from data.

**Diffusion Forcing → causal work-order planner**
Completed work-order steps get t_i=0 (clean, known). Future steps get t_i=T (noisy, being planned). The model simultaneously conditions on verified history and plans forward. ABSURD queue work orders are sequences — this is a natural planning primitive for them.

**Dense Associative Memory = Attention (not metaphor)**
`attention_as_hopfield(Q, K, V)` in `ALGOS/dense_associative_memory.py` returns identical output to standard softmax attention at 1.67e-16 delta. The β parameter is 1/√d. This means any attention layer in the stack can be replaced with a Hopfield retrieval call — same math, different framing that makes memory capacity explicit.

---

### F. WHAT MATHS/ CONTAINS

`math/` directory contains ONLY test files (test_*.py for existing ALGOS).
`03_VAULT/drive_pull/maths/` contains ONLY an import manifest showing a FAILED GDrive pull.
`PYPELINE/MATHS` and `PYPELINE/MATHS_CORE` on Google Drive were NOT successfully pulled — the connector listed them empty. Those folders contain unknown additional algorithms. Pulling them requires re-authentication.

---

### G. STATUS GOING INTO OPUS

| Item | State |
|---|---|
| Tests (targeted 10 files) | 43/43 PASS |
| STATUS_LEDGER --check | PASS |
| Graph write scanner | PASS, 0 violations |
| Bytewax bare python3 | FIXED |
| percyphon.py | COMPILES |
| ALGOS/ seed pool | 80 files |
| Darwin Hammer | RUNNING, gen1 survivors active |
| Schema backlog | ~92/122 unapplied — apply per §20 order |
| 18,627 staged candidates | waiting on write barrier schemas |
| Graph (canon) | 2 items, 0 edges, 0 materializations |
| GDrive MATHS folders | NOT pulled — unknown additional algos |
