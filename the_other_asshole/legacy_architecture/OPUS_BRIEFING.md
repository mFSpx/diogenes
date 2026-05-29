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
