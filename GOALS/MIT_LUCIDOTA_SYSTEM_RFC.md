# RFC-MIT-001 — LUCIDOTA System Design, Runtime, and Research Architecture

Status: **SYNTHESIS / LIMITED-AUDIT RFC**  
Audience: MIT technical review team / external architecture readers  
Generated: **2026-05-27T22:40Z** from local repo/runtime evidence  
Scope: One-file architecture explanation of LUCIDOTA as designed and as currently evidenced.  
Authority: This document summarizes and links local source evidence. It does **not** supersede active specs, schemas, receipts, or operator decisions.

## 0. Claim discipline and evidence bounds

This RFC is intentionally exhaustive at the system-design level, but it is based on a **limited audit/probe**, not a full proof of every historical artifact.

Sources actually inspected for this document include:

- Startup / law layer: `AGENTS.md` instructions provided in-session; `00_PROJECT_BRAIN/TICKLETRUNK.json`; `00_PROJECT_BRAIN/TICKLETRUNK.md`; `00_PROJECT_BRAIN/ACTIVE_SPEC/04_DEV_LIBRARY_REUSE_LAW.md`; `00_PROJECT_BRAIN/BLUEPRINT_FIRST_MODEL_SECOND_PSEUDOLAW.md`.
- Active specs: `00_PROJECT_BRAIN/ACTIVE_SPEC/01_IDENTITY_AND_CLAIM_STATE_LAW.md`, `02_EXECUTION_SPINE.md`, `03_CUSTODY_ETL_PIPELINE.md`, `05_COMPONENT_AUTHORITY_MAP.md`, `08_BOARD_EFFECT_TOURNAMENT_LAW.md`.
- RFC corpus: `00_PROJECT_BRAIN/RFCS/RFC-000` through key subject RFCs `050`, `060`, `070`, `080`, `090`, `110`, `120`, `130`, `140`, `160`, `170`, `190`, `200`, plus `RFC_SUBJECT_REGISTRY.json`.
- Runtime/registry docs: `00_PROJECT_BRAIN/gpu_model_runtime_registry.json`, `00_PROJECT_BRAIN/POSTGRES_AUDIT_CURRENT.md`, `00_PROJECT_BRAIN/DURABLE_WORKFLOW_DECISION.json`, `00_PROJECT_BRAIN/STATUS_LEDGER.md`, `GOALS/CURRENT_HANDOFF.md`, `GOALS/NEXT_GOAL_QUEUE.json`, `GOALS/MODEL_FABRIC_AUDIT.md`.
- Schemas/scripts: ABSURD, CAS, KORPUS, Chrono, GO graph, Indy, language membrane, Percyphon, board effect, model fabric, Rust ETL, and scraper files named throughout this document.
- Fresh probes run during this audit:
  - `python3 scripts/dev_library_scan.py --query <topic>` for named subsystems.
  - `python3 scripts/lucidota_model_registry.py`.
  - `python3 scripts/goal_model_fabric_control.py status --json`.
  - bounded `psql` counts against `lucidota_state` and `lucidota_storage`.

Terminology rule: names such as **Diogenes**, **ABSURD**, **Krampus**, **Santa**, **PercyphonAI**, **Indy_READs**, **FairyFuse**, **Needles**, and **GO-25** are LUCIDOTA-local design handles unless an external source is explicitly named. This document does not present local metaphors as established academic terms.

## 1. Executive thesis

LUCIDOTA is designed as a **local-first sovereign intelligence exocortex**: a constrained-machine operating system for evidence intake, OSINT/research, live coding, investigative writing, graph memory, model-assisted synthesis, and self-audit.

It is **not** designed as:

- a chatbot,
- a SaaS app,
- a prompt pile,
- an unbounded autonomous agent,
- a hidden LLM workflow controller,
- a perfect truth machine,
- or a cleanup project that deletes strange old artifacts to look tidy.

The core design is:

```text
operator intent / raw artifact / external source
  -> custody or command envelope
  -> ABSURD/Postgres work order or deterministic hot lane
  -> bounded worker / scraper / model runner / parser
  -> receipt, event, dead letter, or source-custody row
  -> evidence / claim / hypothesis / artifact candidate
  -> authority mapping + graph promotion gate
  -> optional graph materialization or human-facing output
  -> status, handoff, export, dashboard, or RFC surface
```

The system’s most important engineering posture is:

```text
deterministic first
bounded model second
Postgres/receipts/CAS always
graph truth only through gates
```

## 2. The whole system in one diagram

```text
                 ┌──────────────────────────────────────────────┐
                 │ Operator / build meeting / evidence drop      │
                 └──────────────────────┬───────────────────────┘
                                        │
                         command intent │ raw artifact
                                        │
        ┌───────────────────────────────▼───────────────────────────────┐
        │ Diogenes Kernel / command authority                            │
        │ - control packets                                               │
        │ - authority classes                                             │
        │ - proof custody                                                 │
        │ - graph-promotion guardrails                                    │
        └───────────────┬──────────────────────────────┬────────────────┘
                        │                              │
            authorized work/order              byte custody / archive
                        │                              │
        ┌───────────────▼──────────────┐     ┌────────▼───────────────┐
        │ ABSURD / Postgres queues      │     │ CAS + KORPUS/Krampus   │
        │ - jobs/events/dead letters    │     │ - hash/lock/dedupe     │
        │ - worker contracts            │     │ - componentize/index   │
        │ - retries / leases            │     │ - derived work queue   │
        └───────────────┬──────────────┘     └────────┬───────────────┘
                        │                              │
                        ├──────────────┬───────────────┘
                        │              │
          deterministic workers         │ bounded model / learning lanes
                        │              │
        ┌───────────────▼──────────────▼────────────────┐
        │ Candidate layer                                 │
        │ - evidence refs                                 │
        │ - claims / hypotheses                           │
        │ - timeline / chrono claims                      │
        │ - Indy judgments / LoRA candidates              │
        │ - River/Bytewax/GLiNER hints                    │
        └───────────────┬────────────────────────────────┘
                        │
        ┌───────────────▼────────────────────────────────┐
        │ GO graph / active ontology memory                │
        │ OBJECT + EVENT + EDGE, with staging and journals │
        └───────────────┬────────────────────────────────┘
                        │
        ┌───────────────▼────────────────────────────────┐
        │ Output hyperplexing / artifacts / dashboards    │
        │ template + quotes + model synthesis + smoothing │
        │ default outbound state: draft_only              │
        └────────────────────────────────────────────────┘
```

## 3. Non-negotiable design laws

### 3.1 Custody before cognition

Raw evidence is stored and identified before models, parsers, River, graph promotion, or reports interpret it. The custody path uses byte hashing, CAS, occurrence tracking, component rows, and receipts. This is enforced in doctrine by `ACTIVE_SPEC/03_CUSTODY_ETL_PIPELINE.md` and in storage by KORPUS/CAS schemas.

### 3.2 Workflow path outside the model

The workflow path belongs in code, schema, queue rows, templates, or explicit state machines. Models may extract, summarize, draft, rank, or classify at named nodes; models may not secretly choose the workflow, authorize side effects, materialize graph truth, or claim completion.

### 3.3 Postgres is the durable local control plane

LUCIDOTA has two active local databases:

- `postgresql:///lucidota_state` — queues, contracts, runtime facts, learning hints, Indy state, control plane.
- `postgresql:///lucidota_storage` — KORPUS custody, GO graph, learning artifacts, ETL receipts, investigation artifacts, timeline/chrono data.

ABSURD/Postgres is the current durable-work substrate. DBOS appears as legacy/provenance/compatibility in older artifacts; new durable work targets ABSURD/Postgres.

### 3.4 CAS is the byte-custody spine

CAS is not truth. CAS says: these bytes existed, had this hash, and were stored or journaled. Truth/candidate/promotion is separate.

### 3.5 Graph promotion is gated

No ordinary worker writes canonical graph truth directly. Graph-affecting outputs stage as packets/candidates, then require evidence refs, authority class, preflight, command envelope when applicable, journal, and materialization receipt.

### 3.6 The Dev Library indexes the jungle

The Dev Library/TICKLETRUNK is an access layer over the proof hoard. It makes artifacts findable; it does not make every artifact production. Sovereign originals are not mutated unless explicitly ordered. Reuse means copy/adapt/harden into production lanes.

## 4. Current limited-audit snapshot

### 4.1 Dev Library inventory

`00_PROJECT_BRAIN/TICKLETRUNK.json` was generated at `2026-05-27T17:42:41Z` and reports `1556` indexed entries.

| Category | Count | Role in the system |
|---|---:|---|
| ALGOS | 59 | deterministic primitives and reusable math, e.g. routing, MinHash, Krampus chrono, Percyphon, River-ish helpers |
| SCRIPTS | 727 | active and legacy executable surfaces, workers, audits, adapters |
| MODELS | 33 | local GGUF/model assets and reference vocab/template files |
| LORAS | 80 | LoRA tools plus Indy_READs cartridge files/manifests/train/validation artifacts |
| SCHEMAS | 13 in TICKLETRUNK category, 100+ actual files in `06_SCHEMA/` | DB contracts, JSON schemas, SQL migrations |
| SKILLS | 39 | local Codex/Superpowers/Gmail/Drive/system skills |
| PLUGINS | 11 | Codex/CLAW plugin artifacts |
| SERVICES | 6 | service candidates, including FairyFuse backend/lab |
| BOOKS | 18 | Indy books plus GO ontology files and Indy state |
| SURFACES | 6 | generated HTML/JSON status surfaces |
| SCRAPERS | 12 | Body Capture, Survey, security scan, browser capture, server-test model fetcher |
| WORKFLOWS | 129 | RFCs, active specs, workflow contracts, registry items |
| REPOS | 121 | vendored/referenced repositories, including llama.cpp, PocketFlow, doggystyle/CKDOG1, Rust ETL |
| RUNTIME | 61 | runtime state, model receipts, Indy configs, observation-center snapshots |
| VAULT | 27 | evidence/reference/model/vault material |
| OTHER | 214 | tests/src/tooling and miscellaneous indexed artifacts |

Important interpretation: counts prove indexed existence, not production status.

### 4.2 Fresh Postgres counts

Fresh bounded queries during this RFC produced:

| Area | Count / status |
|---|---:|
| ABSURD jobs: queued | 2 |
| ABSURD jobs: succeeded | 2599 |
| ABSURD jobs: dead_lettered | 78 |
| ABSURD jobs: cancelled | 2 |
| KORPUS file objects | 18,627 |
| KORPUS components | 113,560 |
| KORPUS derived compute queue rows | 2 |
| CAS manifest rows | 18,626 |
| CAS ingest journal rows | 2 |
| Chrono temporal claims | 43,922 |
| Current Chrono timeline projection rows | 18,627 |
| GO graph items | 275,602 |
| GO graph edges | 1,379,933 |
| GO graph journal rows | 1,648,061 |
| Graph materialization helper receipts | 108 |
| Graph promotion materializations | 111 |
| GO staging packets | 22 |

### 4.3 Fresh local model-fabric status

Fresh `scripts/goal_model_fabric_control.py status --json` reported:

| Lane | Port | Live? | Device lane | Model/checkpoint |
|---|---:|---|---|---|
| `deepseek` | 8080 | yes | NVIDIA VRAM model lane | `03_VAULT/models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf` |
| `mamba_cpu` | 8081 | yes | system RAM / CPU | `03_VAULT/models/tensorblock/Falcon3-Mamba-7B-Instruct-GGUF/Falcon3-Mamba-7B-Instruct-Q2_K.gguf` |
| `bonsai` | 8082 | yes | system RAM / CPU | `03_VAULT/models/prism-ml/Ternary-Bonsai-4B-gguf/Ternary-Bonsai-4B-Q2_0.gguf` |
| `mamba_gpu` | 8083 | no | optional preemptible NVIDIA lane | same Falcon3-Mamba file, optional only |
| `needle_0`..`needle_5` | 8090-8095 | yes | system RAM / CPU | `03_VAULT/models/needle/needle.pkl` |

Hardware registry truth:

- GPU: NVIDIA GeForce GTX 1650.
- Total VRAM: 4096 MiB.
- Current probe in registry: 1096 MiB used, 2619 MiB free at `2026-05-27T18:52:55Z`.
- Rule: prefer <=2B quantized GGUF on the 4GB GPU; do not claim arbitrary large-model runtime because `nvidia-smi` exists.

## 5. Models -> VRAM and Models -> RAM contract

### 5.1 Active model assets

| Model asset | Size | Runtime status | Intended role | Memory lane |
|---|---:|---|---|---|
| `DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf` | ~1.12 GB | active strict stack | heavier local synthesis / Indy heavy-hitter | NVIDIA VRAM lane, port 8080 |
| `Falcon3-Mamba-7B-Instruct-Q2_K.gguf` | ~2.57 GB | active strict stack | listener / Mamba lane | CPU/RAM on 8081; optional GPU partial on 8083 |
| `Ternary-Bonsai-4B-Q2_0.gguf` | ~1.07 GB | active strict stack | Bonsai/ternary CPU lane | CPU/RAM, port 8082 |
| `needle.pkl` | not a GGUF LLM | active strict stack | six cheap hot scouts / router swarm | CPU/RAM, ports 8090-8095 |
| `mamba-1.4b-hf-Q2_K.gguf` | ~0.53 GB | legacy reference | watch/reference only | not active loadout |
| `gliner/urchade_gliner_small-v2.1` | HF-style local model | reference/worker candidate | local zero-shot entity span extraction | CPU/local model lane unless separately governed |

### 5.2 Models -> VRAM

The VRAM lane is deliberately narrow:

- Primary evidenced GPU: NVIDIA GeForce GTX 1650, 4096 MiB total VRAM.
- Primary active VRAM model: `DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf`.
- Operational expectation: one bounded quantized synthesis/heavy-hitter lane, not a general promise that every model can fit.
- Governance expectation: record model path, hash, device, prompt/input hash, output hash, and receipt path when invoked.
- Anti-slop rule: do not use the mere existence of CUDA/`nvidia-smi` to imply large-model capacity.

### 5.3 Models -> RAM

The RAM lanes carry the broader always-on fabric:

- `Falcon3-Mamba-7B-Instruct-Q2_K.gguf` is the evidenced listener/Mamba RAM lane on port 8081.
- `Ternary-Bonsai-4B-Q2_0.gguf` is the evidenced Bonsai/ternary RAM lane on port 8082.
- `needle.pkl` supplies six cheap Needle/router swarm workers on ports 8090-8095.
- GLiNER-style extraction workers and similar classifiers are RAM/local-worker candidates unless separately promoted into another governed lane.
- RAM lanes are preferred for cheap routing, listening, classification, chunk handling, and procedural scaffolding so VRAM is reserved for bounded synthesis.

### 5.4 Runtime law

Model execution MUST emit or preserve:

- command envelope UUID when applicable,
- model path,
- model SHA-256,
- device / RAM / VRAM profile,
- input or prompt hash,
- output hash,
- receipt path,
- graph-write status.

Model output is **draft/candidate** until another lane promotes it. A fluent answer is not proof.

### 5.5 Why many small lanes instead of one big model

The machine is constrained. The design uses:

- GPU for one bounded quantized local synthesis lane,
- CPU/RAM for listener/Bonsai/Needle lanes,
- deterministic routing and templates to avoid unnecessary inference,
- KORPUS chunk/component IDs instead of mystery-pile prompts,
- LoRA/adapter candidates as queued artifacts, not hidden runtime mutations.

This is the “punch above the hardware” strategy: small local specialists, not datacenter cosplay.

## 6. Postgres + ABSURD + CAS: the central focus

### 6.1 Postgres roles

Postgres is split by authority and size:

```text
lucidota_state
  -> queues
  -> workflow events
  -> runtime facts
  -> contracts
  -> learning hints
  -> Indy state

lucidota_storage
  -> KORPUS custody
  -> CAS metadata
  -> GO graph/staging/journal
  -> Chrono/hypertimeline data
  -> ETL/investigation artifacts
  -> heavier evidence records
```

### 6.2 ABSURD queue mechanics

ABSURD jobs carry:

- queue name,
- workflow name,
- job kind,
- idempotency key,
- payload,
- status,
- priority,
- attempt count,
- lease metadata,
- result/error,
- timestamps.

Events preserve transitions: enqueued, leased, started, succeeded, failed, dead-lettered, retry scheduled, cancelled, or audit.

Current active queue table includes queues such as:

- `control`
- `chrono`
- `korpus`
- `graph_promotion`
- `surface_cep`
- `river`
- `model_fabric`
- `goal_swarm`
- `document_parse`
- `intake`
- `boring_beast`
- `soak_probe`
- `bitloops_momentary`

Important: some queue names retain legacy DBOS provenance, but `00_PROJECT_BRAIN/DURABLE_WORKFLOW_DECISION.json` says new architecture targets ABSURD/Postgres.

### 6.3 ABSURD worker contract law

A worker must be registered/contracted before trusted side effects. If a queue/job kind has no active contract, the worker records rejection rather than inventing permission.

Evidence files:

- `06_SCHEMA/035_absurd_queue_spine.sql`
- `06_SCHEMA/082_absurd_worker_contract_registry_enforcement.sql`
- `scripts/absurd_queue_spine.py`
- `scripts/absurd_consume_one.py`
- `scripts/absurd_worker_contracts.py`

### 6.4 CAS contract

CAS stores byte identity and custody metadata:

- `lucidota_vault.cas_manifest`
- `lucidota_vault.cas_integrity_check`
- `lucidota_vault.cas_gc_run`
- `lucidota_vault.cas_gc_candidate`
- `lucidota_vault.cas_ingest_journal`

CAS is deliberately boring: SHA-256, path, size, MIME/source, integrity checks, GC candidates, ingest journal stages. It is not a truth layer. It is the substrate that lets higher layers argue from stable bytes.

### 6.5 Why this combination matters

Postgres + ABSURD + CAS provide the core property MIT reviewers should care about:

```text
Every meaningful action can be queried.
Every expensive task can be queued.
Every failure can be dead-lettered.
Every byte can be hashed.
Every candidate can point back to custody.
Every promotion can be audited.
```

This is the architecture’s answer to both LLM hallucination and live-coding chaos.

## 7. DIOGENES Kernel

The Diogenes Kernel is the authority organ, not one daemon.

It includes:

- command envelopes,
- domain-separated control packets,
- permission boundaries,
- receipts,
- proof custody,
- graph-promotion guardrails.

Important evidence:

- `00_PROJECT_BRAIN/RFCS/RFC-050-DIOGENES-KERNEL.md`
- `scripts/kernel_control_packet.py`
- `scripts/spine_kernel_authorization.py`
- `scripts/spine_authority_checker.py`
- `scripts/proof_kernel.py`
- `scripts/graph_promotion_gate.py`
- `00_PROJECT_BRAIN/spine_authority_registry.json`
- `01_REPOS/doggystyle/` CKDOG1 kernel reference/prototype repo
- `check_diogenes.sh` smoke harness, though it contains legacy DBOS-era checks and should be read as historical/integration smoke, not the current architecture decision.

The key separation is:

```text
capability != authority
```

A model can suggest a graph write. A script can technically do a DB insert. A scraper can fetch a page. None of those facts grant permission. Diogenes requires the authority class to be explicit.

## 8. KORPUS / Krampus / KRAMPUSCHEWING

Krampus has two meanings in current LUCIDOTA docs and both must be separated:

1. **KORPUS/Krampus Express/KRAMPUSCHEWING** — the custody/intake stomach for proof-hoard material.
2. **Krampus persona** — a local board-effect audit/enforcement metaphor.

### 8.1 KORPUS/Krampus custody path

KORPUS is the deterministic preserve-now/digest-later organ.

Path:

```text
path/root
  -> file hash
  -> CAS lock/reference
  -> file object
  -> occurrence rows
  -> component rows
  -> tags/entities/concepts/minhash/embedding status
  -> derived compute queue
  -> candidate graph/evidence lanes, not automatic truth
```

Evidence:

- `scripts/korpus_krampii.py`
- `scripts/spine_krampus_worker.py`
- `scripts/krampuschewing_watcher.sh`
- `06_SCHEMA/019_korpus_krampii.sql`
- `06_SCHEMA/020_korpus_derived_compute_queue.sql`
- `06_SCHEMA/006_workflow_registry.sql`

Fresh storage evidence: 18,627 file objects and 113,560 components.

### 8.2 Krampus as board-effect auditor

From `ACTIVE_SPEC/08_BOARD_EFFECT_TOURNAMENT_LAW.md`: Krampus is a LUCIDOTA-local fair audit/enforcement persona. It prefers moves affecting 2 or 4+ board systems, and it may classify slop only from evidence. It is explicitly not allowed to imagine wrongdoing or corruption without receipts, files, DB rows, logs, tests, or contradiction records.

Krampus wins by reducing entropy without destroying proof.

## 9. Santa

Santa is also a LUCIDOTA-local metaphor, not an external technical term.

From `ACTIVE_SPEC/08_BOARD_EFFECT_TOURNAMENT_LAW.md`: Santa is the glow-finding/exploration persona. Santa may settle for one board effect and may assume/search for new glow, but assumptions must be labeled and converted into testable working realities before authority.

Santa is useful because not all valuable discovery begins as a multi-system audit. But Santa cannot promote optimism into graph truth.

## 10. PercyphonAI

PercyphonAI is a zero-VRAM procedural scaffold organ.

Evidence:

- `ALGOS/percyphon.py`
- `00_PROJECT_BRAIN/RFCS/RFC-090-PERCYPHONAI.md`
- `04_RUNTIME/indy_percyphon_hunch_subtleknife_binding.json`

Contract:

- may generate deterministic slots, aliases, UUID-like IDs, personas, and fluid references;
- may help route/label/present work;
- may help create village/witness/scribe-like scaffolds;
- must not claim generated slots are real people;
- must not write canonical graph facts;
- must not require model weights.

PercyphonAI exists because some “AI teammate” structure does not need inference. Stable procedural masks are cheaper, testable, repeatable, and safer than spending VRAM on role decoration.

## 11. Indy_READs, Books, and LoRA cartridges

### 11.1 Indy_READs identity

INDY_READs is the named reading/workflow teammate. She is not just an agent prompt. She has:

- persona config,
- page-locked reading boundaries,
- parser cache/state,
- judgment CSV,
- adapter registry,
- polycareer role modes,
- product-factory outputs.

Evidence:

- `scripts/indy_reads.py`
- `scripts/lucidota_indy_polycareer.py`
- `scripts/lucidota_indy_library_ingest.py`
- `scripts/lucidota_indy_lora_train.py`
- `00_PROJECT_BRAIN/INDY_READS_POLYCAREER_WORKFLOW_WIZARD/ARCHITECTURE.md`
- `04_RUNTIME/indy_reads_persona_config.json`
- `04_RUNTIME/indy_reads_adapter_registry.json`
- `BOOKS/.indy_reads/watch_state.json`

### 11.2 Current book/watch state

Current watch state records six main book files in `BOOKS/`, plus GO ontology JSON files and Indy state files in the Dev Library book category.

Indy’s page-boundary law:

- she may read books/documents from `BOOKS/`,
- she may cache pages and parser results,
- she may collect judgments,
- she may produce notes/briefs,
- she must not claim knowledge of unread pages in page-locked mode,
- she must not mutate active GO terms or graph SQL,
- she must not publish unverified claims as facts.

### 11.3 Current judgment state

`BOOKS/.indy_reads/indy_reads_judgments.csv` currently has only the header row in the limited probe. That means the judgment surface exists, but this audit did not prove a populated human judgment dataset.

### 11.4 LoRA cartridges

Current local LoRA cartridge audit:

- 25 `manifest.json` cartridge manifests under `04_RUNTIME/lora_cartridges/`.
- 75 cartridge files counted at depth 2 (manifests + train + validation files).
- Aggregate from manifests: 1,870 training rows and 224 validation rows.
- All sampled manifests show `training_status: queued`; this audit did not prove trained LoRA weights.
- TICKLETRUNK’s LORAS category count is 80 because it indexes tools plus cartridge files, not only trained adapters.

LoRA rule:

```text
adapter candidates are data/receipts until training + evaluation + authority promotion proves runtime use
```

## 12. Input multiplexing

Input multiplexing solves a real operator problem: not all input deserves the same path.

Evidence:

- `scripts/fast_slow_lane_gate.py`
- `scripts/language_router.py`
- `core/language_membrane.py`
- `06_SCHEMA/020_korpus_derived_compute_queue.sql`
- `RFC-110-INPUT-MULTIPLEXING.md`

Contract:

- FASTLANE: status, health, metadata, check, list, receipt, simple probes.
- SLOWLANE: audit, research, model-heavy synthesis, graph, workflow, benchmark, failure, blocker, proof-heavy work.
- Route decisions preserve packet hash, reason, lane, metadata flags, cache path, flush behavior.
- Fast lane is immediacy, not truth.
- Slow lane is depth, not delay theater.

The gate is deterministic and performs no model calls, network calls, or canonical graph writes.

## 13. Output hyperplexing / multiplexing responses back

Output hyperplexing is the return path: weave multiple authority lanes into one readable artifact without erasing seams.

Evidence:

- `core/language_membrane.py`
- `scripts/template_contract.py`
- `scripts/hypertimeline_engine.py`
- `RFC-120-OUTPUT-HYPERPLEXING.md`

Output lanes include:

- deterministic template lane,
- exact quote / retrieval lane,
- model synthesis lane,
- smoothing/style lane,
- timeline/order lane,
- confidence/falsifier lane,
- operator-action lane.

`core/language_membrane.py` currently names lanes:

- `tera_template`
- `rag_quotes`
- `deepseek_q4`
- `fairyfuse_smoothing`

Default outbound state is `draft_only`. A polished artifact can still be draft-only. Beauty does not grant authority.

Recommended sequence for serious outputs:

1. decision/question,
2. source/custody frame,
3. strongest grounded facts,
4. contradictions/gaps,
5. synthesis/hypothesis,
6. recommended next action,
7. receipts/export refs.

## 14. Scrapers and external intake tools

TICKLETRUNK currently indexes 12 scraper-category artifacts:

| Artifact | Role |
|---|---|
| `01_REPOS/llama.cpp/scripts/fetch_server_test_models.py` | fetches models used by llama.cpp server tests |
| `06_SCHEMA/003_survey_protocol.sql` | Survey / hop-pivot intake schema |
| `06_SCHEMA/011_body_capture.sql` | Body Capture evidence/diff schema |
| `scripts/legacy/lucidota_body_capture.py` | legacy operator-supervised HTTP/body snapshot to local CAS |
| `scripts/legacy/lucidota_dbos_survey.py` | legacy DBOS wrapper for Survey Protocol |
| `scripts/legacy/lucidota_survey.py` | legacy URL/file triage + CAS + pivot hints |
| `scripts/lucidota_body_capture.py` | active/importable Body Capture entrypoint until Rust adapter replaces it |
| `scripts/lucidota_body_capture_evidence.py` | evidence bundle + text/Wayback diff |
| `scripts/lucidota_body_capture_policy.py` | watcher policy evaluator |
| `scripts/lucidota_browser_body_capture.py` | browser capture contract |
| `scripts/lucidota_security_scan.py` | lightweight repo security tripwire |
| `scripts/lucidota_survey.py` | active/importable Survey adapter until Rust port lands |

Scraper law:

- external capture writes custody/evidence/receipts first;
- source policy must be explicit;
- scraped output is source claim/evidence candidate, not final truth;
- graph writes remain gated;
- destructive/external effects require Diogenes/operator authority.

## 15. Tools and services

### 15.1 TOOLS access layer

`TOOLS/` is the human navigation layer over TICKLETRUNK. It has category readmes for:

- ALGOS
- BOOKS
- LORAS
- MODELS
- OTHER
- PLUGINS
- REPOS
- RUNTIME
- SCHEMAS
- SCRAPERS
- SCRIPTS
- SERVICES
- SKILLS
- SURFACES
- VAULT
- WORKFLOWS

Each category repeats the same doctrine: original sandbox/toolbox artifacts remain sovereign; copy/adapt into production; do not mutate originals unless explicitly instructed.

### 15.2 Services

TICKLETRUNK service category currently includes:

- `services/fairyfuse/fairyfuse_backend.py` — FairyFuse resident ternary backend scaffold.
- `services/ternary_lab/README.md`
- `services/ternary_lab/vendor_manifest.json`

Current blocker from status ledger: real BitNet/FairyFuse backend remains not fully wired; symbolic backend is usable as a scaffold/smoothing lane but should not be claimed as benchmarked real ternary model runtime.

## 16. Chrono and hypertimeline

### 16.1 Chrono-Ledger

Chrono-Ledger is append-only temporal evidence over KORPUS file objects.

Evidence:

- `06_SCHEMA/025_chrono_ledger_core.sql`
- `06_SCHEMA/100_chrono_timeline_projection_refresh.sql`
- `scripts/chrono_*`
- `01_REPOS/lucidota_etl/crates/lucidota-chrono-ledger/`

Fresh count evidence:

- 43,922 `lucidota_korpus.temporal_claim` rows.
- 18,627 `current_chrono_timeline_projection` rows.

Important semantics:

- multiple timestamp claims may exist per artifact;
- trust weights express source confidence;
- projection chooses a current dominant timeline row;
- projection does not delete or rewrite the archive.

### 16.2 Hypertimeline engine

`scripts/hypertimeline_engine.py` is the broader ABSURD hypertimeline/circadian engine. It targets chat/social/email/SMS-like event material and creates sequenced events with source kind, entity, action, content, raw payload, provider ID, and sequence.

Current observation-center hunch snapshot:

- `04_RUNTIME/observation_center/hunch_hypertimeline_latest.json`
- Applies to Indy_READs and PercyphonAI.
- 93 parsed/unique hunch IDs.
- Count status: `DISCREPANCY_REQUIRES_REVIEW` because canonical known tracked total is 91 while parsed headings total is 93.
- Reported resolved directional accuracy: 95.652% over resolved hunches.
- Graph writes performed: false.

This is exactly the desired posture: timeline/hunch analysis may be useful, but discrepancy remains review state, not hidden success.

## 17. GO graph and active ontology

Active ontology is **GO / GO-25 plus extensions**, not ROOT-414.

Evidence:

- `OFFICIAL_ONTOLOGY.json`
- `BOOKS/GO_ACTIVE_TERMS.json`
- `06_SCHEMA/016_go_graph_core.sql`
- `scripts/operator_ontology_fidelity_guard.py`
- `scripts/ontology_staging_contract.py`
- `scripts/graph_promotion_gate.py`

Graph primitives:

- OBJECT / graph item — thing, claim, artifact, person, concept, tool, source.
- EVENT / journal/staging/workflow event — changes, observations, ingests, approvals.
- EDGE / graph edge — supports, contradicts, derived-from, happened-before, uses, part-of.

GO classifies; graph rows remember; evidence refs justify; journals make mutation auditable.

## 18. Constant learning

Constant learning is designed as controlled update loops, not a self-mutating belief blob.

Evidence:

- `scripts/absurd_river_worker.py`
- `06_SCHEMA/004_learning_reflex.sql`
- `06_SCHEMA/007_bytewax_stream.sql`
- `06_SCHEMA/038_absurd_river_wrapper.sql`
- `06_SCHEMA/073_absurd_river_claim_packet_job.sql`
- `00_PROJECT_BRAIN/RUVECTOR_ABSURD_SONA_RIVERML_NOTES.md`
- `RFC-140-CONSTANT-LEARNING.md`

Layer contract:

1. observation layer,
2. feature layer,
3. judge layer,
4. neural repair / GLiNER / LoRA / SONA candidates,
5. witness layer,
6. promotion layer.

Hard law:

```text
no direct neural output to canonical graph
no direct River/Treelite judge to SONA mutation
no direct SONA to ABSURD queue mutation
all crossings through typed records
```

## 19. Rust front door and Python wet clay

The ETL plan deliberately separates stable high-volume deterministic work from adaptive glue.

Rust workspace: `01_REPOS/lucidota_etl/` with crates including:

- `lucidota-core`
- `lucidota-db`
- `lucidota-intake`
- `lucidota-kernel`
- `lucidota-launcher`
- `lucidota-audit`
- `lucidota-sqlite-importer`
- `lucidota-workers`
- `lucidota-chrono-ledger`

Rust should own high-volume deterministic work: recursive walking, SHA/BLAKE hashing, CAS locking, MIME/kind inference, archive expansion, metadata flattening, row-level failures, Postgres/COPY or JSONL emission.

Python owns adaptive work: OCR/STT, GLiNER/model glue, River replay, graph-promotion policy, parser repair, live-coding repair lanes.

## 20. Current goals and operating trajectory

### 20.1 Current goal context before this RFC

The previous `GOALS/CURRENT_HANDOFF.md` described a broad objective: full February-to-now reingestion into a graph/db-first sovereign truth engine with chronology, treelite/river lanes, Indy_READs surfaces, organized research/cases/musings, and local-first model work without unsafe canonical graph writes.

This MIT RFC goal temporarily replaced that current handoff so the architecture could be documented in one place.

### 20.2 Next queued goals

`GOALS/NEXT_GOAL_QUEUE.json` includes:

1. **GOAL 5 — System Elegance / Slimming War Game**  
   Audit the rest of LUCIDOTA, reduce LOC/slop without losing capability, reuse FOSS/local parts first, preserve local runnable baseline without cloud dependency.

2. **GOAL 6 — Operator Parse + Safe Automation Continuity**  
   Parse messy operator instructions into typed work orders/conversation commands, support handoff before long-session limits, avoid unsafe autonomous loops.

3. **GOAL 7 — Instruction Hygiene Language Membrane Full Subsystem**  
   Marked complete in the queue; evidence refers to the deterministic language/router subsystem.

### 20.3 Near-term strategic goals

The system’s next useful work should stay centered on:

- keeping Postgres/ABSURD/CAS as the source of operational truth;
- reconciling live queue/dead-letter state;
- filling Indy judgments with sourced page-locked records;
- moving LoRA cartridges from queued datasets to trained/evaluated artifacts only when safe;
- keeping model fabric honest about 4GB VRAM;
- continuing Chrono/hypertimeline reingest without direct graph writes;
- promoting only through graph gates;
- reducing code sprawl by reusing existing scripts and FOSS tools rather than creating new local clones.

## 21. Hypertimeline / chronology of architecture maturation

This is a compressed chronology of local evidence, not a complete project history.

| Date / evidence | Meaning |
|---|---|
| 2026-05-13/14 vault and ingested markdown artifacts | Earlier Diogenes/LUCIDOTA brain snapshots, ontology docs, graph storage rules, and Krampus smoke/drop material were preserved under `03_VAULT/ingested_markdown/` and `03_VAULT/korpus_krampii/` rather than erased. |
| 2026-05-17 status ledger rounds 73-100 | Chrono projection, graph replay, semantic handle generation, queue-to-Chrono bridge, graph promotion E2E, ABSURD soak, system-wide temporal/graph audits, and mega-gate receipts were recorded. |
| 2026-05-24 RFC subject registry | Mandatory subject RFC set was seeded: Diogenes, ABSURD, KORPUS, Local LLM Fabric, PercyphonAI, input/output lanes, Indy_READs, constant learning, ontology, filesystem law. |
| 2026-05-26 SITREP / model fabric / language subsystem | Deterministic language routing, model-fabric status, telemetry, River/Bytewax/GLiNER boundary audits, and KRAMPUSCHEWING preservation lanes were summarized. |
| 2026-05-27 current handoff before this RFC | Model/VRAM truth was corrected: Falcon3 is actual Mamba lane, Spark is Needle scout, Bonsai is CPU/RAM, Bitloops live binary remains missing, graph writes remain guarded. |
| 2026-05-27 this limited audit | Fresh Dev Library scan, Postgres counts, model status, Indy/LoRA state, scraper inventory, and Santa/Krampus law were consolidated into this single MIT-facing RFC. |

## 22. Known gaps, blockers, and falsifiers

### 22.1 Current explicit blockers from handoff/status sources

- Bitloops live binary missing; airlock/provenance may exist, live daemon does not.
- Cohere execute key missing in current shell/context; provider adapter existence is not current execute capability.
- Optional `mamba_gpu_partial` lane is offline in the fresh model-fabric status.
- Local token share was below target after Groq-heavy prior work.
- FairyFuse/BitNet real backend remains not benchmarked/wired; symbolic backend is not the same as real ternary inference.
- Some optional local model weights are not configured/downloaded: BGE, ModernBERT, Ollama models.
- Indy judgment CSV exists but is empty except header in this probe.
- LoRA cartridges are queued datasets, not proven trained/evaluated adapters.
- TICKLETRUNK proves indexed artifacts, not production readiness.
- DBOS names remain in legacy artifacts/queues; current architecture must continue terminology migration to ABSURD/Postgres without destroying provenance.

### 22.2 Architecture falsifiers

The architecture claim would be weakened if:

- ordinary workers can materialize canonical graph truth directly;
- model policy checks load weights or spend VRAM as hidden side effects;
- scraper outputs can bypass custody and source policy;
- CAS bytes cannot be reconciled to DB custody rows;
- ABSURD workers execute unregistered job kinds silently;
- Indy can make forward claims about unread pages;
- LoRA/SONA/River outputs mutate runtime behavior without receipts;
- output hyperplexing erases source/model/template lane boundaries;
- the filesystem cannot distinguish active doctrine, vault, runtime, outputs, labs, and sovereign toolbox originals.

## 23. What MIT should review first

If MIT reviewers only have a short window, review in this order:

1. `00_PROJECT_BRAIN/ACTIVE_SPEC/01_IDENTITY_AND_CLAIM_STATE_LAW.md`
2. `00_PROJECT_BRAIN/ACTIVE_SPEC/02_EXECUTION_SPINE.md`
3. `00_PROJECT_BRAIN/ACTIVE_SPEC/03_CUSTODY_ETL_PIPELINE.md`
4. `00_PROJECT_BRAIN/ACTIVE_SPEC/05_COMPONENT_AUTHORITY_MAP.md`
5. `00_PROJECT_BRAIN/RFCS/RFC-060-ABSURD-WORKFLOWS.md`
6. `00_PROJECT_BRAIN/RFCS/RFC-070-KRAMPUS-KORPUS.md`
7. `00_PROJECT_BRAIN/RFCS/RFC-080-LOCAL-LLM-FABRIC.md`
8. `00_PROJECT_BRAIN/RFCS/RFC-110-INPUT-MULTIPLEXING.md`
9. `00_PROJECT_BRAIN/RFCS/RFC-120-OUTPUT-HYPERPLEXING.md`
10. `00_PROJECT_BRAIN/RFCS/RFC-130-INDY-READS.md`
11. `00_PROJECT_BRAIN/TICKLETRUNK.json`
12. `00_PROJECT_BRAIN/gpu_model_runtime_registry.json`
13. `00_PROJECT_BRAIN/POSTGRES_AUDIT_CURRENT.md`
14. `06_SCHEMA/035_absurd_queue_spine.sql`
15. `06_SCHEMA/005_cas_manifest.sql`
16. `06_SCHEMA/019_korpus_krampii.sql`
17. `06_SCHEMA/016_go_graph_core.sql`

## 24. Final concise description

LUCIDOTA is a local sovereign intelligence workstation where:

- **Postgres** is the durable control/memory substrate,
- **ABSURD** is the queue/workflow nervous system,
- **CAS** is the byte-custody skeleton,
- **KORPUS/Krampus** is the evidence stomach,
- **Diogenes** is the authority immune system,
- **GO graph** is typed working memory,
- **Chrono/hypertimeline** is temporal memory,
- **Indy_READs** is the reading/workflow teammate,
- **PercyphonAI** is zero-VRAM procedural scaffolding,
- **local models** are bounded draft/extraction organs,
- **multiplexing/hyperplexing** keeps input/output lanes honest,
- **Santa/Krampus board metaphors** guide exploration vs audit without pretending to be external science,
- **TICKLETRUNK/TOOLS** makes the proof hoard reusable without making the jungle production truth.

The design is coherent if it keeps this invariant:

```text
fast work is allowed
weird work is allowed
model help is allowed
old artifacts are allowed
but authority, custody, provenance, and graph truth must remain explicit
```

## 25. Final handoff step

"Save This Prompt, Pass on this Handoff:"

This single RFC document is the MIT-facing architecture handoff. It should be refreshed after any material change to model runtime, ABSURD queue state, CAS/KORPUS counts, graph materialization policy, Indy/LoRA training status, or current goal queue. Next pickup action: if the operator approves this RFC, promote or link it from the canonical RFC program and produce a small machine-readable companion index only if needed.

Technical Summary Review and Dev Notes: One lantern-map now covers the organism: Postgres/ABSURD/CAS bones, model/RAM/VRAM muscles, Indy/Percyphon/Krampus/Santa field marks. The cryptid is still alive, but the footprints are now in one place.
