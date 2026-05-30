# RFC-030: Full ETL Pipeline / Custody-First Ingestion

Status: DRAFT  
Subject ID: `full_etl_pipeline`  
Normative role: defines how raw chaos becomes durable candidates without letting interpretation corrupt custody.


## 0. Claim Discipline / ABBA3^5 Gate

ABBA3^5 is a local operator instruction, not an established external domain term. In this RFC it means: do not let confident prose outrun receipts. Every material CLAIM in this RFC must survive these five checks before it is treated as system guidance:

1. **Claim-state:** classify the statement as CLAIM, observation, inference, hypothesis, suggestion, or confidence-rated candidate; factual CLAIMs need receipts or cited sources.
2. **Provenance-count-and-reason:** state which operator instruction, repo file/code, receipt, or external source backs the claim; if only a few docs were consulted, name the count and why those docs were chosen.
3. **Naming-integrity:** preserve names as integral to their statement/context; label local/metaphorical names as local instead of pretending they are established terms.
4. **Reuse-before-reinvention:** search the LUCIDOTA Dev Library and known established concepts before proposing new code; reinvent only for sovereignty, objective superiority, necessity, or explicit operator intent.
5. **Operational-proportionality:** security, logging, tests, audits, and refusal gates are slop if they freeze work, flood storage, or waste operator time without proportional risk/truth benefit.

## 1. Thesis

The ETL pipeline must be a custody-first evidence hydraulic press: raw storage first, derived compute later, expensive cognition queued, and every failure represented as a record. This is the only architecture that fits LUCIDOTA's domain: large messy drops, OSINT artifacts, old code, books, communications, cases, archives, and live operator work.

## 2. Why I Believe This Is True

Local evidence is strong:

- `00_PROJECT_BRAIN/ACTIVE_SPEC/03_CUSTODY_ETL_PIPELINE.md` states the law: raw storage first, derived compute later.
- `06_SCHEMA/023_etl_pipeline.sql` defines `lucidota_etl.pipeline_run`, `object_trace`, `step_checkpoint`, and `dead_letter`.
- `scripts/korpus_krampii.py` is the current Python KORPUS hard-ingest implementation that performs hashing, componentization, routing, and storage-facing work without LLM calls.
- `06_SCHEMA/019_korpus_krampii.sql` defines KORPUS ingestion tables: batch, file object, file occurrence, component, tags, embeddings/status fields, and more.
- `00_PROJECT_BRAIN/ETL_PIPELINE/LIVE_INGEST_NOTES.md` recorded the actual failure chain: synchronous heroic mega-pipeline work jammed raw ingest with River, MinHash, graph promotion, autovacuum/checkpoint pressure, and dashboard confusion.
- `01_REPOS/lucidota_etl/README.md` exists as the Rust hot-path candidate documentation.

The architecture is therefore not speculative. It is a correction to observed failure.

## 3. External Source Anchors

- W3C PROV-O supports recording provenance and lineage. ETL custody is the local machinery that makes provenance possible: <https://www.w3.org/TR/prov-o/>.
- SEPIO supports keeping evidence/provenance connected to claims. ETL provides the evidence side before claims are promoted: <https://ohsu.elsevierpure.com/en/publications/sepio-a-semantic-model-for-the-integration-and-analysis-of-scient/>.
- PostgreSQL row locks / `SKIP LOCKED` support the slow-lane queue pattern for concurrent derived workers: <https://www.postgresql.org/docs/current/sql-select.html>.
- Blueprint First / Model Second supports keeping parsing/routing flow explicit instead of model-hidden: <https://arxiv.org/abs/2508.02721>.

## 4. Hot Lane / Slow Lane

Hot lane MUST stay boring:

- security/quarantine screen,
- hash/CAS,
- MIME/file-kind classification,
- stable metadata,
- cheap text extraction,
- componentization,
- queue insertion,
- receipts.

Slow lane MAY be adaptive:

- OCR,
- Whisper/STT,
- embeddings,
- GLiNER/local extraction,
- River/online learning,
- MinHash/dedupe policy,
- graph-promotion review,
- model-aided repair,
- report synthesis.

The slow lane cannot hold the hot lane hostage.

## 5. Rust / Python Split

Rust SHOULD own stable high-volume work: walking, hashing, CAS locking, archive expansion, standard extraction, row-level failure recording, JSONL/COPY output, and task enqueueing.

Python SHOULD own wet-clay adaptive work: OCR/STT, River/sklearn/scipy work, odd parser repairs, model glue, graph promotion policy, and live operator scripts.

Stable Python lanes MUST either receive a Rust-port candidacy decision or a keep-Python rationale.

## 6. Whole-System Interaction

- ETL feeds **Main Spine** with custody objects and derived tasks.
- ETL gives **Krampus/KORPUS** a deterministic front door.
- ETL gives **ABSURD** work rows instead of heroic synchronous calls.
- ETL gives **Ontology/Graph** candidates rather than truth.
- ETL gives **Local LLM Fabric** bounded chunks, not raw unbounded files.
- ETL gives **Artifact Templates** source-addressed evidence.
- ETL gives **Constant Learning** event/diff/component streams.

## 7. Benefit to the Whole System

The ETL pipeline makes LUCIDOTA robust under real-world chaos. A bad file becomes a dead letter. A large archive becomes child objects. Unsupported media becomes deferred work. Raw evidence is not lost because an OCR worker crashed. The operator can keep moving because slow cognition is restartable.

This is essential for OSINT and self-sovereignty: the system must preserve what happened before deciding what it means.

## 8. Correctness Argument

Custody-first ETL is correct because interpretation is lossy and fallible. Hashing and source preservation are cheaper and more objective than extraction. If extraction comes first, failures corrupt the evidence record. If custody comes first, every later interpretation can be replayed, compared, contradicted, or improved.

The local schema proves the intended shape: `object_trace` records source/virtual path/hash/status; `step_checkpoint` records progress; `dead_letter` records failure; KORPUS component tables preserve derived units without making them truth.

## 9. Falsifiers

This RFC is wrong if:

- raw source can be safely discarded after extraction,
- derived compute can run inside raw ingest without throughput/failure risk,
- model parsing can be trusted as custody,
- failure rows are more expensive than recovery ambiguity,
- Rust hot-path implementation cannot match existing Postgres surfaces.

Observed local notes already falsified the heroic synchronous alternative. A future Rust hot path only wins if it produces the same or better custody receipts, not merely because Rust is faster. The RFC therefore keeps the current Python KORPUS path valid until a side-by-side receipt proves replacement.

## 10. Filesystem / Runtime Consequences

- `ACTIVE_SPEC/03_CUSTODY_ETL_PIPELINE.md` remains the short canonical pipeline map.
- Existing ETL subdocs stay source material until fully merged or archived.
- Rust ETL must remain sidecar/offline until proof receipts exist.
- Python KORPUS remains valid as existing proven path until replacement is verified.
- No AI calls belong in the ETL hot lane.
- All future ingest work should name hot/slow lane behavior explicitly.
