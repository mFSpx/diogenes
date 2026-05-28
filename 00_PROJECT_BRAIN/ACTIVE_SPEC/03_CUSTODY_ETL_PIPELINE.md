# LUCIDOTA Custody ETL Pipeline

Status: ACTIVE SPEC SOURCE — custody-first ingestion and derived-work pipeline  
Purpose: define the full evidence hydraulic press from raw input to derived candidates.

## 1. Pipeline Law

```text
raw storage first
derived compute later
queue everything expensive
no single derived task can hold raw ingest hostage
```

The ETL pipeline does not replace cognition. It protects cognition by giving it stable custody, replayable rows, and bounded work queues.

## 2. Current Target Shape

```text
source/drop/archive/database/message
  -> security/quarantine screen
  -> byte hash + CAS custody
  -> file/object occurrence rows
  -> structural metadata + cheap text extraction
  -> component rows
  -> derived_compute_queue
  -> slow/adaptive workers
  -> evidence/claim/hypothesis candidates
  -> authority gate
  -> review/export/materialization only if authorized
```

## 3. Rust Front Door, Python Wet Clay

Rust ETL owns stable high-volume deterministic work:

- recursive walking with resource caps,
- SHA-256/BLAKE3 hashing,
- CAS locking,
- MIME/file-kind inference,
- archive expansion with depth/size limits,
- standard text/CSV/JSON/SQLite flattening,
- media/container metadata,
- row-level failure records,
- Postgres/COPY or JSONL emission,
- enqueueing derived tasks.

Python owns adaptive work:

- OCR,
- Whisper/STT,
- River replay,
- MinHash policy changes,
- graph-promotion policy,
- odd parser repair,
- model/agent glue,
- one-off live-coding repair lanes.

Stable Python lanes must either graduate to Rust-port candidacy or be explicitly marked keep-Python with evidence.

## 4. Hot Lane / Slow Lane

Hot lane:

- security/quarantine,
- hash/CAS,
- MIME/kind,
- cheap text and metadata,
- componentization,
- queue insertion,
- receipts.

Slow lane:

- OCR/STT,
- embeddings,
- GLiNER/local model extraction,
- River/online learning,
- near-duplicate scans,
- graph-promotion review,
- large archive/database deep work,
- report synthesis.

The hot lane must remain boring and restartable. The slow lane may be weird, adaptive, and abductive, but only through queued work and receipts.

## 5. Database / Queue Contract

Primary existing surfaces:

- `lucidota_korpus.ingest_batch`
- `lucidota_korpus.file_object`
- `lucidota_vault.cas_manifest`
- `lucidota_korpus.file_occurrence`
- `lucidota_korpus.component`
- `lucidota_korpus.derived_compute_queue`
- `lucidota_control.workflow_event`

Queue status vocabulary stays boring:

```text
queued
running
succeeded
failed
dead
```

Expensive workers claim rows with `FOR UPDATE SKIP LOCKED` or the equivalent ABSURD worker-contract path.

## 6. Failure Law

Failures are records, not mysteries.

A bad file/member/row becomes one failed record or dead letter. It must not kill the entire ingest unless the operator requested strict fail-fast behavior.

A stuck object must be explainable by query or receipt, not by reading a prompt transcript.

## 7. Boundaries

The ETL front door must not:

- call AI models,
- run River/MinHash/graph promotion inside raw write transactions,
- move/delete sovereign source material without explicit instruction,
- treat extracted text as truth,
- write canonical graph materialization.

The ETL front door may:

- preserve bytes,
- identify structure,
- extract safe text/metadata,
- enqueue derived work,
- produce receipts,
- mark unsupported work deferred.

## 8. Current Source Material

This file consolidates:

- `00_PROJECT_BRAIN/ETL_PIPELINE/DBOS_CONTRACT.md`
- `00_PROJECT_BRAIN/ETL_PIPELINE/DB_MAPPING.md`
- `00_PROJECT_BRAIN/ETL_PIPELINE/LIVE_INGEST_NOTES.md`
- `00_PROJECT_BRAIN/ETL_PIPELINE/PLAN.md`
- `00_PROJECT_BRAIN/ETL_PIPELINE/WORKER_POLICY.md`
- `06_SCHEMA/023_etl_pipeline.sql`
- `01_REPOS/lucidota_etl/`

Those files remain source/reference until merged, archived, or superseded by receipts.
