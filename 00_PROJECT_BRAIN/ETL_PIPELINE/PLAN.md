# ETL Pipeline Implementation Plan

## Phase 0 — Sidecar only, no live integration

Status: current.

- Keep Python KORPUS as current proven path.
- Keep derived queue dormant unless explicitly backfilled.
- Build ETL Pipeline as offline one-shot command producing JSONL first.

## Phase 1 — Offline JSONL ETL Pipeline MVP

Command:

```bash
cargo run --release -- scan /path/to/drop --jsonl 05_OUTPUTS/etl_pipeline/<stamp>.jsonl --cas-root 03_VAULT/cas
```

Must do for every file:

- walk recursively
- hash SHA-256 and BLAKE3
- CAS lock bytes to existing CAS shape
- infer MIME/file_kind
- emit `UnifiedMetadata` JSONL
- recurse into ZIP/TAR/GZ
- flatten SQLite schema/rows
- extract plain text/CSV/JSON text directly
- mark unsupported docs/media as structurally identified with derived tasks

Exit rule: bad member/file becomes one failed record, not process failure.

## Phase 2 — Postgres writer, still not watcher-integrated

Add command:

```bash
etl_pipeline ingest /path/to/drop --case "KORPUS KRAMPII" --label rust-etl_pipeline-smoke --limit 1000
```

Writes:

- `ingest_batch`
- `file_object`
- `cas_manifest`
- `file_occurrence`
- `component`
- `derived_compute_queue`

Use transaction batches and COPY where possible. Start with normal prepared inserts until semantics are correct; optimize COPY after proof.

## Phase 3 — Full hardwired lanes

Hardwire stable extractors:

- documents: PDF, DOCX, XLSX/ODS, CSV, JSON, JSONL, Parquet
- media: JPEG/TIFF/PNG EXIF, audio tags/duration, video container metadata
- archives: ZIP, TAR, GZ/TGZ, nested recursion with depth/size caps
- databases: SQLite schema/table/row flattening

## Phase 4 — Watcher replacement

Only after Phase 1-3 prove on archived drops:

```text
KRAMPUSCHEWING watcher
→ ETL Pipeline raw storage
→ derived_compute_queue
→ Python enrichment workers
→ dashboard
```

No old synchronous River/MinHash/Graph path returns to the front door.

## Rust-native worker note

Apalis may be used later inside the Rust ETL Pipeline for Rust-only background work such as archive expansion, SQLite flattening, and video demux metadata. It is not the system broker. Postgres remains the DBOS control plane and language-neutral queue.

## Guardrails

- Max archive recursion depth default: 6
- Max extracted member size default: 512 MiB
- Max text per component default: 8 MiB
- Every failed extraction emits a failure record
- Every queue handoff is idempotent via unique `(task_type,target_table,target_uuid)`
- No AI calls in ETL Pipeline
- No River/MinHash/Graph enrichment inside raw write transaction
