# ETL Pipeline — Notes From The Live Ingest

Date: 2026-05-16 America/Vancouver

This is the correction file: do not design the future front door in abstraction. It must fit the system that is already built and the failure modes already observed.

## Current built system facts

- Canonical live dashboard: `05_OUTPUTS/ingestion_observation_dashboard.html`
- Python KORPUS raw ingest script: `scripts/korpus_krampii.py`
- Derived queue migration: `06_SCHEMA/020_korpus_derived_compute_queue.sql`
- Derived queue worker scaffold: `scripts/korpus_derived_compute_worker.py`
- KORPUS schema: `06_SCHEMA/019_korpus_krampii.sql`
- CAS schema: `06_SCHEMA/005_cas_manifest.sql`
- Current raw ingest result: `05_OUTPUTS/korpus_krampii/20260515T231055.korpus.results.jsonl`

## Latest completed ingest run

Label: `krampuschewing-drop-resume-pure-storage`

Result:

- status: succeeded
- roots: `/home/mfspx/LUCIDOTA/KRAMPUSCHEWING/Lucidota`
- observed paths: 24,127
- paths after resume skip: 23,912
- skipped existing paths: 215
- files persisted: 23,906
- new files: 15,690
- duplicate files: 8,216
- components: 120,788
- entities: 494,643
- concepts: 327,764
- errors: 6
- skipped: 0
- elapsed: ~8,910 seconds
- river mode: light
- near-dup mode: off
- graph mode: off

## Actual failure chain observed

The original watcher treated a giant directory drop as a synchronous heroic mega-pipeline:

```text
watch folder
→ wait stable
→ maybe chatdump
→ maybe commdump
→ brain sidecar
→ KORPUS hard ingest
→ rechrono
→ hard truth math
→ move to DIGESTED only if all green
```

For directories, the original KORPUS call forced:

```text
--chronological --workers 1
```

That created a single-file-order bottleneck and then put derived compute inside the hot transaction lane.

Observed traps:

1. River ML pickled model blob was loaded inside the DB transaction path.
2. MinHash near-duplicate comparison ran synchronously per component.
3. Graph promotion wrote graph items/edges synchronously per file/component/entity/concept.
4. Autovacuum and checkpoint pressure fought the ingest under heavy I/O.
5. Dashboard mixed current failures with historical scars until corrected.

## Corrected operational contract

The corrected live contract is:

```text
raw storage first
derived compute later
queue everything expensive
no single derived task can hold raw ingest hostage
```

During bulk ingest:

- River replay: off/light only
- Near duplicate scan: off
- Graph promotion: off
- Embeddings: queued/paused/throttled
- DB locks: watched
- Autovacuum: cancel only if I/O critical

After raw ingest:

- Backfill `lucidota_korpus.derived_compute_queue`
- Run lease-based workers with `FOR UPDATE SKIP LOCKED`
- Enrich slowly and restartably

## What the ETL Pipeline must become

The ETL Pipeline is not a replacement for Python cognition. It is the future front-door evidence hydraulic press.

Rust owns stable deterministic high-volume work:

- recursive walk
- archive expansion
- hashing
- CAS locking
- MIME/magic sniffing
- text extraction for standard formats
- media/container metadata
- EXIF timestamps/GPS
- SQLite row flattening
- COPY/transactional DB write
- enqueue derived tasks

Python keeps volatile/adaptive work:

- River ML replay
- MinHash clustering strategy changes
- graph promotion policy
- OCR and Whisper/STT
- weird one-off repair parsers
- AI/agent workflow layers

## Non-negotiables for ETL Pipeline

1. Do not wire into live watcher until proven offline.
2. Do not replace Python KORPUS mid-ingest.
3. Output must match existing Postgres surfaces, not invent a fantasy schema.
4. Every source byte is CAS-locked before derived interpretation.
5. Archives/databases create child evidence records with provenance.
6. Derived compute is enqueued in Postgres, never performed in the raw storage transaction.
7. Failures are row-level/dead-letter, not process-level explosions.
8. Operator can run it as a one-shot command before any daemonization.
