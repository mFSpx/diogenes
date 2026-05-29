# ETL Pipeline DBOS Contract

The database is the operating system.

The ETL Pipeline and Python sandbox do not need REST, gRPC, Redis, Kafka, or shared memory. They coordinate through Postgres tables only.

## Roles

### Rust ETL Pipeline

Stable compiled front door for incoming data.

Responsibilities:

- recursively discover input objects
- hash bytes
- CAS lock bytes
- identify MIME/file kind
- extract stable structural metadata/text from common formats
- write raw storage rows
- write durable checkpoint rows
- enqueue derived compute rows
- exit without holding derived work hostage

### Python sandbox

Volatile adaptive enrichment lane.

Responsibilities:

- OCR
- Whisper/STT
- River ML replay
- MinHash clustering policy
- graph promotion policy
- weird dirty parser repair
- agent/AI workflows

## DBOS handoff rule

The ETL Pipeline writes primary evidence and task rows in the same transaction:

```text
BEGIN
  INSERT/UPSERT raw evidence rows
  INSERT/UPSERT CAS manifest rows
  INSERT/UPSERT component rows
  INSERT/UPSERT derived_compute_queue rows
  INSERT ETL checkpoint rows
COMMIT
```

After commit, Python workers can claim queue rows with:

```sql
FOR UPDATE SKIP LOCKED
```

No direct process coupling is allowed.

## Durable Python step rule

Every expensive Python task is split into named steps. Each step writes a checkpoint to Postgres.

Example OCR flow:

```text
media_deep_extract
  step 1: extract_image_or_page_frames
  step 2: run_ocr
  step 3: normalize_text_component
  step 4: enqueue_river_and_graph
```

If the process dies during step 3, step 1 and 2 outputs remain in Postgres/CAS. Recovery resumes at step 3.

## Post-mortem rule

Logs are secondary. Truth is queryable state:

- `lucidota_etl.pipeline_run`
- `lucidota_etl.object_trace`
- `lucidota_etl.step_checkpoint`
- `lucidota_korpus.derived_compute_queue`
- `lucidota_control.workflow_event`

A stuck object must be explainable with SELECT statements.

## Queue status vocabulary

Keep it boring:

```text
queued
running
succeeded
failed
dead
```

Failures are records, not mysteries.
