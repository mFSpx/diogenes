# ETL Pipeline Worker Policy

## Default coordination layer

Postgres/DBOS remains the language-neutral control plane.

Rust ETL and Python enrichment coordinate through:

- `lucidota_etl.pipeline_run`
- `lucidota_etl.object_trace`
- `lucidota_etl.step_checkpoint`
- `lucidota_korpus.derived_compute_queue`
- `lucidota_control.workflow_event`

No Redis/Kafka/gRPC is required for the Rust↔Python boundary.

## Apalis placement

Apalis is acceptable only for Rust-native internal ETL work.

Use Apalis for:

- bounded Rust video/container demux jobs
- archive member expansion jobs
- SQLite/table flatten jobs
- document metadata/text extraction jobs when they remain fully Rust-native
- retryable Rust tasks where the payload/state is still checkpointed to Postgres

Do not use Apalis as the cross-language broker.

Do not use Apalis for:

- Python River ML workers
- Whisper/OCR Python workers
- graph promotion Python workers
- any queue that must be consumed by both Rust and Python

## Rule

```text
Apalis = Rust internal execution engine.
Postgres = system control plane and durable queue.
Python = SKIP LOCKED consumer for adaptive enrichment.
```

If Apalis is added, its job state must still be mirrored/checkpointed into Postgres so post-mortem recovery stays DBOS-native.
