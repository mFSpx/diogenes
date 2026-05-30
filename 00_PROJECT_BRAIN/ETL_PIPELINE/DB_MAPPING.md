# ETL Pipeline → Existing LUCIDOTA DB Mapping

ETL Pipeline must write to the system we already have.

## Primary tables

### `lucidota_korpus.ingest_batch`

One row per ETL Pipeline run.

- `batch_label`: e.g. `rust-etl_pipeline-drop`
- `status`: `running|succeeded|failed|cancelled`
- `root_paths`: JSON array of roots
- `options`: JSON with etl_pipeline version, lanes, recursion limits
- counts: updated at end
- `detail`: result paths, elapsed seconds, lane counts

### `lucidota_korpus.file_object`

One row per unique byte object.

ETL Pipeline fields:

- `sha256` ← content SHA-256
- `size_bytes` ← raw bytes length
- `mime` ← `infer`/magic/MIME guess
- `file_kind` ← `text|code|document|image|audio|video|archive|database|binary|unknown`
- `status` ← `indexed|deferred|error`
- `deferred_reason` ← only if raw storage succeeded but lane extraction deferred
- `suffix` ← lower-case extension
- `cas_uri` ← `cas://sha256/<sha256>`
- `locked_relative_path` ← `03_VAULT/cas/aa/bb/<sha256>`
- `first_seen_path` ← original or virtual path
- `minhash_signature` ← optional, may be empty in raw mode
- `detail` ← `UnifiedMetadata.structural_payload`

### `lucidota_vault.cas_manifest`

ETL Pipeline must mirror CAS state.

- `sha256`
- `cas_uri`
- `relative_path`
- `size_bytes`
- `mime`
- `source='rust_etl_pipeline'`

### `lucidota_korpus.file_occurrence`

One row per path/member occurrence.

For archive/database children:

- `absolute_path`: virtual path, e.g. `/drop/foo.zip!bar/baz.pdf`
- `relative_path`: relative virtual path
- `root_path`: physical root drop
- `detail`: parent sha256, archive/database lineage, member index/table/row

### `lucidota_korpus.component`

Rows produced from extracted text or flattened records.

Examples:

- text file: one or more text components
- PDF/docx: text components + metadata component
- XLSX/CSV/Parquet: sheet/table/row-group components
- SQLite: schema components and row-batch components
- image/audio/video: metadata components; OCR/STT enqueued separately

Minimum component mapping:

- `file_uuid`
- `component_index`
- `component_kind`
- `language`
- `title`
- `symbol`
- `start_line`, `end_line`
- `sha256` ← SHA-256 of component content
- `token_count` ← cheap whitespace count
- `content` ← normalized extracted text / row JSON text
- `go_terms` ← basic deterministic route, can be `[]` in raw ETL Pipeline MVP
- `minhash_signature` ← optional raw/minhash lane; can be `[]` if queued
- `entropy`
- `detail` ← lane-specific structural metadata

## Derived queue handoff

ETL Pipeline does not run expensive interpretation. It inserts into:

`lucidota_korpus.derived_compute_queue`

Task types already scaffolded:

- `river_replay_component`
- `near_duplicate_scan`
- `graph_promote_file`
- `graph_promote_component`
- `deferred_parse`
- `media_deep_extract`

Rules:

- any component with content can get `river_replay_component`
- any component with minhash can get `near_duplicate_scan`
- any file/component can get graph promotion tasks
- image/video/PDF pages needing OCR get `media_deep_extract`
- unsupported/broken formats get `deferred_parse`

## Unified Rust struct

```rust
pub struct UnifiedMetadata {
    pub file_hash_sha256: String,
    pub file_hash_blake3: String,
    pub mime_type: String,
    pub file_kind: String,
    pub cas_uri: String,
    pub locked_relative_path: String,
    pub source_path: String,
    pub virtual_path: String,
    pub parent_sha256: Option<String>,
    pub extracted_text: Option<String>,
    pub event_timestamp: Option<DateTime<Utc>>,
    pub geo_location: Option<(f64, f64)>,
    pub structural_payload: serde_json::Value,
    pub derived_tasks: Vec<DerivedTask>,
}
```
