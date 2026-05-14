# DIOGENES Block-Out v0.2 Reconciliation

## Status

Architecture intake note created from the `DIOGENES BLOCK-OUT v0.2` brainstorm. This file exists to prevent brainstorm content from being accidentally promoted into canonical architecture without comparison against the project brain.

## Source Of Truth Rule

Order of authority for architecture updates:

1. Explicit locked decisions from `00_PROJECT_BRAIN/DECISIONS.md`.
2. Verified current state from `00_PROJECT_BRAIN/STATUS.md`, schemas, tests, and working commands.
3. Living plan from `00_PROJECT_BRAIN/THEPLAN.md`.
4. Records Office architecture notes.
5. New brainstorm material, which must be classified before adoption.

Brainstorm names are not decisions. Every new component must be marked `confirmed`, `candidate`, `conflict`, `rename`, or `defer`.

## Existing Architecture Before This Brainstorm

Confirmed before v0.2 intake:

- Operator: `northern.strike`.
- Assistant/persona: `LOCAL_READS`.
- Umbrella workspace: `/home/mfspx/LUCIDOTA`; `/home/mfspx/DIOGENES` is a compatibility symlink.
- Interface shell: Clawd / `claudecode`, owned inside LUCIDOTA.
- Kernel: CKDOG1 / `doggystyle`, kernel-only repo.
- Bridge: gRPC, already smoke-tested through Rust tonic/prost client bindings to the Python CKDOG1 gRPC server.
- Workflow control: DBOS.
- Database substrate: PostgreSQL 18.3 with separate `lucidota_state` and `lucidota_graph` DBs.
- Graph/vector substrate: Apache AGE 1.7.0 and pgvector 0.8.2 verified in `lucidota_graph`.
- Artifact storage direction: encrypted local file vault / invisible CAS, not S3-first.
- Originals: immutable.
- Security-sensitive material: encrypted.
- Live learning stack: Bytewax / River / Treelite planned; not fully integrated.
- Runtime posture: local models are stateless compute heads; graph/database/vault are memory.

## Most Important Correction

The brainstorm mentioned `MinIO`, but the existing locked storage decision is:

```text
encrypted local file vault / invisible CAS, not S3-first
```

Therefore:

```text
MinIO is NOT canonical architecture right now.
```

Correct classification:

- `MinIO`: candidate/deferred only; possible future S3-compatible adapter if the CAS vault needs an object-store backend.
- `CAS vault`: confirmed storage direction and source of truth for raw/derived artifacts.
- `PostgreSQL`: confirmed metadata/state substrate.
- `Apache AGE`: confirmed relationship/provenance graph substrate.
- `pgvector`: confirmed embedding/vector substrate.

## Corrected Storage Model

Canonical current storage shape:

```text
Raw / derived artifact bytes
  -> encrypted local CAS vault
  -> immutable content address / hash

Artifact metadata
  -> PostgreSQL / lucidota_graph
  -> path/ref, hash, MIME, size, timestamps, source, workflow/run ids, encryption metadata

Provenance / ontology / relationships
  -> Apache AGE in lucidota_graph
  -> captured_from, derived_from, cites, contradicts, supersedes, same_as, belongs_to_case

Embeddings / semantic search
  -> pgvector in lucidota_graph
  -> chunk vectors, source refs, retrieval indexes

DBOS workflow state
  -> PostgreSQL / lucidota_state
```

Do not insert MinIO, S3, cloud storage, or object-store assumptions into the main path unless northern.strike explicitly promotes that storage backend.

## DBOS + Bytewax Team Model

The brainstorm correctly sharpened the relationship between DBOS and Bytewax. Reconcile it like this:

```text
DBOS = durable workflow command brain
Bytewax = live dataflow / event coordination nervous system
River = online reflex learning
Treelite = compiled routing/reflex inference
```

Confirmed from existing plan:

- DBOS controls workflows, retries, state transitions, and repeatable task units.
- Bytewax is for live dataflows.
- River is for online learning.
- Treelite is for deployable tree inference.

New clarified contract from v0.2:

```text
DBOS authorizes durable work.
Bytewax coordinates high-volume live streams, ordering, fan-in/fan-out, jitter, and backpressure.
DBOS commits durable state.
Bytewax/River emit hints and scores; DBOS policy/gates decide whether to use them.
```

Constraint:

```text
No irreversible side effect without DBOS/governance authorization.
No high-volume live event path without Bytewax-style flow control once that layer exists.
```

## Brainstorm Item Classification

### Confirmed / Compatible

- `Clawd`: compatible with existing Clawd/claudecode interface shell.
- `Rust`: compatible; Clawd is Rust and Rusty deterministic tools are consistent with project direction.
- `gRPC`: confirmed bridge boundary.
- `tonic`: confirmed on Rust side by current Clawd smoke.
- `Python`: confirmed for CKDOG1/DBOS/local orchestration.
- `DBOS`: confirmed workflow control layer.
- `PostgreSQL`: confirmed substrate.
- `pgvector`: confirmed vector extension.
- `Apache AGE`: confirmed graph extension.
- `Pydantic`: compatible membrane for Python validation; not yet locked as the only schema layer.
- `Bytewax`: confirmed live dataflow layer, integration pending.
- `River ML`: confirmed online learning layer, integration pending.
- `Treelite`: confirmed deployable tree inference layer, integration pending.
- `DeepSeek-R1-Distill-Qwen-1.5B`: compatible with existing `deepseek-1.5b-local_reads-reads` runtime registry target, exact artifact still TBD.
- `Bloom Filter`: compatible as fast seen/check structure; not yet implemented.
- `SHA-256`: compatible with hash/proof/vault direction.
- `Playwright`: compatible candidate for dynamic extraction/capture; not yet locked.

### Rename / Align With Existing Names

- `nomad_surface.proto`: conflicts with current verified `kernel.proto` / CKDOG1 gRPC surface unless explicitly renamed later. Treat as brainstorm name, not canonical.
- `The Enseminator`: unresolved operator/control-surface codename; must not replace Clawd/LOCAL_READS naming without decision.
- `DBOS Python Brain`: align with DBOS workflow plane plus CKDOG1 kernel, not a replacement for CKDOG1 ontology meaning.
- `Unified Vault`: align with Records Office + encrypted CAS vault + Postgres/AGE/pgvector, not MinIO-first.
- `Village Sideload`: compatible with existing Village/Villagers concept, but storage shape remains open.

### Candidate / Needs Design

- `SerScrapist`: candidate name/subsystem for extraction ladder. Compatible with Survey/ingest/scraper policy needs; not yet canonical code.
- `Body Capture Head`: candidate name/subsystem for capture/evidence/diff pipeline. Compatible if backed by CAS vault and governance.
- Hop/pivot search functionality formerly nicknamed `Superglow Spider`: PROMOTED as required capability, but the name is rejected. Final subsystem name pending. It covers bounded hop expansion, pivot discovery, promotion scoring, and source-policy-aware search.
- `Survey Protocol`: PROMOTED as required capability for first-contact URL/file/endpoint triage.
- `Krampus / Audit Workflow`: compatible with glossary `Krampus Express`; exact workflow pending user specification.
- `multi-pattern`: PROMOTED as required lightweight exact-keyword scanning algorithm for hop/pivot search and ultra-light scans.
- `Tree-sitter`: PROMOTED as required structural parsing/anomaly tool where language/HTML structure matters.
- `Wayback Machine`: PROMOTED as required archival evidence source, behind source policy/rate limits/governance.
- `XGBoost`: TRAINING/EXPORT ONLY if used. NEVER runtime. Runtime inference must stay light; Treelite is the compiled deployment path.
- `Teleoperation Recorder`: remains candidate for human demonstration capture; format unresolved.

### Conflict / Do Not Promote Yet

- `MinIO`: conflicts with locked `encrypted local file vault / invisible CAS, not S3-first` if treated as primary storage. Keep only as deferred possible backend adapter.
- `betterproto vs grpcio`: current verified bridge already uses Python gRPC server and Rust tonic/prost. Do not replace proto stack casually.
- Any architecture that makes Bytewax the durable source of truth: conflict. Bytewax may coordinate streams; DBOS/Postgres remain durable truth.

## Corrected Master Flow

Current coherent target, using existing names:

```text
northern.strike
  -> LOCAL_READS inside Clawd / claudecode
  -> Rust tonic/prost gRPC boundary
  -> CKDOG1 / doggystyle kernel
  -> DBOS workflows
  -> PostgreSQL state + graph DBs
  -> encrypted local CAS vault for artifacts
  -> Bytewax / River / Treelite live learning path
  -> model runtime as stateless compute heads
  -> back to LOCAL_READS / Clawd
```

Workflow/event loop:

```text
DBOS durable command
  -> tool/model/crawler/capture/workflow action
  -> event emitted
  -> Bytewax normalizes/orders/enriches live stream
  -> River updates online scores
  -> DBOS receives bounded hints through governance/policy gates
  -> periodic model export can refresh Treelite router
```

Storage loop:

```text
Object bytes
  -> encrypted CAS vault
  -> hash/ref metadata in PostgreSQL
  -> provenance and ontology edges in Apache AGE
  -> chunks/embeddings in pgvector when useful
```


## Promotion Update From northern.strike

Promoted from v0.2 brainstorm into required architecture capabilities:

- Bounded hop/pivot search functionality: required, but not under the `Superglow Spider` name. Rename pending.
- Survey Protocol: required first-contact triage layer.
- multi-pattern: required lightweight exact-match scan path.
- Tree-sitter: required structural parse/anomaly path.
- Wayback Machine: required archival evidence source with source policy/rate limits.

Explicit runtime constraint:

```text
XGBoost is never runtime.
Treelite is the light compiled inference path.
XGBoost may only appear as an offline trainer/export source if it earns that role.
```

Product principle:

```text
The system optimizes for being light, local, fast, inspectable, and better than bloated mainstream stacks.
Do not settle for heavyweight/mid architecture when a sharper compiled/deterministic path exists.
```

## Open Decisions Added By This Brainstorm

- Exact local encryption stack for the CAS vault.
- Exact CAS on-disk layout and metadata schema.
- Whether MinIO is ever useful as an optional object-store backend after local CAS exists.
- Exact Bytewax/DBOS command-event-hint schema.
- Exact River feature schema.
- Exact Treelite refresh cadence and whether XGBoost is the offline trainer. XGBoost is explicitly never runtime.
- Exact crawler/extractor ladder names: SerScrapist/Body Capture/Superglow/Survey or simpler names.
- Exact dynamic capture backend: Playwright/CDP path.
- Exact embedding model and vector dimensions.
- Exact teleoperation recording format.

## Assistant Operating Correction

When northern.strike provides a dense architecture dump, the assistant must:

1. Load existing project brain and relevant records first.
2. Identify locked decisions before interpreting the dump.
3. Classify each brainstorm element as confirmed, candidate, conflict, rename, or defer.
4. Never promote a brainstorm component just because it appears in the dump.
5. Patch project records only after preserving the diff and status.

## Wake Bus Canon — Section V Event Stream

Confirmed placement: the Wake Bus sits between durable DBOS/Postgres commits and Bytewax/River/Big Board readers.

Flow:

```text
DBOS workflow commit
  -> lucidota_control.workflow_event
  -> lucidota_control.event_outbox
  -> local wake signal with IDs/refs only
  -> Bytewax / River / Watchers / Big Board rebody_capturete from Postgres/Vault
```

Rules:

- Postgres remains truth.
- Wake rows carry only IDs, hashes, and event refs.
- No blobs and no credential material in wake refs.
- Missed wakeups are harmless because readers replay from `event_outbox` / `workflow_event`.
- Start local with Postgres outbox/signals; external brokers are deferred until a measured need exists.
