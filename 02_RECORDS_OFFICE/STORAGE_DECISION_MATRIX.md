# DIOGENES / LUCIDOTA Storage Decision Matrix

Updated: 2026-05-13

## Canonical Storage Roles

| Data class | Canonical store | Why |
|---|---|---|
| Workflow events / run state | PostgreSQL via DBOS/control schemas | Durable, inspectable, transactional local state. |
| Source metadata / observations | PostgreSQL | Typed rows, constraints, joins, auditability. |
| Provenance graph edges | Apache AGE on PostgreSQL | Graph traversal while staying inside the local Postgres vault. |
| Embeddings / semantic memory | pgvector on PostgreSQL | Local vector search without another database. |
| Artifact bytes / captures / raw evidence | Local SHA-256 CAS under `03_VAULT/cas` | Immutable, content-addressed, easy to hash/verify/export. |
| Online scoring hints | River/Bytewax tables in PostgreSQL | Durable hint stream, not runtime authority. |
| Lightweight routing artifact | Treelite artifact in ignored local vault + Postgres run row | Fast, inspectable, no XGBoost runtime. |

## Non-Canonical / Deferred

| Candidate | Status | Allowed future use |
|---|---|---|
| Cassandra | Not canonical | Only if a distinct append-only, high-volume event/time-series workload outgrows Postgres. Never for blobs, mutable rows, graph, vectors, CAS, or DBOS state. |
| MinIO / S3-compatible storage | Deferred adapter | Only if remote/object-store sync is explicitly needed. Local CAS stays canonical. |

## Current Rule

Keep the system local-first, light, inspectable, and boring where boring wins:

```text
Postgres + pgvector + AGE + local CAS
```

No heavyweight distributed store enters the core path without a measured workload proving it is necessary.
