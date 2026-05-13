# DBOS / PostgreSQL Topology

## Purpose

Define the local database substrate for LUCIDOTA without smearing responsibilities across the kernel, interface, and workflow layers.

## Current Install Target

- PostgreSQL: 18 from the official PostgreSQL Apt Repository.
- Vector extension: `pgvector` for PostgreSQL 18.
- Graph extension: Apache AGE package for PostgreSQL 18.
- Workflow layer: DBOS Python library, to be added in an isolated project/runtime environment after the database is verified.

## Database Roles

- `lucidota_state`: DBOS workflow state, job history, durable execution metadata, and orchestration bookkeeping.
- `lucidota_graph`: ontology graph, embeddings, claims, evidence refs, document chunks, active village state, retrieval indexes, and derived projections.

The split is intentional. Workflow control and knowledge substrate should be inspectable independently even if they later share one physical Postgres cluster.

## Extension Plan

Enable extensions only in the database that needs them:

- `lucidota_graph`: `vector`, `age`.
- `lucidota_state`: DBOS-required schema only; avoid graph/vector creep unless DBOS requires it.

## Boundary Rules

- `doggystyle` remains the kernel-only repo.
- LUCIDOTA owns the interface fork and orchestration.
- DBOS owns durable workflow execution, not ontology meaning.
- Postgres stores verified state; Markdown does not operate the system.
- External writes remain gated.

## Verification Targets

1. PostgreSQL 18 service/cluster exists and starts.
2. `vector` extension can be created and queried.
3. `age` extension can be created and queried.
4. `lucidota_state` and `lucidota_graph` exist as separate local databases.
5. A minimal DBOS workflow can write/read durable state against `lucidota_state`.

## Notes

Apache AGE for PostgreSQL 18 is currently installed from the PGDG package if available. If that package behaves like an RC or blocks functionality, the fallback is source build from Apache AGE release material for PostgreSQL 18.
