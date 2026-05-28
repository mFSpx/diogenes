# DIOGENES Restore Guide

This document describes how to reconstruct the LUCIDOTA machine from the source-backed DIOGENES mirror.

## What the mirror restores

- repo wiring and control surfaces
- scripts, schemas, algorithms, prompts, docs, tests, and handoffs
- canonical Postgres schema/migration state
- registry rows for code surfaces
- receipts and manifests that explain how to rebuild the system

## What the mirror does not restore

- bulk training data
- binaries and model weights
- `.venv/`, `node_modules/`, and other local build caches
- PostgreSQL data directories and runtime state
- logs, dumps, and large generated exhaust

## Restore steps

1. Check out the source tree.
2. Install the official OS packages needed for the wiring:
   - PostgreSQL 16 client/server
   - `postgresql-16-pgvector`
   - `postgresql-16-age`
   - Python tooling, Rust toolchain, Node/npm, git, and `gh` if remote publication is desired
3. Recreate the canonical PostgreSQL environment:
   - source `scripts/lucidota_pg_user_env.sh`
   - verify `PGHOST=/var/run/postgresql`
   - verify `PGPORT=5432`
   - verify `psql` resolves to the system PostgreSQL 16 binary
4. Apply the schema stack:

```bash
bash scripts/apply_lucidota_control_schema.sh
```

5. Regenerate the backup manifest if needed:

```bash
.venv/bin/python scripts/diogenes_mirror.py --json-out 05_OUTPUTS/diogenes/diogenes_mirror.json
```

6. Re-run targeted tests for the restored wiring.
7. Rebuild only the missing local state that is not intentionally excluded by the mirror policy.

## GitHub publication

If GitHub auth is available, point the current tree at the private `mFSpx/diogenes` remote and push the source-only mirror.
If auth is not available, keep the local commit and write a blocker receipt; do not upload secrets or bulk exhaust.

## Practical rule

The mirror is a source-of-truth backup, not a data lake.
The system becomes by reapplying schema, registry, and runtime wiring — not by restoring excluded exhaust.

## Current mirror receipt

- Dry-run mirror receipt: `05_OUTPUTS/diogenes/diogenes_mirror_20260528T064705282932Z.json`
- Secret scan receipt: `05_OUTPUTS/security/security_quarantine_manifest_20260528T064611Z.json`
