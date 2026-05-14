#!/usr/bin/env bash
set -euo pipefail

DB_URL="${LUCIDOTA_CONTROL_DATABASE_URL:-postgresql://mfspx@/lucidota_state}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

psql "$DB_URL" -v ON_ERROR_STOP=1 -f "$ROOT/06_SCHEMA/001_lucidota_control.sql"
psql "$DB_URL" -v ON_ERROR_STOP=1 -f "$ROOT/06_SCHEMA/002_model_runtime.sql"
psql "$DB_URL" -v ON_ERROR_STOP=1 -f "$ROOT/06_SCHEMA/004_learning_reflex.sql"
psql "$DB_URL" -v ON_ERROR_STOP=1 -f "$ROOT/06_SCHEMA/005_cas_manifest.sql"
psql "$DB_URL" -v ON_ERROR_STOP=1 -f "$ROOT/06_SCHEMA/006_workflow_registry.sql"
psql "$DB_URL" -v ON_ERROR_STOP=1 -f "$ROOT/06_SCHEMA/007_bytewax_stream.sql"
psql "$DB_URL" -v ON_ERROR_STOP=1 -f "$ROOT/06_SCHEMA/008_hop_pivot.sql"
psql "$DB_URL" -v ON_ERROR_STOP=1 -f "$ROOT/06_SCHEMA/009_treelite_router.sql"
psql "$DB_URL" -v ON_ERROR_STOP=1 -f "$ROOT/06_SCHEMA/010_wake_bus.sql"
psql "$DB_URL" -v ON_ERROR_STOP=1 -f "$ROOT/06_SCHEMA/014_indy_runtime.sql"
