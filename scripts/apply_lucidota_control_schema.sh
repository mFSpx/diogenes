#!/usr/bin/env bash
set -euo pipefail

DB_URL="${LUCIDOTA_CONTROL_DATABASE_URL:-postgresql://mfspx@/lucidota_state}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

psql "$DB_URL" -v ON_ERROR_STOP=1 -f "$ROOT/06_SCHEMA/001_lucidota_control.sql"
psql "$DB_URL" -v ON_ERROR_STOP=1 -f "$ROOT/06_SCHEMA/002_model_runtime.sql"
