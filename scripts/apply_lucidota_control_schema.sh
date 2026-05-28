#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Pin the canonical PostgreSQL target before any schema work.
if [[ -f "$ROOT/scripts/lucidota_pg_user_env.sh" ]]; then
  # shellcheck disable=SC1090
  source "$ROOT/scripts/lucidota_pg_user_env.sh"
fi

DB_URL="${LUCIDOTA_CONTROL_DATABASE_URL:-postgresql://mfspx@/lucidota_state}"

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
psql "${LUCIDOTA_GO_STORAGE_DSN:-postgresql:///lucidota_storage}" -v ON_ERROR_STOP=1 -f "$ROOT/06_SCHEMA/005_cas_manifest.sql"
psql "${LUCIDOTA_GO_STORAGE_DSN:-postgresql:///lucidota_storage}" -v ON_ERROR_STOP=1 -f "$ROOT/06_SCHEMA/016_go_graph_core.sql"
psql "${LUCIDOTA_GO_STORAGE_DSN:-postgresql:///lucidota_storage}" -v ON_ERROR_STOP=1 -f "$ROOT/06_SCHEMA/017_indy_reads_library.sql"
psql "${LUCIDOTA_GO_STORAGE_DSN:-postgresql:///lucidota_storage}" -v ON_ERROR_STOP=1 -f "$ROOT/06_SCHEMA/018_investigation_artifact.sql"
psql "${LUCIDOTA_GO_STORAGE_DSN:-postgresql:///lucidota_storage}" -v ON_ERROR_STOP=1 -f "$ROOT/06_SCHEMA/123_absurd_flows.sql"
psql "${LUCIDOTA_GO_STORAGE_DSN:-postgresql:///lucidota_storage}" -v ON_ERROR_STOP=1 -f "$ROOT/06_SCHEMA/121_capability_factory.sql"
psql "${LUCIDOTA_GO_STORAGE_DSN:-postgresql:///lucidota_storage}" -v ON_ERROR_STOP=1 -f "$ROOT/06_SCHEMA/019_korpus_krampii.sql"
psql "${LUCIDOTA_GO_STORAGE_DSN:-postgresql:///lucidota_storage}" -v ON_ERROR_STOP=1 -f "$ROOT/06_SCHEMA/020_chat_dump_timeline.sql"
psql "${LUCIDOTA_GO_STORAGE_DSN:-postgresql:///lucidota_storage}" -v ON_ERROR_STOP=1 -f "$ROOT/06_SCHEMA/021_hard_truth_math.sql"
psql "${LUCIDOTA_GO_STORAGE_DSN:-postgresql:///lucidota_storage}" -v ON_ERROR_STOP=1 -f "$ROOT/06_SCHEMA/022_comm_dump_timeline.sql"
psql "$DB_URL" -v ON_ERROR_STOP=1 -f "$ROOT/06_SCHEMA/122_resource_governor.sql"
psql "$DB_URL" -v ON_ERROR_STOP=1 -f "$ROOT/06_SCHEMA/120_diogenes_mirror.sql"
