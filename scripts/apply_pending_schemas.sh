#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/lucidota_safe_ops_env.sh"

apply_schema() {
  local DSN=$1
  local FILE=$2
  psql "$DSN" -f "$FILE" 2>&1
  if [ $? -eq 0 ]; then
    echo "APPLIED"
  else
    echo "FAILED"
  fi
}

# Core Rust-migration gaps that still need to exist in the live DB.
apply_schema lucidota_state "$ROOT/06_SCHEMA/045_document_ingestion_pipeline.sql"
apply_schema lucidota_state "$ROOT/06_SCHEMA/046_catchme_sensitivity_map.sql"
apply_schema lucidota_state "$ROOT/06_SCHEMA/050_document_claim_packet_bridge.sql"
apply_schema lucidota_state "$ROOT/06_SCHEMA/094_workflow_foundry_runtime.sql"

# Already-queued post-core state migrations.
apply_schema lucidota_state "$ROOT/06_SCHEMA/136_evolution_spine.sql"
apply_schema lucidota_state "$ROOT/06_SCHEMA/137_unified_model_routing_policy.sql"
apply_schema lucidota_state "$ROOT/06_SCHEMA/138_worker_contract_versions.sql"
apply_schema lucidota_state "$ROOT/06_SCHEMA/139_corpse_event.sql"
apply_schema lucidota_state "$ROOT/06_SCHEMA/140_found_secrets.sql"
apply_schema lucidota_state "$ROOT/06_SCHEMA/141_corpse_found_secret_link.sql"
apply_schema lucidota_state "$ROOT/06_SCHEMA/142_feral_and_phenotype_registries.sql"
apply_schema lucidota_state "$ROOT/06_SCHEMA/143_model_startup_receipt.sql"
