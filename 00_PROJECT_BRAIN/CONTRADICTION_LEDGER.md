# CONTRADICTION LEDGER

Conflicts are listed as local truth gaps, not as drama. Each one gets a receipt-backed repair path.

## legacy_manual_skeletons
- Conflict: 05_OUTPUTS/runtime/manuals and api/html files were initial skeletons, while the new canon must live in 00_PROJECT_BRAIN.
- Evidence:
  - 05_OUTPUTS/runtime/manuals/runpod_forge_manual.md
  - 05_OUTPUTS/runtime/api/runpod_artifact_api.md
  - 05_OUTPUTS/runtime/html/runpod_forge_dashboard.html
- Fix: Write final manuals into 00_PROJECT_BRAIN and treat the 05_OUTPUTS copies as legacy output artifacts.

## model_ledger_role_gap
- Conflict: The admission sidecar describes role tags like ingress/egress/classifier/extractor, but the live SQLite/Postgres runtime ledger currently exposes role values like listener/router/heavy_hitter/embedding/reranker/other.
- Evidence:
  - GOALS/MODEL_FABRIC_ADMISSION_SIDECAR.md
  - 06_SCHEMA/002_model_runtime.sql
- Fix: Document the actual ledger roles in API/RUNTIME manual and keep the sidecar as a desired contract, not a false claim.

## bonsai_pid_mismatch
- Conflict: The latest model-fabric status shows a healthy Bonsai endpoint but a dead/stale pid field.
- Evidence:
  - 05_OUTPUTS/goals/goal_model_fabric_control_20260603T021934Z.json
- Fix: Describe this as endpoint-health truth with stale process metadata, not as a live resident guarantee.

## queue_uuid_placeholder_bug
- Conflict: The queue pipeline previously inserted a literal placeholder string into UUID columns.
- Evidence:
  - scripts/conductor_hierarchy_fanout.py
  - 05_OUTPUTS/conductor_hierarchy/conductor_hierarchy_receipt_20260603T021148Z.json
- Fix: Use RETURNING job_uuid::text and commit only real DB-generated UUIDs.

## remote_compact_path
- Conflict: Remote compact prompts are no longer the right route; the lean bootstrap download completed with a custody receipt instead.
- Evidence:
  - 05_OUTPUTS/runpod/talkie_book_lora/remote_talkie_source_custody.json
  - 05_OUTPUTS/runpod/talkie_book_lora/lean_talkie_download.log
- Fix: Keep the remote lane chunked and bootstrap-only; no giant compactor payloads.

## Open status
- The current pass fixes the canonical manual surface and the queue UUID path.
- Remaining runtime contradictions are intentionally documented rather than hidden.
