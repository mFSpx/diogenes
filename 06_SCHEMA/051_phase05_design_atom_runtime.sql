-- FILE: 06_SCHEMA/051_phase05_design_atom_runtime.sql
-- PURPOSE: runtime/idempotency support for deterministic Phase 0.5 design atom extraction.
-- COMPLIANCE: reads allowlisted custody only; writes design_atom candidates only; no graph mutation.

BEGIN;

CREATE SCHEMA IF NOT EXISTS lucidota_archaeology;

CREATE UNIQUE INDEX IF NOT EXISTS ux_phase05_design_atom_source_claim_v1
  ON lucidota_archaeology.design_atom (source_artifact_uuid, extractor_version, md5(normalized_claim));

COMMIT;
