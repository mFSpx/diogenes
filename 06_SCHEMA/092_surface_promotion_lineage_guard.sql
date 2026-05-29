-- FILE: 06_SCHEMA/092_surface_promotion_lineage_guard.sql
-- PURPOSE: Surface promotion lineage idempotency guard.

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_runtime;

ALTER TABLE lucidota_runtime.surface_lineage
  ADD COLUMN IF NOT EXISTS lineage_sha256 text CHECK (lineage_sha256 IS NULL OR lineage_sha256 ~ '^[0-9a-f]{64}$'),
  ADD COLUMN IF NOT EXISTS immutable boolean NOT NULL DEFAULT true;

CREATE UNIQUE INDEX IF NOT EXISTS idx_surface_lineage_sha256
  ON lucidota_runtime.surface_lineage(lineage_sha256) WHERE lineage_sha256 IS NOT NULL;
