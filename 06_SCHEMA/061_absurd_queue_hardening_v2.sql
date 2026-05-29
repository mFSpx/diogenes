-- FILE: 06_SCHEMA/061_absurd_queue_hardening_v2.sql
-- PURPOSE: ABSURD queue schema hardening: payload hash, attempt bounds, timestamps, failure indexes.

BEGIN;
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_control;

ALTER TABLE lucidota_control.absurd_queue_job
  ADD COLUMN IF NOT EXISTS payload_sha256 text,
  ADD COLUMN IF NOT EXISTS last_heartbeat_at timestamptz,
  ADD COLUMN IF NOT EXISTS claimed_at timestamptz,
  ADD COLUMN IF NOT EXISTS schema_hardening_version text NOT NULL DEFAULT '061_absurd_queue_hardening_v2';

UPDATE lucidota_control.absurd_queue_job
SET payload_sha256 = encode(digest(payload::text, 'sha256'), 'hex')
WHERE payload_sha256 IS NULL;

ALTER TABLE lucidota_control.absurd_queue_job
  DROP CONSTRAINT IF EXISTS absurd_queue_job_payload_sha256_hex,
  ADD CONSTRAINT absurd_queue_job_payload_sha256_hex CHECK (payload_sha256 IS NULL OR payload_sha256 ~ '^[0-9a-f]{64}$');

ALTER TABLE lucidota_control.absurd_queue_job
  DROP CONSTRAINT IF EXISTS absurd_queue_job_attempt_bounds_v2,
  ADD CONSTRAINT absurd_queue_job_attempt_bounds_v2 CHECK (attempt_count >= 0 AND max_attempts > 0 AND attempt_count <= max_attempts);

CREATE OR REPLACE FUNCTION lucidota_control.set_absurd_queue_job_hardening_fields()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
  NEW.payload_sha256 := encode(digest(NEW.payload::text, 'sha256'), 'hex');
  IF NEW.status IN ('leased','running') AND NEW.claimed_at IS NULL THEN
    NEW.claimed_at := now();
  END IF;
  IF NEW.status = 'running' THEN
    NEW.last_heartbeat_at := coalesce(NEW.last_heartbeat_at, now());
  END IF;
  RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_set_absurd_queue_job_hardening_fields ON lucidota_control.absurd_queue_job;
CREATE TRIGGER trg_set_absurd_queue_job_hardening_fields
BEFORE INSERT OR UPDATE ON lucidota_control.absurd_queue_job
FOR EACH ROW EXECUTE FUNCTION lucidota_control.set_absurd_queue_job_hardening_fields();

CREATE INDEX IF NOT EXISTS idx_absurd_queue_job_failure_scan_v2
  ON lucidota_control.absurd_queue_job(queue_name, status, error_kind, updated_at DESC)
  WHERE status IN ('failed','dead_lettered');

COMMIT;
