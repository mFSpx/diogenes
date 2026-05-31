-- FILE: 06_SCHEMA/135_simplemem_candidate_status.sql
-- PURPOSE: idempotent add status column to simplemem_candidate.
-- NOTE: The status column was missing from the original schema definition in 053_simplemem_candidate_index.sql.
-- Migration 132 attempted to add a CHECK constraint on the nonexistent column and failed.

BEGIN;

ALTER TABLE lucidota_control.simplemem_candidate
  ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'CANDIDATE';

ALTER TABLE lucidota_control.simplemem_candidate
  ADD CONSTRAINT IF NOT EXISTS chk_simplemem_status
  CHECK (status IN ('CANDIDATE','ACCEPTED','REJECTED'));

COMMIT;
