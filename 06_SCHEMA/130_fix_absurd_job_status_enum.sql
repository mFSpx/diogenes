-- --------------------------------------------------------------------------------
-- This migration corrects the original attempt to rename 'succeeded' to 'COMPLETED'
-- in the absurd_queue_job.status enum. The original migration failed to recognize
-- that absurd_queue_job uses a Postgres enum with values (queued/running/succeeded/
-- failed/dead_lettered/cancelled) while absurd_job.schema.json defines a JSON
-- envelope abstraction with different values (CREATED/RUNNING/COMPLETED/FAILED/
-- DEAD_LETTER). These are distinct schemas and should not be conflated.
-- --------------------------------------------------------------------------------

-- Confirm the 'COMPLETED' value is added to the enum in an idempotent manner
ALTER TYPE absurd_queue_job_status ADD VALUE IF NOT EXISTS 'COMPLETED';

-- --------------------------------------------------------------------------------
-- WARNING: DO NOT RUN THE FOLLOWING UPDATE BLOCK
-- Running this update would break live workers polling for status='succeeded'
-- --------------------------------------------------------------------------------
-- UPDATE absurd_queue_job
-- SET status = 'COMPLETED'
-- WHERE status = 'succeeded';
-- --------------------------------------------------------------------------------
