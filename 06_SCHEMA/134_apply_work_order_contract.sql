-- FILE: 06_SCHEMA/134_apply_work_order_contract.sql
-- PURPOSE: Apply work_order_contract table as defined in 041_boring_beast_loop_contracts.sql
--          and reconcile with work_order.schema.json requirements.
-- COMPLIANCE: Fully idempotent; safe to re-run.

BEGIN;

-- Step 1: Create the table exactly as defined in 041_boring_beast_loop_contracts.sql
-- This is idempotent via IF NOT EXISTS. The table was never applied to the DB,
-- so this ensures it exists with the original schema including work_order_key.
CREATE TABLE IF NOT EXISTS lucidota_control.work_order_contract (
  work_order_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  work_order_key text NOT NULL UNIQUE,
  target_number integer NOT NULL CHECK (target_number BETWEEN 1 AND 20),
  target_name text NOT NULL,
  valid boolean NOT NULL,
  errors jsonb NOT NULL DEFAULT '[]'::jsonb,
  normalized_payload jsonb NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);

-- Step 2: Rename work_order_key to work_order_id to match work_order.schema.json
-- The schema requires "work_order_id" (string) as a required property.
-- We only rename if work_order_key exists AND work_order_id does not exist.
DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema = 'lucidota_control'
      AND table_name = 'work_order_contract'
      AND column_name = 'work_order_key'
  ) AND NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema = 'lucidota_control'
      AND table_name = 'work_order_contract'
      AND column_name = 'work_order_id'
  ) THEN
    EXECUTE 'ALTER TABLE lucidota_control.work_order_contract RENAME COLUMN work_order_key TO work_order_id';
  END IF;
END $$;

-- Step 3: Add status column as required by work_order.schema.json
-- The schema defines status as an enum: CREATED, RUNNING, COMPLETED, FAILED, REJECTED.
-- We use TEXT to allow all enum values; IF NOT EXISTS makes this idempotent.
ALTER TABLE lucidota_control.work_order_contract ADD COLUMN IF NOT EXISTS status TEXT;

COMMIT;
