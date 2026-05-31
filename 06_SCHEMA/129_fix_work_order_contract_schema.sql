-- Fix work order contract schema
ALTER TABLE IF EXISTS lucidota_control.work_order_contract RENAME COLUMN IF EXISTS work_order_key TO work_order_id;
ALTER TABLE IF EXISTS lucidota_control.work_order_contract ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'pending';
ALTER TABLE IF EXISTS lucidota_control.work_order_contract ADD COLUMN IF NOT EXISTS lane TEXT;
