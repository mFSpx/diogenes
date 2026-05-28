-- FILE: 06_SCHEMA/096_worker_command_registry.sql
-- PURPOSE: Worker command registry; handlers must be declared before consume.

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_control;

CREATE TABLE IF NOT EXISTS lucidota_control.worker_command_registry (
  command_key text PRIMARY KEY,
  queue_name text NOT NULL DEFAULT '*',
  job_kind text NOT NULL DEFAULT '*',
  handler text NOT NULL,
  script_path text NOT NULL DEFAULT '',
  active boolean NOT NULL DEFAULT true,
  created_at timestamptz NOT NULL DEFAULT now(),
  detail jsonb NOT NULL DEFAULT '{}'::jsonb
);

INSERT INTO lucidota_control.worker_command_registry(command_key,queue_name,job_kind,handler,script_path,detail) VALUES
('generic.noop','*','*','noop','scripts/absurd_consume_one.py','{}'::jsonb),
('generic.status_ledger_check','*','*','status_ledger_check','scripts/absurd_consume_one.py','{}'::jsonb),
('generic.external_command','*','*','external_command','scripts/absurd_consume_one.py','{"requires_allowlisted_script":true}'::jsonb),
('boring.noop','boring_beast','boring_beast_work_item','noop','scripts/boring_beast.py','{}'::jsonb)
ON CONFLICT(command_key) DO UPDATE SET active=true, detail=EXCLUDED.detail;
