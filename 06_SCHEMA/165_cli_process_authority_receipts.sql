-- CLI process authority receipts: DB-visible proof of auth injection, restarts, and subprocess control.
-- The wrapper writes receipts here; PostgREST exposes the safe view surface.

BEGIN;

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_control;
CREATE SCHEMA IF NOT EXISTS lucidota_canon;

CREATE TABLE IF NOT EXISTS lucidota_control.cli_process_receipt (
    receipt_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    received_at timestamptz NOT NULL DEFAULT now(),
    command_line text NOT NULL DEFAULT '',
    command_sha256 text NOT NULL CHECK (command_sha256 ~ '^[0-9a-f]{64}$'),
    process_pid integer,
    timeout_seconds numeric NOT NULL DEFAULT 0,
    restart_count integer NOT NULL DEFAULT 0,
    auth_env_var text NOT NULL DEFAULT '',
    auth_prompt_seen boolean NOT NULL DEFAULT false,
    auth_injected boolean NOT NULL DEFAULT false,
    status text NOT NULL DEFAULT 'queued' CHECK (status IN ('queued', 'succeeded', 'failed', 'timeout', 'auth_failed')),
    exit_code integer,
    stdout_tail text NOT NULL DEFAULT '',
    stderr_tail text NOT NULL DEFAULT '',
    receipt_path text NOT NULL DEFAULT '',
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_cli_process_receipt_received_at
    ON lucidota_control.cli_process_receipt(received_at DESC, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_cli_process_receipt_status
    ON lucidota_control.cli_process_receipt(status, received_at DESC);

CREATE OR REPLACE VIEW lucidota_canon.cli_process_receipts AS
SELECT
    receipt_uuid,
    received_at,
    command_line,
    command_sha256,
    process_pid,
    timeout_seconds,
    restart_count,
    auth_env_var,
    auth_prompt_seen,
    auth_injected,
    status,
    exit_code,
    stdout_tail,
    stderr_tail,
    receipt_path,
    detail,
    created_at,
    updated_at
FROM lucidota_control.cli_process_receipt;

INSERT INTO lucidota_canon.api_route_catalog (
    route_id, method, path_pattern, description, target, sample_request, sample_response, status
) VALUES (
    'cli_process_receipts',
    'GET',
    '/cli_process_receipts',
    'CLI subprocess authority receipts proving auth injection, timeout recovery, and restart control.',
    'lucidota_canon.cli_process_receipts',
    '{"limit":"1"}',
    '{"status":"succeeded"}',
    'implemented'
)
ON CONFLICT (route_id) DO UPDATE SET
    method = EXCLUDED.method,
    path_pattern = EXCLUDED.path_pattern,
    description = EXCLUDED.description,
    target = EXCLUDED.target,
    sample_request = EXCLUDED.sample_request,
    sample_response = EXCLUDED.sample_response,
    status = EXCLUDED.status,
    updated_at = now();

GRANT SELECT ON lucidota_canon.cli_process_receipts TO lucidota_postgrest_anon, mfspx;
GRANT SELECT ON lucidota_canon.api_route_catalog TO lucidota_postgrest_anon, mfspx;

NOTIFY pgrst, 'reload schema';

COMMIT;
