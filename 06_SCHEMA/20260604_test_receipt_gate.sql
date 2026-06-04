-- FILE: 06_SCHEMA/20260604_test_receipt_gate.sql
-- PURPOSE: DB-owned receipt gate for test execution. Postgres decides; files only feed signatures.

BEGIN;

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_audit;
CREATE SCHEMA IF NOT EXISTS lucidota_canon;

CREATE TABLE IF NOT EXISTS lucidota_audit.test_execution_receipts (
    receipt_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    command_text text NOT NULL,
    scope text NOT NULL,
    dependency_signature text NOT NULL,
    dependency_payload_json jsonb NOT NULL DEFAULT '{}'::jsonb,
    status text NOT NULL CHECK (status IN ('queued','running','passed','failed','skipped','blocked')),
    exit_code integer,
    stdout_sha256 text NOT NULL DEFAULT '',
    stderr_sha256 text NOT NULL DEFAULT '',
    started_at timestamptz NOT NULL DEFAULT now(),
    completed_at timestamptz,
    runner_id text NOT NULL DEFAULT current_user,
    invalidated_by text,
    invalidated_at timestamptz,
    invalidation_reason text NOT NULL DEFAULT '',
    metadata_json jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    CHECK (completed_at IS NULL OR completed_at >= started_at),
    CHECK (stdout_sha256 = '' OR stdout_sha256 ~ '^[0-9a-f]{64}$'),
    CHECK (stderr_sha256 = '' OR stderr_sha256 ~ '^[0-9a-f]{64}$'),
    CHECK (jsonb_typeof(dependency_payload_json) = 'object'),
    CHECK (jsonb_typeof(metadata_json) = 'object')
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_test_execution_receipts_pass_signature
    ON lucidota_audit.test_execution_receipts (scope, dependency_signature)
    WHERE status = 'passed' AND invalidated_by IS NULL;

CREATE INDEX IF NOT EXISTS idx_test_execution_receipts_scope_signature_status_completed
    ON lucidota_audit.test_execution_receipts (scope, dependency_signature, status, completed_at DESC);

CREATE INDEX IF NOT EXISTS idx_test_execution_receipts_scope_started
    ON lucidota_audit.test_execution_receipts (scope, started_at DESC);

CREATE INDEX IF NOT EXISTS idx_test_execution_receipts_runner_started
    ON lucidota_audit.test_execution_receipts (runner_id, started_at DESC);

CREATE INDEX IF NOT EXISTS idx_test_execution_receipts_invalidated
    ON lucidota_audit.test_execution_receipts (invalidated_by, invalidated_at DESC)
    WHERE invalidated_by IS NOT NULL;

CREATE OR REPLACE VIEW lucidota_canon.api_test_execution_receipts AS
SELECT
    receipt_uuid::text AS receipt_uuid,
    command_text,
    scope,
    dependency_signature,
    dependency_payload_json,
    status,
    exit_code,
    stdout_sha256,
    stderr_sha256,
    started_at,
    completed_at,
    runner_id,
    invalidated_by,
    invalidated_at,
    invalidation_reason,
    metadata_json,
    created_at,
    updated_at
FROM lucidota_audit.test_execution_receipts;

REVOKE ALL ON lucidota_audit.test_execution_receipts FROM PUBLIC;
REVOKE ALL ON ALL TABLES IN SCHEMA lucidota_canon FROM lucidota_postgrest_anon;
GRANT SELECT ON lucidota_canon.api_test_execution_receipts TO lucidota_postgrest_anon;

COMMIT;
