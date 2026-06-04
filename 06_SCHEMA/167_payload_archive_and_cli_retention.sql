-- Cold archive lane for high-volume CLI telemetry.
-- Keep hot receipts indexable; move heavy tails to compressed payload archives with hash refs.

BEGIN;

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_control;
CREATE SCHEMA IF NOT EXISTS lucidota_canon;

ALTER TABLE IF EXISTS lucidota_control.cli_process_receipt
    ADD COLUMN IF NOT EXISTS stdout_tail_sha256 text CHECK (stdout_tail_sha256 IS NULL OR stdout_tail_sha256 ~ '^[0-9a-f]{64}$'),
    ADD COLUMN IF NOT EXISTS stderr_tail_sha256 text CHECK (stderr_tail_sha256 IS NULL OR stderr_tail_sha256 ~ '^[0-9a-f]{64}$'),
    ADD COLUMN IF NOT EXISTS stdout_archive_ref text NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS stderr_archive_ref text NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS stdout_archived_at timestamptz,
    ADD COLUMN IF NOT EXISTS stderr_archived_at timestamptz;

CREATE TABLE IF NOT EXISTS lucidota_control.payload_archive (
    archive_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    source_table text NOT NULL,
    source_uuid uuid NOT NULL,
    payload_kind text NOT NULL CHECK (payload_kind IN ('stdout_tail', 'stderr_tail')),
    payload_hash text NOT NULL CHECK (payload_hash ~ '^[0-9a-f]{64}$'),
    payload_bytes integer NOT NULL DEFAULT 0 CHECK (payload_bytes >= 0),
    payload_chars integer NOT NULL DEFAULT 0 CHECK (payload_chars >= 0),
    archive_path text NOT NULL DEFAULT '',
    archived_at timestamptz NOT NULL DEFAULT now(),
    restored_at timestamptz,
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE(source_table, source_uuid, payload_kind)
);

CREATE INDEX IF NOT EXISTS idx_payload_archive_archived_at
    ON lucidota_control.payload_archive(archived_at DESC, payload_kind, source_table);

CREATE INDEX IF NOT EXISTS idx_payload_archive_source
    ON lucidota_control.payload_archive(source_table, source_uuid, payload_kind);

CREATE OR REPLACE VIEW lucidota_canon.payload_archive_status AS
SELECT
    source_table,
    payload_kind,
    count(*) AS archive_count,
    COALESCE(sum(payload_bytes), 0) AS archived_bytes,
    COALESCE(sum(payload_chars), 0) AS archived_chars,
    max(archived_at) AS latest_archived_at,
    count(*) FILTER (WHERE restored_at IS NULL) AS active_archive_count
FROM lucidota_control.payload_archive
GROUP BY source_table, payload_kind
ORDER BY source_table, payload_kind;

INSERT INTO lucidota_canon.api_route_catalog (
    route_id, method, path_pattern, description, target, sample_request, sample_response, status
) VALUES (
    'payload_archive_status',
    'GET',
    '/payload_archive_status',
    'Cold payload archive status for CLI telemetry retention and rehydration.',
    'lucidota_canon.payload_archive_status',
    '{"limit":"10"}',
    '{"source_table":"lucidota_control.cli_process_receipt"}',
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

GRANT SELECT ON lucidota_canon.payload_archive_status TO lucidota_postgrest_anon, mfspx;
GRANT SELECT ON lucidota_canon.api_route_catalog TO lucidota_postgrest_anon, mfspx;

NOTIFY pgrst, 'reload schema';

COMMIT;
