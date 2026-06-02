CREATE TABLE IF NOT EXISTS lucidota_control.worker_contract_version (
    contract_version_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    worker_key text NOT NULL,
    schema_contract_version integer NOT NULL DEFAULT 1,
    compatible_branches text[] NOT NULL DEFAULT '{main}'::text[],
    required_tables text[] DEFAULT '{}',
    required_columns jsonb DEFAULT '{}',
    required_enums jsonb DEFAULT '{}',
    migration_window jsonb DEFAULT NULL,
    contract_hash text NOT NULL,
    declared_at timestamptz DEFAULT now(),
    declared_by text DEFAULT 'system',
    UNIQUE(worker_key, schema_contract_version)
);

CREATE TABLE IF NOT EXISTS lucidota_control.schema_compatibility_matrix (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    from_version integer NOT NULL,
    to_version integer NOT NULL,
    is_compatible boolean NOT NULL,
    requires_migration boolean DEFAULT false,
    notes text,
    created_at timestamptz DEFAULT now(),
    UNIQUE(from_version, to_version)
);

CREATE TABLE IF NOT EXISTS lucidota_control.job_contract_snapshot (
    snapshot_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    job_uuid uuid NOT NULL,
    worker_key text NOT NULL,
    schema_contract_version integer NOT NULL,
    contract_hash text NOT NULL,
    snapshotted_at timestamptz DEFAULT now(),
    compatible_branches text[] NOT NULL DEFAULT '{main}'::text[]
);

CREATE TABLE IF NOT EXISTS lucidota_control.dead_letter_reason (
    reason_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    job_uuid uuid NOT NULL,
    worker_key text NOT NULL,
    reason_kind text NOT NULL CHECK(reason_kind IN ('schema_version_mismatch','branch_incompatible','table_missing','column_missing','enum_missing','migration_window_expired','contract_hash_mismatch','unregistered_worker','handler_error','timeout','unknown')),
    expected text,
    actual text,
    detail jsonb DEFAULT '{}',
    dead_lettered_at timestamptz DEFAULT now()
);

ALTER TABLE lucidota_control.worker_contract_version ADD COLUMN IF NOT EXISTS contract_timestamp timestamptz NOT NULL DEFAULT now();
ALTER TABLE lucidota_control.worker_contract_version ADD COLUMN IF NOT EXISTS contract_version_seq integer NOT NULL DEFAULT 1;
COMMENT ON COLUMN lucidota_control.worker_contract_version.contract_hash IS 'SHA256 of contract fields — triple redundancy with contract_timestamp + contract_version_seq. Aircraft standard: hash catches content tampering, timestamp catches drift, seq catches ordering.';
ALTER TABLE lucidota_control.job_contract_snapshot ADD COLUMN IF NOT EXISTS contract_timestamp timestamptz;
ALTER TABLE lucidota_control.job_contract_snapshot ADD COLUMN IF NOT EXISTS contract_version_seq integer;
