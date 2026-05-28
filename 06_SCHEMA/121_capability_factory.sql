-- FILE: 06_SCHEMA/121_capability_factory.sql
-- PURPOSE: capability factory ledgers under lucidota_investigation.
-- COMPLIANCE: idempotent migration; append-only-ish rows with FK-backed provenance.

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_investigation;

CREATE TABLE IF NOT EXISTS lucidota_investigation.capability_run (
    capability_run_uuid uuid PRIMARY KEY DEFAULT lucidota_investigation.uuid_v7(),
    capability_key text NOT NULL REFERENCES lucidota_investigation.capability_registry(capability_key) ON DELETE RESTRICT,
    run_label text NOT NULL DEFAULT '',
    run_status text NOT NULL DEFAULT 'planned' CHECK (run_status IN (
        'planned',
        'running',
        'blocked',
        'completed',
        'failed',
        'canceled'
    )),
    started_at timestamptz NOT NULL DEFAULT now(),
    finished_at timestamptz,
    summary text NOT NULL DEFAULT '',
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    CHECK (finished_at IS NULL OR finished_at >= started_at)
);

CREATE INDEX IF NOT EXISTS capability_run_capability_started_idx
    ON lucidota_investigation.capability_run(capability_key, started_at DESC);

CREATE INDEX IF NOT EXISTS capability_run_status_started_idx
    ON lucidota_investigation.capability_run(run_status, started_at DESC);

CREATE TABLE IF NOT EXISTS lucidota_investigation.capability_artifact (
    capability_artifact_uuid uuid PRIMARY KEY DEFAULT lucidota_investigation.uuid_v7(),
    capability_run_uuid uuid NOT NULL REFERENCES lucidota_investigation.capability_run(capability_run_uuid) ON DELETE CASCADE,
    artifact_kind text NOT NULL DEFAULT 'other' CHECK (artifact_kind IN (
        'script',
        'receipt',
        'test',
        'schema',
        'log',
        'output',
        'handoff',
        'checkpoint',
        'other'
    )),
    artifact_name text NOT NULL,
    artifact_path text NOT NULL DEFAULT '',
    artifact_sha256 text NOT NULL DEFAULT '' CHECK (artifact_sha256 = '' OR artifact_sha256 ~ '^[0-9a-f]{64}$'),
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    CHECK (artifact_name <> '')
);

CREATE INDEX IF NOT EXISTS capability_artifact_run_idx
    ON lucidota_investigation.capability_artifact(capability_run_uuid, created_at DESC);

CREATE UNIQUE INDEX IF NOT EXISTS capability_artifact_run_name_uidx
    ON lucidota_investigation.capability_artifact(capability_run_uuid, artifact_name);

CREATE TABLE IF NOT EXISTS lucidota_investigation.claim_ledger (
    claim_uuid uuid PRIMARY KEY DEFAULT lucidota_investigation.uuid_v7(),
    capability_run_uuid uuid NOT NULL REFERENCES lucidota_investigation.capability_run(capability_run_uuid) ON DELETE CASCADE,
    claim_key text NOT NULL,
    claim_status text NOT NULL DEFAULT 'asserted' CHECK (claim_status IN (
        'asserted',
        'supported',
        'contradicted',
        'deferred'
    )),
    claim_text text NOT NULL DEFAULT '',
    evidence_refs jsonb NOT NULL DEFAULT '[]'::jsonb CHECK (jsonb_typeof(evidence_refs) = 'array'),
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (capability_run_uuid, claim_key),
    CHECK (claim_key <> ''),
    CHECK (claim_text <> '')
);

CREATE INDEX IF NOT EXISTS claim_ledger_run_idx
    ON lucidota_investigation.claim_ledger(capability_run_uuid, created_at DESC);

CREATE TABLE IF NOT EXISTS lucidota_investigation.lever_ledger (
    lever_uuid uuid PRIMARY KEY DEFAULT lucidota_investigation.uuid_v7(),
    capability_run_uuid uuid NOT NULL REFERENCES lucidota_investigation.capability_run(capability_run_uuid) ON DELETE CASCADE,
    claim_uuid uuid REFERENCES lucidota_investigation.claim_ledger(claim_uuid) ON DELETE SET NULL,
    lever_key text NOT NULL,
    lever_kind text NOT NULL DEFAULT 'unknown' CHECK (lever_kind IN (
        'policy',
        'process',
        'schema',
        'script',
        'test',
        'receipt',
        'unknown'
    )),
    lever_status text NOT NULL DEFAULT 'proposed' CHECK (lever_status IN (
        'proposed',
        'active',
        'retired',
        'blocked'
    )),
    lever_text text NOT NULL DEFAULT '',
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (capability_run_uuid, lever_key),
    CHECK (lever_key <> '')
);

CREATE INDEX IF NOT EXISTS lever_ledger_run_idx
    ON lucidota_investigation.lever_ledger(capability_run_uuid, created_at DESC);

CREATE INDEX IF NOT EXISTS lever_ledger_claim_idx
    ON lucidota_investigation.lever_ledger(claim_uuid);
