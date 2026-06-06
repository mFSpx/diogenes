-- Runtime-only elastic shape receipts and residuals.
-- UUID stays identity; shape signature stays routing/context only.
-- This is intentionally NOT canon truth.

BEGIN;

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_runtime;
CREATE SCHEMA IF NOT EXISTS lucidota_canon;

CREATE TABLE IF NOT EXISTS lucidota_runtime.elastic_shape_receipt (
    receipt_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    artifact_uuid uuid NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    source text NOT NULL CHECK (source IN ('Synthetic', 'Runtime')),
    source_hash text NOT NULL DEFAULT '',
    signature text NOT NULL DEFAULT '',
    collision_signature text NOT NULL DEFAULT '',
    dimensions integer NOT NULL CHECK (dimensions > 0),
    entropy_hint numeric NOT NULL DEFAULT 0,
    shape_vector jsonb NOT NULL DEFAULT '[]'::jsonb,
    active_resonances jsonb NOT NULL DEFAULT '[]'::jsonb,
    fidelity numeric NOT NULL DEFAULT 0,
    residual_mass numeric NOT NULL DEFAULT 0,
    residual_vector jsonb NOT NULL DEFAULT '[]'::jsonb,
    collision boolean NOT NULL DEFAULT false,
    canon_status text NOT NULL DEFAULT 'not_truth_runtime_only',
    route_context jsonb NOT NULL DEFAULT '{}'::jsonb,
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS elastic_shape_receipt_signature_idx
    ON lucidota_runtime.elastic_shape_receipt(signature, created_at DESC);

CREATE INDEX IF NOT EXISTS elastic_shape_receipt_artifact_idx
    ON lucidota_runtime.elastic_shape_receipt(artifact_uuid, created_at DESC);

CREATE TABLE IF NOT EXISTS lucidota_runtime.shape_residuals (
    residual_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    receipt_uuid uuid NOT NULL REFERENCES lucidota_runtime.elastic_shape_receipt(receipt_uuid) ON DELETE CASCADE,
    artifact_uuid uuid NOT NULL,
    collision_signature text NOT NULL DEFAULT '',
    source_hash text NOT NULL DEFAULT '',
    signature text NOT NULL DEFAULT '',
    dimensions integer NOT NULL CHECK (dimensions > 0),
    suggested_dimensions integer NOT NULL DEFAULT 0,
    fidelity numeric NOT NULL DEFAULT 0,
    residual_mass numeric NOT NULL DEFAULT 0,
    residual_vector jsonb NOT NULL DEFAULT '[]'::jsonb,
    route_context jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS shape_residuals_collision_idx
    ON lucidota_runtime.shape_residuals(collision_signature, created_at DESC);

CREATE INDEX IF NOT EXISTS shape_residuals_artifact_idx
    ON lucidota_runtime.shape_residuals(artifact_uuid, created_at DESC);

CREATE OR REPLACE VIEW lucidota_canon.elastic_shape_latest AS
SELECT
    receipt_uuid,
    artifact_uuid,
    created_at,
    source,
    source_hash,
    signature,
    collision_signature,
    dimensions,
    entropy_hint,
    shape_vector,
    active_resonances,
    fidelity,
    residual_mass,
    residual_vector,
    collision,
    canon_status,
    route_context,
    detail,
    updated_at,
    jsonb_build_object(
        'receipt_uuid', receipt_uuid,
        'artifact_uuid', artifact_uuid,
        'source', source,
        'source_hash', source_hash,
        'signature', signature,
        'collision_signature', collision_signature,
        'dimensions', dimensions,
        'entropy_hint', entropy_hint,
        'shape_vector', shape_vector,
        'active_resonances', active_resonances,
        'fidelity', fidelity,
        'residual_mass', residual_mass,
        'residual_vector', residual_vector,
        'collision', collision,
        'canon_status', canon_status,
        'route_context', route_context,
        'created_at', created_at
    ) AS packet
FROM lucidota_runtime.elastic_shape_receipt
ORDER BY created_at DESC, receipt_uuid DESC
LIMIT 1;

CREATE OR REPLACE VIEW lucidota_canon.elastic_shape_current AS
WITH latest AS (
    SELECT *
    FROM lucidota_canon.elastic_shape_latest
)
SELECT
    receipt_uuid,
    artifact_uuid,
    created_at,
    source,
    source_hash,
    signature,
    collision_signature,
    dimensions,
    entropy_hint,
    fidelity,
    collision,
    canon_status,
    route_context,
    jsonb_build_object(
        'receipt_uuid', receipt_uuid,
        'artifact_uuid', artifact_uuid,
        'source', source,
        'source_hash', source_hash,
        'signature', signature,
        'collision_signature', collision_signature,
        'dimensions', dimensions,
        'entropy_hint', entropy_hint,
        'shape_vector', shape_vector,
        'active_resonances', active_resonances,
        'fidelity', fidelity,
        'residual_mass', residual_mass,
        'residual_vector', residual_vector,
        'collision', collision,
        'canon_status', canon_status,
        'route_context', route_context,
        'created_at', created_at
    ) AS packet,
    shape_vector,
    active_resonances,
    residual_mass,
    residual_vector
FROM latest;

CREATE OR REPLACE VIEW lucidota_canon.shape_residuals_current AS
WITH stats AS (
    SELECT
        count(*) AS residual_count,
        max(created_at) AS latest_residual_at
    FROM lucidota_runtime.shape_residuals
),
latest AS (
    SELECT *
    FROM lucidota_runtime.shape_residuals
    ORDER BY created_at DESC, residual_uuid DESC
    LIMIT 1
)
SELECT
    COALESCE(latest.residual_uuid, gen_random_uuid()) AS residual_uuid,
    COALESCE(latest.receipt_uuid, '00000000-0000-0000-0000-000000000000'::uuid) AS receipt_uuid,
    latest.artifact_uuid,
    latest.collision_signature,
    latest.source_hash,
    latest.signature,
    latest.dimensions,
    latest.suggested_dimensions,
    latest.fidelity,
    latest.residual_mass,
    latest.residual_vector,
    latest.route_context,
    stats.residual_count,
    stats.latest_residual_at,
    jsonb_build_object(
        'residual_count', stats.residual_count,
        'latest_residual_at', stats.latest_residual_at,
        'artifact_uuid', latest.artifact_uuid,
        'collision_signature', latest.collision_signature,
        'suggested_dimensions', latest.suggested_dimensions,
        'fidelity', latest.fidelity,
        'residual_mass', latest.residual_mass,
        'source_hash', latest.source_hash,
        'signature', latest.signature,
        'route_context', latest.route_context
    ) AS packet
FROM stats
LEFT JOIN latest ON true;

CREATE OR REPLACE VIEW lucidota_canon.indy_attention_pressure_current AS
WITH latest AS (
    SELECT *
    FROM lucidota_canon.elastic_shape_latest
)
SELECT
    receipt_uuid,
    artifact_uuid,
    created_at,
    source,
    source_hash,
    signature,
    collision_signature,
    dimensions,
    entropy_hint,
    shape_vector,
    active_resonances,
    fidelity,
    residual_mass,
    residual_vector,
    collision,
    canon_status,
    route_context,
    GREATEST(
        0::numeric,
        LEAST(
            1::numeric,
            (COALESCE(fidelity, 0::numeric) * 0.55)
            + (COALESCE(entropy_hint, 0::numeric) * 0.20)
            + (CASE WHEN collision THEN 0.15 ELSE 0 END)
            + ((1::numeric - LEAST(1::numeric, COALESCE(residual_mass, 0::numeric))) * 0.10)
        )
    ) AS pressure_score,
    CASE
        WHEN collision OR residual_mass > 0.25 THEN 'inspect_now'
        WHEN entropy_hint > 0.7 THEN 'creativity_burst'
        WHEN fidelity > 0.85 THEN 'continue'
        ELSE 'explore'
    END AS recommended_action,
    jsonb_build_object(
        'receipt_uuid', receipt_uuid,
        'artifact_uuid', artifact_uuid,
        'source', source,
        'source_hash', source_hash,
        'signature', signature,
        'collision_signature', collision_signature,
        'dimensions', dimensions,
        'entropy_hint', entropy_hint,
        'fidelity', fidelity,
        'residual_mass', residual_mass,
        'collision', collision,
        'canon_status', canon_status,
        'pressure_score', GREATEST(
            0::numeric,
            LEAST(
                1::numeric,
                (COALESCE(fidelity, 0::numeric) * 0.55)
                + (COALESCE(entropy_hint, 0::numeric) * 0.20)
                + (CASE WHEN collision THEN 0.15 ELSE 0 END)
                + ((1::numeric - LEAST(1::numeric, COALESCE(residual_mass, 0::numeric))) * 0.10)
            )
        ),
        'recommended_action', CASE
            WHEN collision OR residual_mass > 0.25 THEN 'inspect_now'
            WHEN entropy_hint > 0.7 THEN 'creativity_burst'
            WHEN fidelity > 0.85 THEN 'continue'
            ELSE 'explore'
        END,
        'route_context', route_context,
        'active_resonances', active_resonances,
        'residual_vector', residual_vector,
        'created_at', created_at
    ) AS packet
FROM latest;

INSERT INTO lucidota_canon.schema_owner_manifest (
    surface_id, canonical_owner, packet_class, surface_kind, approval_required, notes, detail
) VALUES
    ('elastic_shape_latest', 'lucidota_canon', 'runtime_packet', 'view', false, 'Latest elastic shape receipt; UUID is identity and shape is routing only.', '{"source":"runtime"}'::jsonb),
    ('elastic_shape_current', 'lucidota_canon', 'runtime_packet', 'view', false, 'Current elastic shape routing packet for Indy_READs and operator routing.', '{"source":"runtime"}'::jsonb),
    ('shape_residuals_current', 'lucidota_canon', 'runtime_packet', 'view', false, 'Residual torque summary for shape collisions and rehydration drift.', '{"source":"runtime"}'::jsonb),
    ('indy_attention_pressure_current', 'lucidota_canon', 'runtime_packet', 'view', false, 'Indy attention-pressure summary from the latest elastic shape receipt.', '{"source":"runtime"}'::jsonb)
ON CONFLICT (surface_id) DO UPDATE SET
    canonical_owner = EXCLUDED.canonical_owner,
    packet_class = EXCLUDED.packet_class,
    surface_kind = EXCLUDED.surface_kind,
    approval_required = EXCLUDED.approval_required,
    notes = EXCLUDED.notes,
    detail = EXCLUDED.detail,
    updated_at = now();

INSERT INTO lucidota_canon.api_route_catalog (
    route_id,
    method,
    path_pattern,
    description,
    target,
    sample_request,
    sample_response,
    status
) VALUES
    (
        'elastic_shape_latest',
        'GET',
        '/elastic_shape_latest',
        'Latest runtime-only elastic shape receipt.',
        'lucidota_canon.elastic_shape_latest',
        '{"limit":"1"}',
        '{"canon_status":"not_truth_runtime_only"}',
        'implemented'
    ),
    (
        'elastic_shape_current',
        'GET',
        '/elastic_shape_current',
        'Current elastic shape routing packet for Indy_READs and operator paths.',
        'lucidota_canon.elastic_shape_current',
        '{"limit":"1"}',
        '{"canon_status":"not_truth_runtime_only"}',
        'implemented'
    ),
    (
        'shape_residuals_current',
        'GET',
        '/shape_residuals_current',
        'Current residual torque summary for elastic shape collisions.',
        'lucidota_canon.shape_residuals_current',
        '{"limit":"1"}',
        '{"residual_count":0,"canon_status":"not_truth_runtime_only"}',
        'implemented'
    ),
    (
        'indy_attention_pressure_current',
        'GET',
        '/indy_attention_pressure_current',
        'Current Indy attention-pressure summary from the latest elastic shape receipt.',
        'lucidota_canon.indy_attention_pressure_current',
        '{"limit":"1"}',
        '{"pressure_score":0.0,"recommended_action":"explore","canon_status":"not_truth_runtime_only"}',
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

GRANT SELECT ON lucidota_runtime.elastic_shape_receipt TO lucidota_postgrest_anon, mfspx;
GRANT SELECT ON lucidota_runtime.shape_residuals TO lucidota_postgrest_anon, mfspx;
GRANT SELECT ON lucidota_canon.elastic_shape_latest, lucidota_canon.elastic_shape_current, lucidota_canon.shape_residuals_current, lucidota_canon.indy_attention_pressure_current TO lucidota_postgrest_anon, mfspx;
GRANT SELECT ON lucidota_canon.schema_owner_manifest, lucidota_canon.api_route_catalog TO lucidota_postgrest_anon, mfspx;

COMMIT;
