-- Runtime-only resonance receipts and read-only surfaces.
-- This is intentionally NOT canon truth.

BEGIN;

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_runtime;
CREATE SCHEMA IF NOT EXISTS lucidota_canon;

CREATE TABLE IF NOT EXISTS lucidota_runtime.resonance_evolution_receipt (
    receipt_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at timestamptz NOT NULL DEFAULT now(),
    generation bigint NOT NULL,
    source text NOT NULL CHECK (source IN ('Synthetic', 'Runtime')),
    winner_id bigint NOT NULL,
    parent_ids jsonb NOT NULL DEFAULT '[]'::jsonb,
    avg_error numeric NOT NULL,
    latency_us numeric NOT NULL,
    density numeric NOT NULL,
    stability numeric NOT NULL,
    score numeric NOT NULL,
    route_pressure_top jsonb NOT NULL DEFAULT '[]'::jsonb,
    genome jsonb NOT NULL DEFAULT '{}'::jsonb,
    promotion text NOT NULL DEFAULT 'runtime_candidate',
    canon_status text NOT NULL DEFAULT 'not_truth_runtime_only',
    event_type text NOT NULL DEFAULT 'resonance_evolution_generation',
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE OR REPLACE VIEW lucidota_canon.resonance_evolution_latest AS
SELECT
    receipt_uuid,
    created_at,
    generation,
    source,
    winner_id,
    parent_ids,
    avg_error,
    latency_us,
    density,
    stability,
    score,
    route_pressure_top,
    genome,
    promotion,
    canon_status,
    event_type,
    detail,
    updated_at,
    jsonb_build_object(
        'receipt_uuid', receipt_uuid,
        'generation', generation,
        'source', source,
        'winner_id', winner_id,
        'parent_ids', parent_ids,
        'avg_error', avg_error,
        'latency_us', latency_us,
        'density', density,
        'stability', stability,
        'score', score,
        'route_pressure_top', route_pressure_top,
        'genome', genome,
        'promotion', promotion,
        'canon_status', canon_status,
        'event_type', event_type,
        'created_at', created_at
    ) AS packet
FROM lucidota_runtime.resonance_evolution_receipt
ORDER BY generation DESC, created_at DESC
LIMIT 1;

CREATE OR REPLACE VIEW lucidota_canon.resonance_active_runtime_candidate AS
SELECT
    receipt_uuid,
    created_at,
    generation,
    source,
    winner_id,
    route_pressure_top,
    genome,
    score,
    stability,
    canon_status,
    jsonb_build_object(
        'winner_id', winner_id,
        'generation', generation,
        'score', score,
        'stability', stability,
        'route_pressure_top', route_pressure_top,
        'canon_status', canon_status,
        'source', source
    ) AS packet
FROM lucidota_canon.resonance_evolution_latest;

CREATE OR REPLACE VIEW lucidota_canon.resonance_route_pressure_current AS
SELECT
    receipt_uuid,
    created_at,
    generation,
    source,
    route_pressure_top,
    jsonb_build_object(
        'generation', generation,
        'source', source,
        'route_pressure_top', route_pressure_top,
        'receipt_uuid', receipt_uuid
    ) AS packet
FROM lucidota_canon.resonance_evolution_latest;

INSERT INTO lucidota_control.schema_owner_manifest (
    surface_id, canonical_owner, packet_class, surface_kind, approval_required, notes, detail
) VALUES
    ('resonance_evolution_latest', 'lucidota_canon', 'runtime_packet', 'view', false, 'Latest runtime resonance receipt; never canon truth.', '{"source":"runtime"}'::jsonb),
    ('resonance_active_runtime_candidate', 'lucidota_canon', 'runtime_packet', 'view', false, 'Current runtime candidate chosen from resonance receipts.', '{"source":"runtime"}'::jsonb),
    ('resonance_route_pressure_current', 'lucidota_canon', 'runtime_packet', 'view', false, 'Current route pressure telemetry summary for resonance routing.', '{"source":"runtime"}'::jsonb)
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
        'resonance_evolution_latest',
        'GET',
        '/resonance_evolution_latest',
        'Latest runtime-only resonance receipt.',
        'lucidota_canon.resonance_evolution_latest',
        '{"limit":"1"}',
        '{"event_type":"resonance_evolution_generation","canon_status":"not_truth_runtime_only"}',
        'implemented'
    ),
    (
        'resonance_active_runtime_candidate',
        'GET',
        '/resonance_active_runtime_candidate',
        'Latest runtime candidate chosen by the resonance evolver.',
        'lucidota_canon.resonance_active_runtime_candidate',
        '{"limit":"1"}',
        '{"winner_id":0,"canon_status":"not_truth_runtime_only"}',
        'implemented'
    ),
    (
        'resonance_route_pressure_current',
        'GET',
        '/resonance_route_pressure_current',
        'Current route pressure telemetry summary from the resonance evolver.',
        'lucidota_canon.resonance_route_pressure_current',
        '{"limit":"1"}',
        '{"generation":0,"canon_status":"not_truth_runtime_only"}',
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

GRANT SELECT ON lucidota_runtime.resonance_evolution_receipt TO lucidota_postgrest_anon, mfspx;
GRANT SELECT ON lucidota_canon.resonance_evolution_latest, lucidota_canon.resonance_active_runtime_candidate, lucidota_canon.resonance_route_pressure_current TO lucidota_postgrest_anon, mfspx;

COMMIT;
