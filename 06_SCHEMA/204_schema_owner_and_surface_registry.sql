-- Typed manifest surfaces for orchestration ownership and route-to-surface mapping.

BEGIN;

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_control;
CREATE SCHEMA IF NOT EXISTS lucidota_canon;

CREATE TABLE IF NOT EXISTS lucidota_control.schema_owner_manifest (
    surface_id text PRIMARY KEY,
    canonical_owner text NOT NULL,
    packet_class text NOT NULL DEFAULT 'typed_packet',
    surface_kind text NOT NULL DEFAULT 'view',
    approval_required boolean NOT NULL DEFAULT true,
    active boolean NOT NULL DEFAULT true,
    approved_by text NOT NULL DEFAULT '',
    approved_at timestamptz,
    approval_receipt_uuid uuid,
    approval_note text NOT NULL DEFAULT '',
    notes text NOT NULL DEFAULT '',
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE IF EXISTS lucidota_control.schema_owner_manifest
    ADD COLUMN IF NOT EXISTS approved_by text NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS approved_at timestamptz,
    ADD COLUMN IF NOT EXISTS approval_receipt_uuid uuid,
    ADD COLUMN IF NOT EXISTS approval_note text NOT NULL DEFAULT '';

CREATE OR REPLACE FUNCTION lucidota_control.guard_schema_owner_manifest_redefinition()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF OLD.active
       AND NEW.active
       AND (
            OLD.canonical_owner IS DISTINCT FROM NEW.canonical_owner
            OR OLD.packet_class IS DISTINCT FROM NEW.packet_class
            OR OLD.surface_kind IS DISTINCT FROM NEW.surface_kind
            OR OLD.approval_required IS DISTINCT FROM NEW.approval_required
       )
       AND (
            COALESCE(NULLIF(NEW.approved_by, ''), '') = ''
            OR NEW.approved_at IS NULL
            OR NEW.approval_receipt_uuid IS NULL
       ) THEN
        RAISE EXCEPTION
            'schema_owner_manifest active surface redefinition requires approved_by, approved_at, and approval_receipt_uuid'
            USING ERRCODE = '42501';
    END IF;

    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS schema_owner_manifest_redefinition_guard ON lucidota_control.schema_owner_manifest;
CREATE TRIGGER schema_owner_manifest_redefinition_guard
BEFORE UPDATE ON lucidota_control.schema_owner_manifest
FOR EACH ROW
EXECUTE FUNCTION lucidota_control.guard_schema_owner_manifest_redefinition();

CREATE OR REPLACE VIEW lucidota_canon.schema_owner_manifest AS
SELECT
    surface_id,
    canonical_owner,
    packet_class,
    surface_kind,
    approval_required,
    active,
    notes,
    detail,
    created_at,
    updated_at,
    approved_by,
    approved_at,
    approval_receipt_uuid,
    approval_note
FROM lucidota_control.schema_owner_manifest
WHERE active;

CREATE OR REPLACE VIEW lucidota_canon.surface_registry AS
SELECT
    arc.route_id AS surface_id,
    arc.route_id,
    arc.method,
    arc.path_pattern,
    arc.description,
    arc.target,
    arc.status,
    COALESCE(asm.canonical_owner, 'lucidota_canon') AS canonical_owner,
    COALESCE(asm.packet_class, 'typed_packet') AS packet_class,
    COALESCE(asm.surface_kind, 'route') AS surface_kind,
    COALESCE(asm.approval_required, true) AS approval_required,
    COALESCE(asm.active, true) AS active,
    COALESCE(asm.notes, '') AS notes,
    COALESCE(asm.detail, '{}'::jsonb) AS detail,
    arc.updated_at,
    jsonb_build_object(
        'surface_id', arc.route_id,
        'route_id', arc.route_id,
        'target', arc.target,
        'packet_class', COALESCE(asm.packet_class, 'typed_packet'),
        'canonical_owner', COALESCE(asm.canonical_owner, 'lucidota_canon'),
        'approval_required', COALESCE(asm.approval_required, true),
        'approved_by', COALESCE(asm.approved_by, ''),
        'approved_at', asm.approved_at,
        'approval_receipt_uuid', asm.approval_receipt_uuid,
        'approval_note', COALESCE(asm.approval_note, '')
    ) AS packet,
    jsonb_build_object(
        'mode', 'sub_orchestrator',
        'sub_orchestrator_priority', lucidota_control.live_truth_priority_stack(),
        'strict_priority_stack', lucidota_control.live_truth_priority_stack(),
        'surface_ref', arc.route_id,
        'owner_ref', COALESCE(asm.canonical_owner, 'lucidota_canon'),
        'packet_class', COALESCE(asm.packet_class, 'typed_packet'),
        'approved_by', COALESCE(asm.approved_by, ''),
        'approved_at', asm.approved_at,
        'approval_receipt_uuid', asm.approval_receipt_uuid
    ) AS orchestration,
    jsonb_build_array(
        'schema_owner_manifest',
        'surface_registry',
        'renderer_registry',
        'command_registry',
        'capability_registry'
    ) AS next_command_refs,
    COALESCE(asm.approved_by, '') AS approved_by,
    asm.approved_at,
    asm.approval_receipt_uuid,
    COALESCE(asm.approval_note, '') AS approval_note
FROM lucidota_canon.api_route_catalog arc
LEFT JOIN lucidota_control.schema_owner_manifest asm
    ON asm.surface_id = arc.route_id
WHERE arc.status = 'implemented'
  AND COALESCE(asm.active, true);

CREATE OR REPLACE VIEW lucidota_canon.renderer_registry AS
WITH renderers AS (
    SELECT
        script_path AS renderer_id,
        count(*) AS command_count,
        bool_or(active) AS active,
        min(created_at) AS created_at,
        max(created_at) AS updated_at
    FROM lucidota_control.worker_command_registry
    GROUP BY script_path
)
SELECT
    renderer_id,
    'script_path'::text AS renderer_kind,
    command_count,
    active,
    'worker_command_registry'::text AS source_surface,
    created_at,
    updated_at,
    jsonb_build_object(
        'renderer_id', renderer_id,
        'renderer_kind', 'script_path',
        'command_count', command_count,
        'source_surface', 'worker_command_registry'
    ) AS packet,
    jsonb_build_object(
        'mode', 'sub_orchestrator',
        'sub_orchestrator_priority', lucidota_control.live_truth_priority_stack(),
        'strict_priority_stack', lucidota_control.live_truth_priority_stack(),
        'renderer_ref', renderer_id,
        'source_surface', 'worker_command_registry',
        'active', active
    ) AS orchestration,
    jsonb_build_array(
        'schema_owner_manifest',
        'surface_registry',
        'renderer_registry',
        'command_registry',
        'capability_registry'
    ) AS next_command_refs
FROM renderers;

INSERT INTO lucidota_control.schema_owner_manifest (
    surface_id, canonical_owner, packet_class, surface_kind, approval_required, notes, detail
) VALUES
    ('schema_owner_manifest', 'lucidota_control', 'typed_packet', 'table', true, 'Canonical owner map for active surfaces.', '{"source":"control"}'::jsonb),
    ('surface_registry', 'lucidota_canon', 'typed_packet', 'view', true, 'Route-to-surface registry over implemented API routes.', '{"source":"api_route_catalog"}'::jsonb),
    ('renderer_registry', 'lucidota_canon', 'typed_packet', 'view', true, 'Distinct renderer inventory derived from command registry rows.', '{"source":"worker_command_registry"}'::jsonb),
    ('command_registry', 'lucidota_canon', 'typed_packet', 'view', true, 'Typed command registry view over worker command rails.', '{"source":"worker_command_registry"}'::jsonb),
    ('manual_current', 'lucidota_canon', 'typed_packet', 'view', true, 'Manual spine route surface.', '{"source":"manual_current"}'::jsonb),
    ('root_orchestrator_current', 'lucidota_canon', 'typed_packet', 'view', true, 'Root orchestrator current packet.', '{"source":"root_orchestrator_current"}'::jsonb),
    ('daemon_status', 'lucidota_canon', 'typed_packet', 'view', true, 'Daemon status packet.', '{"source":"daemon_status"}'::jsonb),
    ('capability_registry', 'lucidota_canon', 'typed_packet', 'view', true, 'Capability registry packet.', '{"source":"capability_registry"}'::jsonb),
    ('model_registry_current', 'lucidota_canon', 'typed_packet', 'view', true, 'Model registry current packet.', '{"source":"model_registry_current"}'::jsonb),
    ('provider_current', 'lucidota_canon', 'typed_packet', 'view', true, 'Provider current packet.', '{"source":"provider_current"}'::jsonb),
    ('workflow_current', 'lucidota_canon', 'typed_packet', 'view', true, 'Workflow current packet.', '{"source":"workflow_current"}'::jsonb)
ON CONFLICT (surface_id) DO UPDATE SET
    canonical_owner = EXCLUDED.canonical_owner,
    packet_class = EXCLUDED.packet_class,
    surface_kind = EXCLUDED.surface_kind,
    approval_required = EXCLUDED.approval_required,
    active = true,
    notes = EXCLUDED.notes,
    detail = EXCLUDED.detail,
    updated_at = now();

INSERT INTO lucidota_canon.api_route_catalog (
    route_id, method, path_pattern, description, target, sample_request, sample_response, status
) VALUES
    (
        'schema_owner_manifest',
        'GET',
        '/schema_owner_manifest',
        'Canonical owner manifest for active surfaces, packet class, and approval policy.',
        'lucidota_canon.schema_owner_manifest',
        '{"limit":"5"}',
        '{"surface_id":"manual_current","canonical_owner":"lucidota_canon","packet_class":"typed_packet"}',
        'implemented'
    ),
    (
        'surface_registry',
        'GET',
        '/surface_registry',
        'Route-to-surface registry with owner, packet class, and route metadata.',
        'lucidota_canon.surface_registry',
        '{"limit":"5"}',
        '{"surface_id":"manual_current","target":"lucidota_canon.manual_current"}',
        'implemented'
    ),
    (
        'renderer_registry',
        'GET',
        '/renderer_registry',
        'Renderer inventory derived from command registry script paths.',
        'lucidota_canon.renderer_registry',
        '{"limit":"5"}',
        '{"renderer_id":"./scripts/example.py","renderer_kind":"script_path"}',
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

GRANT SELECT ON lucidota_canon.schema_owner_manifest, lucidota_canon.surface_registry, lucidota_canon.renderer_registry TO lucidota_postgrest_anon, mfspx;

NOTIFY pgrst, 'reload schema';

COMMIT;
