-- DB-visible command registry packet: expose worker command rows as typed command/capability references.

BEGIN;

DROP VIEW IF EXISTS lucidota_canon.command_registry CASCADE;

CREATE OR REPLACE VIEW lucidota_canon.command_registry AS
SELECT
    command_key AS command_id,
    CASE
        WHEN queue_name = '*' AND job_kind = '*' THEN command_key
        ELSE queue_name || ':' || job_kind
    END AS route_id,
    handler AS capability_id,
    'worker_command_registry'::text AS surface_id,
    COALESCE(detail ->> 'intent', command_key) AS intent,
    script_path AS renderer_id,
    'worker_command_registry'::text AS surface_ref,
    handler AS capability_ref,
    script_path AS renderer_ref,
    queue_name,
    job_kind,
    active,
    created_at,
    detail,
    jsonb_build_object(
        'command_key', command_key,
        'command_id', command_key,
        'route_id', CASE
            WHEN queue_name = '*' AND job_kind = '*' THEN command_key
            ELSE queue_name || ':' || job_kind
        END,
        'capability_id', handler,
        'surface_id', 'worker_command_registry',
        'surface_ref', 'worker_command_registry',
        'capability_ref', handler,
        'renderer_id', script_path,
        'renderer_ref', script_path,
        'queue_name', queue_name,
        'job_kind', job_kind,
        'handler', handler,
        'script_path', script_path
    ) AS packet,
    jsonb_build_array(
        'surface_registry',
        'renderer_registry',
        'schema_owner_manifest',
        'controller_grant',
        'agent_thread_runtime'
    ) AS next_command_refs,
    jsonb_build_object(
        'mode', 'sub_orchestrator',
        'sub_orchestrator_priority', lucidota_control.live_truth_priority_stack(),
        'strict_priority_stack', lucidota_control.live_truth_priority_stack()
    ) AS orchestration
FROM lucidota_control.worker_command_registry;

INSERT INTO lucidota_canon.api_route_catalog (
    route_id, method, path_pattern, description, target, sample_request, sample_response, status
) VALUES (
    'command_registry',
    'GET',
    '/command_registry',
    'Typed command registry packet for command_id, route_id, capability_id, surface_id, intent, and renderer_id rows.',
    'lucidota_canon.command_registry',
    '{"limit":"1"}',
    '{"command_id":"generic.external_command"}',
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

GRANT SELECT ON lucidota_canon.command_registry TO lucidota_postgrest_anon, mfspx;

NOTIFY pgrst, 'reload schema';

COMMIT;
