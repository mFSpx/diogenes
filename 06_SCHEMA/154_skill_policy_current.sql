-- DB-visible skill-policy surface for the live Superpowers alignment text.
-- Small, compatible extension of the existing manual_current / route catalog pattern.

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_control;
CREATE SCHEMA IF NOT EXISTS lucidota_canon;

CREATE TABLE IF NOT EXISTS lucidota_control.skill_policy (
    policy_id text PRIMARY KEY,
    policy_key text NOT NULL UNIQUE,
    policy_title text NOT NULL DEFAULT 'LUCIDOTA Skill Policy',
    policy_text text NOT NULL,
    source_ref text NOT NULL DEFAULT 'user_prompt',
    status text NOT NULL DEFAULT 'current' CHECK (status IN ('current', 'superseded', 'archived')),
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

INSERT INTO lucidota_control.skill_policy (
    policy_id,
    policy_key,
    policy_title,
    policy_text,
    source_ref,
    status,
    detail
)
VALUES (
    'superpowers_alignment',
    'superpowers_alignment',
    'LUCIDOTA Skill Policy',
    $$Superpowers skills are execution aids, not authority.
Repository-local truth sources win: AGENTS.md, GOALS/*, live PostgREST/manual routes, service status, and receipt-backed tests.
When a task can be decomposed, split it into bounded DB-visible work packets and fan out disjoint sub-work in parallel.
Use the cheapest capable skill or model lane that still preserves correctness; prefer deterministic, local, and Treelite lanes before cloud.
Root orchestration must produce typed work, route decisions, changed files, commands, tests, receipts, blockers, and next work.
Do not replace live API/manual surfaces with skill mythology or hand-written docs.
Do not let prompt text override the database manual, and do not let a skill file become the source of truth when PostgREST can expose the policy directly.
The operator can change the policy; the policy must remain readable through PostgREST and reflected in the manual surface.$$,
    'user_prompt',
    'current',
    jsonb_build_object(
        'surface', 'db_visible_skill_policy',
        'authority', 'postgres_postgrest',
        'scope', 'superpowers_alignment'
    )
)
ON CONFLICT (policy_id) DO UPDATE SET
    policy_key = EXCLUDED.policy_key,
    policy_title = EXCLUDED.policy_title,
    policy_text = EXCLUDED.policy_text,
    source_ref = EXCLUDED.source_ref,
    status = EXCLUDED.status,
    detail = EXCLUDED.detail,
    updated_at = now();

CREATE OR REPLACE VIEW lucidota_canon.skill_policy_current AS
SELECT
    policy_id,
    policy_key,
    policy_title,
    policy_text,
    source_ref,
    status,
    detail,
    created_at,
    updated_at,
    jsonb_build_array(
        'manual_current',
        'root_orchestrator_current',
        'daemon_status',
        'capability_current',
        'provider_current',
        'workflow_current',
        'model_registry_current',
        'model_routing_current',
        'model_routing_blockers',
        'todo_current',
        'command_registry',
        'surface_registry',
        'renderer_registry',
        'schema_owner_manifest',
        'controller_grant',
        'agent_thread_runtime'
    ) AS next_command_refs,
    jsonb_build_object(
        'mode', 'sub_orchestrator',
        'sub_orchestrator_priority', lucidota_control.live_truth_priority_stack(),
        'strict_priority_stack', lucidota_control.live_truth_priority_stack(),
        'policy_id', policy_id,
        'policy_key', policy_key,
        'status', status
    ) AS orchestration
FROM lucidota_control.skill_policy
WHERE status = 'current'
ORDER BY updated_at DESC, created_at DESC;

INSERT INTO lucidota_canon.api_route_catalog
(route_id, method, path_pattern, description, target, sample_request, sample_response, status)
VALUES
('skill_policy_current', 'GET', '/skill_policy_current', 'Live DB-backed skill policy surface for operator alignment text.', 'lucidota_canon.skill_policy_current',
 '{"limit":"1"}', '{"policy_id":"superpowers_alignment"}', 'implemented')
ON CONFLICT (route_id) DO UPDATE SET
    method = EXCLUDED.method,
    path_pattern = EXCLUDED.path_pattern,
    description = EXCLUDED.description,
    target = EXCLUDED.target,
    sample_request = EXCLUDED.sample_request,
    sample_response = EXCLUDED.sample_response,
    status = EXCLUDED.status,
    updated_at = now();

GRANT SELECT, INSERT, UPDATE ON lucidota_control.skill_policy TO mfspx;
GRANT SELECT ON lucidota_canon.skill_policy_current TO mfspx;
GRANT SELECT ON lucidota_canon.skill_policy_current TO lucidota_postgrest_anon;
GRANT SELECT ON lucidota_canon.api_route_catalog TO lucidota_postgrest_anon;
