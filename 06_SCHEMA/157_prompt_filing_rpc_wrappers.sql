-- Canonical-schema RPC wrappers for the prompt ledger.
-- PostgREST exposes rpc routes only for visible schema functions, so the
-- prompt_api implementation functions are mirrored here for live routing.

BEGIN;

CREATE OR REPLACE FUNCTION lucidota_canon.file_prompt(
    source text DEFAULT 'operator',
    source_model text DEFAULT '',
    receiving_model text DEFAULT '',
    target_model text DEFAULT '',
    raw_prompt_text text DEFAULT '',
    normalized_prompt_text text DEFAULT '',
    conversation_session_id text DEFAULT '',
    parent_prompt_id uuid DEFAULT NULL,
    linked_work_order_uuid uuid[] DEFAULT '{}'::uuid[],
    linked_receipt_uuid uuid[] DEFAULT '{}'::uuid[],
    linked_goal_id text DEFAULT '',
    ontology_tags text[] DEFAULT '{}'::text[],
    subsystem_tags text[] DEFAULT '{}'::text[],
    status text DEFAULT 'filed',
    notes text DEFAULT '',
    blockers text DEFAULT '',
    idempotency_key text DEFAULT '',
    source_path text DEFAULT '',
    received_at timestamptz DEFAULT now(),
    received_at_confidence numeric DEFAULT 1,
    received_at_basis text DEFAULT 'provided',
    detail jsonb DEFAULT '{}'::jsonb
) RETURNS jsonb
LANGUAGE sql
SECURITY DEFINER
SET search_path = prompt_api, lucidota_canon, lucidota_control, public
AS $$
    SELECT prompt_api.file_prompt(
        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
        $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22
    );
$$;

CREATE OR REPLACE FUNCTION lucidota_canon.link_prompt_work_order(
    p_prompt_id uuid,
    p_work_order_uuid uuid,
    p_link_reason text DEFAULT '',
    p_linked_goal_id text DEFAULT ''
) RETURNS jsonb
LANGUAGE sql
SECURITY DEFINER
SET search_path = prompt_api, lucidota_canon, lucidota_control, public
AS $$
    SELECT prompt_api.link_prompt_work_order($1, $2, $3, $4);
$$;

CREATE OR REPLACE FUNCTION lucidota_canon.decompose_prompt_to_work_orders(
    prompt_id uuid,
    max_items integer DEFAULT 1,
    task_type text DEFAULT '',
    target_model text DEFAULT ''
) RETURNS jsonb
LANGUAGE sql
SECURITY DEFINER
SET search_path = prompt_api, lucidota_canon, lucidota_control, public
AS $$
    SELECT prompt_api.decompose_prompt_to_work_orders($1, $2, $3, $4);
$$;

INSERT INTO lucidota_canon.api_route_catalog (
    route_id, method, path_pattern, description, target, sample_request, sample_response, status
) VALUES
('file_prompt', 'POST', '/rpc/file_prompt', 'File a prompt into the DB-backed prompt ledger.', 'lucidota_canon.file_prompt',
 '{"source":"operator","raw_prompt_text":"..."}', '{"prompt_id":"...","status":"filed"}', 'implemented'),
('link_prompt_work_order', 'POST', '/rpc/link_prompt_work_order', 'Attach a filed prompt to a work order UUID.', 'lucidota_canon.link_prompt_work_order',
 '{"prompt_id":"...","work_order_uuid":"..."}', '{"linked_work_order_uuid":["..."]}', 'implemented'),
('decompose_prompt_to_work_orders', 'POST', '/rpc/decompose_prompt_to_work_orders', 'Deterministically decompose a prompt into a candidate DB-visible work order and link it.', 'lucidota_canon.decompose_prompt_to_work_orders',
 '{"prompt_id":"..."}', '{"prompt_id":"...","work_order_uuid":"..."}', 'implemented'),
('cloud_packet', 'POST', '/rpc/cloud_packet', 'Bounded prompt packet RPC for cloud/model callers.', 'lucidota_canon.cloud_packet',
 '{"work_order_id":"...","max_chars":8000,"max_items":12}', '{"contract_name":"prompt_api.cloud_packet.v1"}', 'implemented')
ON CONFLICT (route_id) DO UPDATE SET
    method = EXCLUDED.method,
    path_pattern = EXCLUDED.path_pattern,
    description = EXCLUDED.description,
    target = EXCLUDED.target,
    sample_request = EXCLUDED.sample_request,
    sample_response = EXCLUDED.sample_response,
    status = EXCLUDED.status,
    updated_at = now();

GRANT EXECUTE ON FUNCTION lucidota_canon.file_prompt(
    text, text, text, text, text, text, text, uuid, uuid[], uuid[], text, text[], text[], text, text, text, text, text, timestamptz, numeric, text, jsonb
) TO lucidota_postgrest_anon, mfspx;
GRANT EXECUTE ON FUNCTION lucidota_canon.link_prompt_work_order(uuid, uuid, text, text) TO lucidota_postgrest_anon, mfspx;
GRANT EXECUTE ON FUNCTION lucidota_canon.decompose_prompt_to_work_orders(uuid, integer, text, text) TO lucidota_postgrest_anon, mfspx;

NOTIFY pgrst, 'reload schema';

COMMIT;
