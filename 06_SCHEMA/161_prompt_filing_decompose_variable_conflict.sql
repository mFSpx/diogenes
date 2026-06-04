-- Fix prompt_api.decompose_prompt_to_work_orders variable ambiguity and keep decomposition callable.

BEGIN;

CREATE OR REPLACE FUNCTION prompt_api.decompose_prompt_to_work_orders(
    prompt_id uuid,
    max_items integer DEFAULT 1,
    task_type text DEFAULT '',
    target_model text DEFAULT ''
) RETURNS jsonb
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = prompt_api, lucidota_canon, lucidota_control, public
VOLATILE
AS $$
DECLARE
    prompt_row lucidota_control.prompt_record%ROWTYPE;
    classification jsonb;
    prompt_text text;
    prompt_event_id text;
    prompt_raw_artifact_uuid uuid;
    created_work_order_uuid uuid;
    work_order_key text;
    work_kind text;
    lane text;
    payload jsonb;
    prompt_link_count integer := 0;
BEGIN
    SELECT * INTO prompt_row
    FROM lucidota_control.prompt_record
    WHERE prompt_record.prompt_id = $1;

    IF NOT FOUND THEN
        RETURN jsonb_build_object('prompt_id', prompt_id::text, 'error', 'prompt_not_found');
    END IF;

    prompt_text := COALESCE(NULLIF(prompt_row.raw_prompt_text, ''), prompt_row.normalized_prompt_text);
    classification := prompt_api.classify_prompt_text(prompt_text);
    work_kind := COALESCE(NULLIF(task_type, ''), 'prompt_decomposition');
    lane := CASE classification->>'subsystem'
        WHEN 'manual_api' THEN 'audit'
        WHEN 'indy_daemon' THEN 'audit'
        WHEN 'verification' THEN 'audit'
        WHEN 'rust_rewrite' THEN 'slow'
        WHEN 'book_ops' THEN 'external'
        WHEN 'model_orchestration' THEN 'external'
        ELSE 'audit'
    END;

    prompt_event_id := encode(digest('prompt:' || prompt_row.prompt_id::text || ':decompose', 'sha256'), 'hex');
    work_order_key := 'prompt-ledger:' || prompt_row.idempotency_key || ':' || COALESCE(NULLIF(target_model, ''), 'default') || ':' || lane;
    payload := jsonb_build_object(
        'prompt_id', prompt_row.prompt_id::text,
        'prompt_hash', prompt_row.prompt_hash,
        'prompt_source', prompt_row.source,
        'source_model', prompt_row.source_model,
        'receiving_model', prompt_row.receiving_model,
        'target_model', COALESCE(NULLIF(target_model, ''), prompt_row.target_model),
        'subsystem', classification->>'subsystem',
        'ontology_tags', classification->'ontology_tags',
        'go_co_io_tags', classification->'go_co_io_tags',
        'risk', classification->>'risk',
        'parallelizable', (classification->>'parallelizable')::boolean,
        'serialized', (classification->>'serialized')::boolean,
        'executor_role', classification->>'executor_role',
        'acceptance_test', classification->>'acceptance_test',
        'receipt_requirement', classification->>'receipt_requirement',
        'next_action', classification->>'next_action',
        'raw_prompt_excerpt', left(prompt_text, 1200),
        'prompt_status', prompt_row.status
    );

    INSERT INTO lucidota_control.raw_artifact(
        raw_ref, raw_sha256, hash_algo, source, actor, byte_count, char_count, mime_type, storage_hint, detail
    ) VALUES (
        'prompt://' || prompt_row.prompt_id::text,
        prompt_row.prompt_hash,
        'sha256',
        'prompt_ledger',
        'prompt_api',
        octet_length(prompt_text),
        char_length(prompt_text),
        'text/plain',
        'inline',
        jsonb_build_object('prompt_id', prompt_row.prompt_id::text, 'prompt_hash', prompt_row.prompt_hash)
    )
    ON CONFLICT (raw_ref) DO UPDATE
      SET detail = lucidota_control.raw_artifact.detail || EXCLUDED.detail
    RETURNING lucidota_control.raw_artifact.raw_artifact_uuid INTO prompt_raw_artifact_uuid;

    INSERT INTO lucidota_control.event_envelope(
        event_id, ts, source, actor, raw_ref, raw_artifact_uuid, verbatim_hash, hash_algo,
        text, entities, claims, actions_requested, artifacts_referenced, risk_flags, route_candidates,
        board_features, embedding_ref, detail
    ) VALUES (
        prompt_event_id,
        now(),
        'prompt_ledger',
        'worker',
        'prompt://' || prompt_row.prompt_id::text,
        prompt_raw_artifact_uuid,
        prompt_row.prompt_hash,
        'sha256',
        prompt_text,
        '[]'::jsonb,
        '[]'::jsonb,
        jsonb_build_array('decompose_prompt_to_work_orders'),
        jsonb_build_array('prompt://' || prompt_row.prompt_id::text),
        '[]'::jsonb,
        jsonb_build_array('/prompt_recent', '/prompt_unlinked', '/prompt_catalog_status'),
        jsonb_build_object('subsystem', classification->>'subsystem', 'risk', classification->>'risk'),
        NULL,
        jsonb_build_object('prompt_id', prompt_row.prompt_id::text, 'decompose', true)
    )
    ON CONFLICT (event_id) DO UPDATE
      SET detail = lucidota_control.event_envelope.detail || EXCLUDED.detail;

    INSERT INTO lucidota_control.work_order(
        event_id, lane, work_kind, status, payload, idempotency_key
    ) VALUES (
        prompt_event_id,
        lane,
        work_kind,
        'queued',
        payload,
        work_order_key
    )
    ON CONFLICT (idempotency_key) DO UPDATE
      SET status = EXCLUDED.status,
          payload = EXCLUDED.payload,
          updated_at = now()
    RETURNING work_order_uuid INTO created_work_order_uuid;

    PERFORM prompt_api.link_prompt_work_order(prompt_row.prompt_id, created_work_order_uuid, 'auto-decomposed from prompt ledger', COALESCE(prompt_row.linked_goal_id, ''));

    SELECT cardinality(linked_work_order_uuid)
    INTO prompt_link_count
    FROM lucidota_control.prompt_record
    WHERE prompt_id = prompt_row.prompt_id;

    UPDATE lucidota_control.prompt_record
    SET status = CASE WHEN status = 'filed' THEN 'decomposed' ELSE status END,
        subsystem_tags = CASE WHEN cardinality(subsystem_tags) = 0 THEN ARRAY[classification->>'subsystem', 'prompt-ledger']::text[] ELSE subsystem_tags END,
        ontology_tags = CASE WHEN cardinality(ontology_tags) = 0 THEN ARRAY(SELECT jsonb_array_elements_text(classification->'ontology_tags')) ELSE ontology_tags END,
        go_co_io_tags = CASE WHEN cardinality(go_co_io_tags) = 0 THEN ARRAY(SELECT jsonb_array_elements_text(classification->'go_co_io_tags')) ELSE go_co_io_tags END,
        updated_at = now()
    WHERE prompt_id = prompt_row.prompt_id;

    RETURN jsonb_build_object(
        'prompt_id', prompt_row.prompt_id::text,
        'status', 'decomposed',
        'work_order_uuid', created_work_order_uuid::text,
        'linked_work_order_uuid', ARRAY[created_work_order_uuid]::uuid[],
        'subsystem_tags', COALESCE(prompt_row.subsystem_tags, ARRAY[classification->>'subsystem', 'prompt-ledger']::text[]),
        'ontology_tags', COALESCE(prompt_row.ontology_tags, ARRAY(SELECT jsonb_array_elements_text(classification->'ontology_tags'))),
        'go_co_io_tags', COALESCE(prompt_row.go_co_io_tags, ARRAY(SELECT jsonb_array_elements_text(classification->'go_co_io_tags'))),
        'linked_count', prompt_link_count,
        'lane', lane,
        'work_kind', work_kind,
        'prompt_hash', prompt_row.prompt_hash,
        'next_action', classification->>'next_action'
    );
END;
$$;

DROP FUNCTION IF EXISTS lucidota_canon.file_prompt(text, text, text, text, text, text, text, uuid, uuid[], uuid[], text, text[], text[], text, text, text, text, text, timestamptz, numeric, text, jsonb);
DROP FUNCTION IF EXISTS lucidota_canon.link_prompt_work_order(uuid, uuid, text, text);
DROP FUNCTION IF EXISTS lucidota_canon.decompose_prompt_to_work_orders(uuid, integer, text, text);

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
VOLATILE
SET search_path = lucidota_canon, prompt_api, public
AS $$
    SELECT prompt_api.file_prompt(
        source => source,
        source_model => source_model,
        receiving_model => receiving_model,
        target_model => target_model,
        raw_prompt_text => raw_prompt_text,
        normalized_prompt_text => normalized_prompt_text,
        conversation_session_id => conversation_session_id,
        parent_prompt_id => parent_prompt_id,
        linked_work_order_uuid => linked_work_order_uuid,
        linked_receipt_uuid => linked_receipt_uuid,
        linked_goal_id => linked_goal_id,
        ontology_tags => ontology_tags,
        subsystem_tags => subsystem_tags,
        status => status,
        notes => notes,
        blockers => blockers,
        idempotency_key => idempotency_key,
        source_path => source_path,
        received_at => received_at,
        received_at_confidence => received_at_confidence,
        received_at_basis => received_at_basis,
        detail => detail
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
VOLATILE
SET search_path = lucidota_canon, prompt_api, public
AS $$
    SELECT prompt_api.link_prompt_work_order(
        p_prompt_id => p_prompt_id,
        p_work_order_uuid => p_work_order_uuid,
        p_link_reason => p_link_reason,
        p_linked_goal_id => p_linked_goal_id
    );
$$;

CREATE OR REPLACE FUNCTION lucidota_canon.decompose_prompt_to_work_orders(
    prompt_id uuid,
    max_items integer DEFAULT 1,
    task_type text DEFAULT '',
    target_model text DEFAULT ''
) RETURNS jsonb
LANGUAGE sql
SECURITY DEFINER
VOLATILE
SET search_path = lucidota_canon, prompt_api, public
AS $$
    SELECT prompt_api.decompose_prompt_to_work_orders(
        prompt_id => prompt_id,
        max_items => max_items,
        task_type => task_type,
        target_model => target_model
    );
$$;

NOTIFY pgrst, 'reload schema';

COMMIT;
