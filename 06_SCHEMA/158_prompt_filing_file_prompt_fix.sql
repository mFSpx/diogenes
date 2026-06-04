-- Fix prompt_api.file_prompt ambiguity and keep the prompt ledger route callable.

BEGIN;

CREATE OR REPLACE FUNCTION prompt_api.file_prompt(
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
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = prompt_api, lucidota_canon, lucidota_control, public
VOLATILE
AS $$
DECLARE
    normalized text := COALESCE(NULLIF(normalized_prompt_text, ''), prompt_api.normalize_prompt_text(raw_prompt_text));
    prompt_hash text := prompt_api.prompt_hash(COALESCE(raw_prompt_text, normalized_prompt_text));
    classification jsonb := prompt_api.classify_prompt_text(COALESCE(raw_prompt_text, normalized_prompt_text));
    prompt_row lucidota_control.prompt_record%ROWTYPE;
    resolved_idempotency text := COALESCE(
        NULLIF(idempotency_key, ''),
        encode(digest(
            jsonb_build_object(
                'source', source,
                'source_model', source_model,
                'receiving_model', receiving_model,
                'target_model', target_model,
                'prompt_hash', prompt_hash,
                'conversation_session_id', conversation_session_id,
                'parent_prompt_id', parent_prompt_id::text,
                'linked_goal_id', linked_goal_id,
                'status', status
            )::text,
            'sha256'
        ), 'hex')
    );
    work_order_ids uuid[] := COALESCE(linked_work_order_uuid, '{}'::uuid[]);
    receipt_ids uuid[] := COALESCE(linked_receipt_uuid, '{}'::uuid[]);
    ontology text[] := CASE WHEN cardinality(ontology_tags) > 0 THEN ontology_tags ELSE COALESCE(ARRAY(SELECT jsonb_array_elements_text(classification->'ontology_tags')), ARRAY['PROMPT']::text[]) END;
    subsystems text[] := CASE WHEN cardinality(subsystem_tags) > 0 THEN subsystem_tags ELSE ARRAY[COALESCE(classification->>'subsystem', 'mixed'), 'prompt-ledger']::text[] END;
    confidence numeric(4,3) := COALESCE(received_at_confidence::numeric(4,3), 1.000);
    basis text := COALESCE(NULLIF(received_at_basis, ''), 'provided');
    inserted_prompt_id uuid;
BEGIN
    INSERT INTO lucidota_control.prompt_record (
        received_at, source, source_model, receiving_model, target_model,
        raw_prompt_text, normalized_prompt_text, prompt_hash,
        conversation_id, session_id, parent_prompt_id,
        derived_prompt_ids, linked_work_order_uuid, linked_receipt_uuid,
        linked_goal_id, ontology_tags, go_co_io_tags, subsystem_tags,
        status, notes, blockers, source_path, received_at_basis,
        received_at_confidence, detail, idempotency_key
    ) VALUES (
        COALESCE(received_at, now()),
        COALESCE(NULLIF(source, ''), 'operator'),
        COALESCE(source_model, ''),
        COALESCE(receiving_model, ''),
        COALESCE(target_model, ''),
        COALESCE(raw_prompt_text, ''),
        normalized,
        prompt_hash,
        '',
        COALESCE(conversation_session_id, ''),
        parent_prompt_id,
        '{}'::uuid[],
        work_order_ids,
        receipt_ids,
        COALESCE(linked_goal_id, ''),
        ontology,
        COALESCE(ARRAY(SELECT jsonb_array_elements_text(classification->'go_co_io_tags')), ARRAY['GO']::text[]),
        subsystems,
        COALESCE(NULLIF(status, ''), 'filed'),
        COALESCE(notes, ''),
        COALESCE(blockers, ''),
        COALESCE(source_path, ''),
        basis,
        confidence,
        COALESCE(detail, '{}'::jsonb) || classification,
        resolved_idempotency
    )
    ON CONFLICT ON CONSTRAINT prompt_record_idempotency_key_key DO UPDATE SET
        received_at = LEAST(lucidota_control.prompt_record.received_at, EXCLUDED.received_at),
        source_model = EXCLUDED.source_model,
        receiving_model = EXCLUDED.receiving_model,
        target_model = EXCLUDED.target_model,
        raw_prompt_text = CASE WHEN lucidota_control.prompt_record.raw_prompt_text = '' THEN EXCLUDED.raw_prompt_text ELSE lucidota_control.prompt_record.raw_prompt_text END,
        normalized_prompt_text = CASE WHEN lucidota_control.prompt_record.normalized_prompt_text = '' THEN EXCLUDED.normalized_prompt_text ELSE lucidota_control.prompt_record.normalized_prompt_text END,
        prompt_hash = EXCLUDED.prompt_hash,
        conversation_id = COALESCE(NULLIF(lucidota_control.prompt_record.conversation_id, ''), EXCLUDED.conversation_id),
        session_id = COALESCE(NULLIF(lucidota_control.prompt_record.session_id, ''), EXCLUDED.session_id),
        parent_prompt_id = COALESCE(lucidota_control.prompt_record.parent_prompt_id, EXCLUDED.parent_prompt_id),
        derived_prompt_ids = CASE
            WHEN cardinality(lucidota_control.prompt_record.derived_prompt_ids) = 0 THEN EXCLUDED.derived_prompt_ids
            ELSE lucidota_control.prompt_record.derived_prompt_ids
        END,
        linked_work_order_uuid = (
            SELECT COALESCE(array_agg(DISTINCT elem ORDER BY elem), '{}'::uuid[])
            FROM unnest(COALESCE(lucidota_control.prompt_record.linked_work_order_uuid, '{}'::uuid[]) || COALESCE(EXCLUDED.linked_work_order_uuid, '{}'::uuid[])) AS elem
        ),
        linked_receipt_uuid = (
            SELECT COALESCE(array_agg(DISTINCT elem ORDER BY elem), '{}'::uuid[])
            FROM unnest(COALESCE(lucidota_control.prompt_record.linked_receipt_uuid, '{}'::uuid[]) || COALESCE(EXCLUDED.linked_receipt_uuid, '{}'::uuid[])) AS elem
        ),
        linked_goal_id = COALESCE(NULLIF(lucidota_control.prompt_record.linked_goal_id, ''), EXCLUDED.linked_goal_id),
        ontology_tags = CASE
            WHEN cardinality(lucidota_control.prompt_record.ontology_tags) = 0 THEN EXCLUDED.ontology_tags
            ELSE lucidota_control.prompt_record.ontology_tags
        END,
        go_co_io_tags = CASE
            WHEN cardinality(lucidota_control.prompt_record.go_co_io_tags) = 0 THEN EXCLUDED.go_co_io_tags
            ELSE lucidota_control.prompt_record.go_co_io_tags
        END,
        subsystem_tags = CASE
            WHEN cardinality(lucidota_control.prompt_record.subsystem_tags) = 0 THEN EXCLUDED.subsystem_tags
            ELSE lucidota_control.prompt_record.subsystem_tags
        END,
        status = CASE
            WHEN lucidota_control.prompt_record.status = 'archived' THEN 'archived'
            ELSE COALESCE(NULLIF(EXCLUDED.status, ''), lucidota_control.prompt_record.status)
        END,
        notes = CASE WHEN lucidota_control.prompt_record.notes = '' THEN EXCLUDED.notes ELSE lucidota_control.prompt_record.notes END,
        blockers = CASE WHEN lucidota_control.prompt_record.blockers = '' THEN EXCLUDED.blockers ELSE lucidota_control.prompt_record.blockers END,
        source_path = CASE WHEN lucidota_control.prompt_record.source_path = '' THEN EXCLUDED.source_path ELSE lucidota_control.prompt_record.source_path END,
        received_at_basis = COALESCE(NULLIF(lucidota_control.prompt_record.received_at_basis, ''), EXCLUDED.received_at_basis),
        received_at_confidence = GREATEST(lucidota_control.prompt_record.received_at_confidence, EXCLUDED.received_at_confidence),
        detail = lucidota_control.prompt_record.detail || EXCLUDED.detail,
        updated_at = now()
    RETURNING * INTO prompt_row;

    RETURN jsonb_build_object(
        'prompt_id', prompt_row.prompt_id::text,
        'received_at', prompt_row.received_at,
        'source', prompt_row.source,
        'source_model', prompt_row.source_model,
        'receiving_model', prompt_row.receiving_model,
        'target_model', prompt_row.target_model,
        'raw_prompt_text', prompt_row.raw_prompt_text,
        'normalized_prompt_text', prompt_row.normalized_prompt_text,
        'prompt_hash', prompt_row.prompt_hash,
        'conversation_id', prompt_row.conversation_id,
        'session_id', prompt_row.session_id,
        'parent_prompt_id', prompt_row.parent_prompt_id::text,
        'derived_prompt_ids', prompt_row.derived_prompt_ids,
        'linked_work_order_uuid', prompt_row.linked_work_order_uuid,
        'linked_receipt_uuid', prompt_row.linked_receipt_uuid,
        'linked_goal_id', prompt_row.linked_goal_id,
        'ontology_tags', prompt_row.ontology_tags,
        'go_co_io_tags', prompt_row.go_co_io_tags,
        'subsystem_tags', prompt_row.subsystem_tags,
        'status', prompt_row.status,
        'notes', prompt_row.notes,
        'blockers', prompt_row.blockers,
        'source_path', prompt_row.source_path,
        'received_at_basis', prompt_row.received_at_basis,
        'received_at_confidence', prompt_row.received_at_confidence,
        'idempotency_key', prompt_row.idempotency_key,
        'detail', prompt_row.detail
    );
END;
$$;

NOTIFY pgrst, 'reload schema';

COMMIT;
