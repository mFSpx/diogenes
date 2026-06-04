-- Prompt filing system: DB-backed prompt ledger, link surface, and manual integration.
-- Non-destructive, idempotent, PostgREST-readable.

BEGIN;
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS prompt_api;
CREATE SCHEMA IF NOT EXISTS lucidota_control;
CREATE SCHEMA IF NOT EXISTS lucidota_canon;

CREATE OR REPLACE FUNCTION prompt_api.normalize_prompt_text(p_text text)
RETURNS text
LANGUAGE plpgsql
IMMUTABLE
AS $$
#variable_conflict use_variable
DECLARE
    line text;
    cleaned text := '';
    normalized text := replace(replace(coalesce(p_text, ''), E'\r\n', E'\n'), E'\r', E'\n');
BEGIN
    FOREACH line IN ARRAY regexp_split_to_array(normalized, E'\n') LOOP
        cleaned := cleaned || regexp_replace(line, E'[ \t]+$', '') || E'\n';
    END LOOP;
    RETURN btrim(cleaned, E'\n');
END;
$$;

CREATE OR REPLACE FUNCTION prompt_api.prompt_hash(p_text text)
RETURNS text
LANGUAGE sql
IMMUTABLE
AS $$
    SELECT encode(digest(prompt_api.normalize_prompt_text($1), 'sha256'), 'hex');
$$;

CREATE OR REPLACE FUNCTION prompt_api.prompt_go_co_io_tags(p_text text)
RETURNS text[]
LANGUAGE plpgsql
IMMUTABLE
AS $$
#variable_conflict use_variable
DECLARE
    low text := lower(coalesce(p_text, ''));
    tags text[] := ARRAY['GO'];
BEGIN
    IF low ~ '(fix|repair|build|add|create|wire|implement|change|migrate|port|rewrite|update|refactor|delete|retire)' THEN
        tags := array_append(tags, 'CO');
    END IF;
    IF low ~ '(read|inspect|audit|scan|fetch|file|capture|discover|find|list|status|manual|route|prompt|prompt ledger)' THEN
        tags := array_append(tags, 'IO');
    END IF;
    RETURN COALESCE((SELECT array_agg(DISTINCT t ORDER BY t) FROM unnest(tags) AS t), ARRAY['GO']::text[]);
END;
$$;

CREATE OR REPLACE FUNCTION prompt_api.classify_prompt_text(p_text text)
RETURNS jsonb
LANGUAGE plpgsql
IMMUTABLE
AS $$
DECLARE
    low text := lower(prompt_api.normalize_prompt_text(p_text));
    subsystem text;
    ontology_tags text[];
    go_co_io_tags text[];
    risk text;
    parallelizable boolean;
    serialized boolean;
    executor_role text;
    acceptance_test text;
    receipt_requirement text;
    next_action text;
BEGIN
    subsystem := CASE
        WHEN low ~ '(manual|api|route|postgrest|cloud_packet|prompt ledger|prompt filing|prompt file)' THEN 'manual_api'
        WHEN low ~ '(book|books|lora|adapter|training)' THEN 'book_ops'
        WHEN low ~ '(indy daemon|daemon|queue|response)' THEN 'indy_daemon'
        WHEN low ~ '(model|provider|workflow|needle|treelite|router|classifier|summarizer|embedder|reranker|thinker|watcher)' THEN 'model_orchestration'
        WHEN low ~ '(rust|rewrite|port)' THEN 'rust_rewrite'
        WHEN low ~ '(ingest|canon|artifact|legacy|corpse|duplicate|review|slop)' THEN 'ingest_and_slop'
        WHEN low ~ '(test|proof|verify|receipt|smoke)' THEN 'verification'
        ELSE 'mixed'
    END;

    ontology_tags := CASE subsystem
        WHEN 'manual_api' THEN ARRAY['PROMPT', 'API', 'MANUAL', 'ROUTE', 'POSTGREST', 'RECEIPT']
        WHEN 'book_ops' THEN ARRAY['PROMPT', 'BOOK', 'LORA', 'ADAPTER', 'TRAINING']
        WHEN 'indy_daemon' THEN ARRAY['PROMPT', 'DAEMON', 'QUEUE', 'RESPONSE']
        WHEN 'model_orchestration' THEN ARRAY['PROMPT', 'MODEL', 'PROVIDER', 'WORKFLOW', 'NEEDLE', 'TREELITE']
        WHEN 'rust_rewrite' THEN ARRAY['PROMPT', 'RUST', 'REWRITE']
        WHEN 'ingest_and_slop' THEN ARRAY['PROMPT', 'INGEST', 'LEGACY', 'SLOP']
        WHEN 'verification' THEN ARRAY['PROMPT', 'TEST', 'PROOF', 'RECEIPT']
        ELSE ARRAY['PROMPT', 'WORK']
    END;

    go_co_io_tags := prompt_api.prompt_go_co_io_tags(low);

    risk := CASE
        WHEN low ~ '(delete|retire|rewrite|remove|drop|destructive)' THEN 'destructive'
        WHEN low ~ '(migrate|service|daemon|schema|db|port|rewrite)' THEN 'high'
        WHEN low ~ '(test|verify|audit|read|scan|inspect)' THEN 'low'
        ELSE 'medium'
    END;

    parallelizable := NOT (low ~ '(db migration|shared core|service|daemon|rewrite|delete|retire)');
    serialized := NOT parallelizable;

    executor_role := CASE
        WHEN subsystem = 'manual_api' THEN 'router'
        WHEN subsystem = 'book_ops' THEN 'watcher'
        WHEN subsystem = 'indy_daemon' THEN 'listener'
        WHEN subsystem = 'model_orchestration' THEN 'router'
        WHEN subsystem = 'rust_rewrite' THEN 'thinker'
        WHEN subsystem = 'ingest_and_slop' THEN 'watcher'
        WHEN subsystem = 'verification' THEN 'treelite_gate'
        ELSE 'router'
    END;

    acceptance_test := CASE
        WHEN subsystem = 'manual_api' THEN 'curl the live route and verify the prompt ledger packet reflects the current API truth.'
        WHEN subsystem = 'book_ops' THEN 'verify book/LoRA rows are visible through PostgREST and preserve receipts.'
        WHEN subsystem = 'indy_daemon' THEN 'run the daemon once and verify queue/status/response rows transition.'
        WHEN subsystem = 'model_orchestration' THEN 'query model_registry/provider_registry and expose missing roles as blockers.'
        WHEN subsystem = 'rust_rewrite' THEN 'port one bounded module, run A/B equivalence, and keep the API surface stable.'
        WHEN subsystem = 'ingest_and_slop' THEN 'quarantine or promote only after hash and receipt proof.'
        WHEN subsystem = 'verification' THEN 'run receipt-gated tests and verify the route/manual assertions pass.'
        ELSE 'prove the prompt filing surface with a receipt-backed DB/API check.'
    END;

    receipt_requirement := CASE
        WHEN subsystem IN ('manual_api', 'verification') THEN 'receipt-gated verification required'
        WHEN subsystem IN ('rust_rewrite', 'indy_daemon', 'model_orchestration') THEN 'receipt + live API readback required'
        ELSE 'receipt-backed row or quarantine receipt required'
    END;

    next_action := CASE
        WHEN subsystem = 'manual_api' THEN 'file the prompt, expose the ledger route, and link work-order receipts when possible.'
        WHEN subsystem = 'book_ops' THEN 'scan book/LoRA sources and promote only DB-visible work items.'
        WHEN subsystem = 'indy_daemon' THEN 'dequeue the next work item and write indy_responses plus receipts.'
        WHEN subsystem = 'model_orchestration' THEN 'select the live local lane and expose missing roles as blockers.'
        WHEN subsystem = 'rust_rewrite' THEN 'port a bounded module behind a stable API and compare receipts.'
        WHEN subsystem = 'ingest_and_slop' THEN 'classify the artifact, quarantine duplicates, and preserve hash custody.'
        WHEN subsystem = 'verification' THEN 'run the targeted tests and report the receipt path.'
        ELSE 'continue local deterministic routing.'
    END;

    RETURN jsonb_build_object(
        'subsystem', subsystem,
        'ontology_tags', ontology_tags,
        'go_co_io_tags', go_co_io_tags,
        'risk', risk,
        'parallelizable', parallelizable,
        'serialized', serialized,
        'executor_role', executor_role,
        'acceptance_test', acceptance_test,
        'receipt_requirement', receipt_requirement,
        'next_action', next_action
    );
END;
$$;

CREATE TABLE IF NOT EXISTS lucidota_control.prompt_record (
    prompt_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    received_at timestamptz NOT NULL DEFAULT now(),
    source text NOT NULL CHECK (source IN ('operator', 'assistant', 'codex', 'vibe', 'groq', 'local_model', 'system')),
    source_model text NOT NULL DEFAULT '',
    receiving_model text NOT NULL DEFAULT '',
    target_model text NOT NULL DEFAULT '',
    raw_prompt_text text NOT NULL DEFAULT '',
    normalized_prompt_text text NOT NULL DEFAULT '',
    prompt_hash text NOT NULL CHECK (prompt_hash ~ '^[0-9a-f]{64}$'),
    conversation_id text NOT NULL DEFAULT '',
    session_id text NOT NULL DEFAULT '',
    parent_prompt_id uuid REFERENCES lucidota_control.prompt_record(prompt_id) ON DELETE SET NULL,
    derived_prompt_ids uuid[] NOT NULL DEFAULT '{}'::uuid[],
    linked_work_order_uuid uuid[] NOT NULL DEFAULT '{}'::uuid[],
    linked_receipt_uuid uuid[] NOT NULL DEFAULT '{}'::uuid[],
    linked_goal_id text NOT NULL DEFAULT '',
    ontology_tags text[] NOT NULL DEFAULT '{}'::text[],
    go_co_io_tags text[] NOT NULL DEFAULT '{}'::text[],
    subsystem_tags text[] NOT NULL DEFAULT '{}'::text[],
    status text NOT NULL DEFAULT 'filed' CHECK (status IN ('filed', 'decomposed', 'queued', 'executed', 'superseded', 'archived')),
    notes text NOT NULL DEFAULT '',
    blockers text NOT NULL DEFAULT '',
    source_path text NOT NULL DEFAULT '',
    received_at_basis text NOT NULL DEFAULT 'provided',
    received_at_confidence numeric(4,3) NOT NULL DEFAULT 1.000,
    idempotency_key text NOT NULL DEFAULT '',
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE(idempotency_key)
);

CREATE TABLE IF NOT EXISTS lucidota_control.prompt_work_order_link (
    prompt_work_order_link_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    prompt_id uuid NOT NULL REFERENCES lucidota_control.prompt_record(prompt_id) ON DELETE CASCADE,
    work_order_uuid uuid NOT NULL REFERENCES lucidota_control.work_order(work_order_uuid) ON DELETE CASCADE,
    linked_goal_id text NOT NULL DEFAULT '',
    link_reason text NOT NULL DEFAULT '',
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE(prompt_id, work_order_uuid)
);

CREATE INDEX IF NOT EXISTS idx_prompt_record_received_at
    ON lucidota_control.prompt_record(received_at DESC, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_prompt_record_status
    ON lucidota_control.prompt_record(status, received_at DESC);

CREATE INDEX IF NOT EXISTS idx_prompt_work_order_link_prompt
    ON lucidota_control.prompt_work_order_link(prompt_id, created_at DESC);

CREATE OR REPLACE VIEW lucidota_canon.prompts_filed AS
SELECT
    p.prompt_id,
    p.received_at,
    p.source,
    p.source_model,
    p.receiving_model,
    p.target_model,
    p.raw_prompt_text,
    p.normalized_prompt_text,
    p.prompt_hash,
    p.conversation_id,
    p.session_id,
    p.parent_prompt_id,
    p.derived_prompt_ids,
    p.linked_work_order_uuid,
    p.linked_receipt_uuid,
    p.linked_goal_id,
    p.ontology_tags,
    p.go_co_io_tags,
    p.subsystem_tags,
    p.status,
    p.notes,
    p.blockers,
    p.source_path,
    p.received_at_basis,
    p.received_at_confidence,
    p.detail,
    p.created_at,
    p.updated_at,
    cardinality(p.linked_work_order_uuid) AS linked_work_order_count,
    cardinality(p.linked_receipt_uuid) AS linked_receipt_count,
    CASE
        WHEN cardinality(p.linked_work_order_uuid) > 0 THEN 'linked'
        WHEN btrim(p.blockers) <> '' THEN p.blockers
        WHEN btrim(p.notes) <> '' THEN p.notes
        ELSE 'no linked work-order yet'
    END AS explicit_unlinked_reason,
    p.idempotency_key
FROM lucidota_control.prompt_record p;

CREATE OR REPLACE VIEW lucidota_canon.prompt_work_order_links AS
SELECT
    l.prompt_work_order_link_uuid,
    l.prompt_id,
    p.received_at,
    p.source,
    p.source_model,
    p.receiving_model,
    p.target_model,
    l.work_order_uuid,
    l.linked_goal_id,
    l.link_reason,
    p.prompt_hash,
    p.status AS prompt_status,
    p.ontology_tags,
    p.go_co_io_tags,
    p.subsystem_tags,
    l.created_at,
    l.updated_at
FROM lucidota_control.prompt_work_order_link l
JOIN lucidota_canon.prompts_filed p USING (prompt_id);

CREATE OR REPLACE VIEW lucidota_canon.prompt_recent AS
SELECT *
FROM lucidota_canon.prompts_filed
ORDER BY received_at DESC, updated_at DESC, prompt_id DESC;

CREATE OR REPLACE VIEW lucidota_canon.prompt_unlinked AS
SELECT *
FROM lucidota_canon.prompts_filed
WHERE cardinality(linked_work_order_uuid) = 0
ORDER BY received_at DESC, updated_at DESC, prompt_id DESC;

CREATE OR REPLACE VIEW lucidota_canon.prompt_catalog_status AS
WITH prompt_counts AS (
    SELECT
        count(*) AS prompt_count,
        count(*) FILTER (WHERE status = 'filed') AS filed_count,
        count(*) FILTER (WHERE status = 'decomposed') AS decomposed_count,
        count(*) FILTER (WHERE status = 'queued') AS queued_count,
        count(*) FILTER (WHERE status = 'executed') AS executed_count,
        count(*) FILTER (WHERE status = 'superseded') AS superseded_count,
        count(*) FILTER (WHERE status = 'archived') AS archived_count,
        count(*) FILTER (WHERE cardinality(linked_work_order_uuid) = 0) AS unlinked_count,
        count(*) FILTER (WHERE cardinality(linked_work_order_uuid) > 0) AS linked_count,
        max(received_at) AS latest_received_at,
        max(updated_at) AS latest_updated_at
    FROM lucidota_control.prompt_record
),
route_counts AS (
    SELECT count(*) AS prompt_route_count
    FROM lucidota_canon.api_route_catalog
    WHERE route_id IN (
        'prompts_filed', 'prompt_work_order_links', 'prompt_recent', 'prompt_unlinked',
        'prompt_catalog_status', 'file_prompt', 'link_prompt_work_order', 'decompose_prompt_to_work_orders'
    )
)
SELECT
    now() AS refreshed_at,
    pc.prompt_count,
    pc.filed_count,
    pc.decomposed_count,
    pc.queued_count,
    pc.executed_count,
    pc.superseded_count,
    pc.archived_count,
    pc.unlinked_count,
    pc.linked_count,
    pc.latest_received_at,
    pc.latest_updated_at,
    rc.prompt_route_count
FROM prompt_counts pc
CROSS JOIN route_counts rc;

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

CREATE OR REPLACE FUNCTION prompt_api.link_prompt_work_order(
    p_prompt_id uuid,
    p_work_order_uuid uuid,
    p_link_reason text DEFAULT '',
    p_linked_goal_id text DEFAULT ''
) RETURNS jsonb
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = prompt_api, lucidota_canon, lucidota_control, public
VOLATILE
AS $$
DECLARE
    prompt_row lucidota_control.prompt_record%ROWTYPE;
    linked_ids uuid[];
BEGIN
    INSERT INTO lucidota_control.prompt_work_order_link(prompt_id, work_order_uuid, linked_goal_id, link_reason)
    VALUES (p_prompt_id, p_work_order_uuid, COALESCE(p_linked_goal_id, ''), COALESCE(p_link_reason, ''))
    ON CONFLICT (prompt_id, work_order_uuid) DO UPDATE
        SET linked_goal_id = COALESCE(NULLIF(EXCLUDED.linked_goal_id, ''), lucidota_control.prompt_work_order_link.linked_goal_id),
            link_reason = COALESCE(NULLIF(EXCLUDED.link_reason, ''), lucidota_control.prompt_work_order_link.link_reason),
            updated_at = now();

    UPDATE lucidota_control.prompt_record
    SET linked_work_order_uuid = (
            SELECT COALESCE(array_agg(DISTINCT elem ORDER BY elem), '{}'::uuid[])
            FROM unnest(COALESCE(linked_work_order_uuid, '{}'::uuid[]) || ARRAY[p_work_order_uuid]) AS elem
        ),
        linked_goal_id = COALESCE(NULLIF(p_linked_goal_id, ''), lucidota_control.prompt_record.linked_goal_id),
        status = CASE WHEN status = 'filed' THEN 'decomposed' ELSE status END,
        updated_at = now()
    WHERE lucidota_control.prompt_record.prompt_id = p_prompt_id
    RETURNING * INTO prompt_row;

    linked_ids := COALESCE(prompt_row.linked_work_order_uuid, ARRAY[p_work_order_uuid]::uuid[]);
    RETURN jsonb_build_object(
        'prompt_id', prompt_row.prompt_id::text,
        'work_order_uuid', p_work_order_uuid::text,
        'linked_work_order_uuid', linked_ids,
        'status', prompt_row.status,
        'linked_goal_id', prompt_row.linked_goal_id,
        'prompt_hash', prompt_row.prompt_hash
    );
END;
$$;

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

    SELECT cardinality(pr.linked_work_order_uuid)
    INTO prompt_link_count
    FROM lucidota_control.prompt_record pr
    WHERE pr.prompt_id = prompt_row.prompt_id;

    UPDATE lucidota_control.prompt_record pr
    SET status = CASE WHEN status = 'filed' THEN 'decomposed' ELSE status END,
        subsystem_tags = CASE WHEN cardinality(subsystem_tags) = 0 THEN ARRAY[classification->>'subsystem', 'prompt-ledger']::text[] ELSE subsystem_tags END,
        ontology_tags = CASE WHEN cardinality(ontology_tags) = 0 THEN ARRAY(SELECT jsonb_array_elements_text(classification->'ontology_tags')) ELSE ontology_tags END,
        go_co_io_tags = CASE WHEN cardinality(go_co_io_tags) = 0 THEN ARRAY(SELECT jsonb_array_elements_text(classification->'go_co_io_tags')) ELSE go_co_io_tags END,
        updated_at = now()
    WHERE pr.prompt_id = prompt_row.prompt_id;

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

DROP FUNCTION IF EXISTS prompt_api.cloud_packet(uuid, integer, integer, text, text, boolean);
DROP FUNCTION IF EXISTS lucidota_canon.cloud_packet(uuid, integer, integer, text, text, boolean);

CREATE OR REPLACE FUNCTION prompt_api.cloud_packet(
    work_order_id uuid,
    max_chars integer DEFAULT 8000,
    max_items integer DEFAULT 12,
    task_type text DEFAULT '',
    target_model text DEFAULT '',
    include_raw_bodies boolean DEFAULT false
) RETURNS jsonb
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = prompt_api, lucidota_canon, lucidota_learning, lucidota_control, public
STABLE
AS $$
DECLARE
    cap_chars integer := GREATEST(256, LEAST(COALESCE(max_chars, 8000), 12000));
    cap_items integer := GREATEST(1, LEAST(COALESCE(max_items, 12), 32));
    wo record;
    window_rows jsonb := '[]'::jsonb;
    selected_evidence_refs jsonb := '[]'::jsonb;
    event_ids_payload jsonb := '[]'::jsonb;
    source_hashes_payload jsonb := '[]'::jsonb;
    receipt_refs_payload jsonb := '[]'::jsonb;
    raw_bodies jsonb := '[]'::jsonb;
    local_score numeric := 0;
    treelite_score numeric := 0;
    needs_cloud boolean := false;
    summary_text text := '';
    next_action text := '';
    contract_name text := 'prompt_api.cloud_packet.v1';
BEGIN
    SELECT
        w.*,
        COALESCE(w.payload->>'task_type', '') AS payload_task_type,
        COALESCE(w.payload->>'target_model', '') AS payload_target_model,
        COALESCE(w.payload->>'next_action', '') AS payload_next_action,
        COALESCE(w.payload->>'summary', '') AS payload_summary
    INTO wo
    FROM lucidota_control.work_order w
    WHERE w.work_order_uuid = cloud_packet.work_order_id;

    IF NOT FOUND THEN
        RETURN jsonb_build_object(
            'contract_name', contract_name,
            'rules', jsonb_build_object(
                'max_chars', cap_chars,
                'max_items', cap_items,
                'include_raw_bodies', include_raw_bodies,
                'raw_bodies_whitelisted', include_raw_bodies,
                'raw_bodies_default', false,
                'window_view', 'lucidota_canon.bytewax_compact_windows'
            ),
            'work_order_id', work_order_id::text,
            'error', 'work_order_not_found',
            'summary', jsonb_build_object('text', '', 'window_count', 0),
            'selected_evidence_refs', '[]'::jsonb,
            'event_ids', '[]'::jsonb,
            'source_hashes', '[]'::jsonb,
            'scores', jsonb_build_object('local_score', 0, 'treelite_score', 0, 'needs_cloud_reasoning', false),
            'next_action', ''
        );
    END IF;

    WITH limited_windows AS (
        SELECT *
        FROM lucidota_canon.bytewax_compact_windows
        WHERE work_order_uuid = cloud_packet.work_order_id
        ORDER BY created_at DESC, window_end_at DESC
        LIMIT cap_items
    ),
    window_rows_cte AS (
        SELECT COALESCE(
            jsonb_agg(
                jsonb_build_object(
                    'work_order_id', cloud_packet.work_order_id::text,
                    'source', source,
                    'topic', topic,
                    'object_type', object_type,
                    'window_kind', window_kind,
                    'window_start_at', window_start_at,
                    'window_end_at', window_end_at,
                    'event_count', event_count,
                    'dropped_raw_bodies', dropped_raw_bodies,
                    'summary', left(summary, cap_chars),
                    'features', features,
                    'scores', scores,
                    'needs_cloud_reasoning', needs_cloud_reasoning,
                    'event_ids', event_ids,
                    'source_hashes', source_hashes,
                    'receipt_refs', receipt_refs
                )
            ),
            '[]'::jsonb
        ) AS payload
        FROM limited_windows
    ),
    event_id_rows AS (
        SELECT COALESCE(jsonb_agg(event_id), '[]'::jsonb) AS payload
        FROM (
            SELECT DISTINCT elem AS event_id
            FROM limited_windows w, jsonb_array_elements_text(w.event_ids) AS elem
            LIMIT cap_items
        ) s
    ),
    source_hash_rows AS (
        SELECT COALESCE(jsonb_agg(source_hash), '[]'::jsonb) AS payload
        FROM (
            SELECT DISTINCT elem AS source_hash
            FROM limited_windows w, jsonb_array_elements_text(w.source_hashes) AS elem
            LIMIT cap_items
        ) s
    ),
    evidence_rows AS (
        SELECT COALESCE(jsonb_agg(evidence_ref), '[]'::jsonb) AS payload
        FROM (
            SELECT DISTINCT evidence_ref
            FROM (
                SELECT elem AS evidence_ref
                FROM limited_windows w, jsonb_array_elements_text(w.receipt_refs) AS elem
                UNION
                SELECT COALESCE(wr.receipt_path, '') AS evidence_ref
                FROM lucidota_control.work_receipt wr
                WHERE wr.work_order_uuid = cloud_packet.work_order_id
            ) inner_refs
            LIMIT cap_items
        ) s
    ),
    score_rows AS (
        SELECT
            COALESCE(MAX(COALESCE((w.scores->>'local_score')::numeric, 0)), 0) AS local_score,
            COALESCE(MAX(COALESCE((w.scores->>'treelite_score')::numeric, 0)), 0) AS treelite_score,
            COALESCE(bool_or(COALESCE(w.needs_cloud_reasoning, false)), false) AS needs_cloud
        FROM limited_windows w
    )
    SELECT
        w.payload,
        e.payload,
        s.payload,
        r.payload,
        sr.local_score,
        sr.treelite_score,
        sr.needs_cloud
        INTO window_rows, event_ids_payload, source_hashes_payload, selected_evidence_refs, local_score, treelite_score, needs_cloud
    FROM window_rows_cte w, event_id_rows e, source_hash_rows s, evidence_rows r, score_rows sr;

    IF include_raw_bodies THEN
        WITH limited_windows AS (
            SELECT *
            FROM lucidota_canon.bytewax_compact_windows
            WHERE work_order_uuid = cloud_packet.work_order_id
            ORDER BY created_at DESC, window_end_at DESC
            LIMIT cap_items
        ),
        raw_rows AS (
            SELECT DISTINCT left(COALESCE(e.payload->>'raw_body', e.payload->>'body', e.text_surface, ''), cap_chars) AS raw_body
            FROM lucidota_learning.bytewax_abductive_event e
            JOIN limited_windows w ON e.source = w.source
            WHERE e.source_ref IN (
                SELECT elem
                FROM limited_windows w2, jsonb_array_elements_text(w2.event_ids) AS elem
                WHERE w2.source = w.source
            )
            AND COALESCE(e.payload->>'raw_body', e.payload->>'body', '') <> ''
            LIMIT cap_items
        )
        SELECT COALESCE(jsonb_agg(raw_body), '[]'::jsonb)
        INTO raw_bodies
        FROM raw_rows;
    END IF;

    summary_text := left(
        COALESCE(NULLIF(task_type, ''), NULLIF(wo.payload_task_type, ''), wo.work_kind) || ' | ' ||
        COALESCE(NULLIF(target_model, ''), NULLIF(wo.payload_target_model, ''), 'model-unspecified') || ' | ' ||
        COALESCE(NULLIF(wo.payload_summary, ''), left(wo.payload::text, cap_chars), '') ||
        CASE WHEN window_rows <> '[]'::jsonb THEN ' | windows=' || jsonb_array_length(window_rows)::text ELSE '' END,
        cap_chars
    );

    next_action := left(
        COALESCE(NULLIF(task_type, ''), NULLIF(wo.payload_task_type, ''), wo.work_kind, 'review') ||
        CASE WHEN needs_cloud THEN ' -> route to cloud reasoning with bounded packet' ELSE ' -> continue local deterministic path' END,
        cap_chars
    );

    RETURN jsonb_build_object(
        'contract_name', contract_name,
        'rules', jsonb_build_object(
            'max_chars', cap_chars,
            'max_items', cap_items,
            'include_raw_bodies', include_raw_bodies,
            'raw_bodies_whitelisted', include_raw_bodies,
            'raw_bodies_default', false,
            'window_view', 'lucidota_canon.bytewax_compact_windows'
        ),
        'work_order_id', cloud_packet.work_order_id::text,
        'work_order', jsonb_build_object(
            'work_order_uuid', wo.work_order_uuid::text,
            'event_id', wo.event_id,
            'lane', wo.lane,
            'work_kind', wo.work_kind,
            'status', wo.status,
            'idempotency_key', wo.idempotency_key,
            'created_at', wo.created_at,
            'updated_at', wo.updated_at
        ),
        'task_type', COALESCE(NULLIF(task_type, ''), NULLIF(wo.payload_task_type, ''), wo.work_kind),
        'target_model', COALESCE(NULLIF(target_model, ''), NULLIF(wo.payload_target_model, ''), ''),
        'summary', jsonb_build_object(
            'text', summary_text,
            'window_count', COALESCE(jsonb_array_length(window_rows), 0),
            'work_kind', wo.work_kind,
            'lane', wo.lane,
            'status', wo.status
        ),
        'windows', window_rows,
        'selected_evidence_refs', selected_evidence_refs,
        'event_ids', event_ids_payload,
        'source_hashes', source_hashes_payload,
        'receipt_refs', selected_evidence_refs,
        'scores', jsonb_build_object(
            'local_score', local_score,
            'treelite_score', treelite_score,
            'needs_cloud_reasoning', needs_cloud
        ),
        'needs_cloud_reasoning', needs_cloud,
        'raw_bodies', raw_bodies,
        'next_action', next_action
    );
END;
$$;

CREATE OR REPLACE FUNCTION lucidota_canon.cloud_packet(
    work_order_id uuid,
    max_chars integer DEFAULT 8000,
    max_items integer DEFAULT 12,
    task_type text DEFAULT '',
    target_model text DEFAULT '',
    include_raw_bodies boolean DEFAULT false
) RETURNS jsonb
LANGUAGE sql
SECURITY DEFINER
STABLE
AS $$
    SELECT prompt_api.cloud_packet($1, $2, $3, $4, $5, $6);
$$;

INSERT INTO lucidota_canon.api_route_catalog
(route_id, method, path_pattern, description, target, sample_request, sample_response, status)
VALUES
('prompts_filed', 'GET', '/prompts_filed', 'Prompt ledger entries filed through the DB-backed prompt surface.', 'lucidota_canon.prompts_filed',
 '{"limit":"5"}', '{"prompt_id":"...","status":"filed"}', 'implemented'),
('prompt_work_order_links', 'GET', '/prompt_work_order_links', 'Prompt to work-order links with link reasons and live prompt metadata.', 'lucidota_canon.prompt_work_order_links',
 '{"limit":"5"}', '{"prompt_id":"...","work_order_uuid":"..."}', 'implemented'),
('prompt_recent', 'GET', '/prompt_recent', 'Most recently received prompt ledger entries.', 'lucidota_canon.prompt_recent',
 '{"limit":"5"}', '{"prompt_id":"..."}', 'implemented'),
('prompt_unlinked', 'GET', '/prompt_unlinked', 'Prompt ledger entries with no linked work order yet.', 'lucidota_canon.prompt_unlinked',
 '{"limit":"5"}', '{"prompt_id":"...","explicit_unlinked_reason":"..."}', 'implemented'),
('prompt_catalog_status', 'GET', '/prompt_catalog_status', 'Prompt ledger summary and route coverage status.', 'lucidota_canon.prompt_catalog_status',
 '{"limit":"1"}', '{"prompt_count":0,"unlinked_count":0}', 'implemented'),
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

CREATE OR REPLACE VIEW lucidota_canon.manual_current AS
WITH live_routes AS (
    SELECT jsonb_agg(
        jsonb_build_object(
            'route_id', route_id,
            'method', method,
            'path_pattern', path_pattern,
            'description', description,
            'target', target,
            'status', status
        )
        ORDER BY route_id
    ) AS route_list,
    count(*) AS route_count
    FROM lucidota_canon.api_route_catalog
    WHERE route_id IN (
        'manual_current', 'canon_current', 'canon_versions', 'active_goal', 'api_workflow_registry',
        'capability_registry', 'model_registry', 'provider_registry', 'workflow_registry',
        'daemon_status', 'bytewax_compact_windows', 'indy_queue', 'indy_responses',
        'cloud_packet', 'book_source', 'book_scan', 'book_read_queue', 'book_note',
        'lora_candidate', 'lora_adapter', 'training_job', 'book_receipt',
        'ontology_work_batch', 'ontology_work_item', 'todo_current', 'skill_policy_current',
        'root_orchestrator_current', 'prompts_filed', 'prompt_work_order_links',
        'prompt_recent', 'prompt_unlinked', 'prompt_catalog_status',
        'file_prompt', 'link_prompt_work_order', 'decompose_prompt_to_work_orders'
    )
),
goal_row AS (
    SELECT to_jsonb(g) AS current_goal
    FROM lucidota_canon.active_goal g
    ORDER BY updated_at DESC
    LIMIT 1
),
daemon_rows AS (
    SELECT COALESCE(jsonb_agg(to_jsonb(d) ORDER BY d.daemon_name), '[]'::jsonb) AS daemon_status
    FROM lucidota_canon.daemon_status d
),
model_rows AS (
    SELECT COALESCE(jsonb_agg(to_jsonb(m) ORDER BY m.model_id), '[]'::jsonb) AS model_registry
    FROM lucidota_canon.model_registry m
),
provider_rows AS (
    SELECT COALESCE(jsonb_agg(to_jsonb(p) ORDER BY p.provider_key), '[]'::jsonb) AS provider_registry
    FROM lucidota_canon.provider_registry p
),
workflow_rows AS (
    SELECT COALESCE(jsonb_agg(to_jsonb(w) ORDER BY w.workflow_id), '[]'::jsonb) AS workflow_registry
    FROM lucidota_canon.workflow_registry w
),
todo_rows AS (
    SELECT COALESCE(
        jsonb_agg(to_jsonb(t) ORDER BY t.created_at DESC),
        '[]'::jsonb
    ) AS todo_current
    FROM (
        SELECT *
        FROM lucidota_canon.todo_current
        WHERE status IN ('ready', 'queued', 'running')
        ORDER BY created_at DESC
        LIMIT 5
    ) t
),
skill_policy_rows AS (
    SELECT COALESCE(jsonb_agg(to_jsonb(p) ORDER BY p.updated_at DESC), '[]'::jsonb) AS skill_policy_current
    FROM lucidota_canon.skill_policy_current p
),
root_orchestrator_rows AS (
    SELECT COALESCE(jsonb_agg(to_jsonb(r) ORDER BY r.max_updated_at DESC), '[]'::jsonb) AS root_orchestrator_current
    FROM lucidota_canon.root_orchestrator_current r
),
prompt_status_rows AS (
    SELECT COALESCE(jsonb_agg(to_jsonb(p) ORDER BY p.refreshed_at DESC), '[]'::jsonb) AS prompt_catalog_status
    FROM lucidota_canon.prompt_catalog_status p
),
prompt_recent_rows AS (
    SELECT COALESCE(jsonb_agg(to_jsonb(p) ORDER BY p.received_at DESC), '[]'::jsonb) AS prompt_recent
    FROM (
        SELECT *
        FROM lucidota_canon.prompt_recent
        LIMIT 5
    ) p
)
SELECT
    'LUCIDOTA_OPERATOR_MANUAL'::text AS manual_id,
    'LUCIDOTA Operator Manual'::text AS title,
    live_routes.route_count AS node_count,
    now() AS max_updated_at,
    live_routes.route_list,
    jsonb_build_object(
        'read_surface', 'PostgREST safe views and RPCs only',
        'write_surface', 'DB work orders and receipts only',
        'legacy_book_watcher', 'retired as authority',
        'skill_layers', 'execution aids only; live PostgREST/manual truth and GOALS handoffs win; prompt filing law is DB-backed and preserves raw text',
        'skill_policy_surface', 'DB-backed policy current route; live policy text wins over file-only policy snippets',
        'prompt_filing', 'file_prompt -> prompt ledger row -> prompt_work_order_links -> prompt_recent / prompt_unlinked -> prompt_catalog_status',
        'manual_source', 'live route catalog + daemon status + current goal + current todo batches + root orchestrator surface + prompt ledger'
    ) AS auth_expectations,
    jsonb_build_object(
        'book_ingest', 'book_source -> book_scan -> book_read_queue -> book_note -> lora_candidate -> lora_adapter -> training_job -> book_receipt',
        'indy_loop', 'queued row -> /indy_queue -> indy_daemon once/loop -> /indy_responses or receipt row',
        'mamba_role', 'DB queue/receipt/window watcher only; no BOOKS filesystem authority',
        'ontology_loop', 'messy operator text -> ontology_work_batch -> ontology_work_item -> executable route plan',
        'skill_policy', 'skill_policy_current -> operator-readable alignment policy -> manual surface',
        'root_orchestrator', 'root_orchestrator_current -> sub-orchestrator packets -> receipts -> manual update',
        'prompt_filing', 'operator or assistant prompt -> prompt ledger -> explicit linked work-order UUID or explicit unlinked reason'
    ) AS work_order_flow,
    jsonb_build_object(
        'current_goal', goal_row.current_goal,
        'daemon_status', daemon_rows.daemon_status,
        'model_registry', model_rows.model_registry,
        'provider_registry', provider_rows.provider_registry,
        'workflow_registry', workflow_rows.workflow_registry,
        'todo_current', todo_rows.todo_current,
        'skill_policy_current', skill_policy_rows.skill_policy_current,
        'root_orchestrator_current', root_orchestrator_rows.root_orchestrator_current,
        'prompt_catalog_status', prompt_status_rows.prompt_catalog_status,
        'prompt_recent', prompt_recent_rows.prompt_recent
    ) AS live_surface,
    jsonb_build_array(
        'curl -sS http://127.0.0.1:3000/manual_current?limit=1',
        'curl -sS http://127.0.0.1:3000/root_orchestrator_current?limit=1',
        'curl -sS http://127.0.0.1:3000/todo_current?limit=5',
        'curl -sS http://127.0.0.1:3000/skill_policy_current?limit=1',
        'curl -sS http://127.0.0.1:3000/prompt_catalog_status?limit=1',
        'curl -sS http://127.0.0.1:3000/prompt_recent?limit=5',
        '.venv/bin/python scripts/ontology_work_compiler.py --json --text "<objective text>"',
        '.venv/bin/python scripts/indy_daemon.py --once --json',
        '.venv/bin/python scripts/indy_runtime_broker.py snapshot --json',
        '.venv/bin/python scripts/prompt_ledger_capture.py --json'
    ) AS next_commands,
    jsonb_build_array(
        'BOOKS folder watcher authority',
        'hand-written manual slop',
        'raw corpus prompts',
        'unbounded whole-table dumps'
    ) AS retired_surfaces
FROM live_routes
CROSS JOIN goal_row
CROSS JOIN daemon_rows
CROSS JOIN model_rows
CROSS JOIN provider_rows
CROSS JOIN workflow_rows
CROSS JOIN todo_rows
CROSS JOIN skill_policy_rows
CROSS JOIN root_orchestrator_rows
CROSS JOIN prompt_status_rows
CROSS JOIN prompt_recent_rows;

GRANT USAGE ON SCHEMA prompt_api TO mfspx, lucidota_postgrest_anon;
GRANT SELECT, INSERT, UPDATE ON lucidota_control.prompt_record, lucidota_control.prompt_work_order_link TO mfspx;
GRANT SELECT ON lucidota_canon.prompts_filed, lucidota_canon.prompt_work_order_links,
    lucidota_canon.prompt_recent, lucidota_canon.prompt_unlinked, lucidota_canon.prompt_catalog_status TO mfspx;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA prompt_api TO lucidota_postgrest_anon, mfspx;
GRANT EXECUTE ON FUNCTION
    lucidota_canon.file_prompt(text, text, text, text, text, text, text, uuid, uuid[], uuid[], text, text[], text[], text, text, text, text, text, timestamptz, numeric, text, jsonb),
    lucidota_canon.link_prompt_work_order(uuid, uuid, text, text),
    lucidota_canon.decompose_prompt_to_work_orders(uuid, integer, text, text),
    lucidota_canon.cloud_packet(uuid, integer, integer, text, text, boolean)
TO lucidota_postgrest_anon, mfspx;
GRANT SELECT ON lucidota_canon.prompts_filed, lucidota_canon.prompt_work_order_links,
    lucidota_canon.prompt_recent, lucidota_canon.prompt_unlinked, lucidota_canon.prompt_catalog_status,
    lucidota_canon.manual_current, lucidota_canon.api_route_catalog TO lucidota_postgrest_anon;

COMMIT;
