-- prompt_api.cloud_packet RPC and compact Bytewax window surface.

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS prompt_api;
CREATE SCHEMA IF NOT EXISTS lucidota_learning;
CREATE SCHEMA IF NOT EXISTS lucidota_canon;

CREATE TABLE IF NOT EXISTS lucidota_learning.bytewax_compact_window (
    compact_window_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    work_order_uuid uuid REFERENCES lucidota_control.work_order(work_order_uuid) ON DELETE SET NULL,
    work_order_id text NOT NULL DEFAULT '',
    source text NOT NULL,
    topic text NOT NULL,
    object_type text NOT NULL,
    window_kind text NOT NULL CHECK (window_kind IN ('tumbling', 'sliding')),
    window_start_at timestamptz NOT NULL,
    window_end_at timestamptz NOT NULL,
    event_count integer NOT NULL DEFAULT 0 CHECK (event_count >= 0),
    dropped_raw_bodies integer NOT NULL DEFAULT 0 CHECK (dropped_raw_bodies >= 0),
    summary text NOT NULL DEFAULT '',
    features jsonb NOT NULL DEFAULT '{}'::jsonb,
    scores jsonb NOT NULL DEFAULT '{}'::jsonb,
    needs_cloud_reasoning boolean NOT NULL DEFAULT false,
    event_ids jsonb NOT NULL DEFAULT '[]'::jsonb,
    source_hashes jsonb NOT NULL DEFAULT '[]'::jsonb,
    receipt_refs jsonb NOT NULL DEFAULT '[]'::jsonb,
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE(work_order_uuid, source, topic, object_type, window_kind, window_start_at, window_end_at)
);

CREATE OR REPLACE VIEW lucidota_canon.bytewax_compact_windows AS
SELECT
    compact_window_uuid,
    work_order_uuid,
    work_order_id,
    source,
    topic,
    object_type,
    window_kind,
    window_start_at,
    window_end_at,
    event_count,
    dropped_raw_bodies,
    summary,
    features,
    scores,
    needs_cloud_reasoning,
    event_ids,
    source_hashes,
    receipt_refs,
    detail,
    created_at,
    updated_at
FROM lucidota_learning.bytewax_compact_window;

DROP FUNCTION IF EXISTS lucidota_canon.cloud_packet(uuid, integer, integer, text, text, boolean);
DROP FUNCTION IF EXISTS prompt_api.cloud_packet(uuid, integer, integer, text, text, boolean);

CREATE OR REPLACE FUNCTION prompt_api.cloud_packet(
    p_work_order_id uuid,
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
    WHERE w.work_order_uuid = p_work_order_id;

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
            'work_order_id', p_work_order_id::text,
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
        WHERE work_order_uuid = p_work_order_id
        ORDER BY created_at DESC, window_end_at DESC
        LIMIT cap_items
    ),
    window_rows_cte AS (
        SELECT COALESCE(
            jsonb_agg(
                jsonb_build_object(
                    'work_order_id', p_work_order_id,
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
                WHERE wr.work_order_uuid = p_work_order_id
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
            WHERE work_order_uuid = p_work_order_id
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
        'work_order_id', p_work_order_id::text,
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
        'next_action', next_action,
        'raw_bodies', CASE WHEN include_raw_bodies THEN raw_bodies ELSE '[]'::jsonb END
    );
END;
$$;

GRANT USAGE ON SCHEMA prompt_api TO lucidota_postgrest_anon, mfspx;
GRANT EXECUTE ON FUNCTION prompt_api.cloud_packet(uuid, integer, integer, text, text, boolean) TO lucidota_postgrest_anon, mfspx;
GRANT SELECT ON lucidota_canon.bytewax_compact_windows TO lucidota_postgrest_anon, mfspx;

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
SET search_path = prompt_api, lucidota_canon, lucidota_learning, lucidota_control, public
STABLE
AS $$
    SELECT prompt_api.cloud_packet($1, $2, $3, $4, $5, $6);
$$;

GRANT EXECUTE ON FUNCTION lucidota_canon.cloud_packet(uuid, integer, integer, text, text, boolean) TO lucidota_postgrest_anon, mfspx;
