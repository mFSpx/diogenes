-- FILE: 06_SCHEMA/118_exact_sha_bulk_dedupe_alias.sql
-- PURPOSE: Native PostgreSQL exact-SHA dedupe fast path.
-- SCOPE: DEDUP_ONLY_NOT_FOR_LLM_EXTRACTIONS.
--
-- This is the permanent storage-engine version of the historical replay exact-SHA
-- alias pass. It does not perform semantic extraction, schema repair, or model
-- interpretation. It only materializes ATTRIBUTE aliases for already-indexed
-- KRAMPUSCHEWING components whose content SHA-256 is byte-identical to another
-- component and whose source_component_uuid is not already represented in
-- lucidota_go.graph_item.

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_go;

CREATE TABLE IF NOT EXISTS lucidota_go.exact_sha_bulk_alias_batch (
    dedupe_batch_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    source_system text NOT NULL DEFAULT 'postgres_exact_sha_bulk_alias_ledgered.v1',
    fast_path_guard text NOT NULL DEFAULT 'DEDUP_ONLY_NOT_FOR_LLM_EXTRACTIONS'
        CHECK (fast_path_guard = 'DEDUP_ONLY_NOT_FOR_LLM_EXTRACTIONS'),
    operator_confirmed boolean NOT NULL DEFAULT false,
    requested_limit integer,
    receipt_path text NOT NULL DEFAULT '',
    inserted_count integer NOT NULL DEFAULT 0,
    inserted_distinct_sources integer NOT NULL DEFAULT 0,
    token_cost_avoided_estimate bigint NOT NULL DEFAULT 0,
    batch_bad_policy integer NOT NULL DEFAULT 0,
    batch_bad_guard integer NOT NULL DEFAULT 0,
    batch_materializations integer NOT NULL DEFAULT 0,
    batch_helper_receipts integer NOT NULL DEFAULT 0,
    remaining_exact_sha_alias_targets integer NOT NULL DEFAULT 0,
    integrity jsonb NOT NULL DEFAULT '{}'::jsonb,
    status text NOT NULL DEFAULT 'running',
    created_at timestamptz NOT NULL DEFAULT now(),
    completed_at timestamptz
);

CREATE INDEX IF NOT EXISTS exact_sha_bulk_alias_batch_status_idx
    ON lucidota_go.exact_sha_bulk_alias_batch(status, created_at DESC);

CREATE OR REPLACE FUNCTION lucidota_go.fn_exact_sha_bulk_alias_materialize(
    p_limit integer DEFAULT NULL,
    p_batch_uuid uuid DEFAULT gen_random_uuid(),
    p_receipt_path text DEFAULT '',
    p_operator_confirmed boolean DEFAULT false
)
RETURNS jsonb
LANGUAGE plpgsql
AS $$
DECLARE
    v_inserted_count integer := 0;
    v_inserted_distinct_sources integer := 0;
    v_token_cost_avoided_estimate bigint := 0;
    v_batch_bad_policy integer := 0;
    v_batch_bad_guard integer := 0;
    v_batch_materializations integer := 0;
    v_batch_helper_receipts integer := 0;
    v_remaining_exact_sha_alias_targets integer := 0;
    out_status text := 'PASS';
    out_integrity jsonb := '{}'::jsonb;
BEGIN
    IF p_operator_confirmed IS DISTINCT FROM true THEN
        RAISE EXCEPTION 'operator confirmation required for exact-SHA bulk alias fast path'
            USING ERRCODE = 'insufficient_privilege';
    END IF;

    INSERT INTO lucidota_go.exact_sha_bulk_alias_batch(
        dedupe_batch_uuid, operator_confirmed, requested_limit, receipt_path, status
    )
    VALUES (
        p_batch_uuid, true, p_limit, coalesce(p_receipt_path, ''), 'running'
    )
    ON CONFLICT (dedupe_batch_uuid) DO UPDATE SET
        operator_confirmed = true,
        requested_limit = EXCLUDED.requested_limit,
        receipt_path = EXCLUDED.receipt_path,
        status = 'running',
        completed_at = NULL;

    PERFORM pg_advisory_xact_lock(hashtext('lucidota_exact_sha_bulk_alias_v2_ledgered'));

    -- These transaction-local flags satisfy the existing canonical graph write
    -- barrier while keeping this path visibly fenced to the exact-SHA dedupe
    -- helper contract.
    PERFORM set_config('lucidota.graph_promotion_path', 'on', true);
    PERFORM set_config('lucidota.graph_materialization_helper', 'scripts/graph_materialization_helper.py', true);

    DROP TABLE IF EXISTS tmp_exact_sha_bulk_ledgered;
    CREATE TEMP TABLE tmp_exact_sha_bulk_ledgered ON COMMIT DROP AS
    WITH ranked AS (
        SELECT c.component_uuid::text AS component_uuid,
               c.file_uuid::text AS file_uuid,
               c.component_index,
               c.sha256 AS component_sha256,
               coalesce(c.token_count, 0)::integer AS token_count,
               coalesce(c.entropy, 0)::double precision AS entropy,
               f.sha256 AS file_sha256,
               coalesce(nullif(f.locked_relative_path, ''), nullif(f.first_seen_path, ''), nullif(f.cas_uri, ''), '') AS source_path,
               first_value(c.component_uuid::text) OVER (
                   PARTITION BY c.sha256
                   ORDER BY coalesce(f.first_seen_path, ''), c.component_index, c.component_uuid
               ) AS representative_component_uuid,
               row_number() OVER (
                   PARTITION BY c.sha256
                   ORDER BY coalesce(f.first_seen_path, ''), c.component_index, c.component_uuid
               ) AS duplicate_rank,
               count(*) OVER (PARTITION BY c.sha256) AS duplicate_group_size
        FROM lucidota_korpus.component c
        JOIN lucidota_korpus.file_object f ON f.file_uuid = c.file_uuid
        WHERE c.content IS NOT NULL
          AND (
              coalesce(f.first_seen_path, '') LIKE '/home/mfspx/LUCIDOTA/KRAMPUSCHEWING/%'
              OR coalesce(f.locked_relative_path, '') LIKE 'KRAMPUSCHEWING/%'
          )
    ), targets AS (
        SELECT r.*
        FROM ranked r
        WHERE r.duplicate_group_size > 1
          AND r.duplicate_rank > 1
          AND NOT EXISTS (
              SELECT 1
              FROM lucidota_go.graph_item existing
              WHERE existing.payload->'payload'->>'source_component_uuid' = r.component_uuid
          )
        ORDER BY r.component_sha256, r.duplicate_rank, r.component_uuid
        LIMIT greatest(0, coalesce(p_limit, 2147483647))
    ), prepared AS (
        SELECT *,
               ('Exact SHA duplicate alias ' || substring(component_sha256 from 1 for 12) || ' #' || duplicate_rank || ' of ' || duplicate_group_size) AS alias_label,
               jsonb_build_array(
                   'dedupe_batch:' || p_batch_uuid::text,
                   'sha256:' || component_sha256,
                   'component:' || component_uuid,
                   'representative_component:' || representative_component_uuid,
                   coalesce(p_receipt_path, '')
               ) AS evidence_refs
        FROM targets
    ), payloads AS (
        SELECT *,
               jsonb_build_object(
                   'term', 'ATTRIBUTE',
                   'label', alias_label,
                   'status', 'staged',
                   'ternary_valency', 0,
                   'payload', jsonb_build_object(
                       'source_component_uuid', component_uuid,
                       'source_file_uuid', file_uuid,
                       'source_file_sha256', file_sha256,
                       'source_component_sha256', component_sha256,
                       'duplicate_of_component_uuid', representative_component_uuid,
                       'duplicate_group_size', duplicate_group_size,
                       'duplicate_rank', duplicate_rank,
                       'dedupe_policy', 'exact_sha256_duplicate_alias_no_llm_token',
                       'dedupe_batch_uuid', p_batch_uuid::text,
                       'dedupe_bulk_source', 'postgres_exact_sha_bulk_alias_ledgered.v1',
                       'semantic_extraction_required_on_representative', true,
                       'token_cost_avoided_estimate', token_count,
                       'entropy', entropy,
                       'source_path', source_path,
                       'authority', 'deterministic_database_exact_sha_constraint',
                       'fast_path_guard', 'DEDUP_ONLY_NOT_FOR_LLM_EXTRACTIONS'
                   ),
                   'evidence_note', 'Bulk exact-SHA duplicate alias; deterministic no-model set-based graph promotion ledger transaction.',
                   'dedupe_batch_uuid', p_batch_uuid::text,
                   'bulk_insert_receipt_path', coalesce(p_receipt_path, ''),
                   'evidence_refs', evidence_refs
               ) AS candidate_payload
        FROM prepared
    ), packets AS (
        INSERT INTO lucidota_go.graph_promotion_packet(
            source_system, candidate_kind, candidate_payload, evidence_refs,
            authority_class, promotion_status, detail, packet_dedupe_key
        )
        SELECT 'postgres_exact_sha_bulk_alias_ledgered.v1',
               'node',
               candidate_payload,
               evidence_refs,
               'operator_confirmed_finding',
               'operator_confirmed',
               jsonb_build_object(
                   'dedupe_batch_uuid', p_batch_uuid::text,
                   'fast_path_guard', 'DEDUP_ONLY_NOT_FOR_LLM_EXTRACTIONS',
                   'receipt_path', coalesce(p_receipt_path, '')
               ),
               'bulk_exact_sha_alias:' || p_batch_uuid::text || ':' || component_uuid
        FROM payloads
        RETURNING packet_uuid, candidate_payload, evidence_refs
    ), decisions AS (
        INSERT INTO lucidota_go.graph_promotion_decision(
            packet_uuid, decision, decided_by, rationale,
            evidence_refs, operator_confirmed, command_envelope_uuid
        )
        SELECT packet_uuid,
               'operator_confirmed',
               'operator',
               'Operator-confirmed deterministic exact-SHA duplicate alias bulk materialization; dedupe-only, no LLM extraction.',
               evidence_refs,
               true,
               p_batch_uuid
        FROM packets
        RETURNING decision_uuid, packet_uuid
    ), items AS (
        INSERT INTO lucidota_go.graph_item(
            term, label, status, location_at_on_graph,
            location_real_at_added, ternary_valency, payload
        )
        SELECT candidate_payload->>'term',
               candidate_payload->>'label',
               candidate_payload->>'status',
               'KRAMPUSCHEWING/exact_sha_bulk_alias_ledgered/' || p_batch_uuid::text,
               jsonb_build_object(
                   'dedupe_batch_uuid', p_batch_uuid::text,
                   'receipt_path', coalesce(p_receipt_path, ''),
                   'source_system', 'postgres_exact_sha_bulk_alias_ledgered.v1'
               ),
               0,
               candidate_payload
        FROM packets
        RETURNING uuid AS graph_item_uuid, payload
    ), joined AS (
        SELECT i.graph_item_uuid,
               i.payload,
               p.packet_uuid,
               d.decision_uuid,
               p.evidence_refs,
               i.payload->'payload'->>'source_component_uuid' AS source_component_uuid,
               i.payload->'payload'->>'source_component_sha256' AS source_component_sha256,
               (i.payload->'payload'->>'token_cost_avoided_estimate')::integer AS token_cost_avoided_estimate
        FROM items i
        JOIN packets p
          ON p.candidate_payload->'payload'->>'source_component_uuid'
           = i.payload->'payload'->>'source_component_uuid'
        JOIN decisions d ON d.packet_uuid = p.packet_uuid
    ), journals AS (
        INSERT INTO lucidota_go.graph_journal(
            item_uuid, action, reason, before_state, after_state
        )
        SELECT graph_item_uuid,
               'stage',
               'Bulk exact-SHA duplicate alias set-based materialization batch ' || p_batch_uuid::text,
               '{}'::jsonb,
               payload
        FROM joined
        RETURNING journal_uuid, item_uuid
    ), materializations AS (
        INSERT INTO lucidota_go.graph_promotion_materialization(
            packet_uuid, decision_uuid, graph_item_uuid, journal_uuid,
            command_envelope_uuid, evidence_refs, materialization_kind, detail
        )
        SELECT j.packet_uuid,
               j.decision_uuid,
               j.graph_item_uuid,
               jr.journal_uuid,
               p_batch_uuid,
               j.evidence_refs,
               'node',
               jsonb_build_object(
                   'dedupe_batch_uuid', p_batch_uuid::text,
                   'source_system', 'postgres_exact_sha_bulk_alias_ledgered.v1',
                   'fast_path_guard', 'DEDUP_ONLY_NOT_FOR_LLM_EXTRACTIONS'
               )
        FROM joined j
        JOIN journals jr ON jr.item_uuid = j.graph_item_uuid
        RETURNING materialization_uuid, packet_uuid, decision_uuid, graph_item_uuid, journal_uuid, command_envelope_uuid, evidence_refs
    ), helper_receipts AS (
        INSERT INTO lucidota_go.graph_materialization_helper_receipt(
            materialization_uuid, packet_uuid, decision_uuid, graph_item_uuid,
            journal_uuid, command_envelope_uuid, evidence_count, authority_class,
            verification_passed, materializer_report_path, detail
        )
        SELECT materialization_uuid,
               packet_uuid,
               decision_uuid,
               graph_item_uuid,
               journal_uuid,
               command_envelope_uuid,
               jsonb_array_length(evidence_refs),
               'operator_confirmed_finding',
               true,
               coalesce(p_receipt_path, ''),
               jsonb_build_object(
                   'dedupe_batch_uuid', p_batch_uuid::text,
                   'source_system', 'postgres_exact_sha_bulk_alias_ledgered.v1',
                   'bulk_set_based', true
               )
        FROM materializations
        RETURNING materialization_uuid, graph_item_uuid
    )
    SELECT j.graph_item_uuid::text,
           j.packet_uuid::text,
           j.decision_uuid::text,
           m.materialization_uuid::text,
           m.journal_uuid::text,
           j.source_component_uuid,
           j.source_component_sha256,
           j.token_cost_avoided_estimate
    FROM joined j
    JOIN materializations m ON m.graph_item_uuid = j.graph_item_uuid;

    SELECT count(*), count(DISTINCT t.source_component_uuid), coalesce(sum(t.token_cost_avoided_estimate), 0)
    INTO v_inserted_count, v_inserted_distinct_sources, v_token_cost_avoided_estimate
    FROM tmp_exact_sha_bulk_ledgered t;

    SELECT count(*)
    INTO v_batch_materializations
    FROM lucidota_go.graph_promotion_materialization
    WHERE detail->>'dedupe_batch_uuid' = p_batch_uuid::text;

    SELECT count(*)
    INTO v_batch_helper_receipts
    FROM lucidota_go.graph_materialization_helper_receipt
    WHERE detail->>'dedupe_batch_uuid' = p_batch_uuid::text;

    SELECT count(*)
    INTO v_batch_bad_policy
    FROM lucidota_go.graph_item
    WHERE payload->>'dedupe_batch_uuid' = p_batch_uuid::text
      AND payload->'payload'->>'dedupe_policy' <> 'exact_sha256_duplicate_alias_no_llm_token';

    SELECT count(*)
    INTO v_batch_bad_guard
    FROM lucidota_go.graph_item
    WHERE payload->>'dedupe_batch_uuid' = p_batch_uuid::text
      AND payload->'payload'->>'fast_path_guard' <> 'DEDUP_ONLY_NOT_FOR_LLM_EXTRACTIONS';

    WITH ranked AS (
        SELECT c.component_uuid::text AS component_uuid,
               c.sha256,
               row_number() OVER (
                   PARTITION BY c.sha256
                   ORDER BY coalesce(f.first_seen_path, ''), c.component_index, c.component_uuid
               ) AS rn,
               count(*) OVER (PARTITION BY c.sha256) AS group_size
        FROM lucidota_korpus.component c
        JOIN lucidota_korpus.file_object f ON f.file_uuid = c.file_uuid
        WHERE c.content IS NOT NULL
          AND (
              coalesce(f.first_seen_path, '') LIKE '/home/mfspx/LUCIDOTA/KRAMPUSCHEWING/%'
              OR coalesce(f.locked_relative_path, '') LIKE 'KRAMPUSCHEWING/%'
          )
    ), targets AS (
        SELECT r.*
        FROM ranked r
        WHERE r.group_size > 1
          AND r.rn > 1
          AND NOT EXISTS (
              SELECT 1
              FROM lucidota_go.graph_item gi
              WHERE gi.payload->'payload'->>'source_component_uuid' = r.component_uuid
          )
    )
    SELECT count(*) INTO v_remaining_exact_sha_alias_targets FROM targets;

    IF v_inserted_count <> v_inserted_distinct_sources
       OR v_inserted_count <> v_batch_materializations
       OR v_inserted_count <> v_batch_helper_receipts
       OR v_batch_bad_policy <> 0
       OR v_batch_bad_guard <> 0 THEN
        out_status := 'REVIEW';
    END IF;

    out_integrity := jsonb_build_object(
        'inserted_count', v_inserted_count,
        'inserted_distinct_sources', v_inserted_distinct_sources,
        'token_cost_avoided_estimate', v_token_cost_avoided_estimate,
        'batch_bad_policy', v_batch_bad_policy,
        'batch_bad_guard', v_batch_bad_guard,
        'batch_materializations', v_batch_materializations,
        'batch_helper_receipts', v_batch_helper_receipts,
        'remaining_exact_sha_alias_targets', v_remaining_exact_sha_alias_targets
    );

    UPDATE lucidota_go.exact_sha_bulk_alias_batch
    SET inserted_count = v_inserted_count,
        inserted_distinct_sources = v_inserted_distinct_sources,
        token_cost_avoided_estimate = v_token_cost_avoided_estimate,
        batch_bad_policy = v_batch_bad_policy,
        batch_bad_guard = v_batch_bad_guard,
        batch_materializations = v_batch_materializations,
        batch_helper_receipts = v_batch_helper_receipts,
        remaining_exact_sha_alias_targets = v_remaining_exact_sha_alias_targets,
        integrity = out_integrity,
        status = out_status,
        completed_at = now()
    WHERE dedupe_batch_uuid = p_batch_uuid;

    RETURN jsonb_build_object(
        'status', out_status,
        'dedupe_batch_uuid', p_batch_uuid,
        'source_system', 'postgres_exact_sha_bulk_alias_ledgered.v1',
        'fast_path_guard', 'DEDUP_ONLY_NOT_FOR_LLM_EXTRACTIONS',
        'canonical_graph_writes_performed', v_inserted_count > 0,
        'write_scope', 'exact_sha256_duplicate_alias_only',
        'integrity', out_integrity
    );
END;
$$;
