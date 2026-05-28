-- Deterministic bare-steel purge matrix.
-- Stable structural operations move to PostgreSQL; model lanes stay untrusted extraction periphery.

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_go;
CREATE SCHEMA IF NOT EXISTS lucidota_korpus;

CREATE OR REPLACE FUNCTION lucidota_go.fn_normalized_staging_term(raw_term text)
RETURNS text
LANGUAGE sql
IMMUTABLE
AS $$
    SELECT CASE upper(btrim(coalesce(raw_term, '')))
        WHEN '' THEN NULL
        WHEN 'ENTITIES' THEN 'ENTITY'
        WHEN 'NODES' THEN 'ENTITY'
        WHEN 'NODE' THEN 'ENTITY'
        WHEN 'OBJECTS' THEN 'ENTITY'
        WHEN 'OBJECT' THEN 'ENTITY'
        WHEN 'EVENTS' THEN 'EVENT'
        WHEN 'CLAIMS' THEN 'CLAIM'
        WHEN 'ASSERTIONS' THEN 'CLAIM'
        WHEN 'ASSERTION' THEN 'CLAIM'
        WHEN 'EVIDENCES' THEN 'EVIDENCE'
        WHEN 'PROOFS' THEN 'EVIDENCE'
        WHEN 'PROOF' THEN 'EVIDENCE'
        ELSE upper(btrim(coalesce(raw_term, '')))
    END
$$;

CREATE OR REPLACE FUNCTION lucidota_go.fn_auto_alias_staging_packet()
RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE
    raw_term text;
    norm_term text;
    term_exists boolean;
BEGIN
    raw_term := NEW.proposed_term;
    norm_term := lucidota_go.fn_normalized_staging_term(raw_term);

    NEW.proposed_item := coalesce(NEW.proposed_item, '{}'::jsonb)
        || jsonb_build_object(
            'staging_hygiene', jsonb_build_object(
                'raw_proposed_term', raw_term,
                'normalized_proposed_term', norm_term,
                'normalizer', 'lucidota_go.fn_auto_alias_staging_packet.v1'
            )
        );

    IF norm_term IS NULL THEN
        NEW.proposed_term := NULL;
        NEW.status := 'needs_repair';
        RETURN NEW;
    END IF;

    SELECT EXISTS (SELECT 1 FROM lucidota_go.term_registry tr WHERE tr.term = norm_term)
      INTO term_exists;

    IF NOT term_exists THEN
        NEW.proposed_term := NULL;
        NEW.status := 'needs_repair';
        RETURN NEW;
    END IF;

    NEW.proposed_term := norm_term;

    IF NEW.parser_name = 'groq_rickshaw_go25_extractor.v1'
       AND NEW.proposed_term NOT IN ('ENTITY', 'EVENT', 'CLAIM', 'EVIDENCE') THEN
        NEW.proposed_term := NULL;
        NEW.status := 'needs_repair';
    END IF;

    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_pre_stage_hygiene ON lucidota_go.staging_packet;
CREATE TRIGGER trg_pre_stage_hygiene
BEFORE INSERT OR UPDATE OF proposed_term, proposed_item, parser_name
ON lucidota_go.staging_packet
FOR EACH ROW
EXECUTE FUNCTION lucidota_go.fn_auto_alias_staging_packet();

CREATE TABLE IF NOT EXISTS lucidota_korpus.krampus_quarantine_route (
    route_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    component_uuid uuid NOT NULL REFERENCES lucidota_korpus.component(component_uuid) ON DELETE CASCADE,
    route_reason text NOT NULL,
    entropy double precision NOT NULL DEFAULT 0,
    token_count integer NOT NULL DEFAULT 0,
    content_sha256 text NOT NULL DEFAULT '',
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE(component_uuid, route_reason)
);

CREATE INDEX IF NOT EXISTS krampus_quarantine_route_reason_idx
    ON lucidota_korpus.krampus_quarantine_route(route_reason, created_at DESC);

CREATE OR REPLACE FUNCTION lucidota_korpus.fn_high_entropy_noise(entropy_value double precision, token_count_value integer, content_value text)
RETURNS boolean
LANGUAGE sql
IMMUTABLE
AS $$
    SELECT coalesce(entropy_value, 0) > 5.45
        OR (
            coalesce(token_count_value, 0) <= 3
            AND length(coalesce(content_value, '')) >= 40
            AND coalesce(content_value, '') !~* '[aeiouy].*[aeiouy].*[aeiouy]'
        )
$$;

CREATE OR REPLACE FUNCTION lucidota_korpus.fn_route_component_high_entropy_quarantine()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF lucidota_korpus.fn_high_entropy_noise(NEW.entropy, NEW.token_count, NEW.content) THEN
        INSERT INTO lucidota_korpus.krampus_quarantine_route(
            component_uuid, route_reason, entropy, token_count, content_sha256, detail
        ) VALUES (
            NEW.component_uuid,
            'high_entropy_or_keyboard_faceroll_noise',
            NEW.entropy,
            NEW.token_count,
            NEW.sha256,
            jsonb_build_object('router', 'lucidota_korpus.fn_route_component_high_entropy_quarantine.v1')
        )
        ON CONFLICT(component_uuid, route_reason) DO UPDATE SET
            entropy = EXCLUDED.entropy,
            token_count = EXCLUDED.token_count,
            content_sha256 = EXCLUDED.content_sha256,
            detail = EXCLUDED.detail,
            updated_at = now();
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_component_high_entropy_quarantine ON lucidota_korpus.component;
CREATE TRIGGER trg_component_high_entropy_quarantine
AFTER INSERT OR UPDATE OF entropy, token_count, content
ON lucidota_korpus.component
FOR EACH ROW
EXECUTE FUNCTION lucidota_korpus.fn_route_component_high_entropy_quarantine();

CREATE OR REPLACE FUNCTION lucidota_go.fn_ternary_product(value integer, modifier integer)
RETURNS integer
LANGUAGE plpgsql
IMMUTABLE
AS $$
BEGIN
    IF value NOT IN (-1,0,1) OR modifier NOT IN (-1,0,1) THEN
        RAISE EXCEPTION 'ternary operands must be -1, 0, or 1';
    END IF;
    RETURN value * modifier;
END;
$$;

CREATE OR REPLACE FUNCTION lucidota_go.fn_kleene_k3(vals integer[])
RETURNS integer
LANGUAGE sql
IMMUTABLE
AS $$
    SELECT CASE
        WHEN vals IS NULL OR array_length(vals, 1) IS NULL THEN 0
        WHEN EXISTS (SELECT 1 FROM unnest(vals) v WHERE v = -1) THEN -1
        WHEN EXISTS (SELECT 1 FROM unnest(vals) v WHERE v = 0) THEN 0
        ELSE 1
    END
$$;

CREATE OR REPLACE VIEW lucidota_go.ternary_valency_summary AS
SELECT
    count(*) AS graph_item_count,
    count(*) FILTER (WHERE ternary_valency = 1) AS positive_count,
    count(*) FILTER (WHERE ternary_valency = 0) AS neutral_count,
    count(*) FILTER (WHERE ternary_valency = -1) AS negative_count,
    coalesce(sum(ternary_valency), 0) AS net_spatial_polarity,
    lucidota_go.fn_kleene_k3(array_agg(ternary_valency)) AS kleene_k3_state
FROM lucidota_go.graph_item;

CREATE OR REPLACE FUNCTION lucidota_korpus.fn_exact_sha_duplicate_alias_candidates(max_rows integer DEFAULT 500)
RETURNS TABLE(
    component_uuid uuid,
    representative_component_uuid uuid,
    component_sha256 text,
    duplicate_rank bigint,
    duplicate_group_size bigint,
    token_count integer,
    entropy double precision
)
LANGUAGE sql
STABLE
AS $$
    WITH ranked AS (
      SELECT c.component_uuid,
             first_value(c.component_uuid) OVER (PARTITION BY c.sha256 ORDER BY COALESCE(f.first_seen_path,''), c.component_index, c.component_uuid) AS representative_component_uuid,
             c.sha256 AS component_sha256,
             row_number() OVER (PARTITION BY c.sha256 ORDER BY COALESCE(f.first_seen_path,''), c.component_index, c.component_uuid) AS duplicate_rank,
             count(*) OVER (PARTITION BY c.sha256) AS duplicate_group_size,
             c.token_count,
             c.entropy
      FROM lucidota_korpus.component c
      JOIN lucidota_korpus.file_object f ON f.file_uuid=c.file_uuid
      WHERE c.content IS NOT NULL
        AND (COALESCE(f.first_seen_path,'') LIKE '/home/mfspx/LUCIDOTA/KRAMPUSCHEWING/%'
             OR COALESCE(f.locked_relative_path,'') LIKE 'KRAMPUSCHEWING/%')
    )
    SELECT r.component_uuid, r.representative_component_uuid, r.component_sha256,
           r.duplicate_rank, r.duplicate_group_size, r.token_count, r.entropy
    FROM ranked r
    WHERE r.duplicate_group_size > 1
      AND r.duplicate_rank > 1
      AND NOT EXISTS (
        SELECT 1 FROM lucidota_go.graph_item gi
        WHERE gi.payload->'payload'->>'source_component_uuid' = r.component_uuid::text
      )
    ORDER BY r.duplicate_group_size DESC, r.component_sha256, r.duplicate_rank
    LIMIT greatest(1, least(coalesce(max_rows, 500), 5000));
$$;
