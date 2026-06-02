-- FILE: 06_SCHEMA/146_root_rotor_bible_node_tags.sql
-- PURPOSE: compatibility migration for node tagging + API route catalog exposure.
-- This migration is idempotent and safe on already-migrated environments.

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_canon;

-- Preserve legacy rows: backfill missing node_kind and ontology_tags from payload and source path.
ALTER TABLE IF EXISTS lucidota_canon.bible_nodes
    ADD COLUMN IF NOT EXISTS node_kind text,
    ADD COLUMN IF NOT EXISTS ontology_tags text[];

UPDATE lucidota_canon.bible_nodes
SET node_kind = CASE
    WHEN (node_kind IS NULL OR btrim(node_kind) = '') THEN
        NULLIF(
            COALESCE(NULLIF(payload::jsonb->>'node_kind', ''),
                CASE
                    WHEN source_refs->>0 LIKE '06_SCHEMA/%' OR source_refs->>0 LIKE '%.sql' THEN 'SCHEMA'
                    WHEN source_refs->>0 LIKE 'scripts/%' THEN 'WORKFLOW'
                    ELSE 'OBJECT'
                END
            ),
            ''
        )
    ELSE node_kind
END,
ontology_tags = CASE
    WHEN ontology_tags IS NULL OR cardinality(ontology_tags) = 0 THEN
        COALESCE(
            CASE
                WHEN payload_format = 'json'
                     AND payload::jsonb ? 'ontology_tags'
                     AND jsonb_typeof(payload::jsonb->'ontology_tags') = 'array' THEN
                    ARRAY(
                        SELECT jsonb_array_elements_text(payload::jsonb->'ontology_tags')
                        ORDER BY 1
                    )
                ELSE NULL
            END,
            CASE
                WHEN payload::text ~ 'source_refs' THEN ARRAY['OBJECT']
                WHEN source_refs->>0 LIKE '06_SCHEMA/%' OR source_refs->>0 LIKE '%.sql' THEN ARRAY['OBJECT', 'STATE', 'CHURN']
                WHEN source_refs->>0 LIKE 'scripts/%' THEN ARRAY['WORKFLOW', 'OBJECT', 'RECEIPT']
                ELSE ARRAY['OBJECT']
            END
        )
    ELSE ontology_tags
END
WHERE (node_kind IS NULL OR btrim(node_kind) = '' OR ontology_tags IS NULL OR cardinality(ontology_tags) = 0);

-- Keep API materializer in sync if it was already in use without full migration 144.
CREATE OR REPLACE FUNCTION lucidota_canon.fn_bible_node_material(node_row lucidota_canon.bible_nodes)
RETURNS jsonb
LANGUAGE sql
IMMUTABLE
AS $$
    SELECT jsonb_build_object(
        'node_id', node_row.node_id,
        'parent_id', node_row.parent_id,
        'manual_id', node_row.manual_id,
        'node_kind', node_row.node_kind,
        'title', node_row.title,
        'payload', node_row.payload,
        'payload_format', node_row.payload_format,
        'ontology_tags', node_row.ontology_tags,
        'source_refs', node_row.source_refs,
        'evidence_hashes', node_row.evidence_hashes,
        'dependencies', node_row.dependencies,
        'affects_nodes', node_row.affects_nodes,
        'status', node_row.status,
        'valid_from', node_row.valid_from,
        'valid_to', node_row.valid_to
    );
$$;

DO $$
BEGIN
    ALTER TABLE lucidota_canon.bible_nodes
        ALTER COLUMN node_kind SET DEFAULT 'OBJECT',
        ALTER COLUMN ontology_tags SET DEFAULT ARRAY['OBJECT']::text[],
        ALTER COLUMN node_kind SET NOT NULL,
        ALTER COLUMN ontology_tags SET NOT NULL;
EXCEPTION WHEN undefined_column THEN
    NULL;
END;
$$;

-- Ensure check constraints reflect node-tag semantics even on repeated migrations.
DO $$
BEGIN
    ALTER TABLE lucidota_canon.bible_nodes DROP CONSTRAINT IF EXISTS bible_nodes_kind_check;
    ALTER TABLE lucidota_canon.bible_nodes
        ADD CONSTRAINT bible_nodes_kind_check
        CHECK (
            node_kind IN (
                'MANUAL_SECTION', 'OBJECT', 'WORKFLOW', 'EVENT', 'RECEIPT', 'EDGE',
                'STATE', 'BOX', 'CLAIM', 'SOURCE', 'LEDGER', 'SCHEMA', 'CONFIG',
                'SCRIPT', 'ALGORITHM', 'MODEL', 'DAEMON', 'TEST', 'REFERENCE'
            )
        );
EXCEPTION WHEN duplicate_object THEN
    NULL;
END;
$$;

DO $$
BEGIN
    ALTER TABLE lucidota_canon.bible_nodes DROP CONSTRAINT IF EXISTS bible_nodes_ontology_tags_not_empty;
    ALTER TABLE lucidota_canon.bible_nodes
        ADD CONSTRAINT bible_nodes_ontology_tags_not_empty
        CHECK (cardinality(ontology_tags) > 0);
EXCEPTION WHEN duplicate_object THEN
    NULL;
END;
$$;

-- Keep API material surface aligned with tagged nodes.
CREATE OR REPLACE VIEW lucidota_canon.api_bible_nodes AS
SELECT
    node_id,
    parent_id,
    node_sort_key,
    manual_id,
    node_kind,
    title,
    payload,
    payload_format,
    ontology_tags,
    source_refs,
    evidence_hashes,
    dependencies,
    affects_nodes,
    status,
    version,
    valid_from,
    valid_to,
    hash_current,
    previous_hash,
    created_at,
    updated_at
FROM lucidota_canon.bible_nodes
WHERE valid_to IS NULL;

-- Route catalog for DB/API/manual compile surface.
CREATE TABLE IF NOT EXISTS lucidota_canon.api_route_catalog (
    route_id text PRIMARY KEY,
    method text NOT NULL,
    path_pattern text NOT NULL,
    description text NOT NULL,
    target text,
    sample_request text,
    sample_response text,
    status text NOT NULL DEFAULT 'implemented',
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT api_route_catalog_method_check CHECK (method IN ('GET', 'POST', 'PATCH', 'DELETE')),
    CONSTRAINT api_route_catalog_status_check CHECK (status IN ('implemented', 'planned', 'deprecated'))
);

CREATE OR REPLACE VIEW lucidota_canon.api_bible_route_catalog AS
SELECT
    route_id,
    method,
    path_pattern,
    description,
    target,
    sample_request,
    sample_response,
    status,
    created_at,
    updated_at
FROM lucidota_canon.api_route_catalog;

INSERT INTO lucidota_canon.api_route_catalog
(route_id, method, path_pattern, description, target, sample_request, sample_response, status)
VALUES
('nodes', 'GET', '/api_bible_nodes?manual_id=eq.{MANUAL_ID}&order=node_sort_key.asc',
 'List live canonical nodes for a manual.', 'lucidota_canon.api_bible_nodes',
 '{"manual_id":"SYSTEM_ARCH"}', '{"node_id":"1.0.0"...}', 'implemented'),
('manuals', 'GET', '/api_bible_manuals', 'List manual digest rows.', 'lucidota_canon.api_bible_manuals',
 '{"limit":"1"}', '{"manual_id":"SYSTEM_ARCH","node_count":10}', 'implemented'),
('subtree', 'GET', '/api_bible_subtree?root_id=eq.{node_id}', 'Fetch canonical subtree rooted at node id.', 'lucidota_canon.get_subtree(root_id text)',
 '{"root_id":"1"}', '{"node_id":"1"...}', 'implemented'),
('route_catalog', 'GET', '/api_bible_route_catalog', 'List route catalog used by compiler and clients.', 'lucidota_canon.api_bible_route_catalog',
 '{"status":"implemented"}', '{"route_id":"nodes"...}', 'implemented')
('root_law_docs', 'GET', '/root_law_docs', 'Render extensive Root-Law manuals, API routes, and contradiction ledger from PostgREST/route/state evidence.', 'lucidota_canon.api_root_law_docs',
 '{"route":"root_law_docs","manual_ids":["SYSTEM_ARCH","RUNTIME_GOVERNOR","AVIONICS","FLIGHT_MAN","LEDGER"],"emit":"html"}',
 '{"status":"ok","html_path":"05_OUTPUTS/root_rotor_manuals/root_law_docs.html"}', 'implemented')
ON CONFLICT (route_id) DO UPDATE SET
    method = EXCLUDED.method,
    path_pattern = EXCLUDED.path_pattern,
    description = EXCLUDED.description,
    target = EXCLUDED.target,
    sample_request = EXCLUDED.sample_request,
    sample_response = EXCLUDED.sample_response,
    status = EXCLUDED.status,
    updated_at = now();

CREATE INDEX IF NOT EXISTS api_route_catalog_status_idx
    ON lucidota_canon.api_route_catalog(status);
