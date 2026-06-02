-- FILE: 06_SCHEMA/144_canonical_technical_bible.sql
-- Porges Protocol V2: DB-coordinate Canonical Technical Bible.
-- The database stores current canon nodes. Files and compiled manuals are views/receipts.

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_canon;

CREATE OR REPLACE FUNCTION lucidota_canon.fn_bible_node_sort_key(p_node_id text)
RETURNS integer[]
LANGUAGE sql
IMMUTABLE
AS $$
    SELECT COALESCE(array_agg(part::integer ORDER BY ord), ARRAY[]::integer[])
    FROM unnest(regexp_split_to_array(p_node_id, '\.')) WITH ORDINALITY AS x(part, ord)
    WHERE part ~ '^[0-9]+$';
$$;

CREATE TABLE IF NOT EXISTS lucidota_canon.bible_nodes (
    node_id text PRIMARY KEY,
    parent_id text REFERENCES lucidota_canon.bible_nodes(node_id) ON UPDATE CASCADE ON DELETE RESTRICT,
    node_sort_key integer[] NOT NULL,
    manual_id text NOT NULL,
    node_kind text NOT NULL DEFAULT 'OBJECT',
    title text NOT NULL,
    payload text NOT NULL,
    payload_format text NOT NULL DEFAULT 'text',
    ontology_tags text[] NOT NULL DEFAULT ARRAY['OBJECT']::text[],
    source_refs jsonb NOT NULL DEFAULT '[]'::jsonb,
    evidence_hashes jsonb NOT NULL DEFAULT '[]'::jsonb,
    dependencies text[] NOT NULL DEFAULT '{}'::text[],
    affects_nodes text[] NOT NULL DEFAULT '{}'::text[],
    status text NOT NULL DEFAULT 'verified',
    version integer NOT NULL DEFAULT 1,
    valid_from timestamptz NOT NULL DEFAULT now(),
    valid_to timestamptz,
    hash_current character(64) NOT NULL,
    previous_hash character(64),
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT bible_nodes_manual_check CHECK (manual_id IN ('SYSTEM_ARCH', 'RUNTIME_GOVERNOR', 'AVIONICS', 'FLIGHT_MAN', 'LEDGER')),
    CONSTRAINT bible_nodes_kind_check CHECK (node_kind IN ('MANUAL_SECTION', 'OBJECT', 'WORKFLOW', 'EVENT', 'RECEIPT', 'EDGE', 'STATE', 'BOX', 'CLAIM', 'SOURCE', 'LEDGER', 'SCHEMA', 'CONFIG', 'SCRIPT', 'ALGORITHM', 'MODEL', 'DAEMON', 'TEST', 'REFERENCE')),
    CONSTRAINT bible_nodes_ontology_tags_not_empty CHECK (cardinality(ontology_tags) > 0),
    CONSTRAINT bible_nodes_status_check CHECK (status IN ('verified', 'review_required', 'deprecated', 'draft')),
    CONSTRAINT bible_nodes_payload_format_check CHECK (payload_format IN ('text', 'markdown', 'json', 'mermaid', 'sql')),
    CONSTRAINT bible_nodes_version_positive CHECK (version > 0),
    CONSTRAINT bible_nodes_hash_hex CHECK (hash_current ~ '^[0-9a-f]{64}$'),
    CONSTRAINT bible_nodes_previous_hash_hex CHECK (previous_hash IS NULL OR previous_hash ~ '^[0-9a-f]{64}$'),
    UNIQUE (node_id, version)
);

CREATE TABLE IF NOT EXISTS lucidota_canon.bible_dependencies (
    edge_id bigserial PRIMARY KEY,
    from_node_id text NOT NULL REFERENCES lucidota_canon.bible_nodes(node_id) ON UPDATE CASCADE ON DELETE CASCADE,
    to_node_id text NOT NULL REFERENCES lucidota_canon.bible_nodes(node_id) ON UPDATE CASCADE ON DELETE CASCADE,
    edge_kind text NOT NULL,
    evidence jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT bible_dependencies_kind_check CHECK (edge_kind IN ('depends_on', 'affects', 'source_ref', 'review_edge')),
    CONSTRAINT bible_dependencies_no_self_edge CHECK (from_node_id <> to_node_id),
    UNIQUE (from_node_id, to_node_id, edge_kind)
);

CREATE TABLE IF NOT EXISTS lucidota_canon.bible_history (
    history_id bigserial PRIMARY KEY,
    node_id text NOT NULL,
    manual_id text NOT NULL,
    version integer NOT NULL,
    payload text NOT NULL,
    hash_current character(64) NOT NULL,
    old_row jsonb NOT NULL,
    archived_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (node_id, version)
);

CREATE INDEX IF NOT EXISTS bible_nodes_manual_sort_idx
    ON lucidota_canon.bible_nodes(manual_id, node_sort_key);

CREATE INDEX IF NOT EXISTS bible_nodes_status_idx
    ON lucidota_canon.bible_nodes(status, updated_at DESC);

CREATE INDEX IF NOT EXISTS bible_nodes_ontology_tags_idx
    ON lucidota_canon.bible_nodes USING gin(ontology_tags);

CREATE INDEX IF NOT EXISTS bible_nodes_parent_idx
    ON lucidota_canon.bible_nodes(parent_id, node_sort_key);

CREATE INDEX IF NOT EXISTS bible_dependencies_to_idx
    ON lucidota_canon.bible_dependencies(to_node_id, edge_kind);

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

CREATE OR REPLACE FUNCTION lucidota_canon.tg_enforce_canon_integrity()
RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE
    calc_hash character(64);
BEGIN
    NEW.node_sort_key := lucidota_canon.fn_bible_node_sort_key(NEW.node_id);
    NEW.updated_at := now();
    calc_hash := encode(digest(lucidota_canon.fn_bible_node_material(NEW)::text, 'sha256'), 'hex');

    IF TG_OP = 'INSERT' THEN
        NEW.version := COALESCE(NEW.version, 1);
        NEW.hash_current := calc_hash;
        RETURN NEW;
    END IF;

    IF OLD.hash_current IS DISTINCT FROM calc_hash THEN
        INSERT INTO lucidota_canon.bible_history (node_id, manual_id, version, payload, hash_current, old_row)
        VALUES (OLD.node_id, OLD.manual_id, OLD.version, OLD.payload, OLD.hash_current, to_jsonb(OLD));

        NEW.version := OLD.version + 1;
        NEW.previous_hash := OLD.hash_current;
        NEW.hash_current := calc_hash;
    ELSE
        NEW.version := OLD.version;
        NEW.previous_hash := OLD.previous_hash;
        NEW.hash_current := OLD.hash_current;
    END IF;

    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS tr_canon_node_integrity_gate ON lucidota_canon.bible_nodes;
CREATE TRIGGER tr_canon_node_integrity_gate
    BEFORE INSERT OR UPDATE ON lucidota_canon.bible_nodes
    FOR EACH ROW EXECUTE FUNCTION lucidota_canon.tg_enforce_canon_integrity();

CREATE OR REPLACE FUNCTION lucidota_canon.fn_sync_bible_dependency_edges()
RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE
    dep_node text;
BEGIN
    DELETE FROM lucidota_canon.bible_dependencies
    WHERE from_node_id = NEW.node_id
      AND edge_kind IN ('depends_on', 'affects');

    FOREACH dep_node IN ARRAY COALESCE(NEW.dependencies, '{}'::text[]) LOOP
        INSERT INTO lucidota_canon.bible_dependencies(from_node_id, to_node_id, edge_kind)
        VALUES (NEW.node_id, dep_node, 'depends_on')
        ON CONFLICT DO NOTHING;
    END LOOP;

    FOREACH dep_node IN ARRAY COALESCE(NEW.affects_nodes, '{}'::text[]) LOOP
        INSERT INTO lucidota_canon.bible_dependencies(from_node_id, to_node_id, edge_kind)
        VALUES (NEW.node_id, dep_node, 'affects')
        ON CONFLICT DO NOTHING;
    END LOOP;

    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS tr_canon_node_edge_sync ON lucidota_canon.bible_nodes;
CREATE TRIGGER tr_canon_node_edge_sync
    AFTER INSERT OR UPDATE OF dependencies, affects_nodes ON lucidota_canon.bible_nodes
    FOR EACH ROW EXECUTE FUNCTION lucidota_canon.fn_sync_bible_dependency_edges();

CREATE OR REPLACE FUNCTION lucidota_canon.fn_mark_bible_blast_radius()
RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE
    dep_node text;
BEGIN
    IF TG_OP = 'UPDATE' AND OLD.hash_current IS DISTINCT FROM NEW.hash_current THEN
        FOREACH dep_node IN ARRAY COALESCE(NEW.affects_nodes, '{}'::text[]) LOOP
            UPDATE lucidota_canon.bible_nodes
            SET status = 'review_required', updated_at = now()
            WHERE node_id = dep_node
              AND status <> 'deprecated';
        END LOOP;
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS tr_canon_node_blast_radius ON lucidota_canon.bible_nodes;
CREATE TRIGGER tr_canon_node_blast_radius
    AFTER UPDATE ON lucidota_canon.bible_nodes
    FOR EACH ROW EXECUTE FUNCTION lucidota_canon.fn_mark_bible_blast_radius();

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

CREATE OR REPLACE VIEW lucidota_canon.api_bible_edges AS
SELECT
    edge_id,
    from_node_id,
    to_node_id,
    edge_kind,
    evidence,
    created_at
FROM lucidota_canon.bible_dependencies;

CREATE OR REPLACE VIEW lucidota_canon.api_bible_manuals AS
SELECT
    manual_id,
    count(*) AS node_count,
    max(version) AS max_node_version,
    max(updated_at) AS last_updated_at,
    encode(digest(string_agg(hash_current, '' ORDER BY node_sort_key), 'sha256'), 'hex') AS manual_hash
FROM lucidota_canon.bible_nodes
WHERE valid_to IS NULL
GROUP BY manual_id;

CREATE OR REPLACE FUNCTION lucidota_canon.get_subtree(root_id text)
RETURNS SETOF lucidota_canon.api_bible_nodes
LANGUAGE plpgsql
STABLE
AS $$
BEGIN
    RETURN QUERY
    WITH RECURSIVE manual_tree AS (
        SELECT n.*
        FROM lucidota_canon.api_bible_nodes n
        WHERE n.node_id = root_id
        UNION ALL
        SELECT child.*
        FROM lucidota_canon.api_bible_nodes child
        INNER JOIN manual_tree parent ON child.parent_id = parent.node_id
    )
    SELECT * FROM manual_tree ORDER BY node_sort_key;
END;
$$;
