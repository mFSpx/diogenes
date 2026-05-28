-- KORPUS KRAMPII: deterministic mass-ingestion substrate.
-- Purpose: reclaim legacy sprawl by hashing, deduping, componentizing, indexing,
-- and routing every durable artifact through software algorithms instead of LLM
-- "thinking".

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS vector;

CREATE SCHEMA IF NOT EXISTS lucidota_korpus;


CREATE OR REPLACE FUNCTION lucidota_korpus.uuid_v7()
RETURNS uuid
LANGUAGE plpgsql
VOLATILE
AS $$
DECLARE
    ts_hex text := lpad(to_hex(floor(extract(epoch FROM clock_timestamp()) * 1000)::bigint), 12, '0');
    rnd text := encode(gen_random_bytes(10), 'hex');
    variant text := substr('89ab', (get_byte(gen_random_bytes(1), 0) % 4) + 1, 1);
BEGIN
    RETURN (
        substr(ts_hex, 1, 8) || '-' ||
        substr(ts_hex, 9, 4) || '-' ||
        '7' || substr(rnd, 1, 3) || '-' ||
        variant || substr(rnd, 4, 3) || '-' ||
        substr(rnd, 7, 12)
    )::uuid;
END;
$$;

CREATE TABLE IF NOT EXISTS lucidota_korpus.algo_inventory (
    algo_key text PRIMARY KEY,
    relative_path text NOT NULL,
    module_name text NOT NULL,
    title text NOT NULL DEFAULT '',
    summary text NOT NULL DEFAULT '',
    sha256 text NOT NULL DEFAULT '',
    status text NOT NULL DEFAULT 'available' CHECK (status IN ('available','missing','deprecated')),
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS lucidota_korpus.ingest_batch (
    batch_uuid uuid PRIMARY KEY DEFAULT lucidota_korpus.uuid_v7(),
    batch_label text NOT NULL DEFAULT '',
    case_uuid uuid,
    status text NOT NULL DEFAULT 'running' CHECK (status IN (
        'queued',
        'running',
        'succeeded',
        'failed',
        'cancelled'
    )),
    root_paths jsonb NOT NULL DEFAULT '[]'::jsonb,
    options jsonb NOT NULL DEFAULT '{}'::jsonb,
    file_count integer NOT NULL DEFAULT 0,
    new_file_count integer NOT NULL DEFAULT 0,
    duplicate_file_count integer NOT NULL DEFAULT 0,
    component_count integer NOT NULL DEFAULT 0,
    concept_count integer NOT NULL DEFAULT 0,
    entity_count integer NOT NULL DEFAULT 0,
    skipped_count integer NOT NULL DEFAULT 0,
    error_count integer NOT NULL DEFAULT 0,
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    started_at timestamptz NOT NULL DEFAULT now(),
    finished_at timestamptz
);

CREATE INDEX IF NOT EXISTS ingest_batch_started_idx
    ON lucidota_korpus.ingest_batch(started_at DESC);

CREATE TABLE IF NOT EXISTS lucidota_korpus.file_object (
    file_uuid uuid PRIMARY KEY DEFAULT lucidota_korpus.uuid_v7(),
    sha256 text UNIQUE NOT NULL CHECK (sha256 ~ '^[0-9a-f]{64}$'),
    size_bytes bigint NOT NULL CHECK (size_bytes >= 0),
    mime text NOT NULL DEFAULT '',
    file_kind text NOT NULL DEFAULT '',
    status text NOT NULL DEFAULT 'indexed' CHECK (status IN ('indexed','deferred','error','archived')),
    deferred_reason text NOT NULL DEFAULT '',
    suffix text NOT NULL DEFAULT '',
    cas_uri text NOT NULL DEFAULT '',
    locked_relative_path text NOT NULL DEFAULT '',
    first_seen_path text NOT NULL DEFAULT '',
    first_seen_at timestamptz NOT NULL DEFAULT now(),
    last_seen_at timestamptz NOT NULL DEFAULT now(),
    seen_count integer NOT NULL DEFAULT 1,
    component_count integer NOT NULL DEFAULT 0,
    entity_count integer NOT NULL DEFAULT 0,
    concept_count integer NOT NULL DEFAULT 0,
    minhash_signature jsonb NOT NULL DEFAULT '[]'::jsonb,
    graph_item_uuid uuid,
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS file_object_kind_seen_idx
    ON lucidota_korpus.file_object(file_kind, last_seen_at DESC);

CREATE TABLE IF NOT EXISTS lucidota_korpus.file_tag (
    file_tag_uuid uuid PRIMARY KEY DEFAULT lucidota_korpus.uuid_v7(),
    file_uuid uuid NOT NULL REFERENCES lucidota_korpus.file_object(file_uuid) ON DELETE CASCADE,
    tag_key text NOT NULL REFERENCES lucidota_investigation.tag_taxonomy(tag_key),
    value text NOT NULL DEFAULT '',
    confidence_bps integer NOT NULL DEFAULT 50 CHECK (confidence_bps BETWEEN 0 AND 10000),
    source text NOT NULL DEFAULT '',
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE(file_uuid, tag_key, value, source)
);

CREATE TABLE IF NOT EXISTS lucidota_korpus.file_occurrence (
    occurrence_uuid uuid PRIMARY KEY DEFAULT lucidota_korpus.uuid_v7(),
    batch_uuid uuid NOT NULL REFERENCES lucidota_korpus.ingest_batch(batch_uuid) ON DELETE CASCADE,
    file_uuid uuid NOT NULL REFERENCES lucidota_korpus.file_object(file_uuid) ON DELETE CASCADE,
    absolute_path text NOT NULL,
    relative_path text NOT NULL DEFAULT '',
    root_path text NOT NULL DEFAULT '',
    mtime timestamptz,
    status text NOT NULL DEFAULT 'indexed' CHECK (status IN (
        'indexed',
        'duplicate',
        'skipped',
        'error'
    )),
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE(batch_uuid, absolute_path)
);

CREATE INDEX IF NOT EXISTS file_occurrence_batch_idx
    ON lucidota_korpus.file_occurrence(batch_uuid, status, created_at DESC);

CREATE TABLE IF NOT EXISTS lucidota_korpus.component (
    component_uuid uuid PRIMARY KEY DEFAULT lucidota_korpus.uuid_v7(),
    file_uuid uuid NOT NULL REFERENCES lucidota_korpus.file_object(file_uuid) ON DELETE CASCADE,
    component_index integer NOT NULL,
    component_kind text NOT NULL,
    language text NOT NULL DEFAULT '',
    title text NOT NULL DEFAULT '',
    symbol text NOT NULL DEFAULT '',
    start_line integer NOT NULL DEFAULT 0,
    end_line integer NOT NULL DEFAULT 0,
    sha256 text NOT NULL CHECK (sha256 ~ '^[0-9a-f]{64}$'),
    token_count integer NOT NULL DEFAULT 0,
    content text NOT NULL DEFAULT '',
    go_terms jsonb NOT NULL DEFAULT '[]'::jsonb,
    minhash_signature jsonb NOT NULL DEFAULT '[]'::jsonb,
    entropy double precision NOT NULL DEFAULT 0,
    river_decision text NOT NULL DEFAULT '',
    river_score integer NOT NULL DEFAULT 0 CHECK (river_score BETWEEN 0 AND 10000),
    vibe_spike boolean NOT NULL DEFAULT false,
    river_features jsonb NOT NULL DEFAULT '{}'::jsonb,
    embedding vector(384),
    embedding_model text NOT NULL DEFAULT '',
    embedding_status text NOT NULL DEFAULT 'pending' CHECK (embedding_status IN ('pending','queued','embedded','failed','deferred')),
    embedded_at timestamptz,
    graph_item_uuid uuid,
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE(file_uuid, component_index),
    UNIQUE(file_uuid, sha256, component_kind, start_line)
);

CREATE INDEX IF NOT EXISTS component_sha_idx
    ON lucidota_korpus.component(sha256);

CREATE INDEX IF NOT EXISTS component_file_idx
    ON lucidota_korpus.component(file_uuid, component_index);

ALTER TABLE lucidota_korpus.component
    ADD COLUMN IF NOT EXISTS parent_component_uuid uuid;

CREATE TABLE IF NOT EXISTS lucidota_korpus.component_tag (
    component_tag_uuid uuid PRIMARY KEY DEFAULT lucidota_korpus.uuid_v7(),
    component_uuid uuid NOT NULL REFERENCES lucidota_korpus.component(component_uuid) ON DELETE CASCADE,
    tag_key text NOT NULL REFERENCES lucidota_investigation.tag_taxonomy(tag_key),
    value text NOT NULL DEFAULT '',
    confidence_bps integer NOT NULL DEFAULT 50 CHECK (confidence_bps BETWEEN 0 AND 10000),
    source text NOT NULL DEFAULT '',
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE(component_uuid, tag_key, value, source)
);

CREATE INDEX IF NOT EXISTS component_tag_key_idx
    ON lucidota_korpus.component_tag(tag_key, created_at DESC);

CREATE TABLE IF NOT EXISTS lucidota_korpus.entity (
    korpus_entity_uuid uuid PRIMARY KEY DEFAULT lucidota_korpus.uuid_v7(),
    component_uuid uuid REFERENCES lucidota_korpus.component(component_uuid) ON DELETE CASCADE,
    file_uuid uuid REFERENCES lucidota_korpus.file_object(file_uuid) ON DELETE CASCADE,
    entity_kind text NOT NULL,
    value text NOT NULL,
    normalized_value text NOT NULL,
    confidence_bps integer NOT NULL DEFAULT 50 CHECK (confidence_bps BETWEEN 0 AND 10000),
    source text NOT NULL DEFAULT '',
    context text NOT NULL DEFAULT '',
    graph_item_uuid uuid,
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE(component_uuid, entity_kind, normalized_value, source)
);

CREATE INDEX IF NOT EXISTS korpus_entity_norm_idx
    ON lucidota_korpus.entity(entity_kind, normalized_value, created_at DESC);

CREATE TABLE IF NOT EXISTS lucidota_korpus.concept (
    concept_uuid uuid PRIMARY KEY DEFAULT lucidota_korpus.uuid_v7(),
    component_uuid uuid REFERENCES lucidota_korpus.component(component_uuid) ON DELETE CASCADE,
    file_uuid uuid REFERENCES lucidota_korpus.file_object(file_uuid) ON DELETE CASCADE,
    concept_kind text NOT NULL,
    value text NOT NULL,
    normalized_value text NOT NULL,
    go_term text NOT NULL DEFAULT '',
    source text NOT NULL DEFAULT '',
    confidence_bps integer NOT NULL DEFAULT 50 CHECK (confidence_bps BETWEEN 0 AND 10000),
    graph_item_uuid uuid,
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE(component_uuid, concept_kind, normalized_value, source)
);

CREATE INDEX IF NOT EXISTS korpus_concept_norm_idx
    ON lucidota_korpus.concept(concept_kind, normalized_value, created_at DESC);

CREATE TABLE IF NOT EXISTS lucidota_korpus.near_duplicate (
    near_duplicate_uuid uuid PRIMARY KEY DEFAULT lucidota_korpus.uuid_v7(),
    left_component_uuid uuid NOT NULL REFERENCES lucidota_korpus.component(component_uuid) ON DELETE CASCADE,
    right_component_uuid uuid NOT NULL REFERENCES lucidota_korpus.component(component_uuid) ON DELETE CASCADE,
    similarity double precision NOT NULL,
    algorithm text NOT NULL DEFAULT 'minhash',
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK (left_component_uuid < right_component_uuid),
    UNIQUE(left_component_uuid, right_component_uuid, algorithm)
);

CREATE INDEX IF NOT EXISTS near_duplicate_similarity_idx
    ON lucidota_korpus.near_duplicate(similarity DESC, created_at DESC);


CREATE TABLE IF NOT EXISTS lucidota_korpus.file_link (
    file_link_uuid uuid PRIMARY KEY DEFAULT lucidota_korpus.uuid_v7(),
    source_file_uuid uuid REFERENCES lucidota_korpus.file_object(file_uuid) ON DELETE CASCADE,
    source_component_uuid uuid REFERENCES lucidota_korpus.component(component_uuid) ON DELETE CASCADE,
    raw_target text NOT NULL,
    normalized_target text NOT NULL,
    target_file_uuid uuid REFERENCES lucidota_korpus.file_object(file_uuid) ON DELETE SET NULL,
    link_kind text NOT NULL DEFAULT 'markdown' CHECK (link_kind IN ('markdown','wikilink','url','import','other')),
    anchor_text text NOT NULL DEFAULT '',
    status text NOT NULL DEFAULT 'unresolved' CHECK (status IN ('resolved','unresolved','external')),
    graph_edge_uuid uuid,
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE(source_component_uuid, raw_target, link_kind, anchor_text)
);

CREATE INDEX IF NOT EXISTS file_link_source_idx
    ON lucidota_korpus.file_link(source_file_uuid, link_kind, created_at DESC);

CREATE INDEX IF NOT EXISTS file_link_target_idx
    ON lucidota_korpus.file_link(target_file_uuid, created_at DESC);

CREATE TABLE IF NOT EXISTS lucidota_korpus.correlative_link (
    correlative_link_uuid uuid PRIMARY KEY DEFAULT lucidota_korpus.uuid_v7(),
    left_component_uuid uuid NOT NULL REFERENCES lucidota_korpus.component(component_uuid) ON DELETE CASCADE,
    right_component_uuid uuid NOT NULL REFERENCES lucidota_korpus.component(component_uuid) ON DELETE CASCADE,
    link_kind text NOT NULL DEFAULT 'thematic',
    score integer NOT NULL DEFAULT 0 CHECK (score BETWEEN 0 AND 10000),
    algorithm text NOT NULL DEFAULT '',
    reason text NOT NULL DEFAULT '',
    graph_edge_uuid uuid,
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK (left_component_uuid < right_component_uuid),
    UNIQUE(left_component_uuid, right_component_uuid, link_kind, algorithm)
);

CREATE INDEX IF NOT EXISTS correlative_link_score_idx
    ON lucidota_korpus.correlative_link(score DESC, created_at DESC);

CREATE TABLE IF NOT EXISTS lucidota_korpus.vibe_tag (
    vibe_tag_key text PRIMARY KEY,
    label text NOT NULL,
    description text NOT NULL DEFAULT '',
    go_term text NOT NULL DEFAULT 'COMMENT',
    status text NOT NULL DEFAULT 'active' CHECK (status IN ('active','planned','deprecated')),
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS lucidota_korpus.component_vibe_tag (
    component_vibe_tag_uuid uuid PRIMARY KEY DEFAULT lucidota_korpus.uuid_v7(),
    component_uuid uuid NOT NULL REFERENCES lucidota_korpus.component(component_uuid) ON DELETE CASCADE,
    vibe_tag_key text NOT NULL REFERENCES lucidota_korpus.vibe_tag(vibe_tag_key),
    score integer NOT NULL DEFAULT 0 CHECK (score BETWEEN 0 AND 10000),
    source text NOT NULL DEFAULT '',
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE(component_uuid, vibe_tag_key, source)
);

CREATE INDEX IF NOT EXISTS component_vibe_tag_key_idx
    ON lucidota_korpus.component_vibe_tag(vibe_tag_key, score DESC);


CREATE TABLE IF NOT EXISTS lucidota_korpus.river_model_blob (
    model_key text PRIMARY KEY,
    model_kind text NOT NULL DEFAULT 'river_stats',
    payload bytea NOT NULL,
    sample_count bigint NOT NULL DEFAULT 0,
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS lucidota_korpus.river_decision (
    decision_uuid uuid PRIMARY KEY DEFAULT lucidota_korpus.uuid_v7(),
    batch_uuid uuid REFERENCES lucidota_korpus.ingest_batch(batch_uuid) ON DELETE SET NULL,
    file_uuid uuid REFERENCES lucidota_korpus.file_object(file_uuid) ON DELETE CASCADE,
    component_uuid uuid REFERENCES lucidota_korpus.component(component_uuid) ON DELETE CASCADE,
    model_key text NOT NULL,
    decision_route text NOT NULL,
    score integer NOT NULL DEFAULT 0 CHECK (score BETWEEN 0 AND 10000),
    vibe_spike boolean NOT NULL DEFAULT false,
    features jsonb NOT NULL DEFAULT '{}'::jsonb,
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE(component_uuid, model_key)
);

CREATE INDEX IF NOT EXISTS river_decision_spike_idx
    ON lucidota_korpus.river_decision(vibe_spike, score DESC, created_at DESC);

CREATE TABLE IF NOT EXISTS lucidota_korpus.vibe_telemetry (
    telemetry_id bigserial PRIMARY KEY,
    batch_uuid uuid REFERENCES lucidota_korpus.ingest_batch(batch_uuid) ON DELETE SET NULL,
    file_uuid uuid REFERENCES lucidota_korpus.file_object(file_uuid) ON DELETE CASCADE,
    component_uuid uuid REFERENCES lucidota_korpus.component(component_uuid) ON DELETE CASCADE,
    model_key text NOT NULL DEFAULT 'korpus_component_riverml_stats_v1',
    ingested_at timestamptz NOT NULL DEFAULT now(),
    original_file_date timestamptz,
    original_file_date_source text NOT NULL DEFAULT '',
    original_file_date_confidence_bps integer NOT NULL DEFAULT 0 CHECK (original_file_date_confidence_bps BETWEEN 0 AND 10000),
    assigned_cluster integer,
    assigned_cluster_label text NOT NULL DEFAULT '',
    decision_route text NOT NULL DEFAULT '',
    score integer NOT NULL DEFAULT 0 CHECK (score BETWEEN 0 AND 10000),
    anomaly_score double precision,
    vibe_spike boolean NOT NULL DEFAULT false,
    features jsonb NOT NULL DEFAULT '{}'::jsonb,
    stream_vector jsonb NOT NULL DEFAULT '{}'::jsonb,
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE(component_uuid, model_key)
);

CREATE INDEX IF NOT EXISTS vibe_telemetry_original_date_idx
    ON lucidota_korpus.vibe_telemetry(original_file_date, ingested_at);

CREATE INDEX IF NOT EXISTS vibe_telemetry_cluster_idx
    ON lucidota_korpus.vibe_telemetry(assigned_cluster_label, score DESC, vibe_spike);

CREATE TABLE IF NOT EXISTS lucidota_korpus.embedding_queue (
    embedding_job_uuid uuid PRIMARY KEY DEFAULT lucidota_korpus.uuid_v7(),
    component_uuid uuid NOT NULL REFERENCES lucidota_korpus.component(component_uuid) ON DELETE CASCADE,
    status text NOT NULL DEFAULT 'queued' CHECK (status IN ('queued','running','succeeded','failed','deferred')),
    embedding_model text NOT NULL DEFAULT 'ckdog1.kernel.hash_quantized_embedding.v1',
    attempts integer NOT NULL DEFAULT 0,
    last_error text NOT NULL DEFAULT '',
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE(component_uuid, embedding_model)
);

CREATE INDEX IF NOT EXISTS embedding_queue_status_idx
    ON lucidota_korpus.embedding_queue(status, created_at);

INSERT INTO lucidota_korpus.vibe_tag(vibe_tag_key, label, description, go_term)
VALUES
('algorithm', 'Algorithm / Tooling', 'Executable logic, scripts, functions, classes, automation.', 'ALGORITHM'),
('goal', 'Goal / Objective', 'Intent, objective, mission, roadmap, desired outcome.', 'ACTION'),
('friction', 'Friction / Problem', 'Bug, issue, blocker, risk, failure, pain point.', 'FRICTION'),
('evidence', 'Evidence / Source', 'Artifacts, citations, facts, proof-bearing notes.', 'EVIDENCE'),
('relationship', 'Relationship / Dependency', 'Links, imports, calls, see-also, graph connections.', 'RELATIONSHIP'),
('identity', 'Identity / POI', 'People, orgs, aliases, handles, phones, emails, IPs.', 'ENTITY'),
('memory', 'Institutional Memory', 'Notes that preserve operational/project history.', 'COMMENT'),
('visceral_tech', 'Visceral-Tech Lexicon', 'Biological/physical metaphors fused with hard system vocabulary.', 'ATTRIBUTE'),
('ledger_density', 'Ledger Density', 'Tables, checklists, bullets, and execution-ledger structure.', 'ATTRIBUTE'),
('metacognitive_recursion', 'Metacognitive Recursion', 'System/operator self-audit, critique, drift, and evaluation patterns.', 'COMMENT'),
('directive_command', 'Directive Command', 'Imperative/action-command density versus hedging.', 'ACTION'),
('osint_targeting', 'OSINT Targeting', 'Real-world target identifiers, legal/registry terms, money, phones, corporate IDs.', 'ENTITY'),
('forensic_shield', 'Forensic Shield', 'Distress terms wrapped in rigid dossier/forensic structure.', 'ATTRIBUTE'),
('poetic_purge', 'Poetic Purge', 'High line-ending rhyme/rhythm with low markdown/ledger structure.', 'COMMENT'),
('dissociative_index', 'Third-Person Dissociation', 'First-person references displaced by named self/persona/third-person references.', 'ENTITY'),
('tactical_wrath', 'Tactical Wrath', 'Bureaucratic targets combined with strike/filing/escalation verbs.', 'ACTION'),
('paladin_protocol', 'Paladin Protocol', 'Empathic/protective target language fused with bureaucratic/legal nodes.', 'ACTION'),
('resource_exhaustion', 'Resource Exhaustion Signal', 'Scarcity markers co-occurring with physical pain, grief, or medical stress markers.', 'FRICTION'),
('god_mode_sprint', 'God-Mode Sprint Velocity', 'Many concurrent project/swarm entities plus explicit time-boxing syntax.', 'ACTION'),
('conspiracy_grounding', 'Crucifixion by Logic', 'Sprawling hypothesis language forced through verification/evidence terminology.', 'EVIDENCE'),
('chaotic_good_tax', 'Chaotic Good Tax', 'Professional deliverables combined with deliberate absurdity, swearing, or casual sign-offs.', 'ATTRIBUTE'),
('rainmaker_corporate_grit', 'Rainmaker Corporate/Grit Tension', 'Corporate capital vocabulary colliding with street/grit brand language.', 'ATTRIBUTE'),
('capital_velocity', 'Capital Velocity Countdown', 'Deadline, launch, critical-path, quarter, and countdown compression language.', 'TIME'),
('deliverable_grid', 'Deliverable Grid', 'Formal asset packaging: file extensions, phases, tiers, workstreams, deliverables.', 'TOOL'),
('pitch_formatting', 'Pitch Formatting', 'Bold/skimmable pitch formatting density for external readers.', 'ATTRIBUTE'),
('cybernetic_merge', 'Cybernetic Merge Index', 'Command syntax balanced against collaborative agent/hive syntax.', 'RELATIONSHIP'),
('subtle_knife_discipline', 'Protocol Adherence Tracker', 'Formal tags, cuts, seal/assert/break/build structure per 100 words.', 'PATTERN'),
('rapidthink_velocity', 'Rapidthink Velocity', 'All-caps clusters, swear/adrenaline markers, and tight timestamp bursts.', 'TIME')
ON CONFLICT(vibe_tag_key) DO UPDATE SET label=EXCLUDED.label, description=EXCLUDED.description, go_term=EXCLUDED.go_term, updated_at=now();

-- Operational hardening: UUIDv7 defaults, range confidence, ordered duplicate pairs,
-- and updated_at triggers. File/component UUIDs are DB-issued UUIDv7 on first sight;
-- SHA-256 remains the durable dedupe key.
ALTER TABLE lucidota_korpus.algo_inventory ALTER COLUMN updated_at SET DEFAULT now();
ALTER TABLE lucidota_korpus.ingest_batch ALTER COLUMN batch_uuid SET DEFAULT lucidota_korpus.uuid_v7();
ALTER TABLE lucidota_korpus.file_object ALTER COLUMN file_uuid SET DEFAULT lucidota_korpus.uuid_v7();
ALTER TABLE lucidota_korpus.file_occurrence ALTER COLUMN occurrence_uuid SET DEFAULT lucidota_korpus.uuid_v7();
ALTER TABLE lucidota_korpus.component ALTER COLUMN component_uuid SET DEFAULT lucidota_korpus.uuid_v7();
ALTER TABLE lucidota_korpus.entity ALTER COLUMN korpus_entity_uuid SET DEFAULT lucidota_korpus.uuid_v7();
ALTER TABLE lucidota_korpus.concept ALTER COLUMN concept_uuid SET DEFAULT lucidota_korpus.uuid_v7();
ALTER TABLE lucidota_korpus.near_duplicate ALTER COLUMN near_duplicate_uuid SET DEFAULT lucidota_korpus.uuid_v7();
ALTER TABLE lucidota_korpus.file_tag ALTER COLUMN file_tag_uuid SET DEFAULT lucidota_korpus.uuid_v7();
ALTER TABLE lucidota_korpus.component_tag ALTER COLUMN component_tag_uuid SET DEFAULT lucidota_korpus.uuid_v7();

ALTER TABLE lucidota_korpus.entity DROP CONSTRAINT IF EXISTS entity_confidence_bps_check;
ALTER TABLE lucidota_korpus.entity ADD CONSTRAINT entity_confidence_bps_check CHECK (confidence_bps BETWEEN 0 AND 10000);
ALTER TABLE lucidota_korpus.concept DROP CONSTRAINT IF EXISTS concept_confidence_bps_check;
ALTER TABLE lucidota_korpus.concept ADD CONSTRAINT concept_confidence_bps_check CHECK (confidence_bps BETWEEN 0 AND 10000);
ALTER TABLE lucidota_korpus.file_tag DROP CONSTRAINT IF EXISTS file_tag_confidence_bps_check;
ALTER TABLE lucidota_korpus.file_tag ADD CONSTRAINT file_tag_confidence_bps_check CHECK (confidence_bps BETWEEN 0 AND 10000);
ALTER TABLE lucidota_korpus.component_tag DROP CONSTRAINT IF EXISTS component_tag_confidence_bps_check;
ALTER TABLE lucidota_korpus.component_tag ADD CONSTRAINT component_tag_confidence_bps_check CHECK (confidence_bps BETWEEN 0 AND 10000);

ALTER TABLE lucidota_korpus.near_duplicate DROP CONSTRAINT IF EXISTS near_duplicate_check;
ALTER TABLE lucidota_korpus.near_duplicate DROP CONSTRAINT IF EXISTS near_duplicate_order_check;
ALTER TABLE lucidota_korpus.near_duplicate ADD CONSTRAINT near_duplicate_order_check CHECK (left_component_uuid < right_component_uuid);

CREATE OR REPLACE FUNCTION lucidota_korpus.touch_updated_at()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS touch_algo_inventory_updated_at ON lucidota_korpus.algo_inventory;
CREATE TRIGGER touch_algo_inventory_updated_at BEFORE UPDATE ON lucidota_korpus.algo_inventory
FOR EACH ROW EXECUTE FUNCTION lucidota_korpus.touch_updated_at();

DROP TRIGGER IF EXISTS touch_file_object_updated_at ON lucidota_korpus.file_object;
CREATE TRIGGER touch_file_object_updated_at BEFORE UPDATE ON lucidota_korpus.file_object
FOR EACH ROW EXECUTE FUNCTION lucidota_korpus.touch_updated_at();

-- Search + link/signal operational indexes.
ALTER TABLE lucidota_korpus.component
    ADD COLUMN IF NOT EXISTS fts_vector tsvector GENERATED ALWAYS AS (to_tsvector('english', coalesce(title,'') || ' ' || coalesce(symbol,'') || ' ' || coalesce(content,''))) STORED;

CREATE INDEX IF NOT EXISTS component_fts_idx ON lucidota_korpus.component USING GIN (fts_vector);
CREATE INDEX IF NOT EXISTS concept_value_trgm_idx ON lucidota_korpus.concept USING GIN (normalized_value gin_trgm_ops);
CREATE INDEX IF NOT EXISTS entity_value_trgm_idx ON lucidota_korpus.entity USING GIN (normalized_value gin_trgm_ops);


-- Deferred parse + RiverML + async embedding columns.
ALTER TABLE lucidota_korpus.file_object
    ADD COLUMN IF NOT EXISTS status text NOT NULL DEFAULT 'indexed',
    ADD COLUMN IF NOT EXISTS deferred_reason text NOT NULL DEFAULT '';

ALTER TABLE lucidota_korpus.file_object DROP CONSTRAINT IF EXISTS file_object_status_check;
ALTER TABLE lucidota_korpus.file_object ADD CONSTRAINT file_object_status_check CHECK (status IN ('indexed','deferred','error','archived'));

ALTER TABLE lucidota_korpus.file_occurrence DROP CONSTRAINT IF EXISTS file_occurrence_status_check;
ALTER TABLE lucidota_korpus.file_occurrence ADD CONSTRAINT file_occurrence_status_check CHECK (status IN ('indexed','duplicate','deferred','skipped','error'));

ALTER TABLE lucidota_korpus.component
    ADD COLUMN IF NOT EXISTS river_decision text NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS river_score integer NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS vibe_spike boolean NOT NULL DEFAULT false,
    ADD COLUMN IF NOT EXISTS river_features jsonb NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN IF NOT EXISTS embedding vector(384),
    ADD COLUMN IF NOT EXISTS embedding_model text NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS embedding_status text NOT NULL DEFAULT 'pending',
    ADD COLUMN IF NOT EXISTS embedded_at timestamptz;

ALTER TABLE lucidota_korpus.component DROP CONSTRAINT IF EXISTS component_river_score_check;
ALTER TABLE lucidota_korpus.component ADD CONSTRAINT component_river_score_check CHECK (river_score BETWEEN 0 AND 10000);
ALTER TABLE lucidota_korpus.component DROP CONSTRAINT IF EXISTS component_embedding_status_check;
ALTER TABLE lucidota_korpus.component ADD CONSTRAINT component_embedding_status_check CHECK (embedding_status IN ('pending','queued','embedded','failed','deferred'));

DROP TRIGGER IF EXISTS touch_river_model_blob_updated_at ON lucidota_korpus.river_model_blob;
CREATE TRIGGER touch_river_model_blob_updated_at BEFORE UPDATE ON lucidota_korpus.river_model_blob
FOR EACH ROW EXECUTE FUNCTION lucidota_korpus.touch_updated_at();

DROP TRIGGER IF EXISTS touch_embedding_queue_updated_at ON lucidota_korpus.embedding_queue;
CREATE TRIGGER touch_embedding_queue_updated_at BEFORE UPDATE ON lucidota_korpus.embedding_queue
FOR EACH ROW EXECUTE FUNCTION lucidota_korpus.touch_updated_at();
