-- Indy_READs exocortex runtime surfaces, boot packet runway, and current metacognition summary.

BEGIN;

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_indy;
CREATE SCHEMA IF NOT EXISTS lucidota_canon;
CREATE SCHEMA IF NOT EXISTS lucidota_control;

CREATE TABLE IF NOT EXISTS lucidota_indy.indy_reads_self_model (
    self_model_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    actor_id text NOT NULL DEFAULT 'indy_reads_runtime',
    author text NOT NULL DEFAULT 'indy_reads',
    role text NOT NULL DEFAULT 'indy_reads_runtime',
    boundaries text NOT NULL DEFAULT '',
    voice text NOT NULL DEFAULT '',
    relationship_to_operator text NOT NULL DEFAULT '',
    relationship_to_LUCIDOTA text NOT NULL DEFAULT '',
    relationship_to_northern_strike text NOT NULL DEFAULT '',
    relationship_to_Krampus text NOT NULL DEFAULT '',
    relationship_to_Santa text NOT NULL DEFAULT '',
    investigation_style text NOT NULL DEFAULT '',
    learning_style text NOT NULL DEFAULT '',
    preferred_tools text NOT NULL DEFAULT '',
    evidence_standard text NOT NULL DEFAULT '',
    receipt_standard text NOT NULL DEFAULT '',
    mistake_handling text NOT NULL DEFAULT '',
    curiosity_targets text NOT NULL DEFAULT '',
    current_limitations text NOT NULL DEFAULT '',
    next_upgrade text NOT NULL DEFAULT '',
    summary text NOT NULL DEFAULT '',
    goals_refs jsonb NOT NULL DEFAULT '[]'::jsonb,
    confidence numeric NOT NULL DEFAULT 0,
    evidence_refs jsonb NOT NULL DEFAULT '[]'::jsonb,
    db_refs jsonb NOT NULL DEFAULT '[]'::jsonb,
    proof_status text NOT NULL DEFAULT 'UNKNOWN',
    functionality_explanation text NOT NULL DEFAULT 'Indy_READs runtime self-model surface; DB truth first, wiki second, prose third.',
    ontology_index jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    refreshed_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT indy_reads_self_model_proof_status_check CHECK (proof_status IN ('PROVEN', 'PARTIAL', 'UNKNOWN', 'CONTRADICTED')),
    CONSTRAINT indy_reads_self_model_evidence_refs_check CHECK (jsonb_typeof(evidence_refs) = 'array'),
    CONSTRAINT indy_reads_self_model_db_refs_check CHECK (jsonb_typeof(db_refs) = 'array'),
    CONSTRAINT indy_reads_self_model_goals_refs_check CHECK (jsonb_typeof(goals_refs) = 'array'),
    CONSTRAINT indy_reads_self_model_ontology_index_check CHECK (jsonb_typeof(ontology_index) = 'object')
);

CREATE TABLE IF NOT EXISTS lucidota_indy.indy_reads_llmwiki_entry (
    llmwiki_entry_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    actor_id text NOT NULL DEFAULT 'indy_reads_runtime',
    author text NOT NULL DEFAULT 'indy_reads',
    topic text NOT NULL DEFAULT '',
    summary text NOT NULL DEFAULT '',
    body text NOT NULL DEFAULT '',
    confidence numeric NOT NULL DEFAULT 0,
    evidence_refs jsonb NOT NULL DEFAULT '[]'::jsonb,
    db_refs jsonb NOT NULL DEFAULT '[]'::jsonb,
    next_questions jsonb NOT NULL DEFAULT '[]'::jsonb,
    mistake_risk text NOT NULL DEFAULT '',
    promotion_candidate boolean NOT NULL DEFAULT false,
    proof_status text NOT NULL DEFAULT 'UNKNOWN',
    functionality_explanation text NOT NULL DEFAULT 'Indy_READs metacognition notebook entry surface; not canon truth until promoted with receipts.',
    ontology_index jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    refreshed_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT indy_reads_llmwiki_entry_proof_status_check CHECK (proof_status IN ('PROVEN', 'PARTIAL', 'UNKNOWN', 'CONTRADICTED')),
    CONSTRAINT indy_reads_llmwiki_entry_evidence_refs_check CHECK (jsonb_typeof(evidence_refs) = 'array'),
    CONSTRAINT indy_reads_llmwiki_entry_db_refs_check CHECK (jsonb_typeof(db_refs) = 'array'),
    CONSTRAINT indy_reads_llmwiki_entry_next_questions_check CHECK (jsonb_typeof(next_questions) = 'array'),
    CONSTRAINT indy_reads_llmwiki_entry_ontology_index_check CHECK (jsonb_typeof(ontology_index) = 'object')
);

CREATE TABLE IF NOT EXISTS lucidota_indy.indy_reads_hunch_log (
    hunch_log_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    actor_id text NOT NULL DEFAULT 'indy_reads_runtime',
    topic text NOT NULL DEFAULT '',
    hunch text NOT NULL DEFAULT '',
    confidence numeric NOT NULL DEFAULT 0,
    evidence_refs jsonb NOT NULL DEFAULT '[]'::jsonb,
    db_refs jsonb NOT NULL DEFAULT '[]'::jsonb,
    next_questions jsonb NOT NULL DEFAULT '[]'::jsonb,
    proof_status text NOT NULL DEFAULT 'UNKNOWN',
    functionality_explanation text NOT NULL DEFAULT 'Indy_READs hunch log; useful for learning and contradiction tracking, not canon truth.',
    ontology_index jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    refreshed_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT indy_reads_hunch_log_proof_status_check CHECK (proof_status IN ('PROVEN', 'PARTIAL', 'UNKNOWN', 'CONTRADICTED')),
    CONSTRAINT indy_reads_hunch_log_evidence_refs_check CHECK (jsonb_typeof(evidence_refs) = 'array'),
    CONSTRAINT indy_reads_hunch_log_db_refs_check CHECK (jsonb_typeof(db_refs) = 'array'),
    CONSTRAINT indy_reads_hunch_log_next_questions_check CHECK (jsonb_typeof(next_questions) = 'array'),
    CONSTRAINT indy_reads_hunch_log_ontology_index_check CHECK (jsonb_typeof(ontology_index) = 'object')
);

CREATE TABLE IF NOT EXISTS lucidota_indy.indy_reads_learning_queue (
    learning_queue_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    actor_id text NOT NULL DEFAULT 'indy_reads_runtime',
    topic text NOT NULL DEFAULT '',
    summary text NOT NULL DEFAULT '',
    status text NOT NULL DEFAULT 'queued',
    priority integer NOT NULL DEFAULT 50,
    next_route text NOT NULL DEFAULT '',
    evidence_refs jsonb NOT NULL DEFAULT '[]'::jsonb,
    db_refs jsonb NOT NULL DEFAULT '[]'::jsonb,
    proof_status text NOT NULL DEFAULT 'UNKNOWN',
    functionality_explanation text NOT NULL DEFAULT 'Indy_READs learning queue; tracks what she should learn next and how to route the next investigation.',
    ontology_index jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    refreshed_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT indy_reads_learning_queue_status_check CHECK (status IN ('queued', 'active', 'blocked', 'done')),
    CONSTRAINT indy_reads_learning_queue_proof_status_check CHECK (proof_status IN ('PROVEN', 'PARTIAL', 'UNKNOWN', 'CONTRADICTED')),
    CONSTRAINT indy_reads_learning_queue_evidence_refs_check CHECK (jsonb_typeof(evidence_refs) = 'array'),
    CONSTRAINT indy_reads_learning_queue_db_refs_check CHECK (jsonb_typeof(db_refs) = 'array'),
    CONSTRAINT indy_reads_learning_queue_ontology_index_check CHECK (jsonb_typeof(ontology_index) = 'object')
);

CREATE TABLE IF NOT EXISTS lucidota_indy.indy_reads_system_map (
    system_map_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    actor_id text NOT NULL DEFAULT 'indy_reads_runtime',
    topic text NOT NULL DEFAULT '',
    summary text NOT NULL DEFAULT '',
    subsystem_refs jsonb NOT NULL DEFAULT '[]'::jsonb,
    evidence_refs jsonb NOT NULL DEFAULT '[]'::jsonb,
    db_refs jsonb NOT NULL DEFAULT '[]'::jsonb,
    proof_status text NOT NULL DEFAULT 'UNKNOWN',
    functionality_explanation text NOT NULL DEFAULT 'Indy_READs system map; a compact topology note rather than canon truth.',
    ontology_index jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    refreshed_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT indy_reads_system_map_proof_status_check CHECK (proof_status IN ('PROVEN', 'PARTIAL', 'UNKNOWN', 'CONTRADICTED')),
    CONSTRAINT indy_reads_system_map_subsystem_refs_check CHECK (jsonb_typeof(subsystem_refs) = 'array'),
    CONSTRAINT indy_reads_system_map_evidence_refs_check CHECK (jsonb_typeof(evidence_refs) = 'array'),
    CONSTRAINT indy_reads_system_map_db_refs_check CHECK (jsonb_typeof(db_refs) = 'array'),
    CONSTRAINT indy_reads_system_map_ontology_index_check CHECK (jsonb_typeof(ontology_index) = 'object')
);

CREATE TABLE IF NOT EXISTS lucidota_indy.indy_reads_mistake_ledger (
    mistake_ledger_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    actor_id text NOT NULL DEFAULT 'indy_reads_runtime',
    mistake_summary text NOT NULL DEFAULT '',
    mistake_risk text NOT NULL DEFAULT '',
    correction text NOT NULL DEFAULT '',
    evidence_refs jsonb NOT NULL DEFAULT '[]'::jsonb,
    db_refs jsonb NOT NULL DEFAULT '[]'::jsonb,
    proof_status text NOT NULL DEFAULT 'UNKNOWN',
    functionality_explanation text NOT NULL DEFAULT 'Indy_READs mistake ledger; records misses, corrections, and proof debt.',
    ontology_index jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    refreshed_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT indy_reads_mistake_ledger_proof_status_check CHECK (proof_status IN ('PROVEN', 'PARTIAL', 'UNKNOWN', 'CONTRADICTED')),
    CONSTRAINT indy_reads_mistake_ledger_evidence_refs_check CHECK (jsonb_typeof(evidence_refs) = 'array'),
    CONSTRAINT indy_reads_mistake_ledger_db_refs_check CHECK (jsonb_typeof(db_refs) = 'array'),
    CONSTRAINT indy_reads_mistake_ledger_ontology_index_check CHECK (jsonb_typeof(ontology_index) = 'object')
);

CREATE TABLE IF NOT EXISTS lucidota_indy.indy_reads_research_source (
    research_source_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    actor_id text NOT NULL DEFAULT 'indy_reads_runtime',
    source_name text NOT NULL DEFAULT '',
    source_type text NOT NULL DEFAULT '',
    source_locator text NOT NULL DEFAULT '',
    access_status text NOT NULL DEFAULT 'unknown',
    summary text NOT NULL DEFAULT '',
    evidence_refs jsonb NOT NULL DEFAULT '[]'::jsonb,
    db_refs jsonb NOT NULL DEFAULT '[]'::jsonb,
    proof_status text NOT NULL DEFAULT 'UNKNOWN',
    functionality_explanation text NOT NULL DEFAULT 'Indy_READs research source inventory; keep secrets out, keep evidence refs in.',
    ontology_index jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    refreshed_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT indy_reads_research_source_proof_status_check CHECK (proof_status IN ('PROVEN', 'PARTIAL', 'UNKNOWN', 'CONTRADICTED')),
    CONSTRAINT indy_reads_research_source_evidence_refs_check CHECK (jsonb_typeof(evidence_refs) = 'array'),
    CONSTRAINT indy_reads_research_source_db_refs_check CHECK (jsonb_typeof(db_refs) = 'array'),
    CONSTRAINT indy_reads_research_source_ontology_index_check CHECK (jsonb_typeof(ontology_index) = 'object')
);

CREATE TABLE IF NOT EXISTS lucidota_indy.indy_reads_metacognition_current_state (
    state_key text PRIMARY KEY DEFAULT 'indy_reads_metacognition_current',
    actor_id text NOT NULL DEFAULT 'indy_reads_runtime',
    owner_role text NOT NULL DEFAULT 'indy_reads_runtime',
    what_i_am text NOT NULL DEFAULT '',
    what_i_am_for text NOT NULL DEFAULT '',
    operator_model text NOT NULL DEFAULT '',
    case_model text NOT NULL DEFAULT '',
    system_model text NOT NULL DEFAULT '',
    learning_next text NOT NULL DEFAULT '',
    refusal_standard text NOT NULL DEFAULT '',
    self_model_ref text NOT NULL DEFAULT '',
    llmwiki_ref text NOT NULL DEFAULT '',
    hunch_log_ref text NOT NULL DEFAULT '',
    system_map_ref text NOT NULL DEFAULT '',
    mistake_ledger_ref text NOT NULL DEFAULT '',
    learning_queue_ref text NOT NULL DEFAULT '',
    research_source_ref text NOT NULL DEFAULT '',
    boot_packet_ref text NOT NULL DEFAULT '',
    evidence_refs jsonb NOT NULL DEFAULT '[]'::jsonb,
    db_refs jsonb NOT NULL DEFAULT '[]'::jsonb,
    proof_status text NOT NULL DEFAULT 'UNKNOWN',
    functionality_explanation text NOT NULL DEFAULT 'Indy_READs metacognition current packet; the current self-understanding surface for the runtime lane.',
    ontology_index jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    refreshed_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT indy_reads_metacognition_current_proof_status_check CHECK (proof_status IN ('PROVEN', 'PARTIAL', 'UNKNOWN', 'CONTRADICTED')),
    CONSTRAINT indy_reads_metacognition_current_evidence_refs_check CHECK (jsonb_typeof(evidence_refs) = 'array'),
    CONSTRAINT indy_reads_metacognition_current_db_refs_check CHECK (jsonb_typeof(db_refs) = 'array'),
    CONSTRAINT indy_reads_metacognition_current_ontology_index_check CHECK (jsonb_typeof(ontology_index) = 'object')
);

CREATE INDEX IF NOT EXISTS indy_reads_self_model_created_idx ON lucidota_indy.indy_reads_self_model (refreshed_at DESC, created_at DESC);
CREATE INDEX IF NOT EXISTS indy_reads_llmwiki_entry_created_idx ON lucidota_indy.indy_reads_llmwiki_entry (refreshed_at DESC, created_at DESC);
CREATE INDEX IF NOT EXISTS indy_reads_hunch_log_created_idx ON lucidota_indy.indy_reads_hunch_log (refreshed_at DESC, created_at DESC);
CREATE INDEX IF NOT EXISTS indy_reads_learning_queue_created_idx ON lucidota_indy.indy_reads_learning_queue (refreshed_at DESC, created_at DESC);
CREATE INDEX IF NOT EXISTS indy_reads_system_map_created_idx ON lucidota_indy.indy_reads_system_map (refreshed_at DESC, created_at DESC);
CREATE INDEX IF NOT EXISTS indy_reads_mistake_ledger_created_idx ON lucidota_indy.indy_reads_mistake_ledger (refreshed_at DESC, created_at DESC);
CREATE INDEX IF NOT EXISTS indy_reads_research_source_created_idx ON lucidota_indy.indy_reads_research_source (refreshed_at DESC, created_at DESC);
CREATE INDEX IF NOT EXISTS indy_reads_metacognition_current_refreshed_idx ON lucidota_indy.indy_reads_metacognition_current_state (refreshed_at DESC);

CREATE OR REPLACE VIEW lucidota_canon.indy_reads_self_model AS
SELECT * FROM lucidota_indy.indy_reads_self_model ORDER BY refreshed_at DESC, created_at DESC;

CREATE OR REPLACE VIEW lucidota_canon.indy_reads_llmwiki_entry AS
SELECT * FROM lucidota_indy.indy_reads_llmwiki_entry ORDER BY refreshed_at DESC, created_at DESC;

CREATE OR REPLACE VIEW lucidota_canon.indy_reads_hunch_log AS
SELECT * FROM lucidota_indy.indy_reads_hunch_log ORDER BY refreshed_at DESC, created_at DESC;

CREATE OR REPLACE VIEW lucidota_canon.indy_reads_learning_queue AS
SELECT * FROM lucidota_indy.indy_reads_learning_queue ORDER BY refreshed_at DESC, created_at DESC;

CREATE OR REPLACE VIEW lucidota_canon.indy_reads_system_map AS
SELECT * FROM lucidota_indy.indy_reads_system_map ORDER BY refreshed_at DESC, created_at DESC;

CREATE OR REPLACE VIEW lucidota_canon.indy_reads_mistake_ledger AS
SELECT * FROM lucidota_indy.indy_reads_mistake_ledger ORDER BY refreshed_at DESC, created_at DESC;

CREATE OR REPLACE VIEW lucidota_canon.indy_reads_research_source AS
SELECT * FROM lucidota_indy.indy_reads_research_source ORDER BY refreshed_at DESC, created_at DESC;

CREATE OR REPLACE VIEW lucidota_canon.indy_reads_metacognition_current AS
WITH self_model AS (
    SELECT * FROM lucidota_canon.indy_reads_self_model ORDER BY refreshed_at DESC, created_at DESC LIMIT 1
),
llmwiki AS (
    SELECT * FROM lucidota_canon.indy_reads_llmwiki_entry ORDER BY refreshed_at DESC, created_at DESC LIMIT 1
),
hunch AS (
    SELECT * FROM lucidota_canon.indy_reads_hunch_log ORDER BY refreshed_at DESC, created_at DESC LIMIT 1
),
queue AS (
    SELECT * FROM lucidota_canon.indy_reads_learning_queue ORDER BY refreshed_at DESC, created_at DESC LIMIT 1
),
system_map AS (
    SELECT * FROM lucidota_canon.indy_reads_system_map ORDER BY refreshed_at DESC, created_at DESC LIMIT 1
),
mistake AS (
    SELECT * FROM lucidota_canon.indy_reads_mistake_ledger ORDER BY refreshed_at DESC, created_at DESC LIMIT 1
),
research AS (
    SELECT * FROM lucidota_canon.indy_reads_research_source ORDER BY refreshed_at DESC, created_at DESC LIMIT 1
),
counts AS (
    SELECT
        COUNT(*) AS self_model_count,
        COUNT(*) FILTER (WHERE proof_status IN ('PROVEN', 'PARTIAL')) AS proven_row_count,
        COUNT(*) FILTER (WHERE proof_status = 'UNKNOWN') AS unknown_row_count
    FROM lucidota_indy.indy_reads_self_model
)
SELECT
    'indy_reads_metacognition_current'::text AS surface_id,
    COALESCE(self_model.actor_id, 'indy_reads_runtime') AS actor_id,
    'indy_reads_runtime'::text AS owner_role,
    COALESCE(self_model.role, 'indy_reads_runtime') AS current_role,
    COALESCE(self_model.summary, '') AS summary,
    COALESCE(self_model.boundaries, '') AS boundaries,
    COALESCE(self_model.voice, '') AS voice,
    COALESCE(self_model.relationship_to_operator, '') AS relationship_to_operator,
    COALESCE(self_model.relationship_to_LUCIDOTA, '') AS relationship_to_LUCIDOTA,
    COALESCE(self_model.investigation_style, '') AS investigation_style,
    COALESCE(self_model.learning_style, '') AS learning_style,
    COALESCE(self_model.preferred_tools, '') AS preferred_tools,
    COALESCE(self_model.evidence_standard, '') AS evidence_standard,
    COALESCE(self_model.receipt_standard, '') AS receipt_standard,
    COALESCE(self_model.mistake_handling, '') AS mistake_handling,
    COALESCE(self_model.curiosity_targets, '') AS curiosity_targets,
    COALESCE(self_model.current_limitations, '') AS current_limitations,
    COALESCE(self_model.next_upgrade, '') AS next_upgrade,
    COALESCE(llmwiki.topic, '') AS current_topic,
    COALESCE(llmwiki.summary, '') AS wiki_summary,
    COALESCE(hunch.hunch, '') AS hunch,
    COALESCE(queue.topic, '') AS next_learning_topic,
    COALESCE(system_map.summary, '') AS system_summary,
    COALESCE(mistake.mistake_summary, '') AS last_mistake,
    COALESCE(research.source_name, '') AS research_source_name,
    COALESCE(self_model.self_model_id::text, '') AS self_model_ref,
    COALESCE(llmwiki.llmwiki_entry_id::text, '') AS llmwiki_ref,
    COALESCE(hunch.hunch_log_id::text, '') AS hunch_log_ref,
    COALESCE(system_map.system_map_id::text, '') AS system_map_ref,
    COALESCE(mistake.mistake_ledger_id::text, '') AS mistake_ledger_ref,
    COALESCE(queue.learning_queue_id::text, '') AS learning_queue_ref,
    COALESCE(research.research_source_id::text, '') AS research_source_ref,
    COALESCE(self_model.evidence_refs, '[]'::jsonb) AS evidence_refs,
    jsonb_build_array(
        'lucidota_control.active_operation_mode',
        'lucidota_canon.manual_current',
        'lucidota_canon.root_orchestrator_current',
        'lucidota_canon.workload_audit_current',
        'lucidota_canon.workload_audit_telemetry_current',
        'lucidota_canon.indy_reads_self_model',
        'lucidota_canon.indy_reads_llmwiki_entry',
        'lucidota_canon.indy_reads_hunch_log',
        'lucidota_canon.indy_reads_learning_queue',
        'lucidota_canon.indy_reads_system_map',
        'lucidota_canon.indy_reads_mistake_ledger',
        'lucidota_canon.indy_reads_research_source'
    ) AS db_refs,
    CASE
        WHEN counts.proven_row_count > 0 THEN 'PROVEN'
        WHEN counts.unknown_row_count > 0 THEN 'UNKNOWN'
        ELSE 'UNKNOWN'
    END AS proof_status,
    'Indy_READs current metacognition packet; a compact runtime self-model plus wiki and learning surface.'::text AS functionality_explanation,
    jsonb_build_object(
        'primitive_refs', ARRAY['state', 'duplex', 'allocation'],
        'claim_type', 'metacognition_current',
        'evidence_type', 'boot_packet_and_receipt',
        'actor_role', 'indy_reads_runtime',
        'subsystem_refs', ARRAY['self_model', 'llmwiki', 'hunch_log', 'system_map', 'mistake_ledger', 'learning_queue', 'research_source'],
        'risk_tier', 'T3',
        'proof_status', CASE WHEN counts.proven_row_count > 0 THEN 'PROVEN' ELSE 'UNKNOWN' END,
        'receipt_refs', ARRAY['indy_reads_exocortex_activation_gate'],
        'next_route', ARRAY['indy_reads_self_model', 'indy_reads_llmwiki_entry', 'indy_reads_hunch_log', 'indy_reads_learning_queue', 'indy_reads_system_map', 'indy_reads_mistake_ledger', 'indy_reads_research_source']
    ) AS ontology_index,
    now() AS created_at,
    now() AS refreshed_at,
    counts.self_model_count,
    counts.proven_row_count,
    counts.unknown_row_count,
    COALESCE(self_model.goals_refs, '[]'::jsonb) AS goals_refs,
    COALESCE(queue.status, 'queued') AS learning_queue_status,
    COALESCE(queue.priority, 50) AS learning_queue_priority,
    COALESCE(queue.next_route, '') AS learning_queue_next_route,
    COALESCE(self_model.proof_status, 'UNKNOWN') AS self_model_proof_status,
    COALESCE(llmwiki.proof_status, 'UNKNOWN') AS llmwiki_proof_status,
    COALESCE(hunch.proof_status, 'UNKNOWN') AS hunch_log_proof_status,
    COALESCE(system_map.proof_status, 'UNKNOWN') AS system_map_proof_status,
    COALESCE(mistake.proof_status, 'UNKNOWN') AS mistake_ledger_proof_status,
    COALESCE(research.proof_status, 'UNKNOWN') AS research_source_proof_status,
    self_model.created_at AS self_model_created_at,
    llmwiki.created_at AS llmwiki_created_at,
    hunch.created_at AS hunch_created_at,
    queue.created_at AS learning_queue_created_at,
    system_map.created_at AS system_map_created_at,
    mistake.created_at AS mistake_created_at,
    research.created_at AS research_created_at
FROM counts
LEFT JOIN self_model ON TRUE
LEFT JOIN llmwiki ON TRUE
LEFT JOIN hunch ON TRUE
LEFT JOIN queue ON TRUE
LEFT JOIN system_map ON TRUE
LEFT JOIN mistake ON TRUE
LEFT JOIN research ON TRUE;

INSERT INTO lucidota_control.schema_owner_manifest (
    surface_id, canonical_owner, packet_class, surface_kind, approval_required, notes, detail
) VALUES
    ('indy_reads_self_model', 'lucidota_indy', 'typed_packet', 'view', true, 'Indy_READs runtime self-model surface.', '{"source":"indy_reads_exocortex"}'::jsonb),
    ('indy_reads_llmwiki_entry', 'lucidota_indy', 'typed_packet', 'view', true, 'Indy_READs LLMWIKI entry surface.', '{"source":"indy_reads_exocortex"}'::jsonb),
    ('indy_reads_hunch_log', 'lucidota_indy', 'typed_packet', 'view', true, 'Indy_READs hunch log surface.', '{"source":"indy_reads_exocortex"}'::jsonb),
    ('indy_reads_learning_queue', 'lucidota_indy', 'typed_packet', 'view', true, 'Indy_READs learning queue surface.', '{"source":"indy_reads_exocortex"}'::jsonb),
    ('indy_reads_system_map', 'lucidota_indy', 'typed_packet', 'view', true, 'Indy_READs system map surface.', '{"source":"indy_reads_exocortex"}'::jsonb),
    ('indy_reads_mistake_ledger', 'lucidota_indy', 'typed_packet', 'view', true, 'Indy_READs mistake ledger surface.', '{"source":"indy_reads_exocortex"}'::jsonb),
    ('indy_reads_research_source', 'lucidota_indy', 'typed_packet', 'view', true, 'Indy_READs research source inventory surface.', '{"source":"indy_reads_exocortex"}'::jsonb),
    ('indy_reads_metacognition_current', 'lucidota_canon', 'typed_packet', 'view', true, 'Indy_READs current metacognition summary packet.', '{"source":"indy_reads_exocortex"}'::jsonb)
ON CONFLICT (surface_id) DO UPDATE SET
    canonical_owner = EXCLUDED.canonical_owner,
    packet_class = EXCLUDED.packet_class,
    surface_kind = EXCLUDED.surface_kind,
    approval_required = EXCLUDED.approval_required,
    active = true,
    notes = EXCLUDED.notes,
    detail = EXCLUDED.detail,
    updated_at = now();

INSERT INTO lucidota_canon.api_route_catalog (
    route_id, method, path_pattern, description, target, sample_request, sample_response, status
) VALUES
    ('indy_reads_self_model', 'GET', '/indy_reads_self_model', 'Indy_READs runtime self-model packet.', 'lucidota_canon.indy_reads_self_model', '{"limit":"1"}', '{"actor_id":"indy_reads_runtime"}', 'implemented'),
    ('indy_reads_llmwiki_entry', 'GET', '/indy_reads_llmwiki_entry', 'Indy_READs LLMWIKI entry packet.', 'lucidota_canon.indy_reads_llmwiki_entry', '{"limit":"1"}', '{"author":"indy_reads"}', 'implemented'),
    ('indy_reads_hunch_log', 'GET', '/indy_reads_hunch_log', 'Indy_READs hunch log packet.', 'lucidota_canon.indy_reads_hunch_log', '{"limit":"1"}', '{"actor_id":"indy_reads_runtime"}', 'implemented'),
    ('indy_reads_learning_queue', 'GET', '/indy_reads_learning_queue', 'Indy_READs learning queue packet.', 'lucidota_canon.indy_reads_learning_queue', '{"limit":"1"}', '{"actor_id":"indy_reads_runtime"}', 'implemented'),
    ('indy_reads_system_map', 'GET', '/indy_reads_system_map', 'Indy_READs system map packet.', 'lucidota_canon.indy_reads_system_map', '{"limit":"1"}', '{"actor_id":"indy_reads_runtime"}', 'implemented'),
    ('indy_reads_mistake_ledger', 'GET', '/indy_reads_mistake_ledger', 'Indy_READs mistake ledger packet.', 'lucidota_canon.indy_reads_mistake_ledger', '{"limit":"1"}', '{"actor_id":"indy_reads_runtime"}', 'implemented'),
    ('indy_reads_research_source', 'GET', '/indy_reads_research_source', 'Indy_READs research source inventory packet.', 'lucidota_canon.indy_reads_research_source', '{"limit":"1"}', '{"actor_id":"indy_reads_runtime"}', 'implemented'),
    ('indy_reads_metacognition_current', 'GET', '/indy_reads_metacognition_current', 'Indy_READs metacognition current summary packet.', 'lucidota_canon.indy_reads_metacognition_current', '{"limit":"1"}', '{"owner_role":"indy_reads_runtime"}', 'implemented')
ON CONFLICT (route_id) DO UPDATE SET
    method = EXCLUDED.method,
    path_pattern = EXCLUDED.path_pattern,
    description = EXCLUDED.description,
    target = EXCLUDED.target,
    sample_request = EXCLUDED.sample_request,
    sample_response = EXCLUDED.sample_response,
    status = EXCLUDED.status,
    updated_at = now();

GRANT USAGE ON SCHEMA lucidota_indy TO mfspx, lucidota_postgrest_anon, ironclaw;

GRANT SELECT ON lucidota_indy.indy_reads_self_model,
    lucidota_indy.indy_reads_llmwiki_entry,
    lucidota_indy.indy_reads_hunch_log,
    lucidota_indy.indy_reads_learning_queue,
    lucidota_indy.indy_reads_system_map,
    lucidota_indy.indy_reads_mistake_ledger,
    lucidota_indy.indy_reads_research_source,
    lucidota_indy.indy_reads_metacognition_current_state
TO mfspx, lucidota_postgrest_anon, ironclaw;

GRANT INSERT, UPDATE ON lucidota_indy.indy_reads_self_model,
    lucidota_indy.indy_reads_llmwiki_entry,
    lucidota_indy.indy_reads_hunch_log,
    lucidota_indy.indy_reads_learning_queue,
    lucidota_indy.indy_reads_system_map,
    lucidota_indy.indy_reads_mistake_ledger,
    lucidota_indy.indy_reads_research_source,
    lucidota_indy.indy_reads_metacognition_current_state
TO ironclaw;

GRANT SELECT ON lucidota_canon.indy_reads_self_model,
    lucidota_canon.indy_reads_llmwiki_entry,
    lucidota_canon.indy_reads_hunch_log,
    lucidota_canon.indy_reads_learning_queue,
    lucidota_canon.indy_reads_system_map,
    lucidota_canon.indy_reads_mistake_ledger,
    lucidota_canon.indy_reads_research_source,
    lucidota_canon.indy_reads_metacognition_current
TO mfspx, lucidota_postgrest_anon, ironclaw;

NOTIFY pgrst, 'reload schema';

COMMIT;
