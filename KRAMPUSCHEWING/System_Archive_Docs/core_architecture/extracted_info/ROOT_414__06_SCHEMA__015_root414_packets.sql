-- ROOT-414 packet/feedback schema.
-- Purpose: make v0.50 route packets graph- and workflow-addressable.

CREATE SCHEMA IF NOT EXISTS lucidota_root414;

CREATE TABLE IF NOT EXISTS lucidota_root414.primitive (
    ordinal integer PRIMARY KEY CHECK (ordinal BETWEEN 1 AND 414),
    symbol text UNIQUE NOT NULL,
    block_name text NOT NULL DEFAULT '',
    root_version text NOT NULL DEFAULT 'root414-global-v1',
    active boolean NOT NULL DEFAULT true,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS lucidota_root414.primitive_cry_doc (
    doc_id text PRIMARY KEY,
    relative_path text NOT NULL UNIQUE,
    title text NOT NULL DEFAULT '',
    sha256 text NOT NULL,
    size_bytes bigint NOT NULL CHECK (size_bytes >= 0),
    line_count integer NOT NULL DEFAULT 0,
    parser_name text NOT NULL DEFAULT 'root414_machine_clean_parser_v0.50',
    flags jsonb NOT NULL DEFAULT '[]'::jsonb,
    imported_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS lucidota_root414.route_packet (
    packet_id text PRIMARY KEY,
    doc_id text REFERENCES lucidota_root414.primitive_cry_doc(doc_id) ON DELETE SET NULL,
    source_id text NOT NULL DEFAULT '',
    parser_name text NOT NULL DEFAULT 'root414_machine_clean_parser_v0.50',
    raw_text_anchor text NOT NULL DEFAULT '',
    claim text NOT NULL DEFAULT '',
    route_anchor text NOT NULL DEFAULT '',
    route_operator text NOT NULL DEFAULT '',
    route_resolution text NOT NULL DEFAULT '',
    ternary_state jsonb NOT NULL DEFAULT '{}'::jsonb,
    claim_lifecycle text NOT NULL DEFAULT 'CLAIM_UNVERIFIED',
    confidence_bps integer NOT NULL DEFAULT 0 CHECK (confidence_bps IN (0,2,4,6,10,50,69,150)),
    falsifier text NOT NULL DEFAULT '',
    hitl_status text NOT NULL DEFAULT 'pending' CHECK (hitl_status IN ('pending','approved','rejected','needs_repair','comment')),
    packet_json jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS route_packet_doc_idx
    ON lucidota_root414.route_packet(doc_id);

CREATE TABLE IF NOT EXISTS lucidota_root414.route_packet_primitive (
    packet_id text REFERENCES lucidota_root414.route_packet(packet_id) ON DELETE CASCADE,
    ordinal integer REFERENCES lucidota_root414.primitive(ordinal),
    symbol text NOT NULL,
    role text NOT NULL CHECK (role IN ('anchor','vector','local_gate','rejected','mention','unknown')),
    mention_count integer NOT NULL DEFAULT 1 CHECK (mention_count >= 0),
    PRIMARY KEY(packet_id, ordinal, role)
);

CREATE TABLE IF NOT EXISTS lucidota_root414.packet_quality_run (
    quality_run_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    parser_name text NOT NULL DEFAULT 'root414_machine_clean_parser_v0.50',
    source_path text NOT NULL DEFAULT '',
    packet_count integer NOT NULL DEFAULT 0,
    mean_score numeric NOT NULL DEFAULT 0,
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS lucidota_root414.packet_quality_feedback (
    feedback_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    quality_run_id uuid REFERENCES lucidota_root414.packet_quality_run(quality_run_id) ON DELETE CASCADE,
    packet_id text NOT NULL,
    severity text NOT NULL CHECK (severity IN ('info','warn','error')),
    code text NOT NULL,
    message text NOT NULL,
    hint text NOT NULL DEFAULT '',
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS packet_quality_feedback_packet_idx
    ON lucidota_root414.packet_quality_feedback(packet_id, severity);

CREATE TABLE IF NOT EXISTS lucidota_root414.review_card (
    card_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    packet_id text NOT NULL,
    front text NOT NULL,
    back text NOT NULL,
    primitive_symbol text NOT NULL DEFAULT '',
    bps integer NOT NULL DEFAULT 10 CHECK (bps IN (0,2,4,6,10,50,69,150)),
    status text NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','approved','rejected','retired')),
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS lucidota_root414.hitl_review_event (
    review_event_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    packet_id text NOT NULL,
    reviewer text NOT NULL DEFAULT 'Northern.Strike',
    decision text NOT NULL CHECK (decision IN ('approved','rejected','needs_repair','comment')),
    note text NOT NULL DEFAULT '',
    created_at timestamptz NOT NULL DEFAULT now()
);
