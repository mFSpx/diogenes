-- RFC-CHRONO-001: Temporal Claim Ledger -> Lane-Split Projection -> Graph Promotion Gate
-- Derived tables only. temporal_claim remains append-only archive.
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_korpus;
CREATE SCHEMA IF NOT EXISTS lucidota_go;

CREATE TABLE IF NOT EXISTS lucidota_korpus.chrono_claim_normalized (
    claim_uuid uuid PRIMARY KEY REFERENCES lucidota_korpus.temporal_claim(claim_uuid),
    file_uuid uuid,
    artifact_sha256 char(64),
    source_path text,
    candidate_timestamp timestamptz,
    evidence_source text NOT NULL,
    extractor text,
    extractor_version text,
    trust_weight numeric,
    epistemic_flag text NOT NULL CHECK (epistemic_flag IN ('FACT','PROBABLE','POSSIBLE','BULLSHIT','SURE_MAYBE')),
    invalid boolean NOT NULL DEFAULT false,
    chrono_lane text NOT NULL CHECK (chrono_lane IN ('LANE_CASE_EVENT','LANE_ARTIFACT_CUSTODY','LANE_SYSTEM_RUNTIME')),
    source_family text NOT NULL,
    path_family text NOT NULL,
    is_runtime_artifact boolean NOT NULL DEFAULT false,
    is_generated_artifact boolean NOT NULL DEFAULT false,
    is_batch_candidate boolean NOT NULL DEFAULT false,
    weak_pair_group_id text,
    graph_event_type text NOT NULL,
    promotion_block_reason text[] NOT NULL DEFAULT ARRAY[]::text[],
    classification_detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    normalized_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_chrono_claim_normalized_lane ON lucidota_korpus.chrono_claim_normalized(chrono_lane);
CREATE INDEX IF NOT EXISTS idx_chrono_claim_normalized_source_family ON lucidota_korpus.chrono_claim_normalized(source_family);
CREATE INDEX IF NOT EXISTS idx_chrono_claim_normalized_batch ON lucidota_korpus.chrono_claim_normalized(is_batch_candidate, candidate_timestamp);
CREATE INDEX IF NOT EXISTS idx_chrono_claim_normalized_file ON lucidota_korpus.chrono_claim_normalized(file_uuid);
CREATE INDEX IF NOT EXISTS idx_chrono_claim_normalized_sha ON lucidota_korpus.chrono_claim_normalized(artifact_sha256);

CREATE TABLE IF NOT EXISTS lucidota_korpus.chrono_batch_cluster (
    batch_cluster_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp timestamptz NOT NULL,
    lane text NOT NULL CHECK (lane IN ('LANE_ARTIFACT_CUSTODY','LANE_SYSTEM_RUNTIME','LANE_CASE_EVENT')),
    evidence_sources text[] NOT NULL DEFAULT ARRAY[]::text[],
    file_count integer NOT NULL CHECK (file_count >= 0),
    path_prefixes text[] NOT NULL DEFAULT ARRAY[]::text[],
    sample_paths text[] NOT NULL DEFAULT ARRAY[]::text[],
    classification text NOT NULL,
    graph_allowed_as text[] NOT NULL DEFAULT ARRAY[]::text[],
    case_event_blocked boolean NOT NULL DEFAULT false,
    detected_at timestamptz NOT NULL DEFAULT now(),
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE(timestamp, lane, classification)
);
CREATE INDEX IF NOT EXISTS idx_chrono_batch_cluster_timestamp ON lucidota_korpus.chrono_batch_cluster(timestamp);
CREATE INDEX IF NOT EXISTS idx_chrono_batch_cluster_blocked ON lucidota_korpus.chrono_batch_cluster(case_event_blocked);

CREATE TABLE IF NOT EXISTS lucidota_korpus.chrono_file_projection (
    file_uuid uuid PRIMARY KEY,
    selected_claim_uuid uuid NOT NULL REFERENCES lucidota_korpus.temporal_claim(claim_uuid),
    selected_timestamp timestamptz NOT NULL,
    selected_lane text NOT NULL CHECK (selected_lane IN ('LANE_CASE_EVENT','LANE_ARTIFACT_CUSTODY','LANE_SYSTEM_RUNTIME')),
    selected_evidence_source text NOT NULL,
    selected_confidence numeric,
    competing_claim_count integer NOT NULL DEFAULT 0,
    weak_claim_count integer NOT NULL DEFAULT 0,
    strong_claim_count integer NOT NULL DEFAULT 0,
    has_conflict boolean NOT NULL DEFAULT false,
    has_batch_cluster boolean NOT NULL DEFAULT false,
    projection_reason text NOT NULL,
    projection_refreshed_at timestamptz NOT NULL DEFAULT now(),
    detail jsonb NOT NULL DEFAULT '{}'::jsonb
);
CREATE INDEX IF NOT EXISTS idx_chrono_file_projection_lane ON lucidota_korpus.chrono_file_projection(selected_lane);
CREATE INDEX IF NOT EXISTS idx_chrono_file_projection_batch ON lucidota_korpus.chrono_file_projection(has_batch_cluster);

CREATE TABLE IF NOT EXISTS lucidota_go.graph_promotion_candidate (
    promotion_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    source_claim_uuid uuid NOT NULL REFERENCES lucidota_korpus.temporal_claim(claim_uuid),
    file_uuid uuid,
    graph_event_type text NOT NULL,
    chrono_lane text NOT NULL CHECK (chrono_lane IN ('LANE_CASE_EVENT','LANE_ARTIFACT_CUSTODY','LANE_SYSTEM_RUNTIME')),
    timestamp timestamptz NOT NULL,
    confidence numeric,
    blocked boolean NOT NULL DEFAULT true,
    block_reason text[] NOT NULL DEFAULT ARRAY[]::text[],
    promotion_reason text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE(source_claim_uuid)
);
CREATE INDEX IF NOT EXISTS idx_graph_promotion_candidate_blocked ON lucidota_go.graph_promotion_candidate(blocked);
CREATE INDEX IF NOT EXISTS idx_graph_promotion_candidate_lane ON lucidota_go.graph_promotion_candidate(chrono_lane);
CREATE INDEX IF NOT EXISTS idx_graph_promotion_candidate_file ON lucidota_go.graph_promotion_candidate(file_uuid);

CREATE TABLE IF NOT EXISTS lucidota_korpus.chrono_lane_refresh_run (
    run_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    started_at timestamptz NOT NULL DEFAULT now(),
    finished_at timestamptz,
    claims_seen integer NOT NULL DEFAULT 0,
    claims_normalized integer NOT NULL DEFAULT 0,
    file_projection_rows integer NOT NULL DEFAULT 0,
    batch_clusters_upserted integer NOT NULL DEFAULT 0,
    promotion_candidates_upserted integer NOT NULL DEFAULT 0,
    status text NOT NULL DEFAULT 'running',
    report jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE OR REPLACE VIEW lucidota_korpus.chrono_case_event_timeline AS
SELECT p.*, n.graph_event_type, n.source_family, n.path_family
FROM lucidota_korpus.chrono_file_projection p
JOIN lucidota_korpus.chrono_claim_normalized n ON n.claim_uuid = p.selected_claim_uuid
WHERE p.selected_lane='LANE_CASE_EVENT';

CREATE OR REPLACE VIEW lucidota_korpus.chrono_artifact_custody_timeline AS
SELECT p.*, n.graph_event_type, n.source_family, n.path_family
FROM lucidota_korpus.chrono_file_projection p
JOIN lucidota_korpus.chrono_claim_normalized n ON n.claim_uuid = p.selected_claim_uuid
WHERE p.selected_lane='LANE_ARTIFACT_CUSTODY';

CREATE OR REPLACE VIEW lucidota_korpus.chrono_system_runtime_timeline AS
SELECT p.*, n.graph_event_type, n.source_family, n.path_family
FROM lucidota_korpus.chrono_file_projection p
JOIN lucidota_korpus.chrono_claim_normalized n ON n.claim_uuid = p.selected_claim_uuid
WHERE p.selected_lane='LANE_SYSTEM_RUNTIME';

COMMENT ON TABLE lucidota_korpus.temporal_claim IS 'CLAIM_LEDGER archive: append-only temporal evidence ledger. Projection/gating must not delete or overwrite claims.';
COMMENT ON TABLE lucidota_korpus.chrono_claim_normalized IS 'RFC-CHRONO-001 derived normalization: every temporal_claim gets exactly one chrono_lane plus source/path family and promotion block reasons.';
COMMENT ON TABLE lucidota_korpus.chrono_file_projection IS 'FILE_PROJECTION: one selected claim per file. Projection, not archive; not proof of human/world event truth.';
COMMENT ON TABLE lucidota_korpus.chrono_batch_cluster IS 'Weak timestamp batch detector. Batch mtime clusters may graph only as custody/runtime batch events, never case-event truth.';
COMMENT ON TABLE lucidota_go.graph_promotion_candidate IS 'Promotion gate candidate rows. Blocked candidates must not graph-write; all graph writes must cite source_claim_uuid and file_uuid.';

CREATE OR REPLACE FUNCTION lucidota_go.chrono_graph_promotion_candidate_allowed(p_promotion_uuid uuid)
RETURNS boolean
LANGUAGE sql
STABLE
AS $$
    SELECT EXISTS (
        SELECT 1
        FROM lucidota_go.graph_promotion_candidate c
        WHERE c.promotion_uuid = p_promotion_uuid
          AND c.blocked = false
          AND c.source_claim_uuid IS NOT NULL
          AND c.timestamp IS NOT NULL
          AND c.chrono_lane IN ('LANE_CASE_EVENT','LANE_ARTIFACT_CUSTODY','LANE_SYSTEM_RUNTIME')
          AND c.graph_event_type <> ''
          AND c.promotion_reason <> ''
          AND (c.file_uuid IS NOT NULL OR c.chrono_lane='LANE_SYSTEM_RUNTIME')
    );
$$;

CREATE OR REPLACE FUNCTION lucidota_go.chrono_graph_promotion_candidate_block_reasons(p_promotion_uuid uuid)
RETURNS text[]
LANGUAGE sql
STABLE
AS $$
    SELECT CASE
        WHEN c.promotion_uuid IS NULL THEN ARRAY['promotion_candidate_missing']::text[]
        WHEN c.blocked THEN c.block_reason
        ELSE ARRAY[]::text[]
    END
    FROM (SELECT p_promotion_uuid AS promotion_uuid) q
    LEFT JOIN lucidota_go.graph_promotion_candidate c ON c.promotion_uuid=q.promotion_uuid;
$$;

CREATE OR REPLACE VIEW lucidota_go.graph_promotion_candidate_allowed AS
SELECT *
FROM lucidota_go.graph_promotion_candidate
WHERE lucidota_go.chrono_graph_promotion_candidate_allowed(promotion_uuid);

COMMENT ON FUNCTION lucidota_go.chrono_graph_promotion_candidate_allowed(uuid) IS 'RFC-CHRONO-001 graph writer guard: returns true only for lane-qualified, unblocked candidates with required source claim/file/timestamp/reason fields.';
