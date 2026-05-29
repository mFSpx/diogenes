-- LUCIDOTA Round 74: current Chrono projection refresh target.
-- Derived projection only. Temporal claims remain archive.
CREATE TABLE IF NOT EXISTS lucidota_korpus.current_chrono_timeline_projection (
    file_uuid uuid PRIMARY KEY,
    selected_claim_uuid uuid NOT NULL REFERENCES lucidota_korpus.temporal_claim(claim_uuid),
    resolved_timestamp timestamptz NOT NULL,
    evidence_source text NOT NULL,
    trust_weight numeric(3,2) NOT NULL,
    source_sha256 char(64),
    projection_refreshed_at timestamptz NOT NULL DEFAULT now(),
    ranking_method text NOT NULL DEFAULT 'trust_weight_desc_timestamp_asc_created_desc',
    detail jsonb NOT NULL DEFAULT '{}'::jsonb
);
CREATE INDEX IF NOT EXISTS idx_current_chrono_projection_source ON lucidota_korpus.current_chrono_timeline_projection(evidence_source);
