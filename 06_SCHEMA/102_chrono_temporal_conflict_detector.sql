-- LUCIDOTA Round 79: temporal conflict detector. Additive findings only.
CREATE TABLE IF NOT EXISTS lucidota_korpus.temporal_conflict_finding (
    conflict_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    file_uuid uuid NOT NULL REFERENCES lucidota_korpus.file_object(file_uuid),
    claim_count integer NOT NULL,
    distinct_timestamp_count integer NOT NULL,
    source_count integer NOT NULL,
    highest_trust_weight numeric(3,2) NOT NULL,
    lowest_trust_weight numeric(3,2) NOT NULL,
    evidence_sources jsonb NOT NULL DEFAULT '[]'::jsonb,
    selected_claim_uuid uuid REFERENCES lucidota_korpus.temporal_claim(claim_uuid),
    review_state text NOT NULL DEFAULT 'candidate' CHECK (review_state IN ('candidate','reviewed','ignored','resolved')),
    idempotency_key text NOT NULL UNIQUE,
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    detected_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_temporal_conflict_file ON lucidota_korpus.temporal_conflict_finding(file_uuid);
