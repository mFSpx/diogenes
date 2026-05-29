-- FILE: 06_SCHEMA/025_chrono_ledger_core.sql
-- COMPONENT: LUCIDOTA Headless Chrono-Ledger Core
-- COMPLIANCE: Idempotent migration script; no destructive actions; no re-ingestion.
-- DEPENDENCIES: 06_SCHEMA/019_korpus_krampii.sql
-- PURPOSE: Append-only temporal evidence ledger protecting chronology from filesystem mtime drift.

START TRANSACTION;

-- gen_random_uuid() is provided by pgcrypto. uuid-ossp is left available for clusters/tools
-- that still use uuid_generate_* helpers elsewhere in the LUCIDOTA estate.
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 1. CHRONO LEDGER BASE TABLE
CREATE TABLE IF NOT EXISTS lucidota_korpus.temporal_claim (
    claim_uuid UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Generic nullable artifact anchor for cross-schema future linking.
    -- Primary storage anchor is file_uuid -> lucidota_korpus.file_object(file_uuid).
    artifact_uuid UUID,

    file_uuid UUID CONSTRAINT fk_chrono_file_object
        REFERENCES lucidota_korpus.file_object(file_uuid)
        ON DELETE CASCADE,

    candidate_timestamp TIMESTAMPTZ NOT NULL,
    evidence_source TEXT NOT NULL,

    trust_weight NUMERIC(3, 2) NOT NULL CONSTRAINT chk_trust_weight_bounds
        CHECK (trust_weight >= 0.00 AND trust_weight <= 1.00),

    raw_evidence TEXT,
    extractor TEXT NOT NULL DEFAULT '',
    extractor_version TEXT NOT NULL DEFAULT '',
    source_path TEXT NOT NULL DEFAULT '',

    source_sha256 CHAR(64) CONSTRAINT chk_temporal_claim_sha256_hex
        CHECK (source_sha256 ~ '^[0-9a-f]{64}$'),

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    detail JSONB NOT NULL DEFAULT '{}'::jsonb
);

COMMENT ON TABLE lucidota_korpus.temporal_claim IS
    'Append-only temporal evidence graph: multiple weighted timestamp claims per file_object; never rewrites file_object chronology.';

COMMENT ON COLUMN lucidota_korpus.temporal_claim.trust_weight IS
    'Temporal evidence confidence: 1.00 strict filename/log/embedded structured time; 0.80 content time; 0.60 git containment time; 0.10 filesystem time.';

-- 2. PERFORMANCE INDEXES
CREATE INDEX IF NOT EXISTS idx_temporal_claim_file_uuid
    ON lucidota_korpus.temporal_claim(file_uuid);

CREATE INDEX IF NOT EXISTS idx_temporal_claim_candidate_timestamp
    ON lucidota_korpus.temporal_claim(candidate_timestamp);

CREATE INDEX IF NOT EXISTS idx_temporal_claim_evidence_source
    ON lucidota_korpus.temporal_claim(evidence_source);

CREATE INDEX IF NOT EXISTS idx_temporal_claim_source_sha256
    ON lucidota_korpus.temporal_claim(source_sha256)
    WHERE source_sha256 IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_temporal_claim_file_weight_time
    ON lucidota_korpus.temporal_claim(file_uuid, trust_weight DESC, candidate_timestamp ASC, created_at DESC);

-- Existing occurrence table support for DB-only filesystem-time fallback extraction.
CREATE INDEX IF NOT EXISTS idx_file_occurrence_file_uuid_mtime
    ON lucidota_korpus.file_occurrence(file_uuid, mtime)
    WHERE mtime IS NOT NULL;

-- 3. RESOLUTION VIEW (DETERMINISTIC WEIGHTED CONSENSUS)
-- Resolution is dynamic. Adding better claims later shifts resolved time without mutating storage rows.
CREATE OR REPLACE VIEW lucidota_korpus.resolved_chrono_timeline AS
WITH scored_claims AS (
    SELECT
        file_uuid,
        candidate_timestamp,
        evidence_source,
        trust_weight,
        source_path,
        source_sha256,
        raw_evidence,
        extractor,
        extractor_version,
        created_at,
        ROW_NUMBER() OVER (
            PARTITION BY file_uuid
            ORDER BY
                trust_weight DESC,
                candidate_timestamp ASC,
                created_at DESC,
                claim_uuid ASC
        ) AS rank_priority,
        COUNT(*) OVER (PARTITION BY file_uuid) AS claim_count
    FROM lucidota_korpus.temporal_claim
    WHERE file_uuid IS NOT NULL
)
SELECT
    f.file_uuid,
    f.sha256 AS artifact_sha256,
    f.locked_relative_path AS current_path,
    f.first_seen_path,
    sc.candidate_timestamp AS resolved_timestamp,
    sc.evidence_source AS dominant_evidence_source,
    sc.trust_weight AS confidence_score,
    sc.source_path AS provenance_path,
    sc.raw_evidence AS dominant_raw_evidence,
    sc.extractor AS dominant_extractor,
    sc.extractor_version AS dominant_extractor_version,
    sc.claim_count
FROM lucidota_korpus.file_object f
JOIN scored_claims sc ON f.file_uuid = sc.file_uuid
WHERE sc.rank_priority = 1;

COMMENT ON VIEW lucidota_korpus.resolved_chrono_timeline IS
    'Dynamic Chrono-Ledger resolution view: one dominant weighted timestamp per file_uuid, no mutation of original file_object rows.';

COMMIT;
