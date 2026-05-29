-- FILE: 06_SCHEMA/071_chrono_conservation_verifier.sql
-- PURPOSE: claim-linked Chrono projection for conservation verifier CLI.
-- COMPLIANCE: creates derived views only; temporal_claim remains the archive.

BEGIN;

CREATE SCHEMA IF NOT EXISTS lucidota_korpus;

CREATE OR REPLACE VIEW lucidota_korpus.resolved_chrono_timeline_with_claim AS
WITH per_file AS (
  SELECT
    file_uuid,
    COUNT(*) AS claim_count,
    COUNT(DISTINCT candidate_timestamp) AS distinct_timestamp_count,
    COUNT(DISTINCT evidence_source) AS distinct_source_count
  FROM lucidota_korpus.temporal_claim
  WHERE file_uuid IS NOT NULL
  GROUP BY file_uuid
), scored_claims AS (
  SELECT
    tc.claim_uuid,
    tc.file_uuid,
    tc.candidate_timestamp,
    tc.evidence_source,
    tc.trust_weight,
    tc.source_path,
    tc.source_sha256,
    tc.raw_evidence,
    tc.extractor,
    tc.extractor_version,
    tc.created_at,
    ROW_NUMBER() OVER (
      PARTITION BY tc.file_uuid
      ORDER BY tc.trust_weight DESC, tc.candidate_timestamp ASC, tc.created_at DESC, tc.claim_uuid ASC
    ) AS rank_priority,
    pf.claim_count,
    pf.distinct_timestamp_count,
    pf.distinct_source_count
  FROM lucidota_korpus.temporal_claim tc
  JOIN per_file pf ON pf.file_uuid = tc.file_uuid
  WHERE tc.file_uuid IS NOT NULL
)
SELECT
  f.file_uuid,
  f.sha256 AS artifact_sha256,
  f.locked_relative_path AS current_path,
  f.first_seen_path,
  sc.claim_uuid AS selected_claim_uuid,
  sc.candidate_timestamp AS resolved_timestamp,
  sc.evidence_source AS dominant_evidence_source,
  sc.trust_weight AS confidence_score,
  sc.source_path AS provenance_path,
  sc.raw_evidence AS dominant_raw_evidence,
  sc.extractor AS dominant_extractor,
  sc.extractor_version AS dominant_extractor_version,
  sc.claim_count,
  (sc.claim_count > 1 AND (sc.distinct_timestamp_count > 1 OR sc.distinct_source_count > 1)) AS has_conflict
FROM lucidota_korpus.file_object f
JOIN scored_claims sc ON f.file_uuid = sc.file_uuid
WHERE sc.rank_priority = 1;

COMMENT ON VIEW lucidota_korpus.resolved_chrono_timeline_with_claim IS
  'Derived current-best Chrono projection with selected_claim_uuid. This is not the archive; temporal_claim is the archive.';

COMMIT;
