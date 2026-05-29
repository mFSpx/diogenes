-- FILE: scripts/chrono_phase_c_validation.sql
-- PURPOSE: Read-only validation for Chrono-Ledger Phase C cursor/dead-letter/service readiness.

\echo '--- LUCIDOTA CHRONO PHASE C VALIDATION ---'

SELECT
    to_regclass('lucidota_korpus.chrono_replay_cursor') AS replay_cursor_reg,
    to_regclass('lucidota_korpus.chrono_dead_letter') AS dead_letter_reg,
    to_regclass('lucidota_korpus.temporal_claim') AS temporal_claim_reg,
    to_regclass('lucidota_korpus.resolved_chrono_timeline') AS timeline_view_reg;

SELECT cursor_name, last_file_uuid, last_file_first_seen_at, processed_count, last_replay_started_at, last_replay_finished_at, updated_at
FROM lucidota_korpus.chrono_replay_cursor
ORDER BY cursor_name;

SELECT resolved, COUNT(*) AS dead_letters
FROM lucidota_korpus.chrono_dead_letter
GROUP BY resolved
ORDER BY resolved;

SELECT evidence_source, trust_weight, COUNT(*) AS claims
FROM lucidota_korpus.temporal_claim
GROUP BY evidence_source, trust_weight
ORDER BY trust_weight DESC, evidence_source;

WITH violations AS (
  SELECT r.file_uuid
  FROM lucidota_korpus.resolved_chrono_timeline r
  JOIN lucidota_korpus.temporal_claim tc ON tc.file_uuid = r.file_uuid
  WHERE tc.trust_weight > r.confidence_score
     OR (tc.trust_weight = r.confidence_score AND tc.candidate_timestamp < r.resolved_timestamp)
)
SELECT COUNT(*) AS ranking_violations FROM violations;

\echo '--- END LUCIDOTA CHRONO PHASE C VALIDATION ---'
