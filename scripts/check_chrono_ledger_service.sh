#!/usr/bin/env bash
set -euo pipefail
DB="${DATABASE_URL:-postgresql:///lucidota_storage}"
SERVICE="lucidota-chrono-ledger.service"

echo "--- user-systemd service ---"
systemctl --user --no-pager status "$SERVICE" || true

echo "--- recent service logs ---"
journalctl --user -u "$SERVICE" --no-pager -n "${LOG_LINES:-40}" || true

echo "--- chrono db health ---"
psql -X -v ON_ERROR_STOP=1 "$DB" <<'SQL'
SELECT cursor_name, last_file_uuid, last_file_first_seen_at, processed_count, last_replay_started_at, last_replay_finished_at, updated_at
FROM lucidota_korpus.chrono_replay_cursor
ORDER BY cursor_name;
SELECT resolved, count(*) AS dead_letters
FROM lucidota_korpus.chrono_dead_letter
GROUP BY resolved
ORDER BY resolved;
WITH violations AS (
  SELECT r.file_uuid
  FROM lucidota_korpus.resolved_chrono_timeline r
  JOIN lucidota_korpus.temporal_claim tc ON tc.file_uuid = r.file_uuid
  WHERE tc.trust_weight > r.confidence_score
     OR (tc.trust_weight = r.confidence_score AND tc.candidate_timestamp < r.resolved_timestamp)
)
SELECT count(*) AS ranking_violations FROM violations;
SQL
