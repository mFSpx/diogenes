-- Lift the canon current packet into an operator-readable truth spine.

BEGIN;

CREATE OR REPLACE VIEW lucidota_canon.canon_current AS
WITH goal_row AS (
    SELECT to_jsonb(g) AS current_goal
    FROM lucidota_canon.active_goal g
    ORDER BY updated_at DESC
    LIMIT 1
)
SELECT
    node_id,
    parent_id,
    node_sort_key,
    manual_id,
    title,
    node_kind,
    ontology_tags,
    status,
    version,
    hash_current,
    updated_at,
    goal_row.current_goal AS goal,
    jsonb_build_object(
        'statement', 'Postgres/PostgREST is truth; files are cache/export/artifact unless API points to them; DB-worthy state goes to DB; receipts prove the thing happened.'
    ) AS db_law,
    jsonb_build_array(
        'curl -sS http://127.0.0.1:3000/canon_current?limit=1',
        'curl -sS http://127.0.0.1:3000/canon_versions?limit=5',
        './luci canon current --json',
        './luci canon versions --json'
    ) AS next_commands
FROM lucidota_canon.bible_nodes
CROSS JOIN goal_row
WHERE valid_to IS NULL;

GRANT SELECT ON lucidota_canon.canon_current TO lucidota_postgrest_anon, mfspx;

NOTIFY pgrst, 'reload schema';

COMMIT;
