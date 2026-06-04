-- Lift the active goal packet into an operator-readable truth spine.

BEGIN;

CREATE OR REPLACE VIEW lucidota_canon.active_goal AS
SELECT
    goal_id,
    title,
    status,
    active_prompt_path,
    active_prompt_hash,
    current_handoff_path,
    detail,
    created_at,
    updated_at,
    jsonb_build_object(
        'goal_id', goal_id,
        'title', title,
        'status', status,
        'current_handoff_path', current_handoff_path
    ) AS goal,
    jsonb_build_object(
        'statement', 'Postgres/PostgREST is truth; files are cache/export/artifact unless API points to them; DB-worthy state goes to DB; receipts prove the thing happened.'
    ) AS db_law,
    jsonb_build_array(
        'curl -sS http://127.0.0.1:3000/active_goal?limit=1',
        'curl -sS http://127.0.0.1:3000/manual_current?limit=1',
        './luci active goal --json',
        './luci api active goal --json'
    ) AS next_commands
FROM lucidota_control.active_goal;

GRANT SELECT ON lucidota_canon.active_goal TO lucidota_postgrest_anon, mfspx;

NOTIFY pgrst, 'reload schema';

COMMIT;
