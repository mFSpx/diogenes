-- Live root-law docs packet for PostgREST.
-- This keeps the existing route catalog entry honest by exposing a read-only
-- summary packet at /root_law_docs.

BEGIN;

CREATE OR REPLACE VIEW lucidota_canon.api_root_law_docs AS
WITH manual_rows AS (
    SELECT
        count(*) AS manual_count,
        COALESCE(jsonb_agg(manual_id ORDER BY manual_id), '[]'::jsonb) AS manual_ids
    FROM lucidota_canon.api_bible_manuals
    WHERE manual_id IN ('SYSTEM_ARCH', 'RUNTIME_GOVERNOR', 'AVIONICS', 'FLIGHT_MAN', 'LEDGER')
),
route_rows AS (
    SELECT
        count(*) AS route_count,
        count(*) FILTER (WHERE status = 'implemented') AS implemented_route_count,
        COALESCE(
            jsonb_agg(
                jsonb_build_object(
                    'route_id', route_id,
                    'method', method,
                    'path_pattern', path_pattern,
                    'description', description,
                    'target', target,
                    'status', status
                )
                ORDER BY route_id
            ),
            '[]'::jsonb
        ) AS route_catalog
    FROM lucidota_canon.api_route_catalog
    WHERE route_id IN ('manuals', 'nodes', 'root_law_docs', 'route_catalog', 'subtree')
)
SELECT
    'root_law_docs'::text AS route_id,
    'Root-Law API docs'::text AS title,
    'ok'::text AS status,
    now() AS refreshed_at,
    '05_OUTPUTS/root_rotor_manuals/root_law_api_docs.html'::text AS html_path,
    '05_OUTPUTS/root_rotor_manuals/root_law_api_docs.md'::text AS markdown_path,
    '05_OUTPUTS/root_rotor_manuals/root_law_gap_atlas.json'::text AS gap_atlas_path,
    manual_rows.manual_count,
    manual_rows.manual_ids,
    route_rows.route_count,
    route_rows.implemented_route_count,
    route_rows.route_catalog,
    jsonb_build_object(
        'manual_summary', 'system manuals + root rotor route evidence',
        'route_summary', 'route catalog + manual packet sync',
        'artifact', 'root_law_api_docs.html',
        'notes', 'read-only packet synthesized from live DB surfaces'
    ) AS notes
FROM manual_rows
CROSS JOIN route_rows;

GRANT SELECT ON lucidota_canon.api_root_law_docs TO lucidota_postgrest_anon, mfspx;

NOTIFY pgrst, 'reload schema';

COMMIT;
