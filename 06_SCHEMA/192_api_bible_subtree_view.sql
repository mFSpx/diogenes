-- Direct subtree surface for PostgREST.
-- Exposes /api_bible_subtree?root_id=eq.{node_id} as a read-only recursive view.

BEGIN;

CREATE OR REPLACE VIEW lucidota_canon.api_bible_subtree AS
SELECT
    root.node_id AS root_id,
    sub.node_id,
    sub.parent_id,
    sub.node_sort_key,
    sub.manual_id,
    sub.title,
    sub.payload,
    sub.payload_format,
    sub.source_refs,
    sub.evidence_hashes,
    sub.dependencies,
    sub.affects_nodes,
    sub.status,
    sub.version,
    sub.valid_from,
    sub.valid_to,
    sub.hash_current,
    sub.previous_hash,
    sub.created_at,
    sub.updated_at
FROM lucidota_canon.api_bible_nodes root
CROSS JOIN LATERAL lucidota_canon.get_subtree(root.node_id) AS sub;

GRANT SELECT ON lucidota_canon.api_bible_subtree TO lucidota_postgrest_anon, mfspx;

UPDATE lucidota_canon.api_route_catalog
SET target = 'lucidota_canon.api_bible_subtree',
    updated_at = now()
WHERE route_id = 'subtree';

NOTIFY pgrst, 'reload schema';

COMMIT;
