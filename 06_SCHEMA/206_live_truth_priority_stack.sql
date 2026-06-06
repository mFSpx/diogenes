-- Canonical live-truth orchestration priority stack.

BEGIN;

CREATE SCHEMA IF NOT EXISTS lucidota_control;

CREATE OR REPLACE FUNCTION lucidota_control.live_truth_priority_stack()
RETURNS jsonb
LANGUAGE sql
IMMUTABLE
AS $$
    SELECT to_jsonb(ARRAY[
        'live_truth_surfaces',
        'deterministic_local_checks',
        'thin_packets',
        'local',
        'indy_reads',
        'codex',
        'vibe',
        'groq',
        'broader_cloud'
    ]::text[]);
$$;

COMMENT ON FUNCTION lucidota_control.live_truth_priority_stack() IS
    'Canonical sub-orchestrator priority stack used by live-current packets.';

COMMIT;
