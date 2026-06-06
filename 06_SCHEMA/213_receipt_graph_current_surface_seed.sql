-- Lucidota Receipt Graph Current Surface Seed
-- Target: postgresql:///lucidota_state
-- Purpose: make the normalized receipt spine visible for the live Indy_READS lane.

BEGIN;

INSERT INTO lucidota_control.worker (
    worker_id,
    actor_class,
    runtime_kind,
    host_id,
    lane_id,
    active_mode
)
VALUES (
    'indy_reads_runtime',
    'indy_reads',
    'ironclaw',
    'local_host',
    'bonsai_slot_0',
    'running'
)
ON CONFLICT (worker_id) DO UPDATE
SET actor_class = EXCLUDED.actor_class,
    runtime_kind = EXCLUDED.runtime_kind,
    host_id = EXCLUDED.host_id,
    lane_id = EXCLUDED.lane_id,
    active_mode = EXCLUDED.active_mode,
    updated_at = NOW();

INSERT INTO lucidota_canon.model_identifier (
    model_identifier_uuid,
    provider,
    model_family,
    model_id,
    weight_hash,
    quantization,
    adapter_id,
    runtime_backend,
    lane_id,
    context_window,
    kv_cache_policy
)
SELECT
    'b0f0a0b0-0000-4000-8000-000000000001',
    'local',
    'bonsai',
    'bonsai_q1_0',
    'sha256/blake3_stub_bonsai_q1_0',
    'q1',
    NULL,
    'ironclaw',
    'bonsai_slot_0',
    10000,
    'prefix_cache'
WHERE NOT EXISTS (
    SELECT 1
    FROM lucidota_canon.model_identifier
    WHERE model_identifier_uuid = 'b0f0a0b0-0000-4000-8000-000000000001'
);

INSERT INTO lucidota_control.work_order_attempt (
    attempt_uuid,
    work_order_uuid,
    worker_id,
    claimed_at,
    started_at,
    completed_at,
    status,
    proof_status,
    receipt_uuid
)
VALUES (
    'b0f0a0b0-0000-4000-8000-000000000002',
    '99065fe0-267d-411b-a97e-5c18a2ae15d6',
    'indy_reads_runtime',
    NOW(),
    NOW(),
    NULL,
    'running',
    'PARTIAL',
    '46de582f-f35e-5562-a189-92652e562e73'
)
ON CONFLICT (attempt_uuid) DO UPDATE
SET work_order_uuid = EXCLUDED.work_order_uuid,
    worker_id = EXCLUDED.worker_id,
    claimed_at = EXCLUDED.claimed_at,
    started_at = EXCLUDED.started_at,
    completed_at = EXCLUDED.completed_at,
    status = EXCLUDED.status,
    proof_status = EXCLUDED.proof_status,
    receipt_uuid = EXCLUDED.receipt_uuid;

UPDATE lucidota_audit.workload_audit_ledger
SET model_identifier_uuid = 'b0f0a0b0-0000-4000-8000-000000000001',
    work_order_uuid = '99065fe0-267d-411b-a97e-5c18a2ae15d6',
    work_order_attempt_uuid = 'b0f0a0b0-0000-4000-8000-000000000002',
    worker_id = 'indy_reads_runtime'
WHERE receipt_uuid IN (
    '46de582f-f35e-5562-a189-92652e562e73',
    'fa59bb4a-eb85-5628-91da-50335d2120c7'
)
  AND actor_id = 'indy_reads_runtime';

COMMIT;
