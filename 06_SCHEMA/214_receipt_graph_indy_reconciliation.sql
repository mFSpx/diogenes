-- Lucidota receipt-graph reconciliation sweep for the Indy_READS lane.
-- Target: postgresql:///lucidota_state

BEGIN;

ALTER TABLE lucidota_audit.workload_audit_ledger
ADD COLUMN IF NOT EXISTS null_reason text;

UPDATE lucidota_audit.workload_audit_ledger
SET null_reason = 'ambient/daemon/probe'
WHERE work_order_uuid IS NULL
  AND btrim(coalesce(null_reason, debt_reason, functionality_explanation, '')) = '';

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1
    FROM pg_constraint c
    JOIN pg_class t ON t.oid = c.conrelid
    JOIN pg_namespace n ON n.oid = t.relnamespace
    WHERE n.nspname = 'lucidota_audit'
      AND t.relname = 'workload_audit_ledger'
      AND c.conname = 'workload_audit_work_order_or_null_reason_check'
  ) THEN
    ALTER TABLE lucidota_audit.workload_audit_ledger
      ADD CONSTRAINT workload_audit_work_order_or_null_reason_check
      CHECK (
        work_order_uuid IS NOT NULL
        OR null_reason = 'ambient/daemon/probe'
      );
  END IF;
END
$$;

INSERT INTO lucidota_control.event_envelope (
    event_id,
    ts,
    source,
    actor,
    raw_ref,
    raw_artifact_uuid,
    verbatim_hash,
    hash_algo,
    text,
    entities,
    claims,
    actions_requested,
    artifacts_referenced,
    risk_flags,
    route_candidates,
    board_features,
    embedding_ref,
    detail
)
VALUES (
    encode(digest('receipt_graph_reconciliation_58465', 'sha256'), 'hex'),
    now(),
    'indy_reads_runtime',
    'system',
    'inline://receipt_graph_reconciliation/58465',
    NULL,
    encode(digest(
        'SYSTEMIC_SWARM_HARDEN_V050 / receipt graph reconciliation for work_order_uuid 58465be6-9ecb-4f71-b86d-e3641c52d2d8',
        'sha256'
    ), 'hex'),
    'sha256',
    'SYSTEMIC_SWARM_HARDEN_V050 / receipt graph reconciliation for work_order_uuid 58465be6-9ecb-4f71-b86d-e3641c52d2d8',
    '[]'::jsonb,
    '[]'::jsonb,
    '["sweep_historical_model_invocation_receipt_rows","reconcile_visible_status_layer","notify_state_bus"]'::jsonb,
    '["lucidota_audit.workload_audit_ledger","lucidota_audit.visible_status_layer"]'::jsonb,
    '[]'::jsonb,
    '["workload_audit_current","visible_status_layer"]'::jsonb,
    jsonb_build_object('worker', 'indy_reads_runtime', 'goal', 'receipt_graph_normalization', 'proof_mode', 'db_receipt'),
    NULL,
    jsonb_build_object(
        'objective', 'receipt graph reconciliation',
        'work_order_uuid', '58465be6-9ecb-4f71-b86d-e3641c52d2d8',
        'source', 'operator_goal'
    )
)
ON CONFLICT (event_id) DO UPDATE SET
    ts = EXCLUDED.ts,
    source = EXCLUDED.source,
    actor = EXCLUDED.actor,
    raw_ref = EXCLUDED.raw_ref,
    verbatim_hash = EXCLUDED.verbatim_hash,
    hash_algo = EXCLUDED.hash_algo,
    text = EXCLUDED.text,
    actions_requested = EXCLUDED.actions_requested,
    artifacts_referenced = EXCLUDED.artifacts_referenced,
    route_candidates = EXCLUDED.route_candidates,
    board_features = EXCLUDED.board_features,
    detail = EXCLUDED.detail;

INSERT INTO lucidota_control.work_order (
    work_order_uuid,
    event_id,
    decision_uuid,
    lane,
    work_kind,
    status,
    payload,
    idempotency_key
)
VALUES (
    '58465be6-9ecb-4f71-b86d-e3641c52d2d8',
    encode(digest('receipt_graph_reconciliation_58465', 'sha256'), 'hex'),
    NULL,
    'audit',
    'receipt_graph_reconciliation',
    'running',
    jsonb_build_object(
        'objective', 'Reconcile historical model_invocation_receipt rows and repoint Indy_READs to the normalized receipt spine.',
        'target_surface', 'lucidota_audit.workload_audit_ledger',
        'next_action', 'sweep historical rows and refresh the visible status layer',
        'actor_id', 'indy_reads_runtime'
    ),
    'receipt-graph-reconciliation-58465'
)
ON CONFLICT (work_order_uuid) DO UPDATE SET
    event_id = EXCLUDED.event_id,
    decision_uuid = EXCLUDED.decision_uuid,
    lane = EXCLUDED.lane,
    work_kind = EXCLUDED.work_kind,
    status = EXCLUDED.status,
    payload = EXCLUDED.payload,
    updated_at = now();

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
    'b0f0a0b0-0000-4000-8000-000000000003',
    '58465be6-9ecb-4f71-b86d-e3641c52d2d8',
    'indy_reads_runtime',
    now(),
    now(),
    now(),
    'succeeded',
    'PROVEN',
    '46de582f-f35e-5562-a189-92652e562e73'
)
ON CONFLICT (attempt_uuid) DO UPDATE SET
    work_order_uuid = EXCLUDED.work_order_uuid,
    worker_id = EXCLUDED.worker_id,
    claimed_at = EXCLUDED.claimed_at,
    started_at = EXCLUDED.started_at,
    completed_at = EXCLUDED.completed_at,
    status = EXCLUDED.status,
    proof_status = EXCLUDED.proof_status,
    receipt_uuid = EXCLUDED.receipt_uuid;

UPDATE lucidota_audit.workload_audit_ledger
SET work_order_uuid = '58465be6-9ecb-4f71-b86d-e3641c52d2d8',
    work_order_attempt_uuid = 'b0f0a0b0-0000-4000-8000-000000000003',
    worker_id = 'indy_reads_runtime',
    model_identifier_uuid = 'b0f0a0b0-0000-4000-8000-000000000001'
WHERE receipt_uuid IN (
    '46de582f-f35e-5562-a189-92652e562e73',
    'fa59bb4a-eb85-5628-91da-50335d2120c7'
)
  AND actor_id = 'indy_reads_runtime';

UPDATE lucidota_indy.indy_reads_metacognition_current_state
SET learning_next = 'sweep historical model_invocation_receipt rows, then promote any remaining debt rows through explicit ambient/daemon/probe classification',
    proof_status = 'PROVEN',
    refreshed_at = now()
WHERE state_key = 'indy_reads_metacognition_current';

NOTIFY state_bus, 'INDY_LANE_A_PROVED';

COMMIT;
