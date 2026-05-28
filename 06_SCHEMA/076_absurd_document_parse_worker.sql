-- FILE: 06_SCHEMA/076_absurd_document_parse_worker.sql
-- PURPOSE: ABSURD wrapper contract for document parse ingestion jobs.
-- COMPLIANCE: parser output is evidence candidate only; no canonical graph mutation.

BEGIN;
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_control;

INSERT INTO lucidota_control.absurd_queue(queue_name, owner_subsystem, max_attempts, notes)
VALUES ('document_parse', 'Document parsing / evidence intake', 3, 'Queued parser jobs write document_parse_run/span evidence candidates only; no graph truth.')
ON CONFLICT (queue_name) DO UPDATE SET
  owner_subsystem=EXCLUDED.owner_subsystem,
  max_attempts=EXCLUDED.max_attempts,
  notes=EXCLUDED.notes,
  updated_at=now();

CREATE TABLE IF NOT EXISTS lucidota_control.absurd_worker_contract (
  contract_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  worker_key text NOT NULL UNIQUE,
  queue_name text NOT NULL REFERENCES lucidota_control.absurd_queue(queue_name) ON DELETE RESTRICT,
  script_path text NOT NULL,
  input_contract jsonb NOT NULL,
  output_contract jsonb NOT NULL,
  idempotency_rule text NOT NULL,
  retry_policy text NOT NULL,
  dead_letter_policy text NOT NULL,
  canonical_graph_write_allowed boolean NOT NULL DEFAULT false,
  status text NOT NULL CHECK (status IN ('contracted','implemented','verified','blocked')),
  evidence_refs jsonb NOT NULL DEFAULT '[]'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  detail jsonb NOT NULL DEFAULT '{}'::jsonb
);

INSERT INTO lucidota_control.absurd_worker_contract(
  worker_key, queue_name, script_path, input_contract, output_contract,
  idempotency_rule, retry_policy, dead_letter_policy, canonical_graph_write_allowed,
  status, evidence_refs, detail
) VALUES (
  'document_parse_worker',
  'document_parse',
  'scripts/spine_document_parse_worker.py',
  '{"job_kind":"document_parse_ingest","payload_fields":["input_path","storage_database_url","parser_backend"]}'::jsonb,
  '{"writes":["lucidota_korpus.document_parse_run","lucidota_korpus.document_parse_span","lucidota_control.absurd_queue_event"],"truth_status":"not_truth_evidence_candidate","forbidden":["canonical graph mutation"]}'::jsonb,
  'sha256(input_path + source_sha256 + parser_backend)',
  'retry parser failures up to queue max_attempts; unsupported binary documents are deferred, not canonical truth',
  'dead-letter only job metadata and redacted error; do not print source contents',
  false,
  'implemented',
  '["06_SCHEMA/076_absurd_document_parse_worker.sql","scripts/spine_document_parse_worker.py"]'::jsonb,
  '{"round":21,"target":"Document parse worker ABSURD migration"}'::jsonb
)
ON CONFLICT (worker_key) DO UPDATE SET
  queue_name=EXCLUDED.queue_name,
  script_path=EXCLUDED.script_path,
  input_contract=EXCLUDED.input_contract,
  output_contract=EXCLUDED.output_contract,
  idempotency_rule=EXCLUDED.idempotency_rule,
  retry_policy=EXCLUDED.retry_policy,
  dead_letter_policy=EXCLUDED.dead_letter_policy,
  canonical_graph_write_allowed=EXCLUDED.canonical_graph_write_allowed,
  status=EXCLUDED.status,
  evidence_refs=EXCLUDED.evidence_refs,
  updated_at=now(),
  detail=EXCLUDED.detail;

COMMIT;
