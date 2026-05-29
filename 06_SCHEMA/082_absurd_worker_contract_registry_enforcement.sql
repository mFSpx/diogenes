-- FILE: 06_SCHEMA/082_absurd_worker_contract_registry_enforcement.sql
-- PURPOSE: complete required ABSURD worker contract registry rows for core workers.
-- COMPLIANCE: contract registry only; no service migration, no graph mutation.

BEGIN;
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_control;

INSERT INTO lucidota_control.absurd_queue(queue_name, owner_subsystem, max_attempts, notes)
VALUES
  ('chrono', 'Chrono-Ledger', 3, 'Chrono ABSURD wrapper jobs for health/audit/replay bridge; custom service remains supervised.'),
  ('korpus', 'KRAMPUSCHEWING/KORPUS', 3, 'KORPUS intake/componentization wrapper jobs.'),
  ('river', 'River/Bytewax legacy ABSURD compatibility lane', 3, 'Legacy River/Bytewax health tick queue retained only with explicit worker contract provenance.'),
  ('absurd.phase05_streaming_brain', 'River/Bytewax/GLiNER streaming brain', 3, 'Streaming brain wrapper jobs including GLiNER claim-packet extraction.'),
  ('surface_cep', 'Darwinian Surfaces/CEP', 3, 'Surface instruction fan-in to conversation command / ABSURD envelope.'),
  ('graph_promotion', 'Graph Promotion', 2, 'Evidence-gated graph promotion workers; materialization only through promotion path.'),
  ('document_parse', 'Document parsing / evidence intake', 3, 'Queued parser jobs write document_parse evidence candidates.')
ON CONFLICT(queue_name) DO UPDATE SET owner_subsystem=EXCLUDED.owner_subsystem, max_attempts=EXCLUDED.max_attempts, notes=EXCLUDED.notes, updated_at=now();

INSERT INTO lucidota_control.absurd_worker_contract(worker_key, queue_name, script_path, input_contract, output_contract, idempotency_rule, retry_policy, dead_letter_policy, canonical_graph_write_allowed, status, evidence_refs, detail)
VALUES
('chrono_worker','chrono','scripts/absurd_chrono_worker.py','{"job_kind":"chrono_health_check"}'::jsonb,'{"writes":["lucidota_control.absurd_queue_event","lucidota_control.runtime_status_fact"],"forbidden":["temporal claim deletion"]}'::jsonb,'job_kind + payload_sha256','idempotent health/audit; replay jobs preserve temporal claims','dead-letter extractor/bridge metadata only',false,'implemented','["scripts/absurd_chrono_worker.py"]'::jsonb,'{"required_core_worker":true}'::jsonb),
('krampus_worker','korpus','scripts/spine_krampus_worker.py','{"job_kind":"krampus_health_check|korpus_componentize|korpus_lane_job"}'::jsonb,'{"writes":["lucidota_control.absurd_queue_event"],"forbidden":["canonical graph mutation"]}'::jsonb,'job_kind + payload_sha256','retry componentization wrappers only; quarantine unsafe payloads','dead-letter paths/hashes only, never secrets',false,'implemented','["scripts/spine_krampus_worker.py"]'::jsonb,'{"required_core_worker":true}'::jsonb),
('river_worker','absurd.phase05_streaming_brain','scripts/absurd_river_worker.py','{"job_kind":"gliner_zero_shot_extract|gliner_claim_packet_extract"}'::jsonb,'{"writes":["lucidota_learning.gliner_extraction_run","lucidota_learning.gliner_entity_span","lucidota_korpus.document_claim_packet","lucidota_control.workflow_event"],"truth_status":"not_truth_claim_candidate"}'::jsonb,'job_kind + payload_sha256','retry extraction idempotently; no temporal mutation','dead-letter extraction metadata only',false,'implemented','["scripts/absurd_river_worker.py","06_SCHEMA/038_absurd_river_wrapper.sql","06_SCHEMA/073_absurd_river_claim_packet_job.sql"]'::jsonb,'{"required_core_worker":true,"strict_dequeue_enforced":true}'::jsonb),
('river_legacy_worker','river','scripts/absurd_river_worker.py','{"job_kind":"river_bytewax_health_check"}'::jsonb,'{"writes":["lucidota_learning.river_run","lucidota_learning.bytewax_stream_run","lucidota_control.workflow_event"],"truth_status":"runtime_health_not_truth_claim"}'::jsonb,'job_kind + payload_sha256','bounded legacy health tick only; no temporal mutation','dead-letter health metadata only',false,'implemented','["scripts/absurd_river_worker.py","06_SCHEMA/038_absurd_river_wrapper.sql"]'::jsonb,'{"legacy_provenance_lane":true,"strict_dequeue_enforced":true}'::jsonb),
('surface_cep_worker','surface_cep','scripts/spine_surface_cep_worker.py','{"job_kind":"surface_cep_health_check|surface_instruction_fan_in|surface_instruction_compiled_command|conversation_command_work_order"}'::jsonb,'{"writes":["lucidota_control.conversation_command","lucidota_control.workflow_event"],"forbidden":["direct canonical mutation"]}'::jsonb,'payload_sha256 + instruction','retry fan-in idempotently','dead-letter command-envelope metadata only',false,'implemented','["scripts/spine_surface_cep_worker.py"]'::jsonb,'{"required_core_worker":true}'::jsonb),
('graph_promotion_worker','graph_promotion','scripts/absurd_graph_promotion_worker.py','{"job_kind":"graph_promotion_health_check|graph_promotion_packet_defer"}'::jsonb,'{"writes":["lucidota_go.graph_promotion_packet"],"canonical_materialization":"disabled_by_default"}'::jsonb,'candidate_payload_sha256 + decision + evidence_refs','retry validation/packet creation only','dead-letter packet validation only',false,'implemented','["scripts/absurd_graph_promotion_worker.py"]'::jsonb,'{"required_core_worker":true}'::jsonb),
('document_parse_worker','document_parse','scripts/spine_document_parse_worker.py','{"job_kind":"document_parse_ingest"}'::jsonb,'{"writes":["lucidota_korpus.document_parse_run","lucidota_korpus.document_parse_span","lucidota_control.absurd_queue_event"],"truth_status":"not_truth_evidence_candidate","forbidden":["canonical graph mutation"]}'::jsonb,'sha256(input_path + source_sha256 + parser_backend)','retry parser failures up to queue max_attempts; unsupported binary documents are deferred, not canonical truth','dead-letter only job metadata and redacted error; do not print source contents',false,'implemented','["scripts/spine_document_parse_worker.py","06_SCHEMA/076_absurd_document_parse_worker.sql"]'::jsonb,'{"required_core_worker":true}'::jsonb)
ON CONFLICT(worker_key) DO UPDATE SET
  queue_name=EXCLUDED.queue_name, script_path=EXCLUDED.script_path,
  input_contract=EXCLUDED.input_contract, output_contract=EXCLUDED.output_contract,
  idempotency_rule=EXCLUDED.idempotency_rule, retry_policy=EXCLUDED.retry_policy,
  dead_letter_policy=EXCLUDED.dead_letter_policy,
  canonical_graph_write_allowed=EXCLUDED.canonical_graph_write_allowed,
  status=EXCLUDED.status, evidence_refs=EXCLUDED.evidence_refs,
  updated_at=now(), detail=lucidota_control.absurd_worker_contract.detail || EXCLUDED.detail;

COMMIT;
