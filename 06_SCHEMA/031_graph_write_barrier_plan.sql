-- FILE: 06_SCHEMA/031_graph_write_barrier_plan.sql
-- COMPONENT: LUCIDOTA graph write barrier PLAN ONLY
-- COMPLIANCE: non-destructive; no GRANT/REVOKE enforcement; schema numbering gaps are expected.
-- PURPOSE: document confirmed graph tables from inspected schema files and outline future role model.

-- Schema files inspected:
-- - 06_SCHEMA/001_lucidota_control.sql
-- - 06_SCHEMA/002_model_runtime.sql
-- - 06_SCHEMA/003_survey_protocol.sql
-- - 06_SCHEMA/004_learning_reflex.sql
-- - 06_SCHEMA/005_cas_manifest.sql
-- - 06_SCHEMA/006_workflow_registry.sql
-- - 06_SCHEMA/007_bytewax_stream.sql
-- - 06_SCHEMA/008_hop_pivot.sql
-- - 06_SCHEMA/009_treelite_router.sql
-- - 06_SCHEMA/010_wake_bus.sql
-- - 06_SCHEMA/011_body_capture.sql
-- - 06_SCHEMA/012_authorized_extractors.sql
-- - 06_SCHEMA/013_signal_ingress.sql
-- - 06_SCHEMA/014_indy_runtime.sql
-- - 06_SCHEMA/015_root414_packets.sql
-- - 06_SCHEMA/016_go_graph_core.sql
-- - 06_SCHEMA/017_indy_reads_library.sql
-- - 06_SCHEMA/018_investigation_artifact.sql
-- - 06_SCHEMA/019_korpus_krampii.sql
-- - 06_SCHEMA/020_chat_dump_timeline.sql
-- - 06_SCHEMA/020_korpus_derived_compute_queue.sql
-- - 06_SCHEMA/021_hard_truth_math.sql
-- - 06_SCHEMA/022_comm_dump_timeline.sql
-- - 06_SCHEMA/023_etl_pipeline.sql
-- - 06_SCHEMA/025_chrono_ledger_core.sql
-- - 06_SCHEMA/026_chrono_absurd_triggers.sql
-- - 06_SCHEMA/027_chrono_phase_c_ops.sql
-- - 06_SCHEMA/028_ternary_lens_lab.sql

-- Confirmed graph tables:
-- - lucidota_go.term_registry | canonical/supporting-registry | 06_SCHEMA/016_go_graph_core.sql:10
-- - lucidota_go.soul_registry | canonical/supporting-registry | 06_SCHEMA/016_go_graph_core.sql:21
-- - lucidota_go.graph_layer | canonical/supporting-registry | 06_SCHEMA/016_go_graph_core.sql:31
-- - lucidota_go.graph_item | promoted/canonical | 06_SCHEMA/016_go_graph_core.sql:39
-- - lucidota_go.graph_item_layer | canonical/supporting-registry | 06_SCHEMA/016_go_graph_core.sql:93
-- - lucidota_go.graph_edge | promoted/canonical | 06_SCHEMA/016_go_graph_core.sql:104
-- - lucidota_go.graph_edge_layer | canonical/supporting-registry | 06_SCHEMA/016_go_graph_core.sql:155
-- - lucidota_go.graph_journal | canonical/audit | 06_SCHEMA/016_go_graph_core.sql:166
-- - lucidota_go.staging_packet | staging/raw | 06_SCHEMA/016_go_graph_core.sql:199

-- Required future roles:
-- - graph_reader: SELECT only
-- - graph_promoter: constrained workflow/function writes only
-- - graph_admin: migrations/break-glass only

-- NO ACTIVE ENFORCEMENT IN THIS FILE.
-- Do not revoke or grant against guessed roles/tables.

SELECT 'GRAPH_WRITE_BARRIER_PLAN_ONLY' AS plan_status;
SELECT 'No destructive GRANT/REVOKE executed; see 00_PROJECT_BRAIN/GRAPH_WRITE_BARRIER.md' AS note;
