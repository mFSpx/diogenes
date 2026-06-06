-- Migration 020: Case File Workflow — Big Board PostgREST views
-- Schema: lucidota_go
-- Created: 2026-06-06 EARN OUROBOROS HARDEN run
-- Purpose: Surface case/evidence/edge status for the Big Board via PostgREST

BEGIN;

-- Case summary view: one row per case with aggregate counts
CREATE OR REPLACE VIEW lucidota_go.case_summary_view AS
SELECT
    c.uuid AS case_uuid,
    c.label AS case_id,
    c.payload->>'title' AS title,
    c.status,
    c.created_at AS opened_at,
    c.updated_at AS last_activity,
    COUNT(DISTINCT e.uuid) FILTER (WHERE e.term = 'FILE_INGEST') AS evidence_count,
    COUNT(DISTINCT e.uuid) FILTER (WHERE e.term IN ('LEAD','HUNCH','CLUE','TRAIL')) AS open_leads,
    COUNT(DISTINCT e.uuid) FILTER (WHERE e.term IN ('CLAIM','HYPOTHESIS')) AS claim_count
FROM lucidota_go.graph_item c
LEFT JOIN lucidota_go.graph_edge ge ON ge.target_uuid = c.uuid AND ge.edge_type = 'BELONGS_TO'
LEFT JOIN lucidota_go.graph_item e ON e.uuid = ge.source_uuid
WHERE c.term = 'CASE'
GROUP BY c.uuid, c.label, c.payload, c.status, c.created_at, c.updated_at;

-- Evidence status dashboard: counts by status for all ingested files
CREATE OR REPLACE VIEW lucidota_go.evidence_status_view AS
SELECT
    status,
    COUNT(*) AS count,
    COUNT(*) FILTER (WHERE term = 'FILE_INGEST') AS file_ingest_count,
    COUNT(*) FILTER (WHERE term IN ('EVIDENCE','SOURCE','WITNESS')) AS direct_evidence_count
FROM lucidota_go.graph_item
WHERE term IN ('FILE_INGEST','EVIDENCE','SOURCE','WITNESS','DOCUMENT')
GROUP BY status
ORDER BY
    CASE status
        WHEN 'collected' THEN 1 WHEN 'registered' THEN 2 WHEN 'hashed' THEN 3
        WHEN 'ingested' THEN 4 WHEN 'linked_to_case' THEN 5 WHEN 'reviewed' THEN 6
        WHEN 'promoted' THEN 7 WHEN 'challenged' THEN 8 WHEN 'superseded' THEN 9
        WHEN 'archived' THEN 10 WHEN 'error_corrected' THEN 11 WHEN 'lost' THEN 12
        WHEN 'collapsed' THEN 13 WHEN 'rejected' THEN 14
        ELSE 99
    END;

-- Orphan evidence: FILE_INGEST items with no BELONGS_TO case edge
CREATE OR REPLACE VIEW lucidota_go.orphan_evidence_view AS
SELECT
    gi.uuid,
    gi.label AS file_path,
    gi.status,
    gi.payload->>'sha256' AS sha256,
    gi.payload->>'size_bytes' AS size_bytes,
    gi.created_at,
    EXTRACT(EPOCH FROM (NOW() - gi.created_at)) / 3600.0 AS hours_orphaned
FROM lucidota_go.graph_item gi
WHERE gi.term = 'FILE_INGEST'
  AND gi.status NOT IN ('archived','superseded','lost')
  AND NOT EXISTS (
    SELECT 1 FROM lucidota_go.graph_edge ge
    WHERE ge.source_uuid = gi.uuid
      AND ge.edge_type = 'BELONGS_TO'
      AND ge.status = 'current'
  )
ORDER BY gi.created_at DESC;

-- Promotion queue: evidence ready for promotion gate review
CREATE OR REPLACE VIEW lucidota_go.promotion_queue_view AS
SELECT
    gi.uuid,
    gi.label AS file_path,
    gi.status,
    gi.payload->>'sha256' AS sha256,
    gi.payload->>'evidence_class' AS evidence_class,
    gi.created_at,
    gi.updated_at,
    CASE WHEN gi.payload->>'reviewed_by' IS NOT NULL THEN true ELSE false END AS has_review,
    gi.payload->>'reviewed_by' AS reviewer
FROM lucidota_go.graph_item gi
WHERE gi.term = 'FILE_INGEST'
  AND gi.status = 'reviewed'
ORDER BY gi.created_at ASC;

-- Edge type usage: count edges by type and family
CREATE OR REPLACE VIEW lucidota_go.edge_type_usage_view AS
SELECT
    COALESCE(ge.edge_type, 'RELATED_TO') AS edge_type,
    COALESCE(ge.relationship_family, 'unclassified') AS relationship_family,
    COUNT(*) AS edge_count,
    COUNT(*) FILTER (WHERE ge.status = 'current') AS current_edges,
    COUNT(*) FILTER (WHERE ge.status = 'superseded') AS superseded_edges
FROM lucidota_go.graph_edge ge
GROUP BY ge.edge_type, ge.relationship_family
ORDER BY COUNT(*) DESC;

-- Unprocessed staging packets summary
CREATE OR REPLACE VIEW lucidota_go.staging_backlog_view AS
SELECT
    sp.status,
    COUNT(*) AS count,
    MIN(sp.created_at) AS oldest,
    MAX(sp.created_at) AS newest
FROM lucidota_go.staging_packet sp
GROUP BY sp.status
ORDER BY count DESC;

-- Evidence chain of custody log (provenance edges)
CREATE OR REPLACE VIEW lucidota_go.custody_chain_view AS
SELECT
    ge.edge_uuid,
    source.label AS evidence_path,
    target.label AS custodian_name,
    ge.edge_type,
    ge.valid_from AS custody_since,
    ge.valid_to AS custody_until,
    ge.status,
    ge.detail->>'transfer_receipt' AS transfer_receipt
FROM lucidota_go.graph_edge ge
JOIN lucidota_go.graph_item source ON source.uuid = ge.source_uuid
JOIN lucidota_go.graph_item target ON target.uuid = ge.target_uuid
WHERE ge.relationship_family = 'provenance'
  AND ge.edge_type IN ('CUSTODY_OF','AUTHORED_BY')
ORDER BY ge.valid_from DESC;

COMMIT;
