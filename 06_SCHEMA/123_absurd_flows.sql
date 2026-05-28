-- Absurd Flows: deterministic chronological views for Krampuschew.
CREATE OR REPLACE VIEW lucidota_investigation.case_chronology AS
SELECT
    case_uuid,
    case_key,
    title,
    status,
    opened_at,
    closed_at,
    summary,
    detail,
    created_at,
    updated_at
FROM lucidota_investigation.case_file
ORDER BY opened_at ASC, created_at ASC;

CREATE OR REPLACE VIEW lucidota_investigation.artifact_chronology AS
SELECT
    artifact_uuid,
    sha256,
    cas_uri,
    locked_relative_path,
    original_path,
    original_name,
    mime,
    file_kind,
    size_bytes,
    title,
    evidence_date,
    evidence_date_source,
    created_at,
    updated_at
FROM lucidota_investigation.artifact
ORDER BY evidence_date ASC NULLS LAST, created_at ASC;

CREATE OR REPLACE VIEW lucidota_investigation.case_artifact_chain AS
SELECT
    c.case_key,
    c.title AS case_title,
    a.sha256,
    a.original_path,
    a.original_name,
    a.file_kind,
    a.size_bytes,
    ca.evidence_id,
    ca.status AS case_artifact_status,
    ca.created_at AS linked_at
FROM lucidota_investigation.case_artifact ca
JOIN lucidota_investigation.case_file c ON c.case_uuid = ca.case_uuid
JOIN lucidota_investigation.artifact a ON a.artifact_uuid = ca.artifact_uuid
ORDER BY c.opened_at ASC, ca.created_at ASC;
