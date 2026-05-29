-- DIOGENES investigative case/artifact workflow.
-- Local-first evidence: stable case folder shape, SHA-256 provenance, locked CAS
-- bytes, normalized sidecars, entity pivots, and GO graph anchors.

CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE SCHEMA IF NOT EXISTS lucidota_investigation;


CREATE OR REPLACE FUNCTION lucidota_investigation.uuid_v7()
RETURNS uuid
LANGUAGE plpgsql
VOLATILE
AS $$
DECLARE
    ts_hex text := lpad(to_hex(floor(extract(epoch FROM clock_timestamp()) * 1000)::bigint), 12, '0');
    rnd text := encode(gen_random_bytes(10), 'hex');
    variant text := substr('89ab', (get_byte(gen_random_bytes(1), 0) % 4) + 1, 1);
BEGIN
    RETURN (
        substr(ts_hex, 1, 8) || '-' ||
        substr(ts_hex, 9, 4) || '-' ||
        '7' || substr(rnd, 1, 3) || '-' ||
        variant || substr(rnd, 4, 3) || '-' ||
        substr(rnd, 7, 12)
    )::uuid;
END;
$$;

CREATE TABLE IF NOT EXISTS lucidota_investigation.case_file (
    case_uuid uuid PRIMARY KEY DEFAULT lucidota_investigation.uuid_v7(),
    case_key text UNIQUE NOT NULL,
    title text NOT NULL,
    status text NOT NULL DEFAULT 'open' CHECK (status IN (
        'open',
        'active',
        'paused',
        'closed',
        'archived'
    )),
    opened_at timestamptz NOT NULL DEFAULT now(),
    closed_at timestamptz,
    summary text NOT NULL DEFAULT '',
    folder_relative_path text NOT NULL DEFAULT '',
    casefile_relative_path text NOT NULL DEFAULT '',
    graph_item_uuid uuid,
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    CHECK (case_key ~ '^KE26-[0-9]{5}$' OR case_key ~ '^CASE-[0-9]{8}-[A-Z0-9][A-Z0-9_-]{1,80}$')
);

CREATE INDEX IF NOT EXISTS case_file_status_opened_idx
    ON lucidota_investigation.case_file(status, opened_at DESC);

CREATE TABLE IF NOT EXISTS lucidota_investigation.capability_registry (
    capability_key text PRIMARY KEY,
    capability_group text NOT NULL,
    capability_name text NOT NULL,
    lifecycle_status text NOT NULL DEFAULT 'planned' CHECK (lifecycle_status IN (
        'planned',
        'prototype',
        'active',
        'deprecated'
    )),
    run_state text NOT NULL DEFAULT 'planned' CHECK (run_state IN (
        'planned',
        'ran',
        'planned_ran',
        'ready'
    )),
    workflow_name text NOT NULL DEFAULT '',
    command text NOT NULL DEFAULT '',
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

INSERT INTO lucidota_investigation.capability_registry
(capability_key, capability_group, capability_name, lifecycle_status, run_state, workflow_name, command, detail)
VALUES
('entity-ingestion-etl', 'Data Engineering, Processing & Analysis', 'Entity Ingestion & ETL Pipelining', 'active', 'ran', 'artifact-ingest', 'scripts/lucidota_artifact_ingest.py ingest', '{"source_matrix": "Krampus Holiday Pogroms & Associated Systems"}'),
('data-comparison-diffing', 'Data Engineering, Processing & Analysis', 'Data Comparison Algorithms & Diffing', 'prototype', 'ran', 'artifact-pivot', 'scripts/lucidota_artifact_ingest.py pivot', '{"source_matrix": "Krampus Holiday Pogroms & Associated Systems"}'),
('csv-manipulation-parsing', 'Data Engineering, Processing & Analysis', 'CSV Manipulation & Parsing', 'prototype', 'ran', 'artifact-ingest', 'scripts/lucidota_artifact_ingest.py ingest', '{"formats": ["csv", "tsv"], "note": "artifact text/entity path handles CSV now; richer ETL wrappers can attach here"}'),
('metadata-extraction-medd', 'Data Engineering, Processing & Analysis', 'Metadata Extraction / Specialized Processing [MEDD]', 'active', 'ran', 'artifact-ingest', 'scripts/lucidota_artifact_ingest.py ingest', '{"tools": ["exiftool", "file", "ffprobe", "identify"]}'),
('big-data-parquet', 'Data Engineering, Processing & Analysis', 'Big Data Storage & Serialization [Parquet]', 'prototype', 'ran', '', '', '{"note": "available for evidence exports; workflow wrapper pending", "python_modules": ["pandas", "pyarrow"]}'),
('ml-feature-engineering-node-ego-temporal-neighbor', 'Data Engineering, Processing & Analysis', 'Machine Learning Feature Engineering [Node, Ego, Temporal, Neighbor]', 'prototype', 'ran', 'go-graph-search', 'scripts/lucidota_basic_workflows.py graph-search', '{"note": "graph/book features exist; formal feature store pending"}'),
('graph-network-data-processing', 'Data Engineering, Processing & Analysis', 'Graph / Network Data Processing', 'active', 'ran', 'artifact-pivot', 'scripts/lucidota_artifact_ingest.py pivot', '{"surfaces": ["lucidota_go", "lucidota_investigation"]}'),
('json-sidecar-handling', 'Data Engineering, Processing & Analysis', 'JSON Data & Sidecar Handling', 'active', 'ran', 'artifact-ingest', 'scripts/lucidota_artifact_ingest.py ingest', '{"sidecars": ["normalized_json", "normalized_yaml", "entities_jsonl", "custody_jsonl"]}'),
('powershell-scripting', 'Scripting, Automation & Software Engineering', 'PowerShell Scripting & Execution', 'planned', 'ran', '', '', '{"note": "registered software capability; Linux runtime wrapper pending"}'),
('python-backend-routing-hooks', 'Scripting, Automation & Software Engineering', 'Python Scripting & Backend Logic [Routing, Hooks]', 'active', 'ran', 'artifact-ingest', 'scripts/lucidota_artifact_ingest.py', '{"language": "python", "surfaces": ["case-create", "ingest", "pivot", "capabilities"]}'),
('automated-three-way-diffing', 'Scripting, Automation & Software Engineering', 'Automated 3-Way Diffing', 'planned', 'ran', '', '', '{"note": "registered software capability; reproducible workflow pending"}'),
('web-scraping-extraction', 'Scripting, Automation & Software Engineering', 'Web Scraping & Automated Data Extraction', 'prototype', 'ran', 'survey-protocol', 'scripts/lucidota_absurd_survey.py', '{"note": "operator/network policy still applies"}'),
('execution-logging-monitoring', 'Scripting, Automation & Software Engineering', 'Execution Logging & Process Monitoring', 'active', 'ran', 'workflow-event', 'lucidota_control.workflow_event', '{"surface": "ABSURD/control events"}'),
('osint-gathering', 'Intelligence, Reconnaissance & Profiling', 'Open Source Intelligence (OSINT) Gathering', 'prototype', 'ran', 'survey-protocol', 'scripts/lucidota_absurd_survey.py', '{"note": "external reads remain policy-gated"}'),
('threat-intelligence-recon', 'Intelligence, Reconnaissance & Profiling', 'Threat Intelligence & Target Reconnaissance', 'prototype', 'ran', 'artifact-pivot', 'scripts/lucidota_artifact_ingest.py pivot', '{"note": "local pivoting active; external enrichment pending explicit gate"}'),
('industry-archetype-profiling', 'Intelligence, Reconnaissance & Profiling', 'Industry Archetype Modeling & Subject Profiling', 'planned', 'planned_ran', '', '', '{"note": "registered as deterministic profiling/feature-mapping capability; model/rules pending"}'),
('voice-protocol', 'Telephony & Communications', 'Voice Protocol Implementation', 'planned', 'ran', '', '', '{"note": "registered; workflow implementation pending"}'),
('voip-telephony-integration', 'Telephony & Communications', 'VoIP / Telephony Systems Integration', 'planned', 'ran', '', '', '{"note": "registered; workflow implementation pending"}'),
('automated-audio-transcription-logging', 'Telephony & Communications', 'Automated Audio Transcription & Logging', 'prototype', 'ran', 'artifact-ingest', 'scripts/lucidota_artifact_ingest.py ingest', '{"note": "ffprobe metadata active; local ASR integration pending model availability"}'),
('systems-architecture-design', 'Web, Architecture & Infrastructure Engineering', 'Systems Architecture Design', 'planned', 'planned', '', '', '{"source_matrix": "Krampus Holiday Pogroms & Associated Systems"}'),
('infrastructure-scaling-planning', 'Web, Architecture & Infrastructure Engineering', 'Infrastructure Expansion & Scaling Planning', 'planned', 'planned', '', '', '{"source_matrix": "Krampus Holiday Pogroms & Associated Systems"}'),
('web-deployment-domain-management', 'Web, Architecture & Infrastructure Engineering', 'Web Deployment & Domain/URL Management', 'planned', 'ran', '', '', '{"note": "registered; workflow implementation pending"}'),
('web-archiving-preservation', 'Web, Architecture & Infrastructure Engineering', 'Web Archiving & Digital Preservation', 'prototype', 'ran', 'body-capture-capture', 'scripts/lucidota_body_capture.py', '{"source_matrix": "Krampus Holiday Pogroms & Associated Systems"}'),
('system-configuration-settings-management', 'Web, Architecture & Infrastructure Engineering', 'System Configuration & Settings Management', 'active', 'ran', 'model-governor', 'scripts/lucidota_model_governor.py', '{"surfaces": ["environment", "runtime registry", "workflow registry"]}'),
('system-bootstrapping-initialization', 'Web, Architecture & Infrastructure Engineering', 'System Bootstrapping & Initialization [Lucidota]', 'active', 'ran', 'basic-workflows', 'scripts/lucidota_basic_workflows.py smoke', '{"surfaces": ["schema apply", "watcher startup", "smoke tests"]}'),
('routing-endpoint-management', 'Web, Architecture & Infrastructure Engineering', 'Routing & Endpoint Management', 'active', 'ran', 'clawd-go-runtime', 'scripts/lucidota_clawd_runtime.py', '{"surfaces": ["claw cli", "GO route", "workflow_event"]}'),
('technical-build-roadmapping', 'Operations & Project Management', 'Technical Build Roadmapping', 'planned', 'planned', '', '', '{"source_matrix": "Krampus Holiday Pogroms & Associated Systems"}'),
('technical-writing-documentation', 'Operations & Project Management', 'Technical Writing & Documentation [Whitepapers, Markdown]', 'prototype', 'planned_ran', '', '', '{"source_matrix": "Krampus Holiday Pogroms & Associated Systems"}'),
('ui-ux-web-design', 'Operations & Project Management', 'UI/UX Web Design [Splashpage]', 'planned', 'planned', '', '', '{"source_matrix": "Krampus Holiday Pogroms & Associated Systems"}'),
('collaborative-docs-strategy', 'Operations & Project Management', 'Collaborative Documentation & Strategy Planning', 'prototype', 'ran', '', '', '{"source_matrix": "Krampus Holiday Pogroms & Associated Systems"}'),
('rules-protocol-governance', 'Operations & Project Management', 'Rules / Protocol Establishment & Governance', 'active', 'ran', 'workflow-signoff', 'scripts/lucidota_absurd_signoff.py', '{"surfaces": ["workflow_registry", "governance_gate"]}'),
('operational-mechanics-process-mapping', 'Operations & Project Management', 'Operational Mechanics & Process Mapping', 'active', 'ran', 'basic-workflows', 'scripts/lucidota_basic_workflows.py audit', '{"note": "case shape + workflow event traces provide process map anchors"}'),
('interactive-data-viz-dashboard', 'Data Visualization & Analytics', 'Interactive Data Visualization & Dashboard Generation', 'planned', 'ran', '', '', '{"note": "capability registered; dashboard renderer/workflow pending"}'),
('network-cluster-centrality-viz', 'Data Visualization & Analytics', 'Network Cluster & Centrality Analysis Visualization', 'prototype', 'ran', 'artifact-pivot', 'scripts/lucidota_artifact_ingest.py pivot', '{"algorithms": ["degree", "ego", "connected_components"], "note": "networkx available"}'),
('wide-ripping-extraction', 'Advanced Extraction & Indexing', 'Mass Data Extraction / Wide Ripping Operations', 'planned', 'ran', '', '', '{"note": "must be source-policy governed; deterministic batching pending"}'),
('web-spidering-crawler-development', 'Advanced Extraction & Indexing', 'Automated Web Spidering & Crawler Development', 'prototype', 'ran', 'survey-protocol', 'scripts/lucidota_absurd_survey.py', '{"note": "operator/network policy applies"}'),
('database-export-indexing-bulk-management', 'Advanced Extraction & Indexing', 'Database Export Indexing & Bulk DB Management', 'planned', 'ran', '', '', '{"formats": ["csv", "jsonl", "parquet"], "note": "bulk import/export module pending"}'),
('metadata-taxonomy-tagging-substrate', 'Architecture & Systems Design', 'Metadata Taxonomy & Tagging Substrate Design', 'active', 'planned', 'artifact-ingest', 'lucidota_investigation.tag_taxonomy', '{"surface": "tag_taxonomy + artifact_tag"}'),
('screenshot-visual-ui-capture-testing', 'Architecture & Systems Design', 'Automated Screenshot / Visual UI Capture Testing', 'prototype', 'ran', 'body-capture-capture', 'scripts/lucidota_body_capture.py', '{"note": "screenshot browser harness pending; body capture exists"}'),
('developer-target-capture-methodologies', 'Operations, Tactics & Project Management', 'Developer/Target Capture Methodologies', 'planned', 'planned', '', '', '{"note": "playbook module pending", "policy": "lawful, consented, defensive, or public-interest investigations only"}'),
('interteam-knowledge-transfer-handoff', 'Operations, Tactics & Project Management', 'Inter-team Knowledge Transfer & Handoff Operations', 'planned', 'ran', '', '', '{"note": "case handoff pack/report generator pending"}'),
('scraper-fleet-audit-health', 'Data Operations & Fleet Management', 'Scraper Fleet Auditing & Health Monitoring', 'planned', 'ran', '', '', '{"note": "fleet table/health command pending"}'),
('open-data-portal-parsing', 'Data Operations & Fleet Management', 'Open Data Portal (ODP) Interaction & Parsing', 'planned', 'ran', '', '', '{"note": "CKAN/Socrata/ArcGIS parser wrappers pending"}'),
('electoral-data-post-election-tracking', 'Domain-Specific Intelligence & Analysis', 'Electoral Data Analysis & Post-Election Tracking', 'planned', 'ran', '', '', '{"note": "domain workflow pending; registered for data joins and timeline tracking"}'),
('regulatory-financial-enforcement-bcfsa', 'Domain-Specific Intelligence & Analysis', 'Regulatory/Financial Enforcement Tracking [BCFSA]', 'planned', 'ran', '', '', '{"note": "domain workflow pending; source-policy governed"}'),
('gap-analysis-cluster-reporting', 'Domain-Specific Intelligence & Analysis', 'Gap Analysis & Cluster Reporting', 'prototype', 'ran', 'artifact-pivot', 'scripts/lucidota_artifact_ingest.py pivot', '{"note": "local cluster/gap reporting planned from graph entities"}'),
('high-stakes-event-monitoring-election-day', 'Tactical & Real-Time Operations', 'High-Stakes Event Monitoring [Election Day Ops]', 'planned', 'ran', '', '', '{"note": "event monitor/playbook module pending"}'),
('investigative-playbook-standardization', 'Tactical & Real-Time Operations', 'Investigative Playbook Development & Standardization', 'planned', 'ran', '', '', '{"note": "case YAML + protocol registry seed this"}'),
('institutional-memory-structuring', 'Knowledge Management & Organization', 'Institutional Memory Retention & Structuring', 'prototype', 'ran', 'go-graph-search', 'scripts/lucidota_basic_workflows.py graph-search', '{"note": "GO graph + Indy chunks + case index"}'),
('kb-construction-indexing', 'Knowledge Management & Organization', 'Knowledge Base (KB) Construction & Indexing', 'prototype', 'ran', 'indy-reads-book-watch', 'scripts/lucidota_indy_reads_watcher.py', '{"note": "book/KBI path active; case KB indexing pending"}'),
('archival-indexing-organization', 'Knowledge Management & Organization', 'Archival Indexing & Organization', 'active', 'ran', 'artifact-ingest', 'scripts/lucidota_artifact_ingest.py ingest', '{"surfaces": ["CAS", "case folders", "sidecars"]}'),
('network-diagram-documentation', 'Knowledge Management & Organization', 'Network Diagram Documentation', 'planned', 'ran', '', '', '{"note": "Mermaid/Graphviz report generator pending"}'),
('automated-pdf-rendering-doc-generation', 'Data Generation & Reporting', 'Automated PDF Rendering & Document Generation', 'planned', 'ran', '', '', '{"note": "availability check + renderer pending", "tools": ["pandoc?", "weasyprint?", "wkhtmltopdf?"]}'),
('targeted-individual-profiling-deep-intel', 'Intelligence, Reconnaissance & Profiling', 'Targeted Individual Profiling & Deep Intel Gathering', 'planned', 'ran', '', '', '{"note": "requires source policy, redaction, and review gates", "policy": "lawful, consented, defensive, or public-interest investigations only"}'),
('political-crossref-affiliation-analysis', 'Intelligence, Reconnaissance & Profiling', 'Political Data Cross-Referencing & Affiliation Analysis', 'planned', 'ran', '', '', '{"note": "domain joins + provenance-first crossrefs pending"}'),
('forensic-evidence-synthesis-deep-analysis', 'Intelligence, Reconnaissance & Profiling', 'Forensic Evidence Synthesis & Deep Analysis', 'prototype', 'ran', 'artifact-ingest', 'scripts/lucidota_artifact_ingest.py ingest', '{"note": "artifact extraction active; synthesis report pending"}'),
('red-string-connection-mapping', 'Network & Relationship Analysis', 'Complex Relationship / Red String Connection Mapping', 'prototype', 'ran', 'artifact-pivot', 'scripts/lucidota_artifact_ingest.py pivot', '{"algorithms": ["ego graph", "shared entities", "connected components"]}'),
('ally-coalition-network-tracking', 'Network & Relationship Analysis', 'Ally & Coalition Network Tracking', 'planned', 'ran', '', '', '{"note": "relationship taxonomy + reports pending"}'),
('wasp-swarm-architecture-management', 'Architecture & Swarm Operations', 'Distributed Swarm Architecture Design & Management [Lucidota Wasp Swarm]', 'planned', 'planned_ran', '', '', '{"note": "deterministic worker/fleet orchestration, not autonomous agent sprawl"}'),
('scraper-fleet-execution-orchestration', 'Architecture & Swarm Operations', 'Automated Scraper Fleet Execution & Task Orchestration', 'planned', 'ran', '', '', '{"note": "source-policy gated fleet runner pending"}'),
('scalpel-methodology-strategy', 'Tactics, Strategy & Evaluation', 'Precision / Targeted Operational Strategy Development [Scalpel Methodology]', 'planned', 'planned_ran', '', '', '{"policy": "resource prioritization and harm-minimizing investigative planning only"}'),
('impact-kpi-metrics-tracking', 'Tactics, Strategy & Evaluation', 'Operational Impact Measurement & KPI Metrics Tracking', 'planned', 'ran', '', '', '{"note": "workflow/case metrics aggregator pending"}'),
('korpus-krampii-mass-ingestion', 'Knowledge Management & Organization', 'KORPUS KRAMPII Mass Ingestion / Sprawl Reclamation', 'active', 'ran', 'korpus-krampii-mass-ingest', 'scripts/korpus_krampii.py ingest', '{"principle": "store now, parse later; no LLM thinking"}'),
('store-now-parse-later-deferred', 'Data Engineering, Processing & Analysis', 'Store-Now Parse-Later Deferred Artifact Handling', 'active', 'ran', 'korpus-krampii-deferred', 'scripts/korpus_krampii.py deferred', '{"status": "deferred files are CAS locked and queryable"}'),
('riverml-streaming-decisions', 'Data Engineering, Processing & Analysis', 'RiverML Streaming Signal / Anomaly Decision Engine', 'active', 'ran', 'korpus-krampii-mass-ingest', 'lucidota_korpus.river_decision', '{"library": "river", "mode": "online per-component stats"}'),
('vector-space-thought-evolution', 'Knowledge Management & Organization', 'Vector-Space Thought Evolution / Semantic Diffing', 'prototype', 'ran', 'korpus-krampii-embed-pending', 'scripts/korpus_krampii.py embed-pending', '{"embedding_model": "ckdog1.kernel.hash_quantized_embedding.v1", "dimension": 384}')
ON CONFLICT (capability_key) DO UPDATE SET
    capability_group=EXCLUDED.capability_group,
    capability_name=EXCLUDED.capability_name,
    lifecycle_status=EXCLUDED.lifecycle_status,
    run_state=EXCLUDED.run_state,
    workflow_name=EXCLUDED.workflow_name,
    command=EXCLUDED.command,
    detail=EXCLUDED.detail,
    updated_at=now();

CREATE TABLE IF NOT EXISTS lucidota_investigation.artifact (
    artifact_uuid uuid PRIMARY KEY DEFAULT lucidota_investigation.uuid_v7(),
    sha256 text UNIQUE NOT NULL CHECK (sha256 ~ '^[0-9a-f]{64}$'),
    cas_uri text UNIQUE NOT NULL,
    locked_relative_path text NOT NULL,
    original_path text NOT NULL DEFAULT '',
    original_name text NOT NULL DEFAULT '',
    mime text NOT NULL DEFAULT '',
    file_kind text NOT NULL DEFAULT 'unknown' CHECK (file_kind IN (
        'image',
        'document',
        'video',
        'audio',
        'archive',
        'text',
        'binary',
        'unknown'
    )),
    size_bytes bigint NOT NULL CHECK (size_bytes >= 0),
    title text NOT NULL DEFAULT '',
    evidence_date timestamptz,
    evidence_date_source text NOT NULL DEFAULT '',
    exif_json jsonb NOT NULL DEFAULT '{}'::jsonb,
    metadata_json jsonb NOT NULL DEFAULT '{}'::jsonb,
    normalized_text text NOT NULL DEFAULT '',
    analysis_json jsonb NOT NULL DEFAULT '{}'::jsonb,
    graph_item_uuid uuid,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS artifact_kind_date_idx
    ON lucidota_investigation.artifact(file_kind, evidence_date DESC NULLS LAST);

CREATE TABLE IF NOT EXISTS lucidota_investigation.case_artifact (
    case_artifact_uuid uuid PRIMARY KEY DEFAULT lucidota_investigation.uuid_v7(),
    case_uuid uuid REFERENCES lucidota_investigation.case_file(case_uuid) ON DELETE CASCADE,
    artifact_uuid uuid NOT NULL REFERENCES lucidota_investigation.artifact(artifact_uuid) ON DELETE CASCADE,
    evidence_id text UNIQUE NOT NULL,
    artifact_title text NOT NULL DEFAULT '',
    evidence_date timestamptz,
    evidence_date_source text NOT NULL DEFAULT '',
    sidecar_relative_dir text NOT NULL DEFAULT '',
    custody jsonb NOT NULL DEFAULT '{}'::jsonb,
    status text NOT NULL DEFAULT 'locked' CHECK (status IN (
        'locked',
        'indexed',
        'needs_review',
        'superseded',
        'archived'
    )),
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE(case_uuid, artifact_uuid)
);

CREATE INDEX IF NOT EXISTS case_artifact_case_idx
    ON lucidota_investigation.case_artifact(case_uuid, created_at DESC);

CREATE TABLE IF NOT EXISTS lucidota_investigation.tag_taxonomy (
    tag_key text PRIMARY KEY,
    tag_group text NOT NULL,
    label text NOT NULL,
    go_term text NOT NULL DEFAULT '',
    status text NOT NULL DEFAULT 'active' CHECK (status IN ('active','planned','deprecated')),
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

INSERT INTO lucidota_investigation.tag_taxonomy
(tag_key, tag_group, label, go_term, status, detail)
VALUES
('artifact:image', 'artifact_kind', 'Image Artifact', 'EVIDENCE', 'active', '{}'),
('artifact:document', 'artifact_kind', 'Document Artifact', 'EVIDENCE', 'active', '{}'),
('artifact:video', 'artifact_kind', 'Video Artifact', 'EVIDENCE', 'active', '{}'),
('artifact:audio', 'artifact_kind', 'Audio Artifact', 'EVIDENCE', 'active', '{}'),
('artifact:text', 'artifact_kind', 'Text Artifact', 'EVIDENCE', 'active', '{}'),
('artifact:binary', 'artifact_kind', 'Binary Artifact', 'EVIDENCE', 'active', '{}'),
('processing:exif', 'processing', 'EXIF / Metadata Extracted', 'ATTRIBUTE', 'active', '{}'),
('processing:ocr', 'processing', 'OCR Applied', 'ALGORITHM', 'active', '{}'),
('processing:document_text', 'processing', 'Document Text Extracted', 'ALGORITHM', 'active', '{}'),
('processing:ffprobe', 'processing', 'Audio/Video Metadata Probed', 'ALGORITHM', 'active', '{}'),
('processing:entity_extract', 'processing', 'Entities Extracted', 'ALGORITHM', 'active', '{}'),
('processing:ast', 'processing', 'AST Parsed', 'ALGORITHM', 'active', '{}'),
('processing:algorithm', 'processing', 'Algorithm / Function Extracted', 'ALGORITHM', 'active', '{}'),
('entity:name', 'entity_kind', 'Name Entity', 'ENTITY', 'active', '{}'),
('entity:alias', 'entity_kind', 'Alias Entity', 'ENTITY', 'active', '{}'),
('entity:phone', 'entity_kind', 'Phone Entity', 'ATOMIC_ID', 'active', '{}'),
('entity:ip', 'entity_kind', 'IP Address Entity', 'ATOMIC_ID', 'active', '{}'),
('entity:email', 'entity_kind', 'Email Entity', 'ATOMIC_ID', 'active', '{}'),
('entity:url', 'entity_kind', 'URL Entity', 'ATOMIC_ID', 'active', '{}'),
('entity:domain', 'entity_kind', 'Domain Entity', 'ATOMIC_ID', 'active', '{}'),
('entity:address', 'entity_kind', 'Address Entity', 'ENTITY', 'active', '{}'),
('entity:date', 'entity_kind', 'Date Entity', 'TIME', 'active', '{}'),
('entity:identifier', 'entity_kind', 'Identifier Entity', 'ATOMIC_ID', 'active', '{}'),
('component:file', 'component_kind', 'KORPUS File Component', 'EVIDENCE', 'active', '{}'),
('component:markdown_section', 'component_kind', 'Markdown Section Component', 'TERM', 'active', '{}'),
('component:markdown_code_block', 'component_kind', 'Markdown Code Block Child Artifact', 'TOOL', 'active', '{}'),
('component:python_function', 'component_kind', 'Python Function / Algorithm Component', 'ALGORITHM', 'active', '{}'),
('component:python_class', 'component_kind', 'Python Class Component', 'ENTITY', 'active', '{}'),
('component:python_method', 'component_kind', 'Python Method / Algorithm Component', 'ALGORITHM', 'active', '{}'),
('component:code_symbol', 'component_kind', 'Code Symbol Component', 'ALGORITHM', 'active', '{}'),
('component:code_chunk', 'component_kind', 'Code Chunk Component', 'TOOL', 'active', '{}'),
('component:text_chunk', 'component_kind', 'Text Chunk Component', 'COMMENT', 'active', '{}'),
('case:associated', 'case_state', 'Artifact Associated To Case', 'RELATIONSHIP', 'active', '{}'),
('case:unassigned', 'case_state', 'Artifact Not Yet Associated To Case', 'COMMENT', 'active', '{}')
ON CONFLICT (tag_key) DO UPDATE SET
    tag_group=EXCLUDED.tag_group,
    label=EXCLUDED.label,
    go_term=EXCLUDED.go_term,
    status=EXCLUDED.status,
    detail=EXCLUDED.detail,
    updated_at=now();

CREATE TABLE IF NOT EXISTS lucidota_investigation.artifact_tag (
    artifact_tag_uuid uuid PRIMARY KEY DEFAULT lucidota_investigation.uuid_v7(),
    artifact_uuid uuid NOT NULL REFERENCES lucidota_investigation.artifact(artifact_uuid) ON DELETE CASCADE,
    case_uuid uuid REFERENCES lucidota_investigation.case_file(case_uuid) ON DELETE CASCADE,
    tag_key text NOT NULL REFERENCES lucidota_investigation.tag_taxonomy(tag_key),
    value text NOT NULL DEFAULT '',
    confidence_bps integer NOT NULL DEFAULT 50 CHECK (confidence_bps BETWEEN 0 AND 10000),
    source text NOT NULL DEFAULT '',
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE(artifact_uuid, case_uuid, tag_key, value, source)
);

CREATE INDEX IF NOT EXISTS artifact_tag_case_idx
    ON lucidota_investigation.artifact_tag(case_uuid, tag_key, created_at DESC);

CREATE TABLE IF NOT EXISTS lucidota_investigation.artifact_sidecar (
    sidecar_uuid uuid PRIMARY KEY DEFAULT lucidota_investigation.uuid_v7(),
    artifact_uuid uuid NOT NULL REFERENCES lucidota_investigation.artifact(artifact_uuid) ON DELETE CASCADE,
    case_uuid uuid REFERENCES lucidota_investigation.case_file(case_uuid) ON DELETE CASCADE,
    sidecar_kind text NOT NULL CHECK (sidecar_kind IN (
        'casefile',
        'normalized_json',
        'normalized_yaml',
        'entities_jsonl',
        'custody_jsonl',
        'pivot_jsonl',
        'text_extract',
        'frame_ocr',
        'other'
    )),
    relative_path text NOT NULL,
    sha256 text NOT NULL CHECK (sha256 ~ '^[0-9a-f]{64}$'),
    mime text NOT NULL DEFAULT '',
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE(artifact_uuid, case_uuid, sidecar_kind, relative_path)
);

CREATE TABLE IF NOT EXISTS lucidota_investigation.artifact_entity (
    entity_uuid uuid PRIMARY KEY DEFAULT lucidota_investigation.uuid_v7(),
    artifact_uuid uuid REFERENCES lucidota_investigation.artifact(artifact_uuid) ON DELETE CASCADE,
    case_uuid uuid REFERENCES lucidota_investigation.case_file(case_uuid) ON DELETE CASCADE,
    entity_kind text NOT NULL CHECK (entity_kind IN (
        'name',
        'alias',
        'phone',
        'ip',
        'email',
        'url',
        'domain',
        'address',
        'date',
        'money',
        'identifier',
        'hash',
        'organization',
        'location',
        'other'
    )),
    value text NOT NULL,
    normalized_value text NOT NULL,
    confidence_bps integer NOT NULL DEFAULT 50 CHECK (confidence_bps BETWEEN 0 AND 10000),
    source text NOT NULL DEFAULT '',
    context text NOT NULL DEFAULT '',
    graph_item_uuid uuid,
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS artifact_entity_norm_idx
    ON lucidota_investigation.artifact_entity(entity_kind, normalized_value, created_at DESC);

CREATE INDEX IF NOT EXISTS artifact_entity_case_idx
    ON lucidota_investigation.artifact_entity(case_uuid, entity_kind, normalized_value);

CREATE UNIQUE INDEX IF NOT EXISTS artifact_entity_dedupe_idx
    ON lucidota_investigation.artifact_entity(
        artifact_uuid,
        coalesce(case_uuid, '00000000-0000-0000-0000-000000000000'::uuid),
        entity_kind,
        normalized_value,
        source
    );

CREATE TABLE IF NOT EXISTS lucidota_investigation.pivot_job (
    job_uuid uuid PRIMARY KEY DEFAULT lucidota_investigation.uuid_v7(),
    case_uuid uuid REFERENCES lucidota_investigation.case_file(case_uuid) ON DELETE SET NULL,
    seed_kind text NOT NULL DEFAULT 'auto',
    seed_value text NOT NULL DEFAULT '',
    status text NOT NULL DEFAULT 'queued' CHECK (status IN (
        'queued',
        'running',
        'succeeded',
        'failed'
    )),
    depth integer NOT NULL DEFAULT 1,
    local_only boolean NOT NULL DEFAULT true,
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    finished_at timestamptz
);

CREATE INDEX IF NOT EXISTS pivot_job_case_created_idx
    ON lucidota_investigation.pivot_job(case_uuid, created_at DESC);

CREATE TABLE IF NOT EXISTS lucidota_investigation.pivot_candidate (
    candidate_uuid uuid PRIMARY KEY DEFAULT lucidota_investigation.uuid_v7(),
    job_uuid uuid NOT NULL REFERENCES lucidota_investigation.pivot_job(job_uuid) ON DELETE CASCADE,
    source_entity_uuid uuid REFERENCES lucidota_investigation.artifact_entity(entity_uuid) ON DELETE SET NULL,
    candidate_kind text NOT NULL DEFAULT '',
    value text NOT NULL,
    normalized_value text NOT NULL,
    score integer NOT NULL DEFAULT 0,
    reason text NOT NULL DEFAULT '',
    status text NOT NULL DEFAULT 'candidate' CHECK (status IN (
        'candidate',
        'queued',
        'promoted',
        'rejected',
        'seen'
    )),
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE(job_uuid, candidate_kind, normalized_value)
);

CREATE INDEX IF NOT EXISTS pivot_candidate_job_score_idx
    ON lucidota_investigation.pivot_candidate(job_uuid, score DESC, created_at DESC);

-- Operational hardening: UUIDv7 defaults, range confidence, and updated_at triggers.
ALTER TABLE lucidota_investigation.case_file ALTER COLUMN case_uuid SET DEFAULT lucidota_investigation.uuid_v7();
ALTER TABLE lucidota_investigation.artifact ALTER COLUMN artifact_uuid SET DEFAULT lucidota_investigation.uuid_v7();
ALTER TABLE lucidota_investigation.case_artifact ALTER COLUMN case_artifact_uuid SET DEFAULT lucidota_investigation.uuid_v7();
ALTER TABLE lucidota_investigation.artifact_sidecar ALTER COLUMN sidecar_uuid SET DEFAULT lucidota_investigation.uuid_v7();
ALTER TABLE lucidota_investigation.artifact_entity ALTER COLUMN entity_uuid SET DEFAULT lucidota_investigation.uuid_v7();
ALTER TABLE lucidota_investigation.pivot_job ALTER COLUMN job_uuid SET DEFAULT lucidota_investigation.uuid_v7();
ALTER TABLE lucidota_investigation.pivot_candidate ALTER COLUMN candidate_uuid SET DEFAULT lucidota_investigation.uuid_v7();
ALTER TABLE lucidota_investigation.artifact_tag ALTER COLUMN artifact_tag_uuid SET DEFAULT lucidota_investigation.uuid_v7();

ALTER TABLE lucidota_investigation.artifact_entity DROP CONSTRAINT IF EXISTS artifact_entity_confidence_bps_check;
ALTER TABLE lucidota_investigation.artifact_entity ADD CONSTRAINT artifact_entity_confidence_bps_check CHECK (confidence_bps BETWEEN 0 AND 10000);
ALTER TABLE lucidota_investigation.artifact_tag DROP CONSTRAINT IF EXISTS artifact_tag_confidence_bps_check;
ALTER TABLE lucidota_investigation.artifact_tag ADD CONSTRAINT artifact_tag_confidence_bps_check CHECK (confidence_bps BETWEEN 0 AND 10000);

CREATE OR REPLACE FUNCTION lucidota_investigation.touch_updated_at()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS touch_case_file_updated_at ON lucidota_investigation.case_file;
CREATE TRIGGER touch_case_file_updated_at BEFORE UPDATE ON lucidota_investigation.case_file
FOR EACH ROW EXECUTE FUNCTION lucidota_investigation.touch_updated_at();

DROP TRIGGER IF EXISTS touch_capability_registry_updated_at ON lucidota_investigation.capability_registry;
CREATE TRIGGER touch_capability_registry_updated_at BEFORE UPDATE ON lucidota_investigation.capability_registry
FOR EACH ROW EXECUTE FUNCTION lucidota_investigation.touch_updated_at();

DROP TRIGGER IF EXISTS touch_artifact_updated_at ON lucidota_investigation.artifact;
CREATE TRIGGER touch_artifact_updated_at BEFORE UPDATE ON lucidota_investigation.artifact
FOR EACH ROW EXECUTE FUNCTION lucidota_investigation.touch_updated_at();

DROP TRIGGER IF EXISTS touch_case_artifact_updated_at ON lucidota_investigation.case_artifact;
CREATE TRIGGER touch_case_artifact_updated_at BEFORE UPDATE ON lucidota_investigation.case_artifact
FOR EACH ROW EXECUTE FUNCTION lucidota_investigation.touch_updated_at();

DROP TRIGGER IF EXISTS touch_tag_taxonomy_updated_at ON lucidota_investigation.tag_taxonomy;
CREATE TRIGGER touch_tag_taxonomy_updated_at BEFORE UPDATE ON lucidota_investigation.tag_taxonomy
FOR EACH ROW EXECUTE FUNCTION lucidota_investigation.touch_updated_at();
