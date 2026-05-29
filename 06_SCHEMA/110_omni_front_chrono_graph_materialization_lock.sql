-- LUCIDOTA omni-front graph materialization lock.
-- Idempotency table only; temporal_claim remains append-only.
CREATE SCHEMA IF NOT EXISTS lucidota_go;

CREATE TABLE IF NOT EXISTS lucidota_go.chrono_graph_materialization_lock (
    promotion_uuid uuid PRIMARY KEY REFERENCES lucidota_go.graph_promotion_candidate(promotion_uuid),
    source_claim_uuid uuid NOT NULL REFERENCES lucidota_korpus.temporal_claim(claim_uuid),
    file_uuid uuid,
    graph_item_uuid uuid NOT NULL REFERENCES lucidota_go.graph_item(uuid),
    graph_journal_uuid uuid REFERENCES lucidota_go.graph_journal(journal_uuid),
    graph_event_type text NOT NULL,
    chrono_lane text NOT NULL,
    timestamp timestamptz NOT NULL,
    materialized_by text NOT NULL DEFAULT 'scripts/lucidota_omni_front_sprint_orchestrator.py',
    materialized_at timestamptz NOT NULL DEFAULT now(),
    detail jsonb NOT NULL DEFAULT '{}'::jsonb
);
CREATE INDEX IF NOT EXISTS idx_chrono_graph_materialization_lock_claim ON lucidota_go.chrono_graph_materialization_lock(source_claim_uuid);
CREATE INDEX IF NOT EXISTS idx_chrono_graph_materialization_lock_file ON lucidota_go.chrono_graph_materialization_lock(file_uuid);
COMMENT ON TABLE lucidota_go.chrono_graph_materialization_lock IS 'Idempotent RFC-CHRONO-001 materialization ledger. Every row cites source_claim_uuid and file_uuid/runtime exception.';
