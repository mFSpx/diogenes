CREATE TABLE IF NOT EXISTS lucidota_learning.evolution_gate_run (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    run_at timestamptz DEFAULT now(),
    fire boolean,
    reason text,
    batch_stats jsonb,
    events_evaluated int,
    dead_letter_rate numeric,
    success_rate numeric
);

CREATE TABLE IF NOT EXISTS lucidota_learning.evolution_drain_run (
    id uuid PRIMARY KEY,
    started_at timestamptz,
    completed_at timestamptz,
    status text CHECK (status IN ('ok','invariant_fail','groq_error','skip')),
    facts_updated int DEFAULT 0,
    invariant_checks jsonb,
    receipt_path text,
    detail jsonb
);

CREATE TABLE IF NOT EXISTS lucidota_learning.evolution_version (
    version_num serial,
    parent_version int,
    created_at timestamptz DEFAULT now(),
    changes jsonb NOT NULL,
    metrics_snapshot jsonb,
    evolution_drain_id uuid REFERENCES lucidota_learning.evolution_drain_run(id)
);

COMMENT ON SCHEMA lucidota_learning IS 'Schema for storing evolution data'; 
COMMENT ON TABLE lucidota_learning.evolution_gate_run IS 'Table for storing evolution gate runs';
COMMENT ON TABLE lucidota_learning.evolution_drain_run IS 'Table for storing evolution drain runs';
COMMENT ON TABLE lucidota_learning.evolution_version IS 'Table for storing evolution versions';
