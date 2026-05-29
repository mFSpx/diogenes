-- Bytewax mini-stream proof: live dataflow outputs durable bounded hints.

CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE SCHEMA IF NOT EXISTS lucidota_learning;

CREATE TABLE IF NOT EXISTS lucidota_learning.bytewax_stream_run (
    run_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    status text NOT NULL CHECK (status IN ('succeeded','failed')),
    events_in integer NOT NULL DEFAULT 0,
    hints_out integer NOT NULL DEFAULT 0,
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS lucidota_learning.bytewax_hint (
    hint_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    source text NOT NULL,
    phase text NOT NULL,
    status text NOT NULL,
    hint text NOT NULL,
    score integer NOT NULL DEFAULT 0,
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS lucidota_learning.bytewax_abductive_event (
    event_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    source text NOT NULL,
    source_ref text NOT NULL,
    event_time timestamptz NOT NULL DEFAULT now(),
    text_surface text NOT NULL DEFAULT '',
    ontology_terms jsonb NOT NULL DEFAULT '[]'::jsonb,
    epistemic_flag text NOT NULL CHECK (epistemic_flag IN ('FACT','PROBABLE','POSSIBLE','BULLSHIT','SURE_MAYBE')),
    injection_flag boolean NOT NULL DEFAULT false,
    compressed_activity jsonb NOT NULL DEFAULT '{}'::jsonb,
    certainty_trace jsonb NOT NULL DEFAULT '{}'::jsonb,
    payload jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE(source, source_ref)
);

CREATE TABLE IF NOT EXISTS lucidota_learning.bytewax_abductive_hint (
    hint_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    event_uuid uuid REFERENCES lucidota_learning.bytewax_abductive_event(event_uuid) ON DELETE SET NULL,
    source text NOT NULL,
    source_ref text NOT NULL,
    epistemic_flag text NOT NULL CHECK (epistemic_flag IN ('FACT','PROBABLE','POSSIBLE','BULLSHIT','SURE_MAYBE')),
    hypothesis text NOT NULL,
    support_score double precision NOT NULL DEFAULT 0.0,
    contradiction_score double precision NOT NULL DEFAULT 0.0,
    centrality_score double precision NOT NULL DEFAULT 0.0,
    injection_flag boolean NOT NULL DEFAULT false,
    ontology_terms jsonb NOT NULL DEFAULT '[]'::jsonb,
    certainty_trace jsonb NOT NULL DEFAULT '{}'::jsonb,
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE(source, source_ref, hypothesis)
);

CREATE TABLE IF NOT EXISTS lucidota_learning.bytewax_replication_receipt (
    receipt_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    slot_name text NOT NULL,
    plugin text NOT NULL,
    status text NOT NULL,
    events_seen integer NOT NULL DEFAULT 0,
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);
