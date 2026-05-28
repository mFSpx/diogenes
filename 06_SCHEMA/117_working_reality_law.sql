CREATE SCHEMA IF NOT EXISTS lucidota_working_reality;

CREATE TABLE IF NOT EXISTS lucidota_working_reality.working_reality_move (
    move_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    evidence_refs text[] NOT NULL DEFAULT ARRAY[]::text[],
    hypothesis text NOT NULL,
    working_reality text NOT NULL,
    move text NOT NULL,
    result text NOT NULL CHECK (result IN ('PASS','FAIL','CONFLICT','OPEN','STALE','SUPERSEDED')),
    record_for_future boolean NOT NULL DEFAULT true,
    contradiction_refs text[] NOT NULL DEFAULT ARRAY[]::text[],
    rejected_hypotheses jsonb NOT NULL DEFAULT '[]'::jsonb,
    layer_snapshot jsonb NOT NULL DEFAULT '{}'::jsonb,
    receipt_path text NOT NULL DEFAULT '',
    canonical_graph_writes_performed boolean NOT NULL DEFAULT false,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS working_reality_move_result_idx
    ON lucidota_working_reality.working_reality_move (result, created_at DESC);

CREATE INDEX IF NOT EXISTS working_reality_move_record_future_idx
    ON lucidota_working_reality.working_reality_move (record_for_future, created_at DESC);
