-- LUCIDOTA Malkovich Siphon shape watcher schema.
-- Tracks shape vector streams, fidelity drift, and cross-lane correlations.
-- Plane 2 (River ML) watches fidelity drift per source/lane.
-- Plane 3 (Bytewax) watches cross-lane shape correlations.

CREATE SCHEMA IF NOT EXISTS lucidota_learning;

-- Shape vector observations — one row per batch of shapes ingested.
CREATE TABLE IF NOT EXISTS lucidota_learning.shape_observation (
    observation_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    source text NOT NULL,                  -- e.g. 'AhoyStrategy', 'IndyReads:fiction_thriller'
    lane text NOT NULL DEFAULT '',         -- e.g. 'authority_vs_insurgency', 'nonfiction_biography'
    batch_size integer NOT NULL DEFAULT 0,
    avg_fidelity double precision,
    min_fidelity double precision,
    max_fidelity double precision,
    collision_count integer NOT NULL DEFAULT 0,
    collision_rate double precision,
    shape_mean double precision[],        -- average shape vector for this batch
    shape_variance double precision[],    -- per-dimension variance
    dim_count integer NOT NULL DEFAULT 64,
    ingested_at timestamptz NOT NULL DEFAULT now(),
    detail jsonb NOT NULL DEFAULT '{}'::jsonb
);

-- Fidelity drift events — when a source/lane's fidelity shifts significantly.
CREATE TABLE IF NOT EXISTS lucidota_learning.shape_drift_event (
    drift_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    source text NOT NULL,
    lane text NOT NULL DEFAULT '',
    drift_kind text NOT NULL CHECK (drift_kind IN ('fidelity_drop', 'fidelity_spike', 'shape_shift', 'collision_surge')),
    prior_fidelity double precision,
    current_fidelity double precision,
    delta double precision,
    prior_observation_id uuid REFERENCES lucidota_learning.shape_observation(observation_id),
    current_observation_id uuid REFERENCES lucidota_learning.shape_observation(observation_id),
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

-- Cross-lane shape correlations — Bytewax stateful join artifacts.
CREATE TABLE IF NOT EXISTS lucidota_learning.shape_cross_lane_correlation (
    correlation_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    source_a text NOT NULL,
    lane_a text NOT NULL,
    source_b text NOT NULL,
    lane_b text NOT NULL,
    cosine_similarity double precision,
    euclidean_distance double precision,
    sample_count_a integer NOT NULL DEFAULT 0,
    sample_count_b integer NOT NULL DEFAULT 0,
    window_start timestamptz,
    window_end timestamptz,
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (source_a, lane_a, source_b, lane_b, window_start)
);

-- Shape watcher run receipt.
CREATE TABLE IF NOT EXISTS lucidota_learning.shape_watcher_run (
    run_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    status text NOT NULL CHECK (status IN ('succeeded', 'failed', 'partial')),
    mode text NOT NULL CHECK (mode IN ('oneshot', 'daemon')),
    sources_scanned integer NOT NULL DEFAULT 0,
    batches_ingested integer NOT NULL DEFAULT 0,
    drift_events_emitted integer NOT NULL DEFAULT 0,
    cross_lane_pairs_computed integer NOT NULL DEFAULT 0,
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);
