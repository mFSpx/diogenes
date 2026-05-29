-- LUCIDOTA hard-math telemetry: LSM, state transitions, stylometry, semantic isolation.
-- These are document/communication metrics, not diagnoses.

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_hardmath;

CREATE OR REPLACE FUNCTION lucidota_hardmath.uuid_v7()
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
        substr(ts_hex, 1, 8) || '-' || substr(ts_hex, 9, 4) || '-' ||
        '7' || substr(rnd, 1, 3) || '-' || variant || substr(rnd, 4, 3) || '-' || substr(rnd, 7, 12)
    )::uuid;
END;
$$;

CREATE TABLE IF NOT EXISTS lucidota_hardmath.analysis_run (
    run_uuid uuid PRIMARY KEY DEFAULT lucidota_hardmath.uuid_v7(),
    run_kind text NOT NULL,
    status text NOT NULL CHECK (status IN ('running','succeeded','failed','partial')) DEFAULT 'running',
    started_at timestamptz NOT NULL DEFAULT now(),
    finished_at timestamptz,
    detail jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS lucidota_hardmath.lsm_pair (
    lsm_uuid uuid PRIMARY KEY DEFAULT lucidota_hardmath.uuid_v7(),
    run_uuid uuid REFERENCES lucidota_hardmath.analysis_run(run_uuid) ON DELETE SET NULL,
    provider text NOT NULL DEFAULT '',
    conversation_uuid uuid,
    conversation_title text NOT NULL DEFAULT '',
    speaker_a text NOT NULL DEFAULT '',
    speaker_b text NOT NULL DEFAULT '',
    turn_pairs integer NOT NULL DEFAULT 0,
    lsm_score double precision NOT NULL DEFAULT 0,
    function_word_detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    first_at timestamptz,
    last_at timestamptz,
    caveat text NOT NULL DEFAULT 'LSM is style alignment, not proof of dominance/submission.',
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS lsm_pair_score_idx ON lucidota_hardmath.lsm_pair(lsm_score DESC, turn_pairs DESC);

CREATE TABLE IF NOT EXISTS lucidota_hardmath.state_transition (
    transition_uuid uuid PRIMARY KEY DEFAULT lucidota_hardmath.uuid_v7(),
    run_uuid uuid REFERENCES lucidota_hardmath.analysis_run(run_uuid) ON DELETE SET NULL,
    model_kind text NOT NULL DEFAULT 'observed_markov_hmm_proxy',
    from_state text NOT NULL,
    to_state text NOT NULL,
    transition_count integer NOT NULL DEFAULT 0,
    transition_probability double precision NOT NULL DEFAULT 0,
    avg_gap_seconds double precision,
    stress_adjusted_probability double precision,
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS state_transition_prob_idx ON lucidota_hardmath.state_transition(transition_probability DESC, transition_count DESC);

CREATE TABLE IF NOT EXISTS lucidota_hardmath.stylometry_attribution (
    attribution_uuid uuid PRIMARY KEY DEFAULT lucidota_hardmath.uuid_v7(),
    run_uuid uuid REFERENCES lucidota_hardmath.analysis_run(run_uuid) ON DELETE SET NULL,
    source_kind text NOT NULL DEFAULT 'korpus_component',
    source_uuid uuid NOT NULL,
    occurred_at timestamptz,
    observed_label text NOT NULL DEFAULT '',
    predicted_label text NOT NULL DEFAULT '',
    confidence double precision NOT NULL DEFAULT 0,
    top_scores jsonb NOT NULL DEFAULT '{}'::jsonb,
    feature_detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    excerpt text NOT NULL DEFAULT '',
    caveat text NOT NULL DEFAULT 'Stylometry is attribution signal, not identity proof.',
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE(run_uuid, source_kind, source_uuid)
);

CREATE INDEX IF NOT EXISTS stylometry_pred_idx ON lucidota_hardmath.stylometry_attribution(predicted_label, confidence DESC);

CREATE TABLE IF NOT EXISTS lucidota_hardmath.semantic_outlier (
    outlier_uuid uuid PRIMARY KEY DEFAULT lucidota_hardmath.uuid_v7(),
    run_uuid uuid REFERENCES lucidota_hardmath.analysis_run(run_uuid) ON DELETE SET NULL,
    component_uuid uuid NOT NULL,
    file_uuid uuid,
    occurred_at timestamptz,
    mahalanobis_distance double precision NOT NULL,
    z_rank integer NOT NULL DEFAULT 0,
    title text NOT NULL DEFAULT '',
    path text NOT NULL DEFAULT '',
    excerpt text NOT NULL DEFAULT '',
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE(run_uuid, component_uuid)
);

CREATE INDEX IF NOT EXISTS semantic_outlier_distance_idx ON lucidota_hardmath.semantic_outlier(mahalanobis_distance DESC, z_rank ASC);
