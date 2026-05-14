-- Survey / hop-pivot intake schema for LUCIDOTA.
-- Light, local, boring tables: durable surface for Clawd, DBOS, Bytewax, and Big Board.

CREATE SCHEMA IF NOT EXISTS lucidota_survey;

CREATE TABLE IF NOT EXISTS lucidota_survey.artifact (
    artifact_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    cas_uri text UNIQUE NOT NULL,
    sha256 text UNIQUE NOT NULL,
    size_bytes bigint NOT NULL CHECK (size_bytes >= 0),
    mime text NOT NULL DEFAULT '',
    source_url text,
    storage_status text NOT NULL DEFAULT 'local_cas' CHECK (storage_status IN (
        'local_cas',
        'metadata_only',
        'rejected'
    )),
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS lucidota_survey.survey_observation (
    observation_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    target text NOT NULL,
    method text NOT NULL CHECK (method IN ('HEAD', 'GET', 'FILE')),
    status_code integer,
    final_url text,
    content_length bigint,
    mime text NOT NULL DEFAULT '',
    etag text NOT NULL DEFAULT '',
    last_modified text NOT NULL DEFAULT '',
    sha256 text,
    artifact_id uuid REFERENCES lucidota_survey.artifact(artifact_id),
    survey_decision text NOT NULL CHECK (survey_decision IN (
        'metadata_only',
        'stored_artifact',
        'too_large',
        'unsupported',
        'failed'
    )),
    keyword_hits jsonb NOT NULL DEFAULT '[]'::jsonb,
    structure jsonb NOT NULL DEFAULT '{}'::jsonb,
    wayback jsonb NOT NULL DEFAULT '{}'::jsonb,
    notes text NOT NULL DEFAULT '',
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS survey_observation_target_idx
    ON lucidota_survey.survey_observation (target, created_at DESC);

CREATE TABLE IF NOT EXISTS lucidota_survey.pivot_candidate (
    candidate_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    observation_id uuid REFERENCES lucidota_survey.survey_observation(observation_id) ON DELETE CASCADE,
    source_target text NOT NULL,
    candidate text NOT NULL,
    candidate_kind text NOT NULL CHECK (candidate_kind IN ('link', 'keyword', 'archive', 'structural')),
    score integer NOT NULL DEFAULT 0,
    reason text NOT NULL DEFAULT '',
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS pivot_candidate_score_idx
    ON lucidota_survey.pivot_candidate (score DESC, created_at DESC);
