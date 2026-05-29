-- LUCIDOTA chat dump timeline: deterministic OpenAI/Claude export normalization.
-- Purpose: convert huge provider data exports into a contemporaneous message timeline.

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE SCHEMA IF NOT EXISTS lucidota_chatdump;

CREATE OR REPLACE FUNCTION lucidota_chatdump.uuid_v7()
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

CREATE TABLE IF NOT EXISTS lucidota_chatdump.export_object (
    export_uuid uuid PRIMARY KEY DEFAULT lucidota_chatdump.uuid_v7(),
    provider text NOT NULL CHECK (provider IN ('openai','claude','unknown')),
    source_path text NOT NULL,
    source_sha256 text NOT NULL CHECK (source_sha256 ~ '^[0-9a-f]{64}$'),
    size_bytes bigint NOT NULL DEFAULT 0,
    mime text NOT NULL DEFAULT '',
    cas_uri text NOT NULL DEFAULT '',
    locked_relative_path text NOT NULL DEFAULT '',
    status text NOT NULL DEFAULT 'running' CHECK (status IN ('running','succeeded','failed','partial')),
    conversation_count integer NOT NULL DEFAULT 0,
    message_count integer NOT NULL DEFAULT 0,
    earliest_message_at timestamptz,
    latest_message_at timestamptz,
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE(source_sha256, provider)
);

CREATE TABLE IF NOT EXISTS lucidota_chatdump.conversation (
    conversation_uuid uuid PRIMARY KEY DEFAULT lucidota_chatdump.uuid_v7(),
    export_uuid uuid NOT NULL REFERENCES lucidota_chatdump.export_object(export_uuid) ON DELETE CASCADE,
    provider text NOT NULL CHECK (provider IN ('openai','claude','unknown')),
    provider_conversation_id text NOT NULL,
    title text NOT NULL DEFAULT '',
    create_time timestamptz,
    update_time timestamptz,
    create_time_raw text NOT NULL DEFAULT '',
    update_time_raw text NOT NULL DEFAULT '',
    message_count integer NOT NULL DEFAULT 0,
    source_member text NOT NULL DEFAULT '',
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE(provider, provider_conversation_id, export_uuid)
);

CREATE INDEX IF NOT EXISTS chatdump_conversation_time_idx
    ON lucidota_chatdump.conversation(provider, create_time, update_time);

CREATE TABLE IF NOT EXISTS lucidota_chatdump.message (
    message_uuid uuid PRIMARY KEY DEFAULT lucidota_chatdump.uuid_v7(),
    conversation_uuid uuid NOT NULL REFERENCES lucidota_chatdump.conversation(conversation_uuid) ON DELETE CASCADE,
    export_uuid uuid NOT NULL REFERENCES lucidota_chatdump.export_object(export_uuid) ON DELETE CASCADE,
    provider text NOT NULL CHECK (provider IN ('openai','claude','unknown')),
    provider_message_id text NOT NULL DEFAULT '',
    parent_message_id text NOT NULL DEFAULT '',
    role text NOT NULL DEFAULT '',
    author_name text NOT NULL DEFAULT '',
    create_time timestamptz,
    update_time timestamptz,
    create_time_raw text NOT NULL DEFAULT '',
    update_time_raw text NOT NULL DEFAULT '',
    time_source text NOT NULL DEFAULT '',
    time_confidence_bps integer NOT NULL DEFAULT 0 CHECK (time_confidence_bps BETWEEN 0 AND 10000),
    sequence_index integer NOT NULL DEFAULT 0,
    content_text text NOT NULL DEFAULT '',
    content_sha256 text NOT NULL DEFAULT '',
    content_kind text NOT NULL DEFAULT '',
    token_count integer NOT NULL DEFAULT 0,
    attachments jsonb NOT NULL DEFAULT '[]'::jsonb,
    raw jsonb NOT NULL DEFAULT '{}'::jsonb,
    graph_item_uuid uuid,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE(conversation_uuid, provider_message_id, sequence_index)
);

CREATE INDEX IF NOT EXISTS chatdump_message_time_idx
    ON lucidota_chatdump.message(create_time, provider, role);
CREATE INDEX IF NOT EXISTS chatdump_message_conv_idx
    ON lucidota_chatdump.message(conversation_uuid, sequence_index);
CREATE INDEX IF NOT EXISTS chatdump_message_content_sha_idx
    ON lucidota_chatdump.message(content_sha256);

ALTER TABLE lucidota_chatdump.message
    ADD COLUMN IF NOT EXISTS fts_vector tsvector GENERATED ALWAYS AS (to_tsvector('english', coalesce(content_text,''))) STORED;
CREATE INDEX IF NOT EXISTS chatdump_message_fts_idx ON lucidota_chatdump.message USING GIN (fts_vector);

CREATE OR REPLACE FUNCTION lucidota_chatdump.touch_updated_at()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS touch_chatdump_export_updated_at ON lucidota_chatdump.export_object;
CREATE TRIGGER touch_chatdump_export_updated_at BEFORE UPDATE ON lucidota_chatdump.export_object
FOR EACH ROW EXECUTE FUNCTION lucidota_chatdump.touch_updated_at();

CREATE TABLE IF NOT EXISTS lucidota_chatdump.decision_signal (
    decision_signal_uuid uuid PRIMARY KEY DEFAULT lucidota_chatdump.uuid_v7(),
    source_kind text NOT NULL CHECK (source_kind IN ('chat_message','korpus_component')),
    source_uuid uuid NOT NULL,
    provider text NOT NULL DEFAULT '',
    occurred_at timestamptz,
    score integer NOT NULL CHECK (score BETWEEN -10000 AND 10000),
    evidence_count integer NOT NULL DEFAULT 0,
    planning_count integer NOT NULL DEFAULT 0,
    delay_count integer NOT NULL DEFAULT 0,
    support_count integer NOT NULL DEFAULT 0,
    boundary_count integer NOT NULL DEFAULT 0,
    outcome_count integer NOT NULL DEFAULT 0,
    impulsive_count integer NOT NULL DEFAULT 0,
    scarcity_count integer NOT NULL DEFAULT 0,
    risk_count integer NOT NULL DEFAULT 0,
    label text NOT NULL DEFAULT '',
    features jsonb NOT NULL DEFAULT '{}'::jsonb,
    excerpt text NOT NULL DEFAULT '',
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE(source_kind, source_uuid)
);

CREATE INDEX IF NOT EXISTS decision_signal_time_idx
    ON lucidota_chatdump.decision_signal(occurred_at, score DESC);
CREATE INDEX IF NOT EXISTS decision_signal_label_idx
    ON lucidota_chatdump.decision_signal(label, occurred_at DESC);

ALTER TABLE lucidota_chatdump.decision_signal DROP CONSTRAINT IF EXISTS decision_signal_source_kind_check;
ALTER TABLE lucidota_chatdump.decision_signal ADD CONSTRAINT decision_signal_source_kind_check CHECK (source_kind IN ('chat_message','korpus_component','comm_message'));
