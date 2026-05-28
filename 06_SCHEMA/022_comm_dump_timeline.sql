-- LUCIDOTA universal communications dump timeline.
-- Non-exhaustive: email JSON, Facebook exports, SMS/text exports, generic message JSON/JSONL/ZIP/dir.

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE SCHEMA IF NOT EXISTS lucidota_commdump;

CREATE OR REPLACE FUNCTION lucidota_commdump.uuid_v7()
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

CREATE TABLE IF NOT EXISTS lucidota_commdump.export_object (
    export_uuid uuid PRIMARY KEY DEFAULT lucidota_commdump.uuid_v7(),
    source_kind text NOT NULL DEFAULT 'generic' CHECK (source_kind IN ('email','facebook','sms','imessage','whatsapp','signal','generic')),
    source_path text NOT NULL,
    source_sha256 text NOT NULL CHECK (source_sha256 ~ '^[0-9a-f]{64}$'),
    size_bytes bigint NOT NULL DEFAULT 0,
    mime text NOT NULL DEFAULT '',
    cas_uri text NOT NULL DEFAULT '',
    locked_relative_path text NOT NULL DEFAULT '',
    status text NOT NULL DEFAULT 'running' CHECK (status IN ('running','succeeded','failed','partial')),
    thread_count integer NOT NULL DEFAULT 0,
    message_count integer NOT NULL DEFAULT 0,
    earliest_message_at timestamptz,
    latest_message_at timestamptz,
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE(source_sha256, source_kind)
);

CREATE TABLE IF NOT EXISTS lucidota_commdump.thread (
    thread_uuid uuid PRIMARY KEY DEFAULT lucidota_commdump.uuid_v7(),
    export_uuid uuid NOT NULL REFERENCES lucidota_commdump.export_object(export_uuid) ON DELETE CASCADE,
    source_kind text NOT NULL DEFAULT 'generic',
    provider_thread_id text NOT NULL DEFAULT '',
    title text NOT NULL DEFAULT '',
    participants jsonb NOT NULL DEFAULT '[]'::jsonb,
    source_member text NOT NULL DEFAULT '',
    message_count integer NOT NULL DEFAULT 0,
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE(export_uuid, source_member, provider_thread_id)
);

CREATE TABLE IF NOT EXISTS lucidota_commdump.message (
    message_uuid uuid PRIMARY KEY DEFAULT lucidota_commdump.uuid_v7(),
    thread_uuid uuid NOT NULL REFERENCES lucidota_commdump.thread(thread_uuid) ON DELETE CASCADE,
    export_uuid uuid NOT NULL REFERENCES lucidota_commdump.export_object(export_uuid) ON DELETE CASCADE,
    source_kind text NOT NULL DEFAULT 'generic',
    provider_message_id text NOT NULL DEFAULT '',
    sender text NOT NULL DEFAULT '',
    recipients jsonb NOT NULL DEFAULT '[]'::jsonb,
    occurred_at timestamptz,
    occurred_at_raw text NOT NULL DEFAULT '',
    time_source text NOT NULL DEFAULT '',
    time_confidence_bps integer NOT NULL DEFAULT 0 CHECK (time_confidence_bps BETWEEN 0 AND 10000),
    sequence_index integer NOT NULL DEFAULT 0,
    subject text NOT NULL DEFAULT '',
    content_text text NOT NULL DEFAULT '',
    content_sha256 text NOT NULL DEFAULT '',
    content_kind text NOT NULL DEFAULT '',
    attachments jsonb NOT NULL DEFAULT '[]'::jsonb,
    raw jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE(thread_uuid, provider_message_id, sequence_index)
);

CREATE INDEX IF NOT EXISTS commdump_message_time_idx ON lucidota_commdump.message(occurred_at, source_kind, sender);
CREATE INDEX IF NOT EXISTS commdump_message_thread_idx ON lucidota_commdump.message(thread_uuid, sequence_index);
CREATE INDEX IF NOT EXISTS commdump_message_content_sha_idx ON lucidota_commdump.message(content_sha256);
ALTER TABLE lucidota_commdump.message
    ADD COLUMN IF NOT EXISTS fts_vector tsvector GENERATED ALWAYS AS (to_tsvector('english', coalesce(subject,'') || ' ' || coalesce(content_text,''))) STORED;
CREATE INDEX IF NOT EXISTS commdump_message_fts_idx ON lucidota_commdump.message USING GIN (fts_vector);
CREATE INDEX IF NOT EXISTS commdump_sender_trgm_idx ON lucidota_commdump.message USING GIN (sender gin_trgm_ops);

CREATE OR REPLACE FUNCTION lucidota_commdump.touch_updated_at()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS touch_commdump_export_updated_at ON lucidota_commdump.export_object;
CREATE TRIGGER touch_commdump_export_updated_at BEFORE UPDATE ON lucidota_commdump.export_object
FOR EACH ROW EXECUTE FUNCTION lucidota_commdump.touch_updated_at();
