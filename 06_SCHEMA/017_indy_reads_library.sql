-- Indy_READs library ingestion, chunking, embedding, and LoRA staging.
-- Postgres/graph live surface. No Markdown doctrine dependency.

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS vector;

CREATE SCHEMA IF NOT EXISTS lucidota_indy;

CREATE TABLE IF NOT EXISTS lucidota_indy.book_source (
    book_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    graph_item_uuid uuid REFERENCES lucidota_go.graph_item(uuid),
    path text NOT NULL UNIQUE,
    file_name text NOT NULL,
    file_ext text NOT NULL,
    sha256 text NOT NULL,
    size_bytes bigint NOT NULL,
    status text NOT NULL DEFAULT 'located' CHECK (status IN ('located','extracted','chunked','embedded','error','archived')),
    title text NOT NULL DEFAULT '',
    author text NOT NULL DEFAULT '',
    extraction_method text NOT NULL DEFAULT '',
    extraction_error text NOT NULL DEFAULT '',
    text_sha256 text NOT NULL DEFAULT '',
    token_count integer NOT NULL DEFAULT 0,
    chunk_count integer NOT NULL DEFAULT 0,
    embedded_count integer NOT NULL DEFAULT 0,
    payload jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS lucidota_indy.book_chunk (
    chunk_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    book_uuid uuid NOT NULL REFERENCES lucidota_indy.book_source(book_uuid) ON DELETE CASCADE,
    graph_item_uuid uuid REFERENCES lucidota_go.graph_item(uuid),
    chunk_index integer NOT NULL,
    token_count integer NOT NULL CHECK (token_count BETWEEN 1 AND 500),
    char_count integer NOT NULL CHECK (char_count > 0),
    content_sha256 text NOT NULL,
    content text NOT NULL,
    anchor text NOT NULL DEFAULT '',
    payload jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE(book_uuid, chunk_index),
    UNIQUE(book_uuid, content_sha256)
);

CREATE INDEX IF NOT EXISTS book_chunk_book_idx ON lucidota_indy.book_chunk(book_uuid, chunk_index);

CREATE TABLE IF NOT EXISTS lucidota_indy.chunk_embedding (
    chunk_uuid uuid PRIMARY KEY REFERENCES lucidota_indy.book_chunk(chunk_uuid) ON DELETE CASCADE,
    embedding_model text NOT NULL,
    embedding_dim integer NOT NULL DEFAULT 384,
    embedding vector(384) NOT NULL,
    content_sha256 text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS chunk_embedding_hnsw_idx
    ON lucidota_indy.chunk_embedding USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS lucidota_indy.lora_training_example (
    example_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    chunk_uuid uuid NOT NULL REFERENCES lucidota_indy.book_chunk(chunk_uuid) ON DELETE CASCADE,
    dataset_split text NOT NULL DEFAULT 'train' CHECK (dataset_split IN ('train','validation','test')),
    model_ceiling text NOT NULL DEFAULT '1.5b',
    max_source_tokens integer NOT NULL DEFAULT 500 CHECK (max_source_tokens <= 500),
    instruction text NOT NULL,
    input text NOT NULL,
    output text NOT NULL,
    content_sha256 text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE(chunk_uuid, dataset_split, instruction)
);

CREATE TABLE IF NOT EXISTS lucidota_indy.ingest_run (
    run_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    source_root text NOT NULL,
    status text NOT NULL DEFAULT 'running' CHECK (status IN ('running','succeeded','failed')),
    max_tokens integer NOT NULL DEFAULT 500 CHECK (max_tokens <= 500),
    embedding_model text NOT NULL DEFAULT '',
    books_seen integer NOT NULL DEFAULT 0,
    books_ok integer NOT NULL DEFAULT 0,
    books_error integer NOT NULL DEFAULT 0,
    chunks_written integer NOT NULL DEFAULT 0,
    embeddings_written integer NOT NULL DEFAULT 0,
    lora_examples_written integer NOT NULL DEFAULT 0,
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    started_at timestamptz NOT NULL DEFAULT now(),
    finished_at timestamptz
);

CREATE TABLE IF NOT EXISTS lucidota_indy.markdown_artifact (
    artifact_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    graph_item_uuid uuid REFERENCES lucidota_go.graph_item(uuid),
    original_path text NOT NULL UNIQUE,
    archived_path text NOT NULL DEFAULT '',
    sha256 text NOT NULL,
    size_bytes bigint NOT NULL,
    line_count integer NOT NULL DEFAULT 0,
    status text NOT NULL DEFAULT 'ingested' CHECK (status IN ('ingested','archived','error')),
    title text NOT NULL DEFAULT '',
    excerpt text NOT NULL DEFAULT '',
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);
