-- 125: fast Groq-extraction sink (candidate_writer). Flat, no heavy FKs; promote to canon via gate later.
CREATE TABLE IF NOT EXISTS lucidota_korpus.corpus_chunk (
  chunk_uuid uuid PRIMARY KEY, file_uuid uuid, sha256 text NOT NULL, source_path text, mime text,
  chunk_index int NOT NULL, content text NOT NULL, go25 jsonb NOT NULL DEFAULT '{}'::jsonb,
  embedding vector(1024), embedding_model text NOT NULL DEFAULT 'bge-m3', extractor text NOT NULL DEFAULT 'groq',
  created_at timestamptz NOT NULL DEFAULT now(), UNIQUE (sha256, chunk_index));
CREATE INDEX IF NOT EXISTS corpus_chunk_hnsw ON lucidota_korpus.corpus_chunk USING hnsw (embedding vector_cosine_ops);
