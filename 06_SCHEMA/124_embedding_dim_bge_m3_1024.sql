-- FILE: 06_SCHEMA/124_embedding_dim_bge_m3_1024.sql
-- PURPOSE: align embedding columns to bge-m3 (1024-dim), the operator-named GPU embedder.
-- CONTEXT: tables empty at migration; safe drop/re-add of vector column. Needle-384 fast
--          ontology embeds may be added later as a SEPARATE column/table (additive).
BEGIN;
DROP INDEX IF EXISTS lucidota_indy.chunk_embedding_hnsw_idx;
ALTER TABLE lucidota_indy.chunk_embedding DROP COLUMN IF EXISTS embedding;
ALTER TABLE lucidota_indy.chunk_embedding ADD COLUMN embedding vector(1024);
ALTER TABLE lucidota_indy.chunk_embedding ALTER COLUMN embedding_dim SET DEFAULT 1024;
CREATE INDEX chunk_embedding_hnsw_idx ON lucidota_indy.chunk_embedding
  USING hnsw (embedding vector_cosine_ops);

DROP INDEX IF EXISTS lucidota_korpus.component_embedding_hnsw_idx;
ALTER TABLE lucidota_korpus.component DROP COLUMN IF EXISTS embedding;
ALTER TABLE lucidota_korpus.component ADD COLUMN embedding vector(1024);
CREATE INDEX component_embedding_hnsw_idx ON lucidota_korpus.component
  USING hnsw (embedding vector_cosine_ops);
COMMIT;
