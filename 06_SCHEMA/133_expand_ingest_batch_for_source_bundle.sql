-- 
-- Name: 133_expand_ingest_batch_for_source_bundle; Type: MIGRATION; Schema: lucidota_korpus; Owner: -
--

ALTER TABLE lucidota_korpus.ingest_batch
ADD COLUMN IF NOT EXISTS root_path TEXT,
ADD COLUMN IF NOT EXISTS normal_count INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS quarantine_count INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS duplicate_groups JSONB,
ADD COLUMN IF NOT EXISTS safety_policy TEXT DEFAULT 'strict';
