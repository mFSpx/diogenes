-- Krampus Hypertimeline: deprecated/timestamped lore ingestion
-- Closes the Ouroboros loop. Every KRAMPUSCHEWING artifact gets a timeline entry.
-- Schema: lucidota_korpus.krampus_hypertimeline

CREATE TABLE IF NOT EXISTS lucidota_korpus.krampus_hypertimeline (
  entry_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),

  -- File identity
  file_path text NOT NULL,
  file_name text NOT NULL,
  file_ext text NOT NULL DEFAULT '',
  file_size_bytes bigint NOT NULL DEFAULT 0,
  sha256 text NOT NULL,

  -- Timeline position
  file_mtime timestamptz,           -- original file modification time
  ingested_at timestamptz NOT NULL DEFAULT now(),
  timeline_bucket text NOT NULL DEFAULT 'deprecated',  -- 'deprecated', 'archived', 'active', 'corpse'

  -- Status
  status text NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'scanned', 'hashed', 'ingested', 'error')),

  -- Lore
  lore_note text DEFAULT '',
  source_context text DEFAULT '',

  -- Graph linking
  parent_entry_id uuid REFERENCES lucidota_korpus.krampus_hypertimeline(entry_id),
  graph_promoted boolean NOT NULL DEFAULT false,
  graph_promoted_at timestamptz,

  -- Receipt
  receipt_path text DEFAULT '',

  -- Indexes
  UNIQUE(sha256)
);

-- Partition by timeline bucket for performance
CREATE INDEX IF NOT EXISTS idx_hypertimeline_bucket ON lucidota_korpus.krampus_hypertimeline (timeline_bucket, ingested_at DESC);
CREATE INDEX IF NOT EXISTS idx_hypertimeline_mtime ON lucidota_korpus.krampus_hypertimeline (file_mtime DESC);
CREATE INDEX IF NOT EXISTS idx_hypertimeline_ext ON lucidota_korpus.krampus_hypertimeline (file_ext, timeline_bucket);
CREATE INDEX IF NOT EXISTS idx_hypertimeline_status ON lucidota_korpus.krampus_hypertimeline (status);
