-- FILE: 06_SCHEMA/032_bitloops_loop.sql
-- PURPOSE: Bitloops automation loop — harvest table for context/checkpoint events
--          feeding the River self-teaching loop via ABSURD queue bitloops_momentary.
-- COMPLIANCE: Idempotent, non-destructive. No canonical graph writes.

CREATE SCHEMA IF NOT EXISTS bitloops_loop;

CREATE TABLE IF NOT EXISTS bitloops_loop.checkpoint_event (
  id                  BIGSERIAL PRIMARY KEY,
  checkpoint_id       TEXT NOT NULL UNIQUE,
  session_id          TEXT NOT NULL,
  repo_path           TEXT NOT NULL DEFAULT '/home/mfspx/LUCIDOTA',
  context_blob_hash   TEXT,
  devql_query         TEXT,
  ontology_route      TEXT,
  timestamp_custody   TIMESTAMPTZ NOT NULL DEFAULT now(),
  workflow_id         TEXT NOT NULL,
  idempotency_key     TEXT NOT NULL UNIQUE,
  processed_at        TIMESTAMPTZ,
  river_correct       BOOLEAN,
  river_delta_seconds FLOAT
);

CREATE INDEX IF NOT EXISTS idx_bitloops_ce_session
  ON bitloops_loop.checkpoint_event (session_id);

CREATE INDEX IF NOT EXISTS idx_bitloops_ce_ts
  ON bitloops_loop.checkpoint_event (timestamp_custody DESC);
