-- 128_percyphon_gin_index.sql
-- GIN index on percyphon_village.slots for fast JSONB hash-key lookups.
-- Without this, WHERE slots ? 'target_hash' does a full 5000-row table scan.
-- CONCURRENTLY means this is safe to apply on a live database with no table lock.

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_percyphon_village_slots_gin
    ON lucidota_go.percyphon_village
    USING GIN (slots);
