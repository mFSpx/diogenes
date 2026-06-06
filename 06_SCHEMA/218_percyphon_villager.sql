-- 218_percyphon_villager.sql
-- Percyphon Villager: normalized per-slot villager identity table.
--
-- Unlike percyphon_village (which stores the full 128-slot scaffold as a JSONB
-- array per row), this table stores one row per slot.  This enables direct SQL
-- queries on individual slots without JSONB path operators.
--
-- Relationship to percyphon_village:
--   percyphon_village.vuuid = stable village UUID (one row per villager)
--   percyphon_villager records = one row per slot, referencing the same seed
--
-- This table is populated by the village_manager worker (see also 126, 128).
-- Mutation class: candidate_writer (procedural_scaffold_candidate_not_truth).
-- No canonical graph truth may be written via this table.
--
-- Schema: WO-5 PERCYPHON VILLAGE CORE (normalized slot layer)

BEGIN;

CREATE SCHEMA IF NOT EXISTS lucidota_korpus;

CREATE TABLE IF NOT EXISTS lucidota_korpus.percyphon_villager (
    villager_id     UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    slot            INTEGER     NOT NULL CHECK (slot >= 0 AND slot < 128),
    seed            TEXT        NOT NULL,
    name            TEXT        NOT NULL,
    persona         TEXT        NOT NULL,
    identity_hash   TEXT        NOT NULL,
    slot_type       TEXT        NOT NULL CHECK (slot_type IN ('fixed_identity', 'procedural')),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Lookup by seed (all slots for a villager)
CREATE INDEX IF NOT EXISTS percyphon_villager_seed_idx
    ON lucidota_korpus.percyphon_villager (seed);

-- Lookup by slot range (all villagers at a given slot position)
CREATE INDEX IF NOT EXISTS percyphon_villager_slot_idx
    ON lucidota_korpus.percyphon_villager (slot);

-- Composite: fast queries for fixed vs procedural slots
CREATE INDEX IF NOT EXISTS percyphon_villager_slot_type_idx
    ON lucidota_korpus.percyphon_villager (slot_type, slot);

-- Lookup by persona archetype
CREATE INDEX IF NOT EXISTS percyphon_villager_persona_idx
    ON lucidota_korpus.percyphon_villager (persona);

COMMENT ON TABLE lucidota_korpus.percyphon_villager IS
    'Percyphon normalized per-slot villager identities. '
    'One row per slot (128 rows per villager). '
    'Candidate layer only -- no canonical graph truth. '
    'Schema: 218_percyphon_villager.sql.';

COMMENT ON COLUMN lucidota_korpus.percyphon_villager.slot IS
    'Slot index 0-127 (0-indexed). Slots 0-27 = fixed identity mask, slots 28-127 = procedural.';

COMMENT ON COLUMN lucidota_korpus.percyphon_villager.slot_type IS
    'fixed_identity for slots 0-27, procedural for slots 28-127.';

GRANT SELECT ON lucidota_korpus.percyphon_villager TO lucidota_postgrest_anon, mfspx;

COMMIT;
