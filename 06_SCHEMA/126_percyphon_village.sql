-- 126_percyphon_village.sql
-- Percyphon Village: deterministic procedural identity scaffolds for 5000 villagers.
-- Mutation class: candidate_writer (procedural_scaffold_candidate_not_truth only).
-- No canonical graph truth may be written via this table.
-- Authority: WO-5 PERCYPHON VILLAGE CORE

CREATE TABLE IF NOT EXISTS lucidota_go.percyphon_village (
    -- Primary key: deterministic UUID derived from seed hash (vuuid)
    vuuid           TEXT        NOT NULL,

    -- Slot identity columns (fixed slots 1-28 summary fields)
    -- These collapse slot-1 as the "primary" identity handle for quick lookup.
    name            TEXT        NOT NULL,         -- slot-1 Villager-NNNN name
    persona         TEXT        NOT NULL,         -- slot-1 persona archetype
    alias           TEXT        NOT NULL,         -- slot-1 Alias-XXXX
    ternary_state   SMALLINT    NOT NULL          -- slot-1 ternary offset: -1, 0, +1
                    CHECK (ternary_state IN (-1, 0, 1)),

    -- Full 128-slot coordinate array stored as JSONB
    -- Each entry: {slot_index, name, alias, persona, uuid, ternary_offset, coord_128}
    -- Slots 1-28: fixed identity mask (CKDOG1 soul mirror)
    -- Slots 29-128: procedural verbosity expansion
    slots           JSONB       NOT NULL DEFAULT '[]'::jsonb,

    -- Relevance score in basis points (0-10000 = 0.00%-100.00%)
    relevance_confidence_bps    SMALLINT NOT NULL DEFAULT 0
                                CHECK (relevance_confidence_bps BETWEEN 0 AND 10000),

    -- The seed string that produced this scaffold (raw, not hashed)
    seed            TEXT        NOT NULL,

    -- Authority tag — must always equal this literal; enforced by constraint
    authority       TEXT        NOT NULL DEFAULT 'procedural_scaffold_candidate_not_truth'
                    CHECK (authority = 'procedural_scaffold_candidate_not_truth'),

    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

    PRIMARY KEY (vuuid)
);

-- Capacity comment: table supports up to 5000 active villager rows.
-- Beyond 5000, oldest rows are evicted by the village_manager worker (not built yet).
COMMENT ON TABLE lucidota_go.percyphon_village IS
    'Percyphon procedural identity scaffolds. Max 5000 active villagers. '
    'Candidate layer only — no canonical graph truth. '
    'Schema: WO-5 PERCYPHON VILLAGE CORE (126_percyphon_village.sql).';

-- Indexes for routing lookups
CREATE INDEX IF NOT EXISTS percyphon_village_persona_idx
    ON lucidota_go.percyphon_village (persona);

CREATE INDEX IF NOT EXISTS percyphon_village_ternary_idx
    ON lucidota_go.percyphon_village (ternary_state);

CREATE INDEX IF NOT EXISTS percyphon_village_updated_idx
    ON lucidota_go.percyphon_village (updated_at DESC);

-- GIN index for slot JSONB queries (e.g. find by slot coord_128)
CREATE INDEX IF NOT EXISTS percyphon_village_slots_gin_idx
    ON lucidota_go.percyphon_village USING gin (slots);
