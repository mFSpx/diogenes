-- Balanced ternary valency wrapper for GO graph nodes/edges.
-- GO-25 vocabulary remains unchanged; this adds native (+1,0,-1) polarity slots.

ALTER TABLE lucidota_go.graph_item
    ADD COLUMN IF NOT EXISTS ternary_valency integer NOT NULL DEFAULT 0
    CHECK (ternary_valency IN (1, 0, -1));

ALTER TABLE lucidota_go.graph_edge
    ADD COLUMN IF NOT EXISTS ternary_valency integer NOT NULL DEFAULT 0
    CHECK (ternary_valency IN (1, 0, -1));

CREATE INDEX IF NOT EXISTS graph_item_ternary_valency_idx
    ON lucidota_go.graph_item(ternary_valency, term, status, created_at DESC);

CREATE INDEX IF NOT EXISTS graph_edge_ternary_valency_idx
    ON lucidota_go.graph_edge(ternary_valency, edge_type, created_at DESC);
