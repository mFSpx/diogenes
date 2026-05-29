-- FILE: 06_SCHEMA/093_graph_edge_materialization_guard.sql
-- PURPOSE: Edge materialization support receipts/indexes for promotion path.

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_go;

CREATE INDEX IF NOT EXISTS idx_graph_promotion_materialization_edge
  ON lucidota_go.graph_promotion_materialization(graph_edge_uuid)
  WHERE graph_edge_uuid IS NOT NULL;
