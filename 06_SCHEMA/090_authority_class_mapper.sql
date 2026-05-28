-- FILE: 06_SCHEMA/090_authority_class_mapper.sql
-- PURPOSE: Authority-class mapping registry for extraction outputs.

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_archaeology;

CREATE TABLE IF NOT EXISTS lucidota_archaeology.authority_class_mapping (
  mapping_key text PRIMARY KEY,
  output_kind text NOT NULL,
  extractor_name text NOT NULL DEFAULT '*',
  authority_class lucidota_archaeology.authority_class NOT NULL,
  active boolean NOT NULL DEFAULT true,
  created_at timestamptz NOT NULL DEFAULT now(),
  detail jsonb NOT NULL DEFAULT '{}'::jsonb
);

INSERT INTO lucidota_archaeology.authority_class_mapping(mapping_key, output_kind, extractor_name, authority_class, detail) VALUES
('sticker_feature_vector:v1','sticker_feature_vector','sticker_feature_extractor_v1','deterministic_metric','{"reason":"deterministic feature extraction"}'::jsonb),
('gliner_entity_span:v1','gliner_entity_span','gliner','model_computed_finding','{"reason":"model-computed zero-shot span extraction"}'::jsonb),
('claim_packet:v1','claim_packet','*','model_computed_finding','{"reason":"candidate claim, not truth"}'::jsonb),
('operator_label:v1','operator_label','*','operator_defined_label','{"reason":"Operator namespace label"}'::jsonb),
('temporal_claim:v1','temporal_claim','chrono','deterministic_metric','{"reason":"timestamp evidence extractor"}'::jsonb),
('graph_relation_candidate:v1','graph_relation_candidate','*','graph_inferred_relation','{"reason":"candidate relation awaiting promotion"}'::jsonb)
ON CONFLICT (mapping_key) DO UPDATE SET
  output_kind=EXCLUDED.output_kind, extractor_name=EXCLUDED.extractor_name, authority_class=EXCLUDED.authority_class,
  active=true, detail=EXCLUDED.detail;
