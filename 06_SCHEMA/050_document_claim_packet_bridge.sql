-- FILE: 06_SCHEMA/050_document_claim_packet_bridge.sql
-- PURPOSE: executable bridge from parsed document spans to GLiNER-style claim packets.
-- COMPLIANCE:
--   - Parser output remains evidence-candidate material, not truth.
--   - Claim packets are graph-promotion candidates only; this migration writes no graph_item/graph_edge rows.
--   - Idempotency is enforced by source span + extractor + label + matched text hash.

BEGIN;

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_korpus;

CREATE TABLE IF NOT EXISTS lucidota_korpus.document_claim_packet (
  claim_packet_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),

  run_uuid uuid NOT NULL REFERENCES lucidota_korpus.document_parse_run(run_uuid) ON DELETE CASCADE,
  span_uuid uuid NOT NULL REFERENCES lucidota_korpus.document_parse_span(span_uuid) ON DELETE CASCADE,

  source_span_id text NOT NULL,
  evidence_refs jsonb NOT NULL DEFAULT '[]'::jsonb,

  claim_text text NOT NULL,
  label text NOT NULL,
  matched_text text NOT NULL DEFAULT '',
  matched_text_sha256 text NOT NULL CHECK (matched_text_sha256 ~ '^[0-9a-f]{64}$'),

  extractor_name text NOT NULL,
  extractor_version text NOT NULL,
  extractor_backend text NOT NULL,
  model_hash text NOT NULL DEFAULT 'missing_local_gliner_model',

  confidence_bps integer NOT NULL CHECK (confidence_bps BETWEEN 0 AND 10000),
  authority_class text NOT NULL CHECK (authority_class IN (
    'raw_evidence',
    'operator_authored_assertion',
    'operator_defined_label',
    'deterministic_metric',
    'statistical_finding',
    'model_computed_finding',
    'stream_ml_finding',
    'graph_inferred_relation',
    'operator_confirmed_finding',
    'canonical_doctrine',
    'external_action_authorized'
  )),

  review_state text NOT NULL DEFAULT 'candidate_unreviewed' CHECK (review_state IN (
    'candidate_unreviewed',
    'reviewed_defer',
    'reviewed_reject',
    'reviewed_promote_candidate',
    'operator_confirmed'
  )),
  graph_promotion_status text NOT NULL DEFAULT 'not_promoted' CHECK (graph_promotion_status IN (
    'not_promoted',
    'staged',
    'rejected',
    'promoted'
  )),
  truth_status text NOT NULL DEFAULT 'not_truth_claim_candidate' CHECK (truth_status = 'not_truth_claim_candidate'),

  created_at timestamptz NOT NULL DEFAULT now(),
  detail jsonb NOT NULL DEFAULT '{}'::jsonb,

  UNIQUE (source_span_id, extractor_name, extractor_version, label, matched_text_sha256)
);

CREATE INDEX IF NOT EXISTS idx_document_claim_packet_run
  ON lucidota_korpus.document_claim_packet(run_uuid);
CREATE INDEX IF NOT EXISTS idx_document_claim_packet_span
  ON lucidota_korpus.document_claim_packet(span_uuid);
CREATE INDEX IF NOT EXISTS idx_document_claim_packet_review_state
  ON lucidota_korpus.document_claim_packet(review_state, graph_promotion_status);
CREATE INDEX IF NOT EXISTS idx_document_claim_packet_label
  ON lucidota_korpus.document_claim_packet(label);

COMMIT;
