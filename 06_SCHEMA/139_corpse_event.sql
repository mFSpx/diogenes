CREATE TABLE IF NOT EXISTS lucidota_control.corpse_event (
    corpse_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    file_path text NOT NULL,
    file_hash text NOT NULL CHECK(file_hash ~ '^[0-9a-f]{64}$'),
    retired_at timestamptz DEFAULT now(),
    reason_kind text NOT NULL CHECK(reason_kind IN ('superseded','slop','security','refactor','evolution_drain','operator_order','dead_import','test_artifact')),
    reason_detail text,
    replacement_path text,
    replacement_hash text,
    training_eligibility text NOT NULL DEFAULT 'REVIEW' CHECK(training_eligibility IN ('ELIGIBLE','SCRUB','REVIEW')),
    krampuschewing_path text,
    corpus_chunk_uuid uuid,
    ingested_at timestamptz,
    ingestion_receipt_path text,
    detail jsonb DEFAULT '{}'
);

COMMENT ON TABLE lucidota_control.corpse_event IS 'Every delete is a reingestion event. KRAMPUSCHEWING eats the dead. training_eligibility=SCRUB means content must be sanitized before corpus exposure (secrets, case material). ELIGIBLE=ready for direct corpus ingest. REVIEW=needs human classification.';
