-- FILE: 06_SCHEMA/028_ternary_lens_lab.sql
-- COMPONENT: LUCIDOTA Ternary Lens Lab
-- COMPLIANCE: Idempotent scaffold; no destructive actions; research registry only.
-- PURPOSE: Track adapter/lens families and whether they preserve the BitNet/FairyFuse low-bit fast path.

START TRANSACTION;

CREATE SCHEMA IF NOT EXISTS lucidota_ternary;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typnamespace = 'lucidota_ternary'::regnamespace AND typname = 'lens_classification') THEN
        CREATE TYPE lucidota_ternary.lens_classification AS ENUM (
            'usable_now',
            'research_only',
            'needs_conversion',
            'unsafe_for_fastpath',
            'unsupported'
        );
    END IF;
END;
$$;

CREATE TABLE IF NOT EXISTS lucidota_ternary.lens_registry (
    lens_uuid UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    lens_key TEXT NOT NULL UNIQUE,
    lens_family TEXT NOT NULL,
    display_name TEXT NOT NULL,
    source_uri TEXT NOT NULL DEFAULT '',
    local_path TEXT NOT NULL DEFAULT '',
    license_id TEXT NOT NULL DEFAULT 'unknown',
    classification lucidota_ternary.lens_classification NOT NULL,
    fast_path_compatible BOOLEAN NOT NULL DEFAULT FALSE,
    fast_path_contract TEXT NOT NULL DEFAULT 'runtime inference must avoid FP16/FP32 LoRA matmuls in the hot path',
    benchmark_required BOOLEAN NOT NULL DEFAULT TRUE,
    benchmark_evidence TEXT NOT NULL DEFAULT '',
    notes TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    detail JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS lucidota_ternary.backend_vendor_audit (
    audit_uuid UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vendor_key TEXT NOT NULL,
    backend_name TEXT NOT NULL,
    source_uri TEXT NOT NULL DEFAULT '',
    license_id TEXT NOT NULL DEFAULT 'unknown',
    supports_bitnet BOOLEAN NOT NULL DEFAULT FALSE,
    supports_packed_ternary BOOLEAN NOT NULL DEFAULT FALSE,
    supports_adapter_runtime BOOLEAN NOT NULL DEFAULT FALSE,
    adapter_runtime_precision TEXT NOT NULL DEFAULT 'unknown',
    classification lucidota_ternary.lens_classification NOT NULL,
    fast_path_compatible BOOLEAN NOT NULL DEFAULT FALSE,
    evidence JSONB NOT NULL DEFAULT '[]'::jsonb,
    blockers TEXT NOT NULL DEFAULT '',
    audited_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS lucidota_ternary.inference_audit (
    inference_audit_uuid UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    lens_uuid UUID REFERENCES lucidota_ternary.lens_registry(lens_uuid) ON DELETE SET NULL,
    backend_name TEXT NOT NULL,
    hardware_profile TEXT NOT NULL DEFAULT '',
    model_artifact TEXT NOT NULL DEFAULT '',
    adapter_artifact TEXT NOT NULL DEFAULT '',
    runtime_precision TEXT NOT NULL DEFAULT 'unknown',
    introduced_fp16_fp32_hotpath BOOLEAN NOT NULL DEFAULT TRUE,
    fast_path_preserved BOOLEAN NOT NULL DEFAULT FALSE,
    classification lucidota_ternary.lens_classification NOT NULL,
    benchmark_command TEXT NOT NULL DEFAULT '',
    benchmark_evidence TEXT NOT NULL DEFAULT '',
    tokens_per_second NUMERIC,
    latency_ms NUMERIC,
    energy_notes TEXT NOT NULL DEFAULT '',
    audited_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    detail JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_lens_registry_classification
    ON lucidota_ternary.lens_registry(classification);

CREATE INDEX IF NOT EXISTS idx_lens_registry_fast_path
    ON lucidota_ternary.lens_registry(fast_path_compatible, classification);

CREATE INDEX IF NOT EXISTS idx_backend_vendor_audit_vendor
    ON lucidota_ternary.backend_vendor_audit(vendor_key, backend_name);

CREATE INDEX IF NOT EXISTS idx_inference_audit_fast_path
    ON lucidota_ternary.inference_audit(fast_path_preserved, classification);

CREATE OR REPLACE FUNCTION lucidota_ternary.touch_lens_registry_updated_at()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger
        WHERE tgname = 'trg_touch_lens_registry_updated_at'
          AND tgrelid = 'lucidota_ternary.lens_registry'::regclass
    ) THEN
        CREATE TRIGGER trg_touch_lens_registry_updated_at
        BEFORE UPDATE ON lucidota_ternary.lens_registry
        FOR EACH ROW
        EXECUTE FUNCTION lucidota_ternary.touch_lens_registry_updated_at();
    END IF;
END;
$$;

COMMIT;
