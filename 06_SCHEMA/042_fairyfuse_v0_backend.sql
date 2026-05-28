-- FILE: 06_SCHEMA/042_fairyfuse_v0_backend.sql
-- PURPOSE: register FairyFuse v0 symbolic ternary backend as usable-now command envelope router.
-- COMPLIANCE: idempotent; no model download; no FP16/FP32 LoRA fast path claim.

BEGIN;
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_ternary;

INSERT INTO lucidota_ternary.lens_registry(
  lens_key, lens_family, display_name, source_uri, local_path, license_id,
  classification, fast_path_compatible, benchmark_required, benchmark_evidence, notes, detail
) VALUES (
  'command_envelope_router_v0',
  'symbolic_ternary_router',
  'FairyFuse v0 Command Envelope Router',
  '',
  'services/fairyfuse/fairyfuse_backend.py',
  'local_project',
  'usable_now',
  true,
  false,
  'python3 ALGOS/ternary_lens_router.py --raw-command proof --normalized-intent system.verify --context {} --lens-key command_envelope_router_v0 --database-url postgresql:///lucidota_state',
  'Deterministic integer/hash ternary command router. No model weights, no adapter matmul, no FP16/FP32 LoRA hot path.',
  '{"backend":"FAIRYFUSE_V0_SYMBOLIC_TERNARY","general_ternary_lora":false,"command_envelope_router":true}'::jsonb
) ON CONFLICT (lens_key) DO UPDATE SET
  lens_family=EXCLUDED.lens_family,
  display_name=EXCLUDED.display_name,
  local_path=EXCLUDED.local_path,
  classification=EXCLUDED.classification,
  fast_path_compatible=EXCLUDED.fast_path_compatible,
  benchmark_required=EXCLUDED.benchmark_required,
  benchmark_evidence=EXCLUDED.benchmark_evidence,
  notes=EXCLUDED.notes,
  detail=EXCLUDED.detail,
  updated_at=now();

INSERT INTO lucidota_ternary.backend_vendor_audit(
  vendor_key, backend_name, license_id, supports_bitnet, supports_packed_ternary,
  supports_adapter_runtime, adapter_runtime_precision, classification, fast_path_compatible, evidence, blockers
) VALUES (
  'lucidota_local_fairyfuse_v0',
  'FAIRYFUSE_V0_SYMBOLIC_TERNARY',
  'local_project',
  false,
  true,
  false,
  'integer_hash_ternary',
  'usable_now',
  true,
  '["services/fairyfuse/fairyfuse_backend.py","06_SCHEMA/042_fairyfuse_v0_backend.sql"]'::jsonb,
  ''
);
COMMIT;
