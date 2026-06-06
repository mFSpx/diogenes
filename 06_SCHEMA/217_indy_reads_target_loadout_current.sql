-- Indy_READs target loadout canonicalization.
-- Purpose: separate Indy_READs intended model body from the emergency closure/runtime substitute stack.

BEGIN;

DROP VIEW IF EXISTS lucidota_canon.indy_reads_vram_coprocessor_fabric_current;
DROP VIEW IF EXISTS lucidota_canon.indy_reads_target_model_loadout_current;

CREATE VIEW lucidota_canon.indy_reads_target_model_loadout_current AS
WITH rows AS (
    SELECT *
    FROM (VALUES
        (10, 'needle_26m_router_swarm', 'Needle 26M router/reader swarm', 20, 1, 'shared_weight_cpu_preload', 0::numeric, 4096, 'reader_hot', true, false, true, true, false, false, true, NULL::numeric, NULL::numeric,
         '03_VAULT/models/needle/needle.pkl', '.venv/bin/python + scripts/lucidota_start_needle_swarm.sh', 'Target: 20 logical slots on one shared/preloaded Needle 26M weight. Current proof is one shared worker/process, not 20 physical processes.'),
        (20, 'bonsai_8b_1bit_dual_lane', 'Bonsai 8B 1-bit dual lane', 2, 1, 'shared_weight_vram_preload_q4kv_admitted_10k', 625::numeric, 10000, 'orchestrator_hot', true, false, true, true, false, false, false, NULL::numeric, NULL::numeric,
         'hf:prism-ml/Bonsai-8B-gguf:Q1_0', 'scripts/lucidota_start_bonsai_ternary_llama.sh', 'Target: two logical Bonsai lanes, one shared weight. Shared-weight proof remains false until a dedicated receipt proves the exact sharing semantics.'),
        (30, 'bimamba_mamba2_1p3b_ternary', 'BiMamba/Mamba2 1.3B ternary resident target', 1, 0, 'recurrent_no_transformer_kv', 0::numeric, 8192, 'target_resident_missing', true, false, false, false, false, true, false, NULL::numeric, NULL::numeric,
         '', 'missing until proven', 'Canonical resident target, but no BiMamba/Mamba2 1.3B ternary artifact/runtime proof is present.'),
        (40, 'rwkv_small_reader_warm', 'RWKV-small reader', 1, 0, 'recurrent_warm_preemptible', 0::numeric, 8192, 'reader_warm_preemptible', true, false, false, false, false, true, false, NULL::numeric, NULL::numeric,
         '', 'missing until proven', 'Warm/preemptible target reader lane; artifact and runtime proof are not yet present.'),
        (50, 'lora_book_adapter_hot_lanes', 'LoRA book adapter-hot lanes', 6, 1, 'inherits_base_runtime_kv_adapter_hot', 0::numeric, 4096, 'adapter_hot', true, false, true, true, false, false, false, NULL::numeric, NULL::numeric,
         'BOOKS/.indy_reads + LoRA adapter registry', 'ironclaw-indy-reads + admitted base runtime', 'Book lanes are adapter-hot targets; concrete adapters remain independently receipted.'),
        (60, 'mistral_code_7b_tern_2bit_swapout', 'Mistral Code 7B tern/2-bit swapout experiment', 1, 0, 'q4_swapout_experiment', 0::numeric, 4096, 'swapout_experiment', true, false, false, false, true, true, false, NULL::numeric, NULL::numeric,
         '', 'missing until proven', 'Swapout experiment target; not resident and no local artifact proof yet.'),
        (70, 'deepseek_1p5b_auxiliary_swapout', 'DeepSeek 1.5B auxiliary coder lane', 1, 1, 'q4_auxiliary_swapout', 0::numeric, 4096, 'auxiliary_swapout', false, true, true, true, true, false, false, NULL::numeric, NULL::numeric,
         '03_VAULT/models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf', 'scripts/lucidota_start_deepseek_llama.sh', 'Emergency closure/current substitute coder lane; not Indy_READs intended model body.'),
        (80, 'ram_mamba_bonsai_tiny_overflow', 'RAM Mamba/Bonsai/tiny-model overflow lanes', 3, 3, 'host_ram_overflow_no_active_vram_kv', 0::numeric, 4096, 'ram_overflow', false, true, true, true, false, false, false, NULL::numeric, NULL::numeric,
         '03_VAULT/models/tensorblock/Falcon3-Mamba-7B-Instruct-GGUF/Falcon3-Mamba-7B-Instruct-Q2_K.gguf + tiny lanes', 'scripts/lucidota_start_mamba_llama.sh + Bonsai/Needle fallbacks', 'Emergency closure/current substitute overflow fabric; not Indy_READs intended model body.')
    ) AS v(target_rank, model_key, display_name, logical_lane_count, physical_process_count, kv_policy, kv_allocated_now_mb, max_context_admitted, preemption_group, intended_target_status, current_substitute_status, admitted_runtime_status, resident_now_status, swapout_candidate_status, missing_artifact_status, shared_weight_proven, toks_in_per_s_measured, toks_out_per_s_measured, artifact_path, launcher_path, reason)
)
SELECT
    target_rank,
    model_key,
    display_name,
    logical_lane_count,
    physical_process_count,
    kv_policy,
    kv_allocated_now_mb,
    max_context_admitted,
    preemption_group,
    toks_in_per_s_measured,
    toks_out_per_s_measured,
    artifact_path,
    launcher_path,
    jsonb_build_object('status', intended_target_status, 'reason', CASE WHEN intended_target_status THEN 'canonical Indy_READs target body row' ELSE reason END) AS intended_target,
    jsonb_build_object('status', current_substitute_status, 'reason', CASE WHEN current_substitute_status THEN reason ELSE 'not a current substitute' END) AS current_substitute,
    jsonb_build_object('status', admitted_runtime_status, 'reason', CASE WHEN admitted_runtime_status THEN 'admitted by current runtime/admission surface or deterministic non-model lane' ELSE reason END) AS admitted_runtime,
    jsonb_build_object('status', resident_now_status, 'reason', CASE WHEN resident_now_status THEN 'resident/hot in current closure or target fabric' ELSE 'not resident now' END) AS resident_now,
    jsonb_build_object('status', swapout_candidate_status, 'reason', CASE WHEN swapout_candidate_status THEN 'eligible for swapout/preemption, not target body anchor' ELSE 'not a swapout candidate' END) AS swapout_candidate,
    jsonb_build_object('status', missing_artifact_status, 'reason', CASE WHEN missing_artifact_status THEN reason ELSE 'artifact not currently marked missing for this row' END) AS missing_artifact,
    jsonb_build_object(
        'shared_weight_proven', shared_weight_proven,
        'logical_lane_count', logical_lane_count,
        'physical_process_count', physical_process_count,
        'kv_policy', kv_policy,
        'kv_allocated_now_mb', kv_allocated_now_mb,
        'max_context_admitted', max_context_admitted,
        'preemption_group', preemption_group,
        'toks_in_per_s_measured', toks_in_per_s_measured,
        'toks_out_per_s_measured', toks_out_per_s_measured,
        'artifact_path', artifact_path,
        'launcher_path', launcher_path,
        'evidence_refs', jsonb_build_array('05_OUTPUTS/model_runtime/strict_model_stack_admission_latest.json', '05_OUTPUTS/runtime/needle_kv_probe_latest.json', '05_OUTPUTS/system_map/runtime_closure_final_proof_20260605T134428110330Z.json')
    ) AS receipt_fields,
    'indy_reads.target_loadout.v1'::text AS schema,
    now() AS refreshed_at
FROM rows;

CREATE VIEW lucidota_canon.indy_reads_vram_coprocessor_fabric_current AS
WITH kv_cases AS (
    SELECT *
    FROM (VALUES
        (16000, 'q4', 'q8', 0.5::numeric, 1.0::numeric),
        (16000, 'q4', 'q4', 0.5::numeric, 0.5::numeric),
        (12000, 'q4', 'q8', 0.5::numeric, 1.0::numeric),
        (12000, 'q4', 'q4', 0.5::numeric, 0.5::numeric),
        (10000, 'q4', 'q4', 0.5::numeric, 0.5::numeric)
    ) AS v(context_tokens, k_quant, v_quant, k_bytes, v_bytes)
), pressure AS (
    SELECT
        context_tokens,
        k_quant,
        v_quant,
        round((context_tokens::numeric * 32 * 8 * 128 * (k_bytes + v_bytes)) / 1048576, 1) AS per_lane_kv_mb,
        round((2 * context_tokens::numeric * 32 * 8 * 128 * (k_bytes + v_bytes)) / 1048576, 1) AS dual_lane_kv_mb,
        CASE WHEN round((2 * context_tokens::numeric * 32 * 8 * 128 * (k_bytes + v_bytes)) / 1048576, 1) <= 625 THEN 'admit' ELSE 'reject' END AS admission_decision
    FROM kv_cases
), fabric_components AS (
    SELECT jsonb_build_array(
        jsonb_build_object(
            'fabric_key', 'treelite_stack_deterministic_gate_asset',
            'intended_target', jsonb_build_object('status', true, 'reason', 'deterministic gate asset in target fabric'),
            'current_substitute', jsonb_build_object('status', false),
            'admitted_runtime', jsonb_build_object('status', true),
            'resident_now', jsonb_build_object('status', true),
            'swapout_candidate', jsonb_build_object('status', false),
            'missing_artifact', jsonb_build_object('status', false),
            'receipt_fields', jsonb_build_object('kv_policy', 'no_transformer_kv_gate_asset', 'logical_lane_count', 1, 'physical_process_count', 1, 'shared_weight_proven', true, 'kv_allocated_now_mb', 0, 'max_context_admitted', 0, 'preemption_group', 'deterministic_hot_gate', 'toks_in_per_s_measured', null, 'toks_out_per_s_measured', null)
        ),
        jsonb_build_object(
            'fabric_key', 'fft_gpu_batch_kernels_where_measured',
            'intended_target', jsonb_build_object('status', true, 'reason', 'GPU batch kernels where measurement receipts exist'),
            'current_substitute', jsonb_build_object('status', false),
            'admitted_runtime', jsonb_build_object('status', true),
            'resident_now', jsonb_build_object('status', false),
            'swapout_candidate', jsonb_build_object('status', false),
            'missing_artifact', jsonb_build_object('status', false),
            'receipt_fields', jsonb_build_object('kv_policy', 'no_transformer_kv_gpu_kernel', 'logical_lane_count', 1, 'physical_process_count', 0, 'shared_weight_proven', false, 'kv_allocated_now_mb', 0, 'max_context_admitted', 0, 'preemption_group', 'gpu_batch_when_measured', 'toks_in_per_s_measured', null, 'toks_out_per_s_measured', null)
        ),
        jsonb_build_object(
            'fabric_key', 'bernoulli_venturi_adversarial_harness_reserve',
            'intended_target', jsonb_build_object('status', true, 'reason', 'hot gate/harness reserve'),
            'current_substitute', jsonb_build_object('status', false),
            'admitted_runtime', jsonb_build_object('status', true),
            'resident_now', jsonb_build_object('status', true),
            'swapout_candidate', jsonb_build_object('status', false),
            'missing_artifact', jsonb_build_object('status', false),
            'receipt_fields', jsonb_build_object('kv_policy', 'no_transformer_kv_harness_reserve', 'logical_lane_count', 3, 'physical_process_count', 1, 'shared_weight_proven', true, 'kv_allocated_now_mb', 0, 'max_context_admitted', 0, 'preemption_group', 'hot_gate_harness_reserve', 'toks_in_per_s_measured', null, 'toks_out_per_s_measured', null)
        )
    ) AS components
)
SELECT
    'indy_reads_vram_coprocessor_fabric_current'::text AS fabric_id,
    now() AS refreshed_at,
    jsonb_build_object(
        'formula', 'context_tokens * layers * kv_heads * head_dim * (k_bytes + v_bytes) / 1048576',
        'layers', 32,
        'kv_heads', 8,
        'head_dim', 128,
        'q4_bytes', 0.5,
        'q8_bytes', 1.0,
        'dual_lane_kv_budget_mb', 625,
        'decision_rule', 'admit when dual_lane_kv_mb <= 625; otherwise reject'
    ) AS kv_governor,
    (SELECT jsonb_agg(jsonb_build_object(
        'context_tokens', context_tokens,
        'k_quant', k_quant,
        'v_quant', v_quant,
        'per_lane_kv_mb', per_lane_kv_mb,
        'dual_lane_kv_mb', dual_lane_kv_mb,
        'admission_decision', admission_decision
    ) ORDER BY context_tokens DESC, k_quant, v_quant) FROM pressure) AS bonsai_kv_pressure,
    fabric_components.components AS fabric_components,
    jsonb_build_object(
        'shared_weight_proven', false,
        'logical_lane_count', 2,
        'physical_process_count', 1,
        'kv_policy', 'bonsai_dual_logical_lane_q4kv_governed',
        'kv_allocated_now_mb', 625,
        'max_context_admitted', 10000,
        'preemption_group', 'orchestrator_hot',
        'toks_in_per_s_measured', null,
        'toks_out_per_s_measured', null
    ) AS receipt_fields
FROM fabric_components;

INSERT INTO lucidota_control.schema_owner_manifest (
    surface_id, canonical_owner, packet_class, surface_kind, approval_required, notes, detail
) VALUES
    ('indy_reads_target_model_loadout_current', 'lucidota_canon', 'typed_packet', 'view', true, 'Indy_READs target model loadout canonicalization surface.', '{"source":"217_indy_reads_target_loadout_current.sql","postgrest_safe":true}'::jsonb),
    ('indy_reads_vram_coprocessor_fabric_current', 'lucidota_canon', 'typed_packet', 'view', true, 'Indy_READs VRAM/coprocessor fabric canonicalization surface.', '{"source":"217_indy_reads_target_loadout_current.sql","postgrest_safe":true}'::jsonb)
ON CONFLICT (surface_id) DO UPDATE SET
    canonical_owner = EXCLUDED.canonical_owner,
    packet_class = EXCLUDED.packet_class,
    surface_kind = EXCLUDED.surface_kind,
    approval_required = EXCLUDED.approval_required,
    active = true,
    notes = EXCLUDED.notes,
    detail = EXCLUDED.detail,
    updated_at = now();

INSERT INTO lucidota_canon.api_route_catalog (
    route_id, method, path_pattern, description, target, sample_request, sample_response, status
) VALUES
    ('indy_reads_target_model_loadout_current', 'GET', '/indy_reads_target_model_loadout_current', 'Indy_READs target model loadout canonicalization rows.', 'lucidota_canon.indy_reads_target_model_loadout_current', '{"order":"target_rank.asc"}', '{"model_key":"needle_26m_router_swarm","logical_lane_count":20}', 'implemented'),
    ('indy_reads_vram_coprocessor_fabric_current', 'GET', '/indy_reads_vram_coprocessor_fabric_current', 'Indy_READs VRAM/coprocessor fabric canonicalization packet.', 'lucidota_canon.indy_reads_vram_coprocessor_fabric_current', '{"limit":"1"}', '{"fabric_id":"indy_reads_vram_coprocessor_fabric_current"}', 'implemented')
ON CONFLICT (route_id) DO UPDATE SET
    method = EXCLUDED.method,
    path_pattern = EXCLUDED.path_pattern,
    description = EXCLUDED.description,
    target = EXCLUDED.target,
    sample_request = EXCLUDED.sample_request,
    sample_response = EXCLUDED.sample_response,
    status = EXCLUDED.status,
    updated_at = now();

GRANT SELECT ON lucidota_canon.indy_reads_target_model_loadout_current,
    lucidota_canon.indy_reads_vram_coprocessor_fabric_current
TO mfspx, lucidota_postgrest_anon, ironclaw;
GRANT SELECT ON lucidota_canon.api_route_catalog TO mfspx, lucidota_postgrest_anon, ironclaw;

NOTIFY pgrst, 'reload schema';

COMMIT;
