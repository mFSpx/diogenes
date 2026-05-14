-- Model runtime registry for LUCIDOTA.

CREATE SCHEMA IF NOT EXISTS lucidota_runtime;

CREATE TABLE IF NOT EXISTS lucidota_runtime.model_candidate (
    model_id text PRIMARY KEY,
    role text NOT NULL CHECK (role IN (
        'listener',
        'router',
        'tactician',
        'heavy_hitter',
        'embedding',
        'reranker',
        'other'
    )),
    source_url text NOT NULL,
    local_path text,
    license text NOT NULL DEFAULT 'unknown',
    parameter_count bigint,
    quantization text,
    expected_vram_mb integer CHECK (expected_vram_mb IS NULL OR expected_vram_mb >= 0),
    benchmark_status text NOT NULL DEFAULT 'untested' CHECK (benchmark_status IN (
        'untested',
        'testing',
        'accepted',
        'rejected',
        'watch'
    )),
    notes text NOT NULL DEFAULT '',
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS lucidota_runtime.resident_loadout (
    loadout_id text PRIMARY KEY,
    active boolean NOT NULL DEFAULT false,
    description text NOT NULL,
    target_gpu text NOT NULL DEFAULT 'GTX 1650 4GB',
    budget_vram_mb integer NOT NULL DEFAULT 4096,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS lucidota_runtime.resident_loadout_slot (
    loadout_id text NOT NULL REFERENCES lucidota_runtime.resident_loadout(loadout_id) ON DELETE CASCADE,
    slot_name text NOT NULL,
    model_id text NOT NULL REFERENCES lucidota_runtime.model_candidate(model_id),
    instance_count integer NOT NULL DEFAULT 1 CHECK (instance_count > 0),
    expected_vram_mb integer CHECK (expected_vram_mb IS NULL OR expected_vram_mb >= 0),
    priority integer NOT NULL DEFAULT 100,
    notes text NOT NULL DEFAULT '',
    PRIMARY KEY (loadout_id, slot_name)
);

CREATE TABLE IF NOT EXISTS lucidota_runtime.adapter_cartridge (
    adapter_id text PRIMARY KEY,
    target_model_id text REFERENCES lucidota_runtime.model_candidate(model_id),
    root_anchor text,
    source_manifest jsonb NOT NULL DEFAULT '{}'::jsonb,
    license_status text NOT NULL DEFAULT 'unknown' CHECK (license_status IN (
        'unknown',
        'lawful_private',
        'redistributable',
        'blocked',
        'needs_review'
    )),
    local_path text,
    expected_vram_mb integer CHECK (expected_vram_mb IS NULL OR expected_vram_mb >= 0),
    activation_status text NOT NULL DEFAULT 'planned' CHECK (activation_status IN (
        'planned',
        'available',
        'testing',
        'accepted',
        'rejected'
    )),
    notes text NOT NULL DEFAULT '',
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

INSERT INTO lucidota_runtime.model_candidate (
    model_id,
    role,
    source_url,
    local_path,
    license,
    parameter_count,
    quantization,
    expected_vram_mb,
    benchmark_status,
    notes
) VALUES
    (
        'needle-26m',
        'router',
        'https://github.com/cactus-compute/needle',
        '01_REPOS/needle',
        'MIT',
        26000000,
        'upstream JAX checkpoint, quantization TBD',
        NULL,
        'testing',
        'Function-call router candidate. Local benchmark required before tok/s claims are accepted.'
    ),
    (
        'mamba2-1.3b-listener',
        'listener',
        'TBD',
        NULL,
        'unknown',
        1300000000,
        '4-bit target',
        NULL,
        'watch',
        'Always-on stream listener target; exact model artifact not selected.'
    ),
    (
        'deepseek-1.5b-indy_reads-reads',
        'heavy_hitter',
        'TBD',
        NULL,
        'unknown',
        1500000000,
        '4-bit target',
        NULL,
        'watch',
        'Resident Indy_Reads heavy-hitter/adaptor host target; exact artifact not selected.'
    )
ON CONFLICT (model_id) DO UPDATE SET
    role = EXCLUDED.role,
    source_url = EXCLUDED.source_url,
    local_path = EXCLUDED.local_path,
    license = EXCLUDED.license,
    parameter_count = EXCLUDED.parameter_count,
    quantization = EXCLUDED.quantization,
    expected_vram_mb = EXCLUDED.expected_vram_mb,
    benchmark_status = EXCLUDED.benchmark_status,
    notes = EXCLUDED.notes,
    updated_at = now();

INSERT INTO lucidota_runtime.resident_loadout (
    loadout_id,
    active,
    description,
    target_gpu,
    budget_vram_mb
) VALUES (
    'gtx1650-special-forces-v0',
    true,
    '1 listener, 6 routers, 2 tacticians, 1 resident heavy hitter; exact artifacts pending benchmarks.',
    'GTX 1650 4GB',
    4096
)
ON CONFLICT (loadout_id) DO UPDATE SET
    active = EXCLUDED.active,
    description = EXCLUDED.description,
    target_gpu = EXCLUDED.target_gpu,
    budget_vram_mb = EXCLUDED.budget_vram_mb;

INSERT INTO lucidota_runtime.resident_loadout_slot (
    loadout_id,
    slot_name,
    model_id,
    instance_count,
    expected_vram_mb,
    priority,
    notes
) VALUES
    (
        'gtx1650-special-forces-v0',
        'listener',
        'mamba2-1.3b-listener',
        1,
        NULL,
        10,
        'Long-running stream listener.'
    ),
    (
        'gtx1650-special-forces-v0',
        'router_swarm',
        'needle-26m',
        6,
        NULL,
        20,
        'Function-call/JSON routing lane.'
    ),
    (
        'gtx1650-special-forces-v0',
        'indy_reads_heavy_hitter',
        'deepseek-1.5b-indy_reads-reads',
        1,
        NULL,
        40,
        'Resident adapter host target.'
    )
ON CONFLICT (loadout_id, slot_name) DO UPDATE SET
    model_id = EXCLUDED.model_id,
    instance_count = EXCLUDED.instance_count,
    expected_vram_mb = EXCLUDED.expected_vram_mb,
    priority = EXCLUDED.priority,
    notes = EXCLUDED.notes;


-- Cleanup obsolete transitional persona slot name from the temporary Local_Reads rename.
DELETE FROM lucidota_runtime.resident_loadout_slot
WHERE loadout_id = 'gtx1650-special-forces-v0'
  AND slot_name = 'local_reads_heavy_hitter';
DELETE FROM lucidota_runtime.model_candidate
WHERE model_id = 'deepseek-1.5b-local_reads-reads';
