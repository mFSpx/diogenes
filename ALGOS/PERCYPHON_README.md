# Percyphon -- Zero-VRAM Procedural Entity Generator

## What It Is

Percyphon is a deterministic, zero-VRAM procedural entity generator that produces
128-slot xxhash128 identity scaffolds from any seed string.  It requires no GPU, no
model inference, no API calls, and no randomness.  Every output is a pure function
of its seed.

Constitutional constraint: every scaffold carries the authority tag
`procedural_scaffold_candidate_not_truth`.  Percyphon emits candidates for routing
and identity; it never writes canonical graph truth.

## 128-Slot Architecture

Each villager scaffold contains exactly 128 slots of coordinate data:

| Range | Count | Type | Description |
|-------|-------|------|-------------|
| Slots 1-28 | 28 | fixed identity mask | Mirrors CKDOG1 soul positions. Stable per seed. |
| Slots 29-128 | 100 | procedural verbosity expansion | Runtime fluid domain slots. |

### Per-Slot Fields

Each `ProceduralSlot` contains:

- **slot_index** (1-indexed, 1-128)
- **name** -- e.g. `Villager-1234` (deterministic from seed + slot index)
- **alias** -- e.g. `Alias-ABCD` (short hex handle)
- **persona** -- one of six archetypes: `ledger`, `runner`, `witness`, `archivist`,
  `carrier`, `scribe`
- **uuid** -- deterministic UUID derived from SHA-256 of `{seed}:fixed:{slot_index}`
  or `{seed}:fluid:{slot_index}:{villager_ref}`
- **ternary_offset** -- per-slot ternary state: -1, 0, or +1 (modulated by
  `psyche_wrath_velocity` for fixed slots and `psyche_forensic_shield_ratio`
  for procedural slots)
- **coord_128** -- 128-bit integer coordinate via xxhash128 (SHA-256[:16] fallback)

### Scaffold Object

The top-level scaffold returned by `procedural_entity_generator`:

- `schema` -- versioned envelope (`lucidota.percyphon.scaffold.v1`)
- `zero_vram` -- always `True` (load-bearing, non-negotiable)
- `seed` -- concatenation of up to 5000 villager seeds
- `slots` -- array of 128 `ProceduralSlot` dicts
- `slot_count`, `fluid_slot_count`
- `source_count` -- min(5000, len(villagers))
- `uuid` -- deterministic village UUID from seed
- `name` -- name of slot 1 (primary identity handle)
- `relevance_confidence_bps` -- relevance in basis points (0-10000)
- `authority` -- always `"procedural_scaffold_candidate_not_truth"`

## Hash Backend

Percyphon attempts to use `xxhash.xxh128()` for 128-bit coordinate hashing.
If the `xxhash` package is not installed, it falls back to SHA-256, taking
the first 16 bytes of the digest:

```python
# Primary (when xxhash is available):
xxhash.xxh128(data).intdigest()

# Fallback (xxhash not available):
int.from_bytes(hashlib.sha256(data.encode()).digest()[:16], "big")
```

To install xxhash:

```bash
pip install xxhash
```

## File Inventory

| File | Purpose |
|------|---------|
| `ALGOS/percyphon.py` | Core generator: scaffold, matrix, slot naming, ternary state, relevance scoring |
| `ALGOS/percyphon_comms_filter.py` | Read-only keyword-set comms pattern classifier for villager enrichment |
| `scripts/percyphon_village_seed.py` | Deterministic village seeder -- upserts 5000 scaffolds into percyphon_village |
| `scripts/percyphon_kernel_bridge.py` | Routes Percyphon scaffolds through Diogenes kernel authorization gate |
| `scripts/luci_percyphon.py` | Runtime surface renderer (reads PostgREST) |
| `scripts/luci_percyphon_emit.py` | One-shot scaffold emitter (stdout + optional DB write) |
| `06_SCHEMA/126_percyphon_village.sql` | Core percyphon_village table (JSONB slot array, one row per villager) |
| `06_SCHEMA/128_percyphon_gin_index.sql` | GIN index on percyphon_village.slots for JSONB lookup |
| `06_SCHEMA/208_percyphon_runtime.sql` | Runtime views: percyphon_village_matrix, percyphon_current |
| `06_SCHEMA/218_percyphon_villager.sql` | Normalized per-slot villager table (one row per slot, 128 rows per villager) |
| `tests/test_percyphon_kernel_bridge.py` | Primary test file |
| `00_PROJECT_BRAIN/organ_registry/40_percyphon_math.json` | Organ registry with 117 capability entries |
| `ALGOS/PERCYPHON_README.md` | This file -- Percyphon village documentation |

### Procedural Matrix Generator

`procedural_matrix_generator()` produces a deterministic 129x5000 matrix snapshot:

- **Rows 1-28**: Identity trunk (fixed identity mask rows)
- **Rows 29-128**: Procedural engine bands (fluid domain expansion rows)
- **Row 129**: Interlock summary / pressure seal
- **Columns 1-5000**: Villager souls / vUUID coordinate axis

Each cell is a compact `{term}:{hash_prefix}` token, deterministic from the
seed and row/column indices. The matrix is advisory geometry only (authority:
`procedural_scaffold_candidate_not_truth`).

```python
from ALGOS import percyphon

matrix = percyphon.procedural_matrix_generator(
    ['seed-a', 'seed-b'],
    row_count=129,         # must be 129
    column_count=5000,     # must be 5000
    source="Runtime",
)
print(matrix['schema'])  # lucidota.percyphon.matrix.v1
print(len(matrix['rows']))  # 129
```

The helper functions `_matrix_terms(villagers, column_count)` and
`_matrix_row_label(row_index)` support matrix construction by normalizing
seed terms and classifying row types.

## Integration with RETE Bandit and BRAG Pipeline

### BRAG ABSURD Worker

The BRAG pipeline (`scripts/brag_absurd_worker.py`) chains four algorithmic stages:

1. **RETE bandit gate** (`ALGOS.rete_bandit_gate.apply_rete_bandit`) -- routes each
   document packet to the correct ontology pass and algorithm pool
2. **Percyphon** (`ALGOS.percyphon`) -- generates deterministic 128-slot xxhash128
   identity scaffolds
3. **LTC** (`ALGOS.ltc`) -- orders chunks by temporal evidence flow via ODE
4. **XHash** -- wraps every shape with a hash

The worker imports percyphon directly:
```python
from ALGOS import percyphon
```

### Routing Stack Position

Percyphon occupies Layer 1 of the four-layer routing stack (documented in
OPUS_BRIEFING.md):

- **Layer 1 -- Percyphon** (zero-VRAM alias router): generates identity scaffolds
- **Layer 2 -- Diogenes Kernel**: the authority kernel that decides routing
- **Layer 3 -- ALGOS/ (200+ deterministic algorithms)**: the computational bulk
- **Layer 4 -- LLM models**: handles edges, summarization, extraction

### RETE Bandit Gate

The RETE bandit gate (`ALGOS/rete_bandit_gate.py`) does not directly reference
Percyphon, but it is called from the BRAG pipeline where Percyphon scaffolds are
already present. The typical data flow is:

```
Raw document -> RETE bandit gate (route decision)
             -> Percyphon scaffold (identity hash for routing coordinates)
             -> LTC ordering -> XHash wrap -> PostgREST ingestion
```

## Identity Flow Through the System

```
                         Seed string (ontology anchor, domain handle, etc.)
                                     |
                                     v
                      procedural_entity_generator()
                                     |
                     +---------------+---------------+
                     |                               |
                     v                               v
              Fixed slots 1-28              Procedural slots 29-128
              (identity mask, mirrors       (fluid domain expansion,
               CKDOG1 soul positions)        verbosity coordinates)
                     |                               |
                     +-------+-----------+-----------+
                             |           |
                             v           v
                    128-slot scaffold with:
                    - name / alias / persona per slot
                    - ternary_offset (-1, 0, +1)
                    - coord_128 (128-bit xxhash)
                    - relevance_confidence_bps (0-10000)
                             |
                     +-------+-------+
                     |               |
                     v               v
            percyphon_village   percyphon_villager
            (JSONB array,       (normalized table,
             1 row per seed)     128 rows per seed)
                     |               |
                     v               v
            percyphon_current      SQL queries on
            (latest surface)       individual slots
                     |
                     v
            PostgREST API -> luci_percyphon.py (render)
```

### From Seed to Surfaces

1. **Seed creation**: Seeds are generated deterministically from ontology anchors
   (GO-25), domain handles, runtime concepts, workflow nodes, and identity markers.
   See `scripts/percyphon_village_seed.py` for the 50-entry sample corpus and
   synthetic expansion to 5000.

2. **Scaffold generation**: `procedural_entity_generator()` in `percyphon.py`
   takes a list of villager seeds, generates 128 slots with names, aliases,
   personas, ternary offsets, and 128-bit coordinates.

3. **DB ingestion**: `percyphon_village_seed.py` upserts into
   `lucidota_go.percyphon_village`. The `scaffold_log_entry()` function in
   `percyphon.py` can also log to `lucidota_go.percyphon_scaffold_log`.

4. **Surface views**: `208_percyphon_runtime.sql` creates normalized views
   (`percyphon_village_matrix`, `percyphon_current`) with 128 individual slot
   columns for PostgREST consumption.

5. **Comms enrichment**: `percyphon_comms_filter.py` provides read-only comms
   pattern classification (VPN, proxy, burn_phone, identity_change,
   anonymous_relay) via `enrich_villager()`.

## DB Schema -- Percyphon Tables

### `lucidota_go.percyphon_village` (126_percyphon_village.sql)

One row per villager. All 128 slots stored as a JSONB array.

| Column | Type | Description |
|--------|------|-------------|
| vuuid | TEXT PK | Deterministic UUID from seed |
| name | TEXT | Slot-1 identity name |
| persona | TEXT | Slot-1 persona archetype |
| alias | TEXT | Slot-1 alias handle |
| ternary_state | SMALLINT | Slot-1 ternary offset (-1, 0, 1) |
| slots | JSONB | Full 128-slot array |
| relevance_confidence_bps | SMALLINT | 0-10000 |
| seed | TEXT | Seed string |
| authority | TEXT | Always `procedural_scaffold_candidate_not_truth` |
| updated_at | TIMESTAMPTZ | Last update timestamp |

### `lucidota_korpus.percyphon_villager` (218_percyphon_villager.sql)

Normalized per-slot representation. One row per slot per villager.

| Column | Type | Description |
|--------|------|-------------|
| villager_id | UUID PK | Auto-generated |
| slot | INTEGER | 0-indexed slot (0-127) |
| seed | TEXT | Villager seed |
| name | TEXT | Slot name |
| persona | TEXT | Persona archetype |
| identity_hash | TEXT | Deterministic hash |
| slot_type | TEXT | `fixed_identity` or `procedural` |
| created_at | TIMESTAMPTZ | Row creation time |

## Audit Findings

### Bug: Slot Boundary Overflow in `procedural_entity_generator`

File: `ALGOS/percyphon.py`, line 135.

```python
# Current (BUG):
for slot_index in range(1, 30):         # generates 29 slots (1..29)
    ...

for idx in range(fluid_slots):          # fluid_slots=100, indices 29..128
    slot_index = 29 + idx
```

**Problem**: `range(1, 30)` generates 29 fixed slots instead of 28. The
procedural loop starts at `slot_index = 29 + 0 = 29`, so **slot 29 is generated
twice** -- once as the last fixed slot and once as the first fluid slot.

**Result**: Total scaffold has 129 entries (not 128), with slot 29 duplicated
and different hash-derived values (different seeds). `fluid_slot_count` reports
101 instead of 100.

**Fix**: Change `range(1, 30)` to `range(1, 29)` on line 135.

### Edge Case: Empty Villagers List

When `villagers` is `None` or empty, the generator falls back to a baseline seed:
```python
seed = "|".join(str(v) for v in (villagers or [])[:5000]) or "lucidota-villager-baseline"
```
This is safe but produces a single deterministic villager. In the procedural
loop, `villagers[idx % len(villagers)]` would divide by zero if `villagers`
is empty. The fallback handles this:
```python
villager_ref = villagers[idx % len(villagers)] if villagers else seed
```

### Edge Case: xxhash Not Installed

When `xxhash` is not available, the SHA-256 fallback is used. This produces
different 128-bit coordinates than xxhash would, which means scaffolds
generated on different machines with different packages will not match.
Install xxhash for deterministic cross-machine reproducibility.

**Current status (2026-06-05)**: `xxhash` has been installed on this system
(`pip install xxhash --break-system-packages`). The SHA-256 fallback remains
in the code as a safety net for environments without xxhash.

### Issue: psycopg2 Top-Level Import

`percyphon.py` has `import psycopg2` at the module level (line 17), which
means even pure hash functions (`_sha256_hex`, `_xxhash128_int`,
`_uuid_from_sha256`, `_slot_name`, etc.) cannot be imported without
`psycopg2` installed. Since `psycopg2` is only used in
`scaffold_log_entry()`, it could be moved to a lazy import inside that
function to keep the rest of the module importable in lightweight
environments.

**Current status (2026-06-05)**: `psycopg2-binary` has been installed on
this system as a workaround (`pip install psycopg2-binary --break-system-packages`).

### Edge Case: `_ternary_state` Clamping

The `_ternary_state` function computes:
```python
spread = int(h[:2], 16) % 3      # 0, 1, or 2
raw = base_offset + spread - 1   # shift to -1..+1 range
return max(-1, min(1, raw))      # clamp
```
If `base_offset` is outside [-1, 1] (e.g. extreme `psyche_wrath_velocity`
values), the clamp catches it. But for large `base_offset` values (e.g. > 2),
all slots would clamp to 1, losing the per-slot spread. In normal usage
`psyche_wrath_velocity` and `psyche_forensic_shield_ratio` are 0.0 and the
clamp is inactive.

### BPS Snap in `scaffold_log_entry`

The function snaps `relevance_confidence_bps` to the nearest value in
`{0, 2, 4, 6, 10, 50, 69, 150}`. This set matches the valid BPS values in
`percyphon_comms_filter.py`, ensuring consistency. However, the BPS rounding
means the DB value may not reflect the raw hash-derived score.

### `scaffold_log_entry` Returns None

The function signature claims `-> dict`, but it returns `None` when a row
already exists (ON CONFLICT DO NOTHING + no RETURNING row). Callers should
handle `None`.

### Matrix Generator Parameter Enforcement

`procedural_matrix_generator` strictly enforces `row_count == 129` and
`column_count == 5000`. Passing other values raises `ValueError`, which
prevents accidental misconfiguration of the matrix shape.

### Gaps (from OPUS_BRIEFING.md)

- **percyphon_scaffold_log** table referenced in `scaffold_log_entry()` does
  not exist in any applied SQL schema -- it is a forward-reference.
- Per-slot independent psyche modulation not built (currently scalar).
- Scaffold revocation work order type not in ABSURD queue.
- Village population management (which 5000 are active) is not automated.
- The 88 procedural domain slots are generated but not consumed by downstream
  routing or scoring functions.

## Usage Examples

```bash
# Generate a scaffold for a single seed
python3 -c "
from ALGOS import percyphon
s = percyphon.villager_scaffold('my-seed')
print(s['name'], s['slot_count'], 'slots')
"

# Seed the full village (5000 rows) -- dry run
python3 scripts/percyphon_village_seed.py --count 5000 --dry-run

# Seed to database (requires running Postgres + LUCIDOTA_GO_STORAGE_DSN)
python3 scripts/percyphon_village_seed.py --count 5000

# Classify a comms pattern
python3 -c "
from ALGOS.percyphon_comms_filter import classify_comms_pattern
print(classify_comms_pattern('Using WireGuard VPN with kill switch'))
"

# Run comms filter smoke test
python3 ALGOS/percyphon_comms_filter.py

# View village surfaces from PostgREST
python3 scripts/luci_percyphon.py current
python3 scripts/luci_percyphon.py matrix --limit 5
```

## Design Principles

1. **Zero VRAM, always.** `zero_vram: True` is load-bearing and non-negotiable.
2. **Deterministic.** Same seed always produces the same scaffold.
3. **No model calls.** Pure arithmetic and hashing.
4. **No randomness.** All variation comes from hash diffusion.
5. **Candidate layer only.** Every output is `procedural_scaffold_candidate_not_truth`.
6. **No canonical graph truth.** Percyphon does not write to graph tables.
