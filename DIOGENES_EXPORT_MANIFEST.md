# DIOGENES Export Manifest

This repository is the source-backed wiring layer for LUCIDOTA.
`/DIOGENES` is intended to mirror the project state that matters for rebuilding the machine:
schemas, scripts, prompts, configs, manifests, docs, tests, handoffs, and the thin orchestration surfaces that wire the system together.

## Included by default

- `AGENTS.md`
- `00_PROJECT_BRAIN/`
- `01_REPOS/README.md`
- `01_REPOS/PocketFlow/`
- `ALGOS/`
- `GOALS/`
- `README.md`
- `TOOLS/`
- `core/`
- `scripts/`
- `src/`
- `tests/`
- `06_SCHEMA/`
- small repo wiring files such as `check_diogenes.sh`, `claw`, and top-level policy/docs

## Excluded by default

- virtual environments and caches
- generated build products
- node modules
- Rust `target/`
- runtime state and vault data
- PostgreSQL data directories, WAL, dumps, and local instances
- logs
- large outputs and receipts when they become exhaust
- binaries, model weights, training corpora, archives, and unpacked large drops

## Explicit exclusions

The backup lane excludes, unless a file is later promoted by hand:

- `.venv/`
- `node_modules/`
- `target/`
- `__pycache__/`
- `.pytest_cache/`
- `.mypy_cache/`
- `.ruff_cache/`
- `.cache/`
- PostgreSQL data directories and WAL
- `*.pyc`, `*.so`, `*.dll`, `*.exe`
- `*.bin`, `*.pt`, `*.pth`, `*.safetensors`, `*.gguf`, `*.onnx`
- `*.db`, `*.sqlite*`, `*.dump`, `*.bak`
- `*.log`, archives, caches, and bulk generated exhaust

## Live mirror command

Use the mirror runner to generate a current manifest:

```bash
.venv/bin/python scripts/diogenes_mirror.py --dry-run --json-out 05_OUTPUTS/diogenes/diogenes_mirror_dry_run.json
```

For a DB-backed snapshot:

```bash
.venv/bin/python scripts/diogenes_mirror.py --json-out 05_OUTPUTS/diogenes/diogenes_mirror.json
```

## Current scan receipt

- Dry-run mirror receipt: `05_OUTPUTS/diogenes/diogenes_mirror_20260528T064705282932Z.json`
- Included files: `88,481`
- Excluded files: `365,449`
- Registry candidates: `882`
- Kind counts:
  - `db-native`: `279`
  - `runtime-core`: `338`
  - `experimental`: `87,641`
  - `quarantine`: `223`

The manifest is the backup receipt. The mirror excludes data exhaust on purpose; it preserves wiring, code, schemas, and operational contracts.
