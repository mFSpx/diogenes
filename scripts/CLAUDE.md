# SCRIPTS — Worker/Gate/Receipt Layer

This directory contains ~459 scripts forming the active workforce of LUCIDOTA.

## Key Rules

- **Receipts, not prose.** Every script that does work must produce a receipt in `05_OUTPUTS/`.
- **Mutation class required.** Every script must be one of: `read_only`, `receipt_only`, `custody_writer`, `queue_writer`, `candidate_writer`, `authority_gate`, `materializer`, `external_effect`.
- **No canonical graph writes from ordinary workers.** Use the promotion gate.
- **Prefer reading from Postgres over file-based state.**
- **New scripts must be registered** in `ALLOWED_EXTERNAL_COMMANDS` in `scripts/absurd_queue_spine.py` if they're used as queue workers.
- **Search `dev_library_scan.py --query <topic>` before inventing new scripts.**

## Conventions

- `scripts/_lib/` — shared scaffolding for CLI tools
- `scripts/ironclaw_host_os/` — ironclaw integration scripts
- Name: `snake_case.py` for Python, `snake_case.sh` for shell
