# API MANUAL — LIVE OPERATOR SURFACE

This document is a thin human-facing mirror of the live PostgREST manual/control API.
It is not authority by itself; the DB routes are authority.

## What to read first

```bash
curl -sS http://127.0.0.1:3000/manual_current?limit=1
curl -sS http://127.0.0.1:3000/active_goal?limit=1
curl -sS http://127.0.0.1:3000/daemon_status?limit=1
curl -sS http://127.0.0.1:3000/api_route_catalog?limit=25
```

## Live operator manual

`/manual_current` now exposes a DB-backed operator packet with:

- route list
- auth expectations
- work-order flow
- live goal / daemon / registry surface
- next executable commands
- retired surfaces

If `/manual_current` and the route catalog disagree, the DB route wins.

## Command front door

Use `luci` or `clawd` as the operator shell. Current helpful commands:

```bash
.venv/bin/python scripts/luci_help_manual.py manual
.venv/bin/python scripts/indy_daemon.py --once --json
.venv/bin/python scripts/indy_runtime_broker.py snapshot --json
.venv/bin/python scripts/indy_runtime_broker.py packet --work-order-id <uuid> --json
.venv/bin/python scripts/prompt_api_client.py --work-order-id <uuid> --json
```

## Prompt ledger commands

```bash
curl -sS http://127.0.0.1:3000/prompts_filed?limit=5
curl -sS http://127.0.0.1:3000/prompt_recent?limit=5
curl -sS http://127.0.0.1:3000/prompt_unlinked?limit=5
curl -sS http://127.0.0.1:3000/prompt_catalog_status?limit=1
curl -sS -X POST http://127.0.0.1:3000/rpc/file_prompt \
  -H 'content-type: application/json' \
  -d '{"source":"operator","raw_prompt_text":"<prompt>"}'
curl -sS -X POST http://127.0.0.1:3000/rpc/link_prompt_work_order \
  -H 'content-type: application/json' \
  -d '{"p_prompt_id":"<uuid>","p_work_order_uuid":"<uuid>"}'
curl -sS -X POST http://127.0.0.1:3000/rpc/decompose_prompt_to_work_orders \
  -H 'content-type: application/json' \
  -d '{"prompt_id":"<uuid>"}'
```

## Live routes that matter

- `/manual_current`
- `/canon_current`
- `/canon_versions`
- `/active_goal`
- `/api_route_catalog`
- `/api_workflow_registry`
- `/capability_registry`
- `/model_registry`
- `/provider_registry`
- `/workflow_registry`
- `/daemon_status`
- `/bytewax_compact_windows`
- `/indy_queue`
- `/indy_responses`
- `/rpc/cloud_packet`
- `/prompts_filed`
- `/prompt_work_order_links`
- `/prompt_recent`
- `/prompt_unlinked`
- `/prompt_catalog_status`
- `/book_source`
- `/book_scan`
- `/book_read_queue`
- `/book_note`
- `/lora_candidate`
- `/lora_adapter`
- `/training_job`
- `/book_receipt`

## DB truth rules

- Postgres/PostgREST is the manual and control surface.
- Files are cache, export, or evidence unless the API points to them.
- `BOOKS` file watching is legacy; DB rows and work orders are authoritative.
- Prompt filing is DB-backed: raw prompt text is preserved, idempotency prevents duplicates, and unlinked prompts must stay visible.
- Cloud prompts come only from bounded PostgREST packets.
- Receipts are proof, not decoration.

## Indy_READs daemon posture

- `scripts/indy_daemon.py` is the DB-driven front door.
- It snapshots the manual/registry surface first.
- It then polls the queue and runs `indy_reads.py chat --respond-once --json` when work exists.
- `/indy_responses` and the live dialogue row are the response surface.

## Book / LoRA posture

- The book pipeline is DB-visible: source, scan, read queue, note, candidate, adapter, training job, receipt.
- The old BOOKS watcher script is legacy only; it is not authority.

## Registry posture

- `model_registry` and `provider_registry` are the live role discovery surface.
- `workflow_registry` is the live workflow contract surface.
- `daemon_status` is the live health surface.

## Finish standard

Nothing is done because the file exists.
A slice is done when:

1. the live route exists,
2. the route is readable,
3. a test or smoke proves the behavior,
4. a receipt exists,
5. and the manual reflects the change.
