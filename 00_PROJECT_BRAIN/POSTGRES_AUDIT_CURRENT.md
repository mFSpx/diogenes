# LUCIDOTA POSTGRES AUDIT — CURRENT

Generated: `2026-05-26T23:39:40.181545Z`

## Databases

- State DB: `postgresql:///lucidota_state`
- Storage DB: `postgresql:///lucidota_storage`

## State DB schemas

| Schema | Tables | Notes |
|---|---:|---|
| lucidota_control | 60 | queue/workflow/contracts/runtime facts |
| lucidota_learning | 14 | river/bytewax/learning hints |
| lucidota_runtime | 10 | runtime state and receipts |
| lucidota_go | 1 | graph promotion evidence |
| lucidota_indy | 3 | Indy_READs state |
| lucidota_pivot | 4 | routing/selection pivots |
| lucidota_ternary | 3 | ternary lab |
| lucidota_vault | 5 | vault/secret custody |
| lucidota_bus | 2 | bus/wake events |

## Storage DB schemas

| Schema | Tables | Notes |
|---|---:|---|
| lucidota_korpus | 43 | raw/derived custody, present in storage not state |
| lucidota_go | 23 | storage-side graph staging |
| lucidota_learning | 8 | storage-side learning artifacts |
| lucidota_archaeology | 20 | audit/archaeology receipts |
| lucidota_spine | 12 | spine/runtime contracts |
| lucidota_runtime | 5 | runtime artifacts |
| lucidota_vault | 5 | vault artifacts |
| lucidota_indy | 6 | Indy artifacts |
| lucidota_chatdump | 4 | chat timeline imports |
| lucidota_commdump | 3 | comm timeline imports |
| lucidota_etl | 5 | ETL pipeline receipts |
| lucidota_hardmath | 5 | hard-truth math audit |
| lucidota_investigation | 10 | evidence/investigation |
| lucidota_semantic | 1 | semantic staging |
| lucidota_workflow_foundry | 1 | workflow foundry |

## Current runtime facts

```json
{
  "absurd_queue_health": {
    "dead_letter_rows": 45,
    "dead_lettered": 78,
    "fact": "absurd_queue_stale_contract_failures_reconciled",
    "failed": 0,
    "queued": 0,
    "queues_treated": [
      "control",
      "graph_promotion",
      "marrow_loop",
      "model_fabric"
    ],
    "reconciled_at": "2026-05-26T14:28:13.00939-07:00",
    "resolution": "manual_stale_reconcile_complete",
    "succeeded": 2455
  },
  "autonomous_hitman_loop": {
    "command": "systemd-inhibit --who=lucidota-hitman-loop --what=idle:sleep:shutdown:handle-lid-switch --mode=block /home/mfspx/LUCIDOTA/scripts/lucidota_hitman_loop.sh",
    "lock_file": "/tmp/lucidota_hitman_loop.lock",
    "log": "/tmp/lucidota_hitman_loop.log",
    "loop_script": "scripts/lucidota_hitman_loop.sh",
    "next_start": "@reboot + */5 * * * *",
    "status": "scheduled"
  },
  "llxprt_groq_login_wired": {
    "model": "llama-3.1-8b-instant",
    "provider": "groq",
    "script": "scripts/llxprt_groq_login_bind.sh",
    "status": "wired"
  },
  "loop_next_order_payload": {
    "next_single_order": "python3 scripts/system_runtime_facts_refresh.py --execute",
    "next_step": "monitor_and_await_new_failures",
    "queue_repair_state": "no_failed_absurd_jobs",
    "updated_at": "2026-05-26T14:28:18.458992-07:00"
  },
  "runtime_facts_refresh": {
    "absurd_queue": {
      "failed": 0,
      "queued": 0,
      "succeeded": 2465,
      "total": 2545
    },
    "chrono": {
      "files_covered": 18627,
      "temporal_claims": 43922
    },
    "chrono_service_rc": 0,
    "conversation_command": {
      "executed": 4,
      "total": 27
    },
    "graph": {
      "graph_edges": 1379886,
      "graph_items": 275547,
      "materializations": 9
    }
  }
}
```

## Operational reading

- `lucidota_state` is the active control plane for queues, contracts, learning hints, and runtime facts.
- `lucidota_storage` holds the heavier custody and ETL/graph/workflow evidence.
- Current queue state: succeeded `2455`, dead-lettered `78`, unresolved dead-letter rows `0`.
- Current graph/materialization snapshot from runtime facts: `graph_items=275547`, `graph_edges=1379886`, `materializations=9`.
- Storage-side KORPUS exists in `lucidota_storage`, not `lucidota_state`. That distinction matters.

## What to audit next if the DB changes

1. Re-run `python3 scripts/system_runtime_facts_refresh.py --execute`
2. Re-run this doc refresh script
3. Re-check queue counts, schema counts, and unresolved dead-letter rows
4. Keep receipts in `05_OUTPUTS/system_docs/`
