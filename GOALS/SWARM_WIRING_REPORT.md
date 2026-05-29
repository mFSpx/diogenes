# Swarm Wiring Report — Current Proof Slice

Status: partial but real; local swarm is pushing tokens, Groq is wired but current key is rejected by Groq (401).

## What is wired now

- Local fabric status: `05_OUTPUTS/goals/goal_model_fabric_control_20260526T192322Z.json` (10/10 health endpoints online after probes).
- Local invocation CLI added: `scripts/local_model_chat_cli.py` via `scripts/model_runner_cli.py local-chat`.
- Usage ledger added: `scripts/swarm_usage_ledger.py`.
- Current usage ledger receipt: `05_OUTPUTS/goals/swarm_usage_ledger_20260526T192035307489Z.json`.
- Groq current delegate attempt: `05_OUTPUTS/goals/groq_goal_delegate_20260526T191923970797Z.json`; subreceipt is blocked by `groq_http_error:401`, so no new Groq tokens were honestly counted.
- Historical Groq execute receipts remain summarized in `GOALS/SITREP_CURRENT_WIRED_STATUS.md`; this report counts only the current proof slice.

## Current token accounting

| lane | accounted tokens | share | target |
|---|---:|---:|---:|
| main_agent | 0 | 0.000 | 0.25 |
| local | 504 | 1.000 | 0.20 |
| groq | 0 | 0.000 | 0.55 |

Main-agent account tokens are not available from repo receipts, so this ledger does not fake the requested 25%.

## Local token-push receipts

| model lane | tokens | source | receipt |
|---|---:|---|---|
| deepseek | 50 | provider_usage | `05_OUTPUTS/model_invocations/local_model_chat_deepseek_execute_20260526T191641067444Z.json` |
| mamba_cpu | 66 | provider_usage | `05_OUTPUTS/model_invocations/local_model_chat_mamba_cpu_execute_20260526T191655393582Z.json` |
| bonsai | 64 | provider_usage | `05_OUTPUTS/model_invocations/local_model_chat_bonsai_execute_20260526T191738743883Z.json` |
| mamba_gpu | 66 | provider_usage | `05_OUTPUTS/model_invocations/local_model_chat_mamba_gpu_execute_20260526T191746663431Z.json` |
| needle_0 | 43 | char_estimate | `05_OUTPUTS/model_invocations/local_model_chat_needle_0_execute_20260526T191751451114Z.json` |
| needle_1 | 43 | char_estimate | `05_OUTPUTS/model_invocations/local_model_chat_needle_1_execute_20260526T191755407292Z.json` |
| needle_2 | 43 | char_estimate | `05_OUTPUTS/model_invocations/local_model_chat_needle_2_execute_20260526T191759119058Z.json` |
| needle_3 | 43 | char_estimate | `05_OUTPUTS/model_invocations/local_model_chat_needle_3_execute_20260526T191803095040Z.json` |
| needle_4 | 43 | char_estimate | `05_OUTPUTS/model_invocations/local_model_chat_needle_4_execute_20260526T191807424347Z.json` |
| needle_5 | 43 | char_estimate | `05_OUTPUTS/model_invocations/local_model_chat_needle_5_execute_20260526T191811971551Z.json` |

## Current blockers

- `05_OUTPUTS/model_invocations/groq_model_catalog_20260526T191821498646Z.json`: groq_http_error:401
- `05_OUTPUTS/model_invocations/groq_chat_execute_20260526T191923949590Z.json`: groq_http_error:401

## Anti-slop decision

No claim of Groq/current 55% compliance is made until a valid Groq key produces fresh PASS receipts. No claim of main-window 25% compliance is made without an authoritative account token export.
