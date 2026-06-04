# MODEL FABRIC ADMISSION SIDECAR

## Core correction

Do not hardcode model names in prompts.

Model names, files, endpoints, ports, quantization, context, slots, VRAM/RAM cost, speed, launch command, health probe, smoke test, and role fitness live in the DB model ledger.

Agents may remember obvious current profiles, but the system source of truth is the ledger. If a model is not in the ledger, discover/register it or emit MODEL_UNREGISTERED. Do not silently skip it.

## Purpose

Make LUCIDOTA able to run whatever model the operator wants, whenever resources allow, without dying, freezing, squatting VRAM, duplicating weights, or dropping local models in favor of cloud convenience.

This sidecar is an admission/scheduling contract, not a model list.

## Required DB concepts

The model fabric must have ledger rows equivalent to:

- model_id
- display_name
- provider_type: local_llama_cpp | local_worker | api_openai_compat | api_groq | api_mistral | api_other
- artifact_path or remote_model_id
- launch_command_template
- stop_command_template
- endpoint_base_url
- health_probe
- smoke_test
- model_role_tags: ingress, egress, classifier, sequence_watcher, embedder, extractor, reasoner, coder, auditor
- quantization
- context_limit
- preferred_prompt_budget
- supports_slots
- slot_count
- supports_shared_kv
- supports_gpu
- vram_estimate_mb
- ram_estimate_mb
- cold_start_seconds
- warm_tokens_per_second
- prompt_tokens_per_second
- max_parallel_requests
- current_state: cold | starting | warm | busy | degraded | failed | cooling | stopped
- last_probe_at
- last_receipt_path
- admission_policy_json
- failure_policy_json

Names are data. Policies are data. The conductor queries them.

## Runtime law

Before any model call:

1. Resolve the desired role, not a hardcoded model name.
2. Query model ledger for candidates.
3. Check current resource state: RAM, VRAM, ports, existing processes, API limits.
4. Score candidates by role fit, resource fit, warm/cold cost, context budget, expected latency, and current availability.
5. If local candidate fits, use it before external API unless policy says otherwise.
6. If no candidate fits, emit MODEL_ADMISSION_DENIED with reason and next command.
7. If selected model is cold, open it through its ledger launch template.
8. Wait for health probe.
9. Run smoke test.
10. Register active lane/slot.
11. Execute call.
12. Write model-use receipt.
13. Release, keep warm, or cool down according to policy.

No silent fallback. No fake success. No “model not used because agent forgot.”

## Slot/opening law

“Open the model” means:

- process exists or API credential exists,
- endpoint is reachable,
- slot/capacity is allocated,
- prompt budget fits,
- smoke test passes,
- active lane row is written,
- receipt exists.

For local llama.cpp profiles, opening must account for:

- one weight load where possible,
- multiple slots if supported,
- shared/unified KV if configured,
- port conflicts,
- VRAM admission,
- CPU/RAM admission,
- prompt/context budget.

## LLM step-stripping law

If an LLM proposes multiple steps, the conductor must steal those steps and turn them into typed work:

LLM says: “I will inspect, patch, test, summarize.”
System does:
- inspect_job
- patch_job
- test_job
- summary_job
- receipts
- DB state transition

LLM can propose or verbalize. It does not own multi-step execution.

## Model-use receipt

Every call writes:

MODEL_USE_RECEIPT {
  request_id,
  operator_request_id,
  selected_model_id,
  selected_role,
  provider_type,
  endpoint,
  process_id,
  slot_id,
  prompt_tokens,
  output_tokens,
  prompt_budget,
  admission_decision,
  resource_before,
  resource_after,
  latency_ms,
  success,
  fallback_used,
  fallback_reason,
  receipt_path
}

## Non-drop tests

Fail the run if:

- a model role is needed and no ledger query occurs,
- an external API is used before checking eligible local lanes,
- a model is called without admission receipt,
- a cold local model is skipped without MODEL_ADMISSION_DENIED,
- VRAM/RAM is not checked before launching a local model,
- endpoint is assumed without health probe,
- prompt exceeds budget and is still sent,
- multiple llama.cpp processes load duplicate weights when one shared-slot profile could satisfy the route,
- conductor hardcodes model names instead of resolving from ledger,
- LLM executes multi-step plans instead of compiling work orders.

## Resource scheduler

The scheduler owns:

- warm/cold model state
- port allocation
- VRAM/RAM reservation
- prompt/context budget
- slot allocation
- API token bucket/backoff
- process lifetime
- health probes
- timing receipts
- queue sequencing

LLMs do not decide resource safety. They request capabilities. Scheduler admits or denies.

## Required scripts

Create or verify:

scripts/model_ledger_discover.py
scripts/model_ledger_register.py
scripts/model_fabric_status.py
scripts/model_fabric_admit.py
scripts/model_fabric_open.py
scripts/model_fabric_call.py
scripts/model_fabric_release.py
scripts/model_fabric_receipt.py
scripts/model_resource_probe.sh
scripts/model_prompt_budget_gate.py
scripts/model_api_rate_gate.py

## Required CLI shape

Examples:

python3 scripts/model_fabric_status.py
python3 scripts/model_fabric_admit.py --role ingress --budget-tokens 3000
python3 scripts/model_fabric_open.py --role ingress
python3 scripts/model_fabric_call.py --role ingress --input packet.json
python3 scripts/model_fabric_release.py --policy keep-warm
python3 scripts/model_ledger_register.py --discover-local

The conductor calls roles. The ledger maps roles to actual models.

## Completion definition

DONE means:

- model ledger exists in Postgres,
- active model lanes are queryable,
- resources are measured before launch,
- model profiles are selected by role and policy,
- local models are not dropped silently,
- external APIs are only used after admission logic,
- every model call has a receipt,
- every skipped model has a reason,
- every multi-step LLM plan becomes work orders,
- operator can ask for any registered model/profile and scheduler either runs it or explains exactly why not.
