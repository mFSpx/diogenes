# ROOT MANUAL — LUCIDOTA CANON

## What is canon

Canon is the current, receipt-backed source of operational truth. Manuals are the user-facing canon after this pass. If docs, DB state, runtime behavior, and receipts disagree, receipts win and the docs must be repaired.

## Non-negotiable laws

- Search the Dev Library before inventing new tools or workflows.
- Prefer reuse over reinvention unless sovereign originals must stay intact.
- Blueprint first, model second: keep the workflow visible in source, schema, or queue objects.
- Receipts over claims: a PASS requires a command, output, file, or DB row.
- No remote compact bombs: chunk by subsystem, volume, schema, endpoint, or command family.
- No Docker dependency for core operator flow unless explicitly required by a lane.
- No fake success, no markdown-only completion, no hidden model policy.

## Authority stack

1. Operator instruction
2. Receipts and runtime evidence
3. Canonical docs updated from receipts
4. Work queues and workflow rows
5. Supporting skeletons / archives / proof-hoard artifacts

## What is true today

- RunPod/Talkie SSH is live.
- Talkie source custody PASS exists.
- The remote bootstrap/download path completed to a custody receipt.
- Model fabric status is available through the local control scripts.
- LoRA work orders are staged, not yet trained.
- Queues are durable and must use real UUIDs, not placeholders.

## Operator law

- Operate in bounded chunks.
- Do not narrate instead of doing.
- Every repetitive action becomes a work order.
- Every model call must be admitted by role and resource fit.
- Every skipped action must have a blocker receipt or explicit reason.

## Evidence anchors

- `00_PROJECT_BRAIN/TICKLETRUNK.md`
- `00_PROJECT_BRAIN/ACTIVE_SPEC/04_DEV_LIBRARY_REUSE_LAW.md`
- `00_PROJECT_BRAIN/BLUEPRINT_FIRST_MODEL_SECOND_PSEUDOLAW.md`
- `05_OUTPUTS/goals/goal_model_fabric_control_20260603T021934Z.json`
- `05_OUTPUTS/runpod/talkie_book_lora/remote_talkie_source_custody.json`
- `05_OUTPUTS/model_runtime/talkie_source_custody.json`

## Current operational posture

Sellable standard means the operator can read the manuals, find the live commands, reproduce the gates, and see exactly what blocks the next move.
