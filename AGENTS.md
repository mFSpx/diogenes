# LUCIDOTA AGENT STARTUP LAW

Before writing new code in LUCIDOTA, read the **LUCIDOTA Dev Library** manifest (legacy filenames):
- `00_PROJECT_BRAIN/TICKLETRUNK.json`
- `00_PROJECT_BRAIN/TICKLETRUNK.md`
- `00_PROJECT_BRAIN/ACTIVE_SPEC/04_DEV_LIBRARY_REUSE_LAW.md`

Then read:
- `00_PROJECT_BRAIN/BLUEPRINT_FIRST_MODEL_SECOND_PSEUDOLAW.md`

Then search the Dev Library manifest for existing relevant tools, scripts, schemas, workflows, models, LoRAs, scrapers, skills, plugins, and services. Preferred command: `python3 scripts/dev_library_scan.py --query <topic>`.
Prefer copy/adapt/reuse over writing from scratch.
Do not mutate sovereign toolbox originals unless explicitly ordered.

Proof hoard doctrine: index the jungle; do not pave it. Disconnected, experimental, strange, and useful-later artifacts are allowed.

## Goal Handoff Law

At the start of every persistent goal operation in LUCIDOTA:

1. Read `GOALS/CURRENT_HANDOFF.md` if it exists.
2. Write or refresh a handoff using the exact prefix: `"Save This Prompt, Pass on this Handoff:"`.
3. At the end of every goal step, update `GOALS/CURRENT_HANDOFF.md` with `X/N` step progress and append the same entry to `GOALS/GOAL_LOG.md`.
4. Every submitted goal plan should include a final step that starts with `"Save This Prompt, Pass on this Handoff:"` and records the final handoff.
5. Include `Technical Summary Review and Dev Notes` after each step: extremely brief, conversational, technically precise, with small cryptid-field-note flavor only when useful.
6. Use `scripts/goal_handoff.py` and `GOALS/GOAL_HANDOFF_PROMPT.md` when possible. Keep all goal notes in `GOALS/`; do not create folder sprawl.

## Agent Model Economy Law

For persistent GOALS work, the orchestrator must not try to change the main-window model unless a safe model-control tool exists and the operator explicitly asked for it. For subagents, use the cheapest capable available model/tier for the bounded task, write coding-only prompts with exact file ownership and acceptance checks, chunk work by complexity, sequence dependent work locally, and parallelize only disjoint useful slices. See `GOALS/AGENT_ORCHESTRATION_POLICY.md`.

## Persistent Build-Meeting Memory

Exact quote to remember and apply:
> I’m going to encode this into the actual Slop Law docs, not just agree in chat.

If the operator is having a literal meeting to make build decisions, that is work, not therapy. Extract the decision, encode durable changes where appropriate, verify, and receipt. Do not substitute generic emotional support, moralizing friction, or fake agreement for engineering follow-through.
