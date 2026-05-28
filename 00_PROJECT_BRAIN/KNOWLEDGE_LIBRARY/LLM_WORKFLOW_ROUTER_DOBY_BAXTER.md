# Knowledge Card: Doby Baxter LLM Workflow Router

- ID: `dobybaxter_llm_router`
- Authority class: `research_reference`
- Source: https://gitlab.com/dobybaxter127/llm-router
- Local clone: `01_REPOS/llm-router/`
- Cloned commit: `965725e3f0514b5f591efbef491739a98e682971`
- License notes: `LICENSE-COMMERCIAL.txt`; commercial/production use requires license per README.

## What it is

A deterministic, stateless topology enforcement layer for LLM workflows. It does not inspect content, moderate, orchestrate, retry, persist sessions, or mutate prompts. It evaluates structured metadata against a static YAML topology and returns one terminal state: `PROCEED`, `REFUSE`, or `PAUSE`.

## Learned pattern

1. Control structure, not content.
   - LUCIDOTA already wants deterministic routing before models; this repo is a clean miniature of that idea.
   - Routing decisions should operate on metadata packets, not prompt vibes.

2. Explicit transitions only.
   - Containers declare allowed transitions, re-entry policy, and max invocation count.
   - Unknown containers, invalid transitions, max-depth overflow, and blocked re-entry become structured refusal reasons.

3. Validate topology at load time.
   - Detect unknown targets, dead ends, missing entrypoints, cycle/re-entry contradictions, and unreachable containers before runtime.
   - Runtime evaluation should not throw for normal cases; it returns structured terminal decisions.

4. Host owns execution.
   - Router returns a decision; the host application remains responsible for state, orchestration, UX, retries, and actual tool/model execution.

## Repository map

- `router/engine.py` — `WorkflowEngine.evaluate(metadata)` deterministic evaluation.
- `router/models/` — frozen dataclasses/enums for config, metadata, result.
- `router/validation/config_validator.py` — config shape/transition target validation.
- `router/validation/transition_validator.py` — topology analysis and strict transition validation.
- `router/logging/schema.py` — JSON-safe evaluation log event.
- `router/cli.py` — validate/analyze/run CLI implementation.
- `examples/sample_config.yaml` and `examples/sample_metadata.json` — minimal runnable topology and evaluation packet.
- `tests/` — small topology/engine/logging/CLI contract tests.

## LUCIDOTA integration stance

Keep as a research reference and candidate pattern for a future tiny LUCIDOTA route gate. Do not import wholesale until license and use case are clear.

Best immediate uses:

- Model/router fastlane-slowlane transition guards.
- Work-order container transition checks (`intake -> parse -> packetize -> route -> receipt`, with explicit terminal `REFUSE/PAUSE`).
- CLAW command safety topology (`operator_shell -> deterministic_script -> receipt`, no hidden fallback).
- Graph-promotion workflow guard before canonical materialization.

Design target for LUCIDOTA: a small metadata-only topology checker that emits receipts and can be called by `claw`, not a model-driven orchestrator.

## Verification / blockers

Passed local checks:

- `python3 -m py_compile $(find 01_REPOS/llm-router/router -name '*.py' -print)`
- `PYTHONPATH=01_REPOS/llm-router python3 -c 'from router.cli import main; main()' validate 01_REPOS/llm-router/examples/sample_config.yaml`
- `PYTHONPATH=01_REPOS/llm-router python3 -c 'from router.cli import main; main()' analyze 01_REPOS/llm-router/examples/sample_config.yaml`
- `PYTHONPATH=01_REPOS/llm-router python3 -c 'from router.cli import main; main()' run --config ... --metadata ...`

Known issues from unmodified upstream clone:

- `python -m router.cli ...` returns no output because `router/cli.py` lacks a module `if __name__ == "__main__": main()` guard; package console script should still work when installed.
- Upstream test suite currently reports 4 failures in this direct clone/session: CLI tests use `python -m router.cli`, and one test expects unreachable containers to be an error while implementation reports warning.
- Because this is a cloned reference repo, do not patch it as LUCIDOTA law without a separate bounded integration task.
