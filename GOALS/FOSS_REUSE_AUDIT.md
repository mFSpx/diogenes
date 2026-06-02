# GOAL 3 FOSS Reuse Audit

Purpose: check whether an existing FOSS tool already solves External Plugin Build Mode better than a local reinvention.

## Current answer

Do not build a new coding agent, editor extension, provider gateway, or llama.cpp wrapper inside GOALS. GOALS remains the tiny handoff/adapter contract layer: it tells any external agent what slice to run, which existing LUCIDOTA lane/provider to use, and what receipt proves it.

## Reuse candidates checked

- LiteLLM: reuse if LUCIDOTA needs a real OpenAI-compatible gateway/router across many providers, retries, fallbacks, cost tracking, or budgets. Do not recreate that in GOALS. Source checked: `https://docs.litellm.ai/docs/`.
- OpenCode: reuse as a BYO-provider terminal coding agent when the operator wants a FOSS agent shell. It already supports many providers, local models, custom OpenAI-compatible providers, and provider credentials/config. Source checked: `https://opencode.ai/docs/providers/` and `https://github.com/anomalyco/opencode`.
- Aider: reuse as the Git-native terminal pair-programming agent for local repo edits; it can use many LLMs and local OpenAI-compatible endpoints. Source checked: `https://aider.chat/docs/llms.html`.
- Continue: reuse for IDE-side VS Code/JetBrains coding with multiple providers, local model runners, Groq, Cohere, and llama.cpp. Source checked: `https://docs.continue.dev/customize/model-providers/overview`.
- llxprt-code: Gemini CLI derivation with hook and session surfaces that matches this repo’s local coding workflow and is already available as `01_REPOS/llxprt-code` (upstream: `https://github.com/vybestack/llxprt-code`).
- llama.cpp: already local in this repo; reuse its existing server launchers and OpenAI-compatible surface. Do not write another local model server.

## Local integration decision

GOALS should export compact instructions/manifest, not become the agent. Existing tools can point at:

1. `GOALS/CURRENT_HANDOFF.md` for the active slice.
2. `GOALS/AGENT_ORCHESTRATION_POLICY.md` for cheapest-capable model/task sizing.
3. `GOALS/EXTERNAL_PLUGIN_BUILD_MODE.md` for the adapter contract.
4. `GOALS/plugin_build_mode_bootstrap.json` for local/cloud lane commands and receipt expectations. Cloud entries are optional; the baseline contract still has to work when only local lanes are present.
5. `GOALS/MODEL_FABRIC_AUDIT.md` for known local/cloud LLM lanes.

## Efficient next reuse pitches

1. Add a deterministic exporter that emits one small handoff packet for OpenCode/Aider/Continue/Codex/Claude without new runtime services.
2. If many providers become active, point agents at LiteLLM instead of adding local provider routing code.
3. Keep llama.cpp lanes behind existing strict admission + launch scripts; no automatic heavy model startup from GOALS.

## Chain/continuity re-check — 2026-05-26

- AICTX (`https://aictx.org/`, `https://github.com/oldskultxo/aictx`) is the closest FOSS match for repo-local agent continuity: work state, next actions, decisions, failures, validation evidence, CLI/MCP, Codex/Claude/Copilot/generic-agent support. It is MIT-licensed and current, but it is a full package/runtime with `.aictx/`, MCP/plugin install flow, and a small unaffiliated maintainer footprint. Decision: do **not** install inside this unattended under-100-LOC GOALS tranche; keep our 25-line `goal_chain.py` + 63-line `goal_handoff.py` now, and mark AICTX as the first candidate if GOALS continuity outgrows the tiny local protocol.
- Maestro remains a heavier task/contract/evidence harness. Good pattern source; too much runtime/document surface for the present no-sprawl requirement.
- OpenCode and LiteLLM remain reuse targets for actual coding-agent/provider-router work, not for the tiny GOALS baton itself.

## Language/router re-check — 2026-05-26

Checked current routing/template prior art: vLLM Semantic Router, DSPy, Pellet/OpenAI-compatible model routing, SPL-style structured prompt routing, and semantic-router/intent-classifier approaches. These are useful patterns for larger provider routing/prompt-pipeline optimization, but too large for the immediate under-100-LOC language membrane. Local decision: reuse existing `core/language_membrane.py`, `scripts/template_contract.py`, `scripts/fast_slow_lane_gate.py`, FairyFuse ternary smoothing, and GO ontology JSON; add only the 48-LOC `scripts/language_router.py` bridge.

## Current FOSS re-check — 2026-05-26

Browsed current public docs again before adding local code. Result unchanged: reuse external agents/routers instead of rebuilding them inside GOALS.

- LiteLLM remains the right future gateway if we need provider fan-out, spend tracking, budgets, retries, or OpenAI-compatible proxy behavior. Do not recreate that in GOALS: https://docs.litellm.ai/
- Aider remains the Git-native coding agent candidate; it supports many providers and OpenAI-compatible/local endpoints. GOALS should emit a packet Aider can consume, not become Aider: https://aider.chat/docs/llms.html and https://aider.chat/docs/llms/openai-compat.html
- Continue remains IDE-side provider/model integration; it already lists Groq, Cohere, llama.cpp, and local options: https://docs.continue.dev/customize/model-providers/overview
- OpenCode remains a BYO-provider terminal coding-agent candidate; current docs/community reports confirm OpenAI-compatible/local-provider patterns exist, but local tool-call reliability depends on provider/model config. GOALS should hand it explicit coding packets, not hide work in prose: https://dev.opencode.ai/docs/providers/

Local action from this re-check: add the tiny `scripts/goal_agent_packet.py` exporter/router and normalize `GOALS/plugin_build_mode_bootstrap.json` adapter fields. No new agent package or gateway was installed.

## Current FOSS re-check — 2026-05-26 04:20Z

Verified current docs again before changing GOAL 3 wiring. Decision unchanged: do not install or rebuild a gateway here. Keep GOALS as tiny adapter/packet layer and let established tools do their jobs.

- LiteLLM remains the likely FOSS gateway if LUCIDOTA later needs centralized provider routing/budgets: https://docs.litellm.ai/
- Aider can connect to OpenAI-compatible endpoints, including local/provider gateways: https://aider.chat/docs/llms/openai-compat.html
- OpenCode supports OpenAI-compatible providers through project config: https://dev.opencode.ai/docs/providers/
- Continue supports OpenAI-compatible providers and llama-cpp-family local surfaces: https://docs.continue.dev/customize/model-providers/top-level/openai

GOALS action: reuse those harnesses/gateways by exporting compact coding packets and adapter metadata; do not create another coding agent, daemon, or model gateway in GOALS.
