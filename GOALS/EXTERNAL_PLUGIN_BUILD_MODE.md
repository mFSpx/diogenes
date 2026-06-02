# External Plugin Build Mode

Codex-first, BYO-LLM: any CLI/agent/runtime can point at GOALS, read the handoff, then execute only the assigned coding slice. Core LUCIDOTA still has to run local-first; cloud is optional capacity, never the baseline dependency.

Adapter Contract: name, provider/local lane, endpoint/env key names only, model tier intent, command to dry-run, command to execute, expected receipt path, safety limits, and rollback/check command. Never store secret values here.

Workflow: operator slop is allowed in the incoming goal; orchestrator slop is not. The orchestrator translates vibes into small steps, picks cheapest capable agents/tools, writes explicit coding-only prompts, and records X/N handoffs. If a local lane exists, baseline routing stays local-first; cloud lanes are only optional accelerators.

Reuse rule: do not build a new coding agent, model gateway, or local model server here. Reuse FOSS tools such as LiteLLM, OpenCode, Aider, Continue, `llxprt-code` (upstream at https://github.com/vybestack/llxprt-code), and llama.cpp when they fit; GOALS stays the tiny handoff/adapter contract layer.

three next pitches: (1) adapter registry JSON generated from existing scripts, (2) one dry-run router that chooses local/Groq/Cohere/llama.cpp by task tier, (3) optional CLI handoff packet exporter for Codex/Claude/generic CLI agents without changing GOALS law.

Pass-on intent packet: use `scripts/goal_dev_control.py` for away-time windows and dev-supply routing. It records the operator time budget, current handoff cadence, effective LOC/hour, and the selected agent lane in a receipt under `05_OUTPUTS/goals/`.
Proof packet rule: external agents must return changed files plus evidence commands/receipts. A handoff without evidence is commentary, not completion.

Packet exporter: `scripts/goal_agent_packet.py` is the tiny BYO-agent bridge. Codex, Claude, OpenCode, Aider, Continue, or a generic CLI reads the emitted JSON, executes only the coding slice, and returns changed files plus command/receipt proof. It is a router/exporter, not a daemon and not a model gateway.

Registry Source Rule: packet export prefers `adapter_registry` in `GOALS/plugin_build_mode_bootstrap.json` over duplicated hard-coded adapter facts. Tiny fallback defaults are allowed only so recovery packets still work if the manifest is damaged; the manifest remains the GOALS source of truth for provider type, endpoint labels, env key names, dry-run/execute commands, receipt globs, and stop/rollback commands. Cloud entries are optional registry facts, not a requirement for baseline operation.

Model Fabric Start Control: `scripts/goal_model_fabric_control.py start --target <lane> --json` is the explicit-start GOALS wrapper for existing launchers. It is not a daemon; it writes pid/log/status receipts and should start heavy lanes one or two at a time unless the strict stack plan says otherwise.
