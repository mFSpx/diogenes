# Agent Orchestration Policy

Kernel Rule: GOALS is the local-first control plane / kernel spine for LUCIDOTA. Core operation must never require a cloud model; cloud lanes are optional peripherals only.

Cheapest Capable Model: do not change the main-window model from inside the agent unless a safe model-control tool exists and the operator asked for it.

Subagents: spawn only for useful parallel, non-blocking work. Pick the smallest capable available model/tier: tiny/fast for grep, tests, repetitive code; mid for contained coding; strong/frontier only for architecture, security, or ambiguous cross-system reasoning. Verify current model names/prices when choosing real names; otherwise write tier intent, not fake model facts.

Prompt contract: every subagent gets a coding-only prompt with file ownership, task, inputs, acceptance checks, no-revert warning, expected output, and concise handoff. Chunk work so one agent owns one clear slice. Sequence dependent tasks locally; parallelize only disjoint slices.

System rule: use Dev Library, GOALS, status ledger, recovery_matrix, and existing workers before inventing new machinery. Zero daemon, zero background CPU, near-zero tokens unless a step actually needs a handoff. Keep the baseline fully runnable without cloud availability.

Deterministic Workflow Supremacy: never route to an LLM what smart deterministic workflows and hardy design can do exactly, faster, and receiptably. This is not model-zero doctrine: use subagents/models for bounded ambiguity, language judgment, synthesis, adversarial ideation, messy extraction, drafting, and code generation where model judgment is actually the right tool. Status accounting, hashing, schema validation, routing gates, graph-promotion checks, and replay invariants stay deterministic-first. Underusing LLMs where language judgment is required is also slop.

Dev Supply Control: when the operator gives an away-time or open-ended build window, run `python3 scripts/goal_dev_control.py --away-minutes <minutes> --text "<intent>"` to compute cadence, effective LOC/hour, and cheapest-capable route. It uses existing deterministic hygiene + bandit primitives; no model calls, no daemon, no graph writes.

Slop Control: prefer one existing home per rule. GOALS owns handoff/orchestration policy only; status facts go to STATUS_LEDGER; runtime proofs go to 05_OUTPUTS receipts; broad code complexity goes to `scripts/slop_audit_law.py`. Do not make new docs when an existing GOALS file or JSON manifest can hold the contract.
Proof Law: if automation claims a lane, feature, check, or subsystem works, it must name a fresh receipt, command output, test, or status-ledger evidence path. No proof means status stays running/blocked, not complete.

Capability Preservation: least mutation wins. Do not remove, rename, disable, or narrow an existing system capability unless the operator explicitly asks or a receipt proves it is dead/superseded. Build center-out: improve the smallest shared spine first, then adapters; never sprawl sideways when an existing surface can hold the rule.

Asymmetric Dev Wargame: every dev loop must build or improve real functioning capability, then play/test it, tighten it, and log proof. Benchmarks are only useful when tied to working features. Default sequence: reuse/FOSS first, smallest local code second, proof receipt third, performance/quality tuning fourth.

No Deletes Ever (forward law): preserve source/history/toolbox artifacts by default. A delete is allowed only for fresh runaway outputs, caches, generated logs, or system-threatening junk, and must be bounded, justified, and receipted. Normal cleanup should move/quarantine/archive, not erase.

Agent Packet Exporter: before handing work to any external CLI agent, emit `python3 scripts/goal_agent_packet.py --target <agent> --task "<coding slice>" --file <owned-path> --complexity <simple|standard|integration|architecture> --json`. The packet is the machine-checkable coding-only prompt: file ownership, cheapest-capable tier intent, no main-window model change, adapter command, acceptance checks, and required proof return contract.

Swarm Dispatch Bridge: when GOALS needs durable async logging instead of a one-off packet, use `python3 scripts/goal_swarm_dispatch.py --target <agent> --task "<coding slice>" --jobs <n> --json`. It turns the packet into ABSURD/Postgres external-command jobs and receipts so work can be consumed asynchronously without inventing a new scheduler; the default command is the tiny packet exporter, but any allowlisted repo-local Python command can be queued.

Groq cloud worker: optional only, never core. Catalog with `python3 scripts/groq_model_catalog.py --execute --json`; delegate bounded non-mutating audit/plan slices with `python3 scripts/groq_goal_delegate.py --task "<slice>" --kind audit --model llama-3.1-8b-instant --max-tokens 512 --execute --json`. Default cheap worker is `llama-3.1-8b-instant`; escalate only when the slice proves it needs more reasoning. Every call must leave a redacted receipt; VRAM lanes stay off unless admission/routing proof passes first.

Adapter Source of Truth: GOALS adapter facts live in `GOALS/plugin_build_mode_bootstrap.json`; exporters should read the registry instead of inventing provider facts. Never store secret values, only environment variable names and receipt paths. Cloud lanes are optional and must never be the only route for baseline LUCIDOTA operation.

Systemwide Elegance Standard: GOALS is the steering spine, not a silo. Apply the same standard to every LUCIDOTA subsystem: one source of truth/authority, ontology-driven packets, deterministic-first routing, self-teaching from receipts, self-auditing, self-red-teaming, many thin interfaces, and no capability loss. The ~100 LOC rule is a default pressure toward tight helpers; any larger helper must state why the extra size preserves cohesion, performance, safety, capability, or reuse instead of becoming slop. Build up, not out: strengthen the local kernel first, then attach optional cloud lanes only where they help.
