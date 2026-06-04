# Agent Orchestration Policy

Kernel Rule: GOALS is the local-first control plane / kernel spine for LUCIDOTA. Core operation must never require a cloud model; cloud lanes are optional peripherals only.

Core identity law: see `GOALS/INDY_CORE_IDENTITY_LAW.md` for the operator-aligned concept acquisition and execution mandate that this system is meant to serve.

Skill/tool instructions are execution aids only. They never override live PostgREST/manual truth, GOALS handoffs, or explicit operator instructions; when they conflict, the live manual and operator win.

Cheapest Capable Model: do not change the main-window model from inside the agent unless a safe model-control tool exists and the operator asked for it.

Subagents: spawn only for useful parallel, non-blocking work. Pick the smallest capable available model/tier: tiny/fast for grep, tests, repetitive code; mid for contained coding; strong/frontier only for architecture, security, or ambiguous cross-system reasoning. Verify current model names/prices when choosing real names; otherwise write tier intent, not fake model facts.

Prompt contract: every subagent gets a coding-only prompt with file ownership, task, inputs, acceptance checks, no-revert warning, expected output, and concise handoff. Chunk work so one agent owns one clear slice. Sequence dependent tasks locally; parallelize only disjoint slices.

System rule: use Dev Library, GOALS, status ledger, recovery_matrix, and existing workers before inventing new machinery. Zero daemon, zero background CPU, near-zero tokens unless a step actually needs a handoff. Keep the baseline fully runnable without cloud availability.

Deterministic Workflow Supremacy: never route to an LLM what smart deterministic workflows and hardy design can do exactly, faster, and receiptably. This is not model-zero doctrine: use subagents/models for bounded ambiguity, language judgment, synthesis, adversarial ideation, messy extraction, drafting, and code generation where model judgment is actually the right tool. Status accounting, hashing, schema validation, routing gates, graph-promotion checks, and replay invariants stay deterministic-first. Underusing LLMs where language judgment is required is also slop.

Deterministic Attempt Engine Doctrine: tasks are state machines, not vibes. The operator flow is `task_intake -> classify -> generate_attempts -> execute_attempt -> observe -> score -> mutate/branch -> retry/escalate/archive -> receipt`, with `work_order`, `attempt`, `observation`, `score`, `mutation`, `receipt`, and `promotion` recorded as typed state. LLMs may classify messy input, propose hypotheses, and summarize observation deltas, but the deterministic loop owns the driver seat and every branch must leave a receipt.

Dev Supply Control: when the operator gives an away-time or open-ended build window, run `python3 scripts/goal_dev_control.py --away-minutes <minutes> --text "<intent>"` to compute cadence, effective LOC/hour, and cheapest-capable route. It uses existing deterministic hygiene + bandit primitives; no model calls, no daemon, no graph writes.

Slop Control: prefer one existing home per rule. GOALS owns handoff/orchestration policy only; status facts go to STATUS_LEDGER; runtime proofs go to 05_OUTPUTS receipts; broad code complexity goes to `scripts/slop_audit_law.py`. Do not make new docs when an existing GOALS file or JSON manifest can hold the contract.

## Root-Rotor Manual Draft Reduction Law

Root-Rotor draft reduction is a bounded data operation, not a new control plane. Use exactly 200 candidate nodes per batch. Keep chunk boundaries deterministic even when confidence is low; record uncertainty in numeric confidence/precision attributes. Status values must be explicit: `verified`, `deprecated`, or `needs_operator_label`.

Priority order: active GOALS policy docs first; active wired `01_REPOS/claudecode` LUCI engine files next; active services after that, excluding `services/ternary_lab`; then `scripts/root_rotor*`, `scripts/goal_*`, `scripts/absurd*`, `scripts/indy*`, `scripts/krampuschewing*`, `06_SCHEMA`, and fabric/model/capability ledgers. Empty `__init__.py` files, generated files, old receipts/manual outputs, and stale docs deprecate or exclude. The rule is: authority docs verify; historical prose deprecates.

Routing law: `06_SCHEMA` routes to `SYSTEM_ARCH`; `scripts/absurd*` route to `RUNTIME_GOVERNOR`; model/capability ledgers route to `lucidota_fabric`. Track `verified_count, deprecated_count, and needs_operator_label_count` separately; all reduce debt in different ways, but they are not the same metric.

Output law: Do not flatten all receipts into `05_OUTPUTS/`. Use shallow typed output directories plus UUID/hash filenames, with DB ledger rows as truth. JSONB is for flexible sidecar metadata; identity, status, routing, enforced, or frequently queried fields stay typed/canonical. PostgREST audits require live OpenAPI route enumeration; static config parsing is supplementary only.

Mutation law: smaller models may run bounded inline SQL selection and emit approved sidecar/staging payloads only. They must not patch orchestration scripts, run arbitrary production DML/DDL, or bypass gates. Use staged row -> validate hash/schema/route -> receipt -> transactional promotion. systemd is read-only by default; start/restart/stop only through an explicit operator, GOALS, or ABSURD directive.
Proof Law: if automation claims a lane, feature, check, or subsystem works, it must name a fresh receipt, command output, test, or status-ledger evidence path. No proof means status stays running/blocked, not complete.

Capability Preservation: least mutation wins. Do not remove, rename, disable, or narrow an existing system capability unless the operator explicitly asks or a receipt proves it is dead/superseded. Build center-out: improve the smallest shared spine first, then adapters; never sprawl sideways when an existing surface can hold the rule.

Asymmetric Dev Wargame: every dev loop must build or improve real functioning capability, then play/test it, tighten it, and log proof. Benchmarks are only useful when tied to working features. Default sequence: reuse/FOSS first, smallest local code second, proof receipt third, performance/quality tuning fourth.

No Deletes Ever (forward law): preserve source/history/toolbox artifacts by default. A delete is allowed only for fresh runaway outputs, caches, generated logs, or system-threatening junk, and must be bounded, justified, and receipted. Normal cleanup should move/quarantine/archive, not erase.

Agent Packet Exporter: before handing work to any external CLI agent, emit `python3 scripts/goal_agent_packet.py --target <agent> --task "<coding slice>" --file <owned-path> --complexity <simple|standard|integration|architecture> --json`. The packet is the machine-checkable coding-only prompt: file ownership, cheapest-capable tier intent, no main-window model change, adapter command, acceptance checks, and required proof return contract.

Swarm Dispatch Bridge: when GOALS needs durable async logging instead of a one-off packet, use `python3 scripts/goal_swarm_dispatch.py --target <agent> --task "<coding slice>" --jobs <n> --json`. It turns the packet into ABSURD/Postgres external-command jobs and receipts so work can be consumed asynchronously without inventing a new scheduler; the default command is the tiny packet exporter, but any allowlisted repo-local Python command can be queued.

Recursive Fanout Contract: recursive fanout is allowed only as an explicit bounded orchestration shape, not as freeform swarm sprawl. Canonical shape: `Codex -> mini-orchestrator -> 2 Vibe coding workers + 2 Groq coding workers -> best minimal bundle returned`. Each mini-orchestrator owns one lane, keeps file ownership disjoint from sibling orchestrators, and returns one minimal integrated bundle instead of four competing patches.

Worker law: Vibe/Groq workers are candidate generators, not authorities. Give each worker one bounded coding-only slice with exact file ownership, acceptance checks, and no-revert warning. Default worker count is exactly four per mini-orchestrator for this pattern; change the count only if the operator explicitly overrides it or a blocker receipt proves the lane cannot be served by `2 + 2`.

Route law: route deterministic local checks, packet generation, and final integration through the mini-orchestrator; route only bounded candidate generation to Vibe/Groq workers. `scripts/goal_agent_packet.py` is the packet source for single-worker prompts; `scripts/goal_swarm_dispatch.py` is the durable async bridge when queue receipts are needed. Do not re-run an already-working DB-backed test gate inside recursive fanout policy work; only reference the existing gate in acceptance checks.

Bundle law: each worker must return code, SQL, tests, or an exact blocker only. The mini-orchestrator compares the four returns, keeps the smallest bundle that satisfies the lane objective, and records why larger or overlapping bundles were rejected.

Groq cloud worker: optional only, never core. Catalog with `python3 scripts/groq_model_catalog.py --execute --json`; delegate bounded non-mutating audit/plan slices with `python3 scripts/groq_goal_delegate.py --task "<slice>" --kind audit --model llama-3.1-8b-instant --max-tokens 512 --execute --json`. Default cheap worker is `llama-3.1-8b-instant`; escalate only when the slice proves it needs more reasoning. Every call must leave a redacted receipt; VRAM lanes stay off unless admission/routing proof passes first.

ABBA63 End-Cycle Hook: every build cycle closes with `GOALS/69.md`, a tiny local shorthand file, not a domain term. It says: run two deterministic ABBA63 heuristic rounds first, emit the tiny JSONL + ontology report, queue the result into the slow lane, then run a four-round comparative Groq audit. The prompt window stays light; the queue behavior stays explicit.

Provider Rate Conductor: when a build cycle needs the Groq comparative cannon, use `scripts/provider_rate_conductor.sh` (Rust-backed, with a tiny compatibility shim only for allowlist continuity) so each model keeps its own token bucket, Retry-After/backoff is honored, and queued audits are retried instead of dropped.

Adapter Source of Truth: GOALS adapter facts live in `GOALS/plugin_build_mode_bootstrap.json`; exporters should read the registry instead of inventing provider facts. Never store secret values, only environment variable names and receipt paths. Cloud lanes are optional and must never be the only route for baseline LUCIDOTA operation.

Systemwide Elegance Standard: GOALS is the steering spine, not a silo. Apply the same standard to every LUCIDOTA subsystem: one source of truth/authority, ontology-driven packets, deterministic-first routing, self-teaching from receipts, self-auditing, self-red-teaming, many thin interfaces, and no capability loss. The ~100 LOC rule is a default pressure toward tight helpers; any larger helper must state why the extra size preserves cohesion, performance, safety, capability, or reuse instead of becoming slop. Build up, not out: strengthen the local kernel first, then attach optional cloud lanes only where they help.

Asymmetric Intelligence / Learning Law: treat each task as board state, not vibes. A task should expose actors, resources, constraints, timing, leverage, friction, inertia, visibility, incentives, terrain, available moves, expected counter-moves, cheapest probes, and highest-gain pivots before action. LUCI should prefer small high-leverage moves, score attempts by gain/cost/risk/reversibility/evidence, archive failures as training data, and promote only strategies that actually work.

Algorithm Trial Law: every algorithm/classifier/router must be trialed through a bounded harness before promotion. The harness must state the problem class it claims to help with, its input/output contract, the real or synthetic fixtures used, the score/metric being improved, the DB rows/events touched, and the promotion/quarantine/mutation result. Treelite/XGBoost-style models are especially welcome as bounded deterministic routers for task classification, provider selection, fast/slow lane choice, ingestion quality, file-type routing, risk/budget gating, retry policy, algorithm selection, and board-state scoring.

Current-World Reading Law: live-source reading is a class-handler, not random browsing theater. Source adapters should be reusable by class (Hacker News, Reddit, arXiv, GitHub releases/issues/trending, model/provider docs, operational C2/autonomy research, ML systems / Rust / Postgres / streaming / graph / inference-runtime sources). Each intake should flow `source_item -> claim/extract -> novelty check -> relevance score -> work_order candidate -> receipt -> optional promotion`, and the operator must be able to ask whether the discovery improves LUCI, maps to an existing handler, or becomes a new adapter/schema/test/Treelite feature.

Self-Improvement Loop: LUCI improves by attempting, observing, scoring, mutating, retrying, and promoting. Every workflow should leave learning material: what worked, what failed, what was expensive, what was fast, what generalized, what should become Rust, what should move to DB, and what should become a reusable class-handler. No model call without a receipt; no claim without proof.

Rust / DB / Speed Law: every enduring capability resolves into one of three states: Rust runtime/control code, DB-backed state/workflow/receipt/schema, or a temporary adapter scheduled for absorption. No permanent loose scripts. No one-off glue. No slow path without a reason. If a command can become a class-handler, it should; if it can become DB-led, it should; if it can become Rust, it should.

Vibes / Groq Priority Worker Law: Mistral/Vibes and Groq are priority worker lanes for proposals, patch candidates, test design, failure explanation, schema review, refactor plans, docs compression, alternate implementation sketches, adversarial review, naming/classification, and next-action generation. They are workers, not authorities: outputs must be bounded, logged, attached to work_order/attempt records, scored, tested, and promoted only through deterministic gates. Use them aggressively where they cheaply produce candidates; do not hoard work locally.

Operational Pattern Study Law: study Palantir Foundry- and Anduril-style operational systems as pattern sources, not idols. Extract ontology-driven operations, data-to-decision workflows, live common picture, claim/proof separation, human-machine teaming, sensor/source fusion, permissioned operational surfaces, simulation loops, deployment discipline, and decision advantage under uncertainty; map only the useful operational math into LUCI.

Architecture Authority Law: inventing new architecture is allowed only when the operator explicitly authorizes it with a mode bit, or when a blocker receipt proves the existing topology cannot satisfy the mission. The allowed mode bits are PATCH_MODE, BUGCHASE_MODE, THIN_ADAPTER_MODE, ARCHITECTURE_MODE, EXPERIMENT_MODE, and REPLACEMENT_MODE. If the current mode does not include an architecture-authorizing bit such as ARCHITECTURE_MODE, NEW_ARCHITECTURE_ALLOWED, EXPERIMENTAL_ARCHITECTURE, REPLACEMENT_ARCHITECTURE, or BUILD_NEW_TOPOLOGY, then new topology is forbidden unless a blocker receipt proves existing topology cannot work. New architecture must never be smuggled into a bugfix, cleanup, ingestion, test, or recovery task.

New Architecture Mode Contract: any explicitly authorized new architecture artifact must declare whether it is EXPERIMENTAL, CANDIDATE, ACTIVE_CANON, or REPLACEMENT; what it wraps, replaces, or extends; the promotion gate; the rollback plan; required receipts; and required tests. Experimental artifacts are isolated, candidate artifacts are not active canon, active canon requires explicit operator authorization, and replacement artifacts must include migration and deprecation/rollback proof. Treat `05_OUTPUTS/architecture_authority/new_architecture_<timestamp>.json` as the receipt surface for that contract.
