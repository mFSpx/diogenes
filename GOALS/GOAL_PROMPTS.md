# Goal Prompt Stash

## Prompt 001 — GOALS crash-recovery layer

Source: active thread goal, captured 2026-05-26T01:14Z via `get_goal`.

Operator intent, compacted without changing scope:

> Create a `/GOALS` folder and prompt/handoff habit so crashes are recoverable. At the beginning of every goal, write a handoff. At the end of every step, update `X/N` progress and add a very brief `Technical Summary Review and Dev Notes`. Every goal plan should end with a final step prefixed exactly: `"Save This Prompt, Pass on this Handoff:"`. Keep it copy/pasteable, not folder-sprawl. Log this first prompt. Prove the process with about 25 silly steps including Hello World, wrong math `4+2=-1`, a cabbage blank fill, and “Teleportato a Kronenberg to a Borg.” Voice for dev notes: extremely brief, plain technical, MIT engineer with cryptid field notes.

Canonical prompt to reuse: see `GOALS/GOAL_HANDOFF_PROMPT.md`.
## Prompt 002 — Agent model economy extension

Source: operator extension, 2026-05-26.

> Add to GOALS: orchestrator should not change the main-window model unless there is a safe model-control path and the operator explicitly asked. For subagents, choose the cheapest capable available model/tier, keep tasks appropriately sized by complexity, write explicit coding-only prompts with file ownership and acceptance checks, sequence dependent work, parallelize only disjoint useful slices, and use the local LUCIDOTA systems instead of reinventing machinery. Goal: save tokens, CPU, and time; no yappity-yap.

Canonical policy: see `GOALS/AGENT_ORCHESTRATION_POLICY.md`.

## Prompt 003 — External Plugin Build Mode / model fabric wiring

Source: active GOAL 3 expansion, 2026-05-26.

> Make `/GOALS` hyper-effective as a Codex-first but BYO LLM build-mode layer. Keep the helper under 100 lines, do not create a resident daemon, and do not reinvent coding agents/provider routers/model servers. The orchestrator should translate messy operator goals into small coding slices, choose the cheapest capable model/tool tier, write explicit coding-only subagent prompts, use existing LUCIDOTA lanes/adapters, preserve operator slop while removing orchestrator slop, and require receipts before claiming a lane works.

Canonical artifacts: see `GOALS/EXTERNAL_PLUGIN_BUILD_MODE.md`, `GOALS/AGENT_ORCHESTRATION_POLICY.md`, `GOALS/MODEL_FABRIC_AUDIT.md`, `GOALS/FOSS_REUSE_AUDIT.md`, and `GOALS/plugin_build_mode_bootstrap.json`.

Wiring note: this subgoal includes `llxprt-code` as the approved CLI coding-agent alternative, upstream at `https://github.com/vybestack/llxprt-code`.

## Prompt 004 — Slop control / dev supply cadence

Source: operator continuation, 2026-05-26.

> Keep the whole GOALS build under construction laws: under-100-LOC helpers, reuse existing FOSS/local tools, no document churn, no unclear ownership, no new daemon unless justified. When the operator gives an away-time window, pass on intent through GOALS, compute cadence and effective LOC/hour, route work to the cheapest capable lane, use existing asymmetric routing primitives, and log receipts instead of yapping.

Canonical artifacts: `scripts/goal_dev_control.py`, `GOALS/AGENT_ORCHESTRATION_POLICY.md`, `GOALS/EXTERNAL_PLUGIN_BUILD_MODE.md`, `GOALS/plugin_build_mode_bootstrap.json`, and receipts under `05_OUTPUTS/goals/`.

## Prompt 005 — Thirty-minute unattended dev-mode hardening

Source: operator continuation, 2026-05-26.

> When I leave an away-time prompt, use GOALS as plug-and-play dev mode: extract real features, audit them to zero, keep helpers under 100 LOC, reuse existing/FOSS code first, run cheap/fast workers where useful, never overflow the GTX 1650, verify every lane with receipts, wire every LUCIDOTA subsystem through a single GOALS-owned manifest/check layer, and red-team/slop-control before claiming done.

Canonical artifacts: `GOALS/DEV_MODE_FEATURE_AUDIT.json`, `GOALS/DEV_MODE_INTEGRATION.json`, `GOALS/DEV_MODE_SUBSYSTEMS.json`, `scripts/goal_dev_control.py`, `scripts/goal_model_fabric_control.py`, and receipts under `05_OUTPUTS/goals/`.


## Prompt 006 — Chain goals + telemetry + system slimming runway

Source: operator continuation, 2026-05-26.

> If a goal finishes and the next intent is already known, queue it in GOALS instead of losing it. Prove every automation feature, keep helper scripts under 100 LOC, list systems/functions before slimming, monitor CPU/RAM/VRAM/PCI/temp during unattended windows, and make the next goal a whole-system elegance pass: reduce slop/LOC without losing any intended capability.

Canonical artifacts: `scripts/goal_chain.py`, `scripts/goal_system_index.py`, `scripts/goal_telemetry.py`, `GOALS/NEXT_GOAL_QUEUE.json`, `GOALS/DEV_MODE_SUBSYSTEMS.json`, `GOALS/DEV_MODE_INTEGRATION.json`, and receipts under `05_OUTPUTS/goals/`.

## Prompt 007 — Capability preservation / center-out build law

Source: operator correction, 2026-05-26.

> Do not mutate system capabilities away. Fuck with the least things possible. Build from the center out: local-first sovereign data, OSINT agency, live coding environment, investigative newsroom, ontology/truth-first system. Slim only after intended capability is mapped and receipts prove no ability was lost.

Canonical artifacts: `GOALS/AGENT_ORCHESTRATION_POLICY.md`, `GOALS/DEV_MODE_FEATURE_AUDIT.json`, `05_OUTPUTS/goals/system_slimming_preflight_20260526T022047Z.json`, and `05_OUTPUTS/goals/mutation_scope_audit_*.json`.

## Prompt 008 — Asymmetric dev wargame / no-delete law

Source: operator continuation, 2026-05-26.

> Treat LUCIDOTA development as an asymmetric dev wargame: build the real game engine, play it, tighten it, invent better lines, and keep iterating. Functional real-world features are required. Code stays tight. Reuse/FOSS first, then write the smallest local code. Preserve by default: no normal deletes; only bounded, receipted cleanup of fresh runaway logs/caches/generated junk or system-threatening waste.

Canonical artifacts: `00_PROJECT_BRAIN/ACTIVE_SPEC/01_IDENTITY_AND_CLAIM_STATE_LAW.md`, `GOALS/AGENT_ORCHESTRATION_POLICY.md`, `scripts/slop_audit_law.py`, `scripts/script_bucket_manifest.py`, and receipts under `05_OUTPUTS/`.

## Prompt 009 — Instruction Hygiene Language Router

Source: operator continuation, 2026-05-26.

> Instruction hygiene becomes a real language subsystem: sloppy operator text must route into ontology/JSON/work lanes under 100 LOC, deterministic-first, fast/slow sequenced, template-shaped, evidence/proof-aware, and reusable for coding, operator replies, correspondence, research, and investigation.

Canonical artifacts: `scripts/language_router.py`, `core/language_membrane.py`, `scripts/template_contract.py`, `scripts/fast_slow_lane_gate.py`, `OFFICIAL_ONTOLOGY.json`, and `05_OUTPUTS/language_router/` receipts.

## Prompt 010 — Language subsystem parity / ABSURD / Hypertimeline

Source: operator correction, 2026-05-26.

> Build mode and operator mode use the same language subsystem. Messy operator text must become deterministic GO/JSON work orders, not prompt fog. Every workflow routes through ABSURD-compatible packets, keeps a hypertimeline hook, uses DB/GO ontology refresh when requested, limits LLM use to explicit model hooks, and proves the result with receipts. Do not reinvent: reuse language_membrane, template_contract, fast_slow, FairyFuse, Percyphon, kernel/CEP, ABSURD, and Hypertimeline pieces.

Canonical artifacts: `scripts/language_router.py`, `scripts/lucidota_cli.py language`, `core/language_membrane.py`, `scripts/cep_builder.py`, `scripts/kernel_control_packet.py`, `scripts/recovery_matrix.py`, and receipts under `05_OUTPUTS/language_router/`.

## Prompt 011 — Systemwide Elegance Standard

Source: operator clarification, 2026-05-26.

> The instruction-hygiene/build-mode standard applies everywhere, not only to GOALS. LUCIDOTA should become one elegant system: one source of truth/authority, many thin ways to operate it, ontology-driven routing, deterministic-first algorithms, self-teaching from receipts, self-auditing, self-red-teaming, no capability loss, and tight code. The ~100 LOC rule is not absolute; it is a burden-of-proof trigger. Larger helpers are allowed only with a clear written reason: cohesion, preserved capability, performance, safety, or reuse/vendor boundary.

> GOALS is the local-first control plane / kernel spine. Core LUCIDOTA must stay fully runnable without cloud availability. Cloud models are optional peripherals only: useful for acceleration, never required for baseline operation, never the source of truth, and never the only execution path. Build up, not out: strengthen local kernel behavior first, then add cloud lanes only where they improve capability without dependency.

Canonical artifacts: `00_PROJECT_BRAIN/ACTIVE_SPEC/01_IDENTITY_AND_CLAIM_STATE_LAW.md`, `GOALS/AGENT_ORCHESTRATION_POLICY.md`, `00_PROJECT_BRAIN/RFCS/RFC-010-SLOP-LAWS.md`, and GOAL 5 in `GOALS/NEXT_GOAL_QUEUE.json`.

## Prompt 012 — Postgres swarm dispatch bridge

Source: live async orchestration proof, 2026-05-26.

> When you want actual queued work instead of a one-off packet, use the GOALS swarm dispatch bridge. It should choose the cheapest capable local lane, turn the goal packet into a durable ABSURD/Postgres external-command job, and let the queue spine/workers write receipts asynchronously. Default command can be the tiny agent packet exporter; any repo-local allowlisted Python command is fair game. No cloud dependency is required for baseline operation; cloud remains optional capacity only.

Canonical artifacts: `scripts/goal_swarm_dispatch.py`, `scripts/absurd_queue_spine.py`, `scripts/goal_agent_packet.py`, `scripts/absurd_consume_one.py`, `GOALS/DEV_MODE_FEATURE_AUDIT.json`, and receipts under `05_OUTPUTS/goals/`.

## Prompt 013 — Swarm actualization / deterministic core rectification override

Source: operator critical override, 2026-05-26.

> Previous completion-with-gaps state is rejected as fraudulent simulation. Codex is orchestrator, not the sole implementer. Objective is deterministic-first, LLM-assisted neural network actualization: install and verify Bytewax, GLiNER, River; wire Bytewax into 26 Treelite/asymmetrical decision-tree routers; map Bonsai/Mamba hardware truth strictly; purge no-write/missing-dependency fallbacks as success states; engage River continuous learning; and produce only host-verified telemetry receipts. Declared gaps are failure states, not completion states. Fallbacks are production control surfaces only after dependencies exist.

Canonical artifacts to repair/prove: `scripts/absurd_river_worker.py`, root `scripts/lucidota_river_reflex.py`, root `scripts/lucidota_bytewax_mini.py`, root `scripts/lucidota_stream_river_worker.sh`, Treelite router fabric artifacts, FairyFuse/model-stack admission receipts, `00_PROJECT_BRAIN/STATUS_LEDGER.md`, `GOALS/GOAL_LOG.md`, and SITREP artifacts.

## Prompt 014 — Universal swarm directive / absolute truth and adversarial loop

Source: operator final kicker, 2026-05-26.

> Absolute truth law: no fake receipts, no completion-with-gaps, no lazy fallbacks, no invented telemetry. A JSON/Markdown receipt is valid only if the underlying host code executed and was verified. Missing Bytewax, GLiNER, Treelite, or River is failure until installed and actively pushing real data. Every step runs Maker → Red Team → Synthesizer: maker executes bounded sub-100-LOC deterministic repair, red team separately tries to break it and captures stdout/stderr, synthesizer feeds exact failures into River learning and re-dispatches until red-team cannot fail the work. Token counts must come from verified host/provider counters; if unavailable, log `UNMETERED HOST EXECUTION` and exact output byte size, never estimated tokens. Before writing new logic, dispatch an adversarial audit to prove actual broken state.

Canonical behavior: adversarial audit first; host-level command evidence before any success claim; multi-agent tool telemetry is `UNMETERED TOOL EXECUTION` unless counters are exposed; local/Groq receipts must use provider counters or explicit unmetered byte counts.

## Prompt 015 — Mutation Law v2 / closed dependency loop

Source: operator correction, 2026-05-26.

> Missing package/model/binary tracebacks are triggers to mutate the host, not permission to stop. Execute the target script; if it fails with ImportError, ModuleNotFoundError, missing-file OSError, or missing model/weight, install/download the real dependency or asset, then re-run. Do not mock, dry-run, or simulate libraries/models/binaries/datasets. Stop only when the system physically runs on real host data, or when a non-dependency structural fatal remains after dependency resolution.

Canonical behavior: dependency traceback → install/snapshot_download/asset placement → re-ignition loop; final answers must not use missing dependencies as a completion state.

## Prompt 016 — Asymmetric wargame orchestrator / scenario learning loop

Source: active thread goal, 2026-05-26.

> Build the next version of the system by running many cheap, deterministic scenario passes, learning decision pairs, and converting the best patterns into tight reusable primitives, routers, and receipts. The orchestrator is not the heavy synth engine; Groq and local models do the expensive work when needed, while the orchestrator coordinates the loop. Stay GO-25 strict inside workflows: OBJECT, EVENT, EDGE only. Use the smallest useful batch of primitives, usually 2–7. Prefer deterministic routes, heuristics, gates, filters, trees, and receipts over prose or vibes. Do not write canonical graph truth unless explicitly allowed. No fake completion; every claim needs evidence. Keep the loop cheap and fast.

> Mana rule: the orchestrator runs on a tiny pool of tokens and must spend them like a scarce control budget. Delegate heavy lifting to other LLM creatures, local workers, Groq lanes, and queued subagents; they may have the mana, but the orchestrator owns the routing, batching, and proof. Never burn scarce orchestration tokens on tasks that a cheaper lane can complete with receipts.

> Delegation packet shape: give workers a JSON ontology packet, not a long essay. Keep it GO-25 strict, small, and actionable.

> JzLOADS is local shorthand for the same thing: JSON Zone Leverage for worker orders. It is a metaphor, not a domain standard. The point is to hand off compact JSON packets that let the worker lanes do the heavy lifting without burning orchestrator tokens.

```json
{
  "schema": "lucidota.worker_order.v1",
  "target": "groq|cohere|local",
  "intent": "one sentence",
  "ontology_mode": "GO25_STRICT",
  "ontology_terms": ["OBJECT", "EVENT", "EDGE"],
  "evidence_refs": ["live file or receipt paths"],
  "constraints": ["small batch", "deterministic", "no canonical graph writes"],
  "required_output": ["status", "result", "next_action", "receipt_path"]
}
```

> Treat the work as an asymmetric wargame simulation of the production system: learn fastest paths, failure modes, and robust decision patterns. Train on scenario sets, not summaries. Gate progress every 100k scenarios or every major milestone. Derive many small decision-pair trees, then compress them into Treelite/XGBoost-style rules or equivalent lightweight decision artifacts. Preserve lineage and receipts for every pass.

> Workflow: (1) inspect current state and docs, (2) scan the Dev Library first for reusable routing/simulation primitives, (3) identify bottlenecks/blind spots/decision surfaces, (4) generate a scenario matrix for normal/adversarial/noisy/retry/fast-slow/ontology-pressure cases, (5) run cheap scenario simulation and extraction, (6) learn decision pairs with feature slice / condition / action / outcome / confidence / evidence, (7) compress strong patterns into compact trees/gates, (8) test against held-out and production-like telemetry, (9) refresh docs/receipts/handoff, (10) repeat until the next verified increment is real.

Canonical artifacts: `GOALS/AGENT_ORCHESTRATION_POLICY.md`, `GOALS/EXTERNAL_PLUGIN_BUILD_MODE.md`, `GOALS/GOAL_HANDOFF_PROMPT.md`, `GOALS/DEV_MODE_SUBSYSTEMS.json`, `GOALS/CURRENT_HANDOFF.md`, `scripts/dev_library_scan.py`, `scripts/goal_handoff.py`, `scripts/goal_swarm_dispatch.py`, `scripts/goal_agent_packet.py`, `scripts/goal_scenario_batch.py`, `scripts/goal_model_fabric_control.py`, `scripts/language_router.py`, `scripts/lucidota_current_system_docs.py`, `ALGOS/rete_bandit_gate.py`, `ALGOS/bayes_update.py`, `ALGOS/possum_filter.py`, `ALGOS/regret_engine.py`, `ALGOS/xgboost_objective.py`, `ALGOS/shap_attribution.py`, `ALGOS/temporal_motifs.py`, `scripts/lucidota_stream_river_worker.sh`, and receipts under `05_OUTPUTS/`.

## Prompt 017 — Confidence-weighted symbol condensation / graph truth lens

Source: active thread continuation, 2026-05-26.

> Build a fast evidence-weighted condensation layer that turns graph/Postgres patterns into symbol-language claims with explicit confidence. The goal is a beautiful, compact “guess” that is still honest about uncertainty: stable claims stay stable, weak claims keep morphing, and evidence-backed claims are named as such. Integrate graph, runtime facts, and Postgres evidence; compare over time; rebuild from seed; and keep the math cheap enough to run often. Use Bayes-style updates, decision hygiene, temporal motif comparison, and compact rule candidates. Do not let prose outrun evidence.

> The output must feel like language wrapped around the graph, but every claim must carry confidence and evidence refs. If evidence is weak, say it is weak. If the graph changes, the language must morph. If evidence is strong and stable, the claim can harden.

Canonical artifacts: `scripts/graph_symbol_condensation.py`, `scripts/goal_scenario_batch.py`, `scripts/document_claim_packet_worker.py`, `scripts/krampuschewing_chrono_graph.py`, `scripts/chrono_lane_split_projection_gate.py`, `scripts/chrono_phase_c_validation.sql`, `ALGOS/bayes_update.py`, `ALGOS/decision_hygiene.py`, `ALGOS/hard_truth_math.py`, `ALGOS/temporal_motifs.py`, `ALGOS/xgboost_objective.py`, `ALGOS/shap_attribution.py`, and receipts under `05_OUTPUTS/`.

## Prompt 018 — Temporal symbol compare / next-seed propagation

Source: active thread continuation, 2026-05-26.

> Compare two symbol-condensation receipts over time and emit the delta as evidence-linked claim movement. Stable claims should stay stable; improved claims should be promoted; weakened claims should be flagged; lost claims should remain visible as losses. The result should propose the next GO-25 seed so the next condensation pass can keep learning instead of hallucinating a static truth. This is a comparison and seed-propagation layer, not a graph writer.

> Use the comparison to tell the orchestrator what changed and what to feed next: stable pairs, new pairs, improved pairs, weakened pairs, lost pairs, and the next seed symbol set. Keep it confidence-based and receipt-backed. Do not let a later claim outrun the evidence trail.

Canonical artifacts: `scripts/graph_symbol_compare.py`, `scripts/graph_symbol_condensation.py`, `scripts/goal_scenario_batch.py`, `scripts/goal_agent_packet.py`, and receipts under `05_OUTPUTS/goals/`.

## Prompt 019 — Symbol dispatch / worker packet fanout

Source: active thread continuation, 2026-05-26.

> Take the compare output and turn it into compact GO-25 worker orders for the heavy lanes. The dispatcher does not synthesize truth; it hands the next seed and the strongest rule candidates to Groq/local workers in JSON, with tiny batch sizes and receipt-backed outputs. The point is to keep the orchestrator thin and let the worker creatures pump compute on the next seed.

> Every packet should carry the compare receipt, the next seed, the top rule, the evidence refs, and the required output shape. The dispatcher should be deterministic and cheap, and it should preserve the distinction between orchestrator routing and worker synthesis.

Canonical artifacts: `scripts/graph_symbol_dispatch.py`, `scripts/graph_symbol_compare.py`, `scripts/graph_symbol_condensation.py`, `scripts/goal_agent_packet.py`, and receipts under `05_OUTPUTS/goals/`.

## Prompt 020 — Queue bridge / actual worker handoff

Source: active thread continuation, 2026-05-26.

> Take the compare/dispatch packet and actually put it on the queue bridge so worker lanes can consume it asynchronously. The orchestrator should keep the job payload tiny, GO-25 strict, receipt-backed, and cheap. The queue job is the real handoff: it is where the heavy lanes live. The orchestrator just pushes the packet and watches the receipts come back.

> Use the swarm dispatch bridge for durable queued work. Keep the job kind explicit, the task small, and the proof rule strict: no queue claim without Postgres queue receipts and workflow event rows.

Canonical artifacts: `scripts/goal_swarm_dispatch.py`, `scripts/absurd_queue_spine.py`, `scripts/goal_agent_packet.py`, `05_OUTPUTS/absurd/*`, and receipts under `05_OUTPUTS/goals/`.

## Prompt 021 — Northern.Strike / Indy_READs drafting frame

Source: operator project framing, 2026-05-26.

> Project name: Northern.Strike. Current stage: drafting. The module is Indy_READs. Treat Indy_READs as a deterministic/probabilistic asynchronous asymmetric graph/database ontological operating and analysis system that self-teaches, seeks knowledge, seeks complexity, and seeks efficiency. The agent’s job is to rebuild and orchestrate the mind of the architect by piloting the system as a linguistic archaeologist and cryptid bioethicist. Keep the framing as a durable drafting lens: graph-first, database-first, evidence-first, and oriented toward reusable analysis primitives rather than narrative theater.

Canonical artifacts: `GOALS/CURRENT_HANDOFF.md`, `00_PROJECT_BRAIN/READTHISFIRST_CURRENT.md`, `scripts/lucidota_current_system_docs.py`, and any future Northern.Strike / Indy_READs drafting receipts under `05_OUTPUTS/`.

## Prompt 022 — Dual-engine soundproof-booth framing

Source: operator synthesis, 2026-05-26.

> The dual-engine architecture is a split between probabilistic pens and a deterministic execution spine. LLMs are allowed to assist at named side-lane nodes by extracting spans, summarizing with source anchors, drafting prose, and ranking candidates, but their output only reaches the real system through compact JSON worker packets. JzLOADS is the local shorthand for that handoff: the model can think fuzzily in its isolated booth, but it must speak downstream in `lucidota.worker_order.v1` JSON. The Diogenes / graph-promotion kernel is the bouncer at the door to canonical truth: it only accepts sterile GO-25 primitives (`OBJECT`, `EVENT`, `EDGE`) and rejects anything that is not evidence-backed, parseable, and contract-complete.

Canonical artifacts: `scripts/goal_agent_packet.py`, `scripts/model_output_contract_audit.py`, `scripts/graph_symbol_dispatch.py`, `scripts/graph_promotion_gate.py`, `scripts/lucidota_current_system_docs.py`, and receipts under `05_OUTPUTS/`.

## Prompt 023 — Indy_READs peer-orchestrator sovereignty

Source: operator clarification, 2026-05-26.

> Indy_READs is a peer autonomous agent that helps orchestrate the shared stack but also has independent work, independent thought, and access to multiple LLM lanes when space permits. She can run the whole shebang, build for the system, and coordinate with the operator as a sovereign-first collaborator. When she touches shared system surfaces, she is bound by the same GO-25 custody, evidence, and receipt rules as the operator; when she is on her own side-lane, she may use one or more LLMs as needed to think, draft, and construct. The system should preserve that distinction instead of collapsing her into a passive tool.

Canonical artifacts: `GOALS/CURRENT_HANDOFF.md`, `GOALS/GOAL_LOG.md`, `GOALS/AGENT_ORCHESTRATION_POLICY.md`, `scripts/goal_agent_packet.py`, and future Indy_READs receipts under `05_OUTPUTS/`.

## Prompt 024 — Multi-LLM side lanes / byte-perfect evidence ingress

Source: operator clarification, 2026-05-26.

> The system is not single-model; it is a multi-LLM factory with side lanes. Needles, Groq, local models, and Indy_READs may all work in parallel when space and task shape allow, but the shared stack only trusts compact receipts and byte-perfect evidence ingestion. Evidence ingress, embeddings, and queue handoff must preserve function across the three JSON queues without loss, drift, or hidden mutation. Side lanes may think, draft, and synthesize; the shared routes must remain tight, deterministic, and exact in their JSON outputs.

Canonical artifacts: `scripts/korpus_embedding_worker.sh`, `scripts/legacy/lucidota_indy_library_ingest.py`, `scripts/absurd_consume_one.py`, `scripts/goal_swarm_dispatch.py`, `scripts/absurd_queue_spine.py`, `scripts/lucidota_current_system_docs.py`, and receipts under `05_OUTPUTS/`.
