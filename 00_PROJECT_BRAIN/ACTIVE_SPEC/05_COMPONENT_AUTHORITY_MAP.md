<!--
DEV NOTE (anti-slop provenance, 2026-05-24): A file/report/taxonomy is proof of the truth it actually declares: this document exists, has this title, and asserts this scoped content. It is not proof of unrelated external facts, not permission to rename its scope as "the manual," and not evidence that the operator personally coined a local label. Names are integral to their own statements and contexts; do not erase, swap, or flatten them. When an agent says "this is the user's rule," it must cite direct operator instructions or accepted doctrine/RFC/receipt lineage, state exactly how many source docs were consulted and why those docs were chosen, and distinguish direct evidence from inference or compression. If the system lacks receipts for a factual CLAIM, classify it as hypothesis, observation, inference, suggestion, or confidence-rated candidate instead.
-->

# LUCIDOTA Component Authority Map

Status: ACTIVE SPEC SOURCE — named component roles, boundaries, and evidence paths  
Purpose: map named components to scoped duties, inputs, outputs, authority boundaries, and evidence. It is a component authority map, not proof that every named component is active runtime.

## 1. Diogenes Kernel

- **What:** authority kernel: command envelope, permissions, receipts, graph-promotion guard.
- **Why:** prevents live coding, agents, or scripts from becoming ambient authority.
- **How:** control packets, kernel authorization checks, authority registry, graph gate, proof kernel.
- **Inputs:** operator intent, command envelopes, evidence refs, worker requests.
- **Outputs:** route decisions, receipts, approval/defer/reject records.
- **Authority:** gate/receipt authority; not a general truth oracle.
- **Evidence:** `scripts/kernel_control_packet.py`, `scripts/spine_kernel_authorization.py`, `scripts/spine_authority_checker.py`, `scripts/graph_promotion_gate.py`, `scripts/proof_kernel.py`.

## 2. ABSURD Workflows

- **What:** durable Postgres-backed work-order/event/dead-letter spine.
- **Why:** makes work restartable, inspectable, and non-magical.
- **How:** queue rows, worker contracts, `FOR UPDATE SKIP LOCKED`, events, receipts, dead letters.
- **Inputs:** command envelopes, derived compute tasks, ingest jobs, graph-promotion jobs.
- **Outputs:** completed/failed/dead-lettered work, workflow events, receipts.
- **Authority:** queue/work authority only; ordinary workers do not materialize graph truth.
- **Evidence:** `06_SCHEMA/035_absurd_queue_spine.sql`, `scripts/absurd_queue_spine.py`, `scripts/absurd_consume_one.py`, `scripts/absurd_worker_contracts.py`.

## 3. Krampus Express / KORPUS

- **What:** preserve-now, digest-later intake stomach for large drops, documents, archives, old code, and evidence jungles.
- **Why:** lets the active system get small without deleting memory.
- **How:** hash, CAS, classify, componentize, queue derived work, preserve failures.
- **Inputs:** filesystem drops, archives, vault material, recovered docs/code.
- **Outputs:** custody rows, components, indexes, derived queue tasks, dashboards.
- **Authority:** custody/candidate authority; no direct canonical truth.
- **Evidence:** `scripts/korpus_krampii.py`, `scripts/krampuschewing_watcher.sh`, `scripts/spine_krampus_worker.py`, `06_SCHEMA/019_korpus_krampii.sql`.

## 4. Full ETL Pipeline

- **What:** deterministic evidence hydraulic press.
- **Why:** high-volume raw storage must not be hostage to expensive interpretation.
- **How:** Rust front door for hot lane; Python workers for adaptive slow lane.
- **Inputs:** drops/files/databases/messages.
- **Outputs:** CAS, metadata, components, derived tasks, receipts.
- **Authority:** custody/queue authority only.
- **Evidence:** `ACTIVE_SPEC/03_CUSTODY_ETL_PIPELINE.md`, `01_REPOS/lucidota_etl/`, `06_SCHEMA/023_etl_pipeline.sql`.

## 5. PercyphonAI

- **What:** zero/low-VRAM procedural entity/reasoning mask organ.
- **Why:** creates stable low-cost scaffolding without spending model memory or pretending to be truth.
- **How:** deterministic SHA-256 seeded slots, aliases, UUID-like identities, ternary offsets.
- **Inputs:** source labels/villagers/context signals.
- **Outputs:** procedural slots and fluid slot metadata.
- **Authority:** metadata/scaffold authority only.
- **Evidence:** `ALGOS/percyphon.py`.

## 6. Indy_READs

- **What:** named teammate/read-intake/synthesis identity, more than a disposable agent.
- **Why:** gives the operator a stable companion lens for books, intake, memory, role routing, and synthesis.
- **How:** library ingest, watcher, polycareer role modes, Glow Hunter, LoRA cartridge prep, GO-25 alignment.
- **Inputs:** books, documents, operator questions, intake artifacts.
- **Outputs:** reading memory, role routes, synthesis notes, LoRA training candidates, reports.
- **Authority:** companion/synthesis/candidate authority; not graph truth or external-action authority.
- **Evidence:** `scripts/indy_reads.py`, `scripts/lucidota_indy_*`, `00_PROJECT_BRAIN/INDY_READS_POLYCAREER_WORKFLOW_WIZARD/`.

## 7. Local Model / RAM / VRAM Fabric

- **What:** constrained local inference fabric.
- **Why:** sovereignty over data and operation; punch above limited hardware.
- **How:** safe-ops profile, model governor, staged residency, DeepSeek/Mamba/Bonsai/LoRA lanes, Needle scouts, context reaper.
- **Inputs:** bounded prompts, hashes, model paths, routing decisions.
- **Outputs:** draft text, extraction candidates, model receipts, runtime budget decisions.
- **Authority:** advisory/extraction/draft authority only.
- **Evidence:** `00_PROJECT_BRAIN/gpu_model_runtime_registry.json`, `04_RUNTIME/inference_os/`, `scripts/lucidota_model_governor.py`, `scripts/model_runner_*`, `pypeline/math/model_vram_scheduler.py`.

## 8. Language Membrane / Multiplexing / Hyperplexing

- **What:** input/output routing membrane for deterministic templates, retrieval, model synthesis, and smoothing lanes.
- **Why:** separates structure from prose and keeps output explainable.
- **How:** fast regex/RETE routing, semantic neighbor lane, exact quote RAG, deterministic templates, draft-only weaving.
- **Inputs:** operator text, retrieved sources, model drafts, route context.
- **Outputs:** lane-tagged draft surfaces and routing metadata.
- **Authority:** draft/surface authority only.
- **Evidence:** `core/language_membrane.py`, `scripts/language_router.py`, `scripts/lucidota_cli.py language`, `scripts/fast_slow_lane_gate.py`, `scripts/template_contract.py`, `scripts/cep_builder.py`, `scripts/kernel_control_packet.py`, `scripts/hypertimeline_engine.py`.
- **Parity rule:** build-mode and operator-mode language routing share the same deterministic subsystem; CLI/operator surfaces may adapt I/O, but not the underlying GO/JSON/work-order semantics.

## 9. Constant Learning

- **What:** learning from committed events, diffs, failures, and receipts.
- **Why:** every coding truth can improve investigation truth and reporting truth if labeled honestly.
- **How:** River/Bytewax lanes, derived compute queue, telemetry features, model-diff labs, future SONA/LoRA repair lanes.
- **Inputs:** workflow events, diffs, receipts, claim packets, dead letters, telemetry.
- **Outputs:** scores, anomaly hints, repair candidates, LoRA/data candidates.
- **Authority:** hint/judge/candidate authority only.
- **Evidence:** `scripts/absurd_river_worker.py`, `06_SCHEMA/073_absurd_river_claim_packet_job.sql`, `00_PROJECT_BRAIN/RUVECTOR_ABSURD_SONA_RIVERML_NOTES.md`.

## 10. Artifact Generation Templates

- **What:** deterministic report/letter/case-packet/surface generation.
- **Why:** operator-visible output should be reproducible and source-anchored, not free-prose fog.
- **How:** templates plus bounded context, exact quotes, receipts, export bundles.
- **Inputs:** evidence candidates, case workspace, timeline, operator intent.
- **Outputs:** drafts, packets, exports, review surfaces.
- **Authority:** draft/report authority; final external action requires explicit operator authorization.
- **Evidence:** `scripts/template_contract.py`, `scripts/case_packet_*`, `scripts/export_bundle.py`, `05_OUTPUTS/templates/`.

## 11. Board-Game / Simulation Lab

- **What:** board-game making and simulation lab, including Ahoy.
- **Why:** simulations can mine algorithms, policies, and test harnesses without polluting production truth.
- **How:** isolated game engines, strategy datasets, receipts, exported models when useful.
- **Inputs:** rules, game states, policies, synthetic training rows.
- **Outputs:** simulations, strategy receipts, algorithm candidates.
- **Authority:** paused lab / reusable-prior authority only.
- **Boundary:** Ahoy belongs under `/home/mfspx/BOARD_GAMES/AHOY/`; it has been externalized from the active LUCIDOTA repo tree and is not active LUCIDOTA truth spine.

## 12. Active Ontology

- **What:** GO-25 plus approved extensions.
- **Why:** gives the operator a compact active conceptual skin without dragging archived ROOT-414 gravity into production.
- **How:** `OFFICIAL_ONTOLOGY.json`, graph schema, fidelity guard, staging contracts.
- **Inputs:** candidate objects/events/edges, operator approvals, ontology staging.
- **Outputs:** graph items/edges/candidates, fidelity reports.
- **Authority:** ontology label/candidate authority; truth still requires spine gate.
- **Evidence:** `OFFICIAL_ONTOLOGY.json`, `BOOKS/GO_ACTIVE_TERMS.json`, `06_SCHEMA/016_go_graph_core.sql`, `scripts/operator_ontology_fidelity_guard.py`.
