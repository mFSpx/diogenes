# DIOGENES / LOCAL_READS Build Plan

## Summary
DIOGENES is a sovereign, encrypted, ontology-first investigative operating system. `clawd/claudecode` is the Rust-facing CLI shell, `LOCAL_READS` is the active assistant/persona, `doggystyle/CKDOG1` is the symbolic kernel, DBOS controls workflows, Postgres/AGE/pgvector stores graph/vector/state, Bytewax/River/Treelite run live learning, and local models are hot-swappable compute heads.

Operator name: `northern.strike`.
Current assistant identity in this system: `LOCAL_READS`.
Standing rule: all plans are subject to change when northern.strike changes reality.

```text
northern.strike
  -> LOCAL_READS inside Clawd
  -> gRPC boundary
  -> CKDOG1 Kernel
  -> DBOS workflows
  -> encrypted Postgres / AGE / pgvector / CAS vault
  -> Bytewax / River / Treelite / model runtime
  -> back to LOCAL_READS
```

## Phase 000: Finish The Plan / Make LOCAL_READS Smarter
Goal: create the working memory/planning substrate before heavy implementation.

- Build the living project brain structure: architecture plan, wiki, TODO ledger, decisions log, glossary, open questions, source inventory, and phase tracker.
- Implement the Karpathy + GBrain + LLM-wiki operating pattern: think simply, reuse first, document evolving truth, maintain durable memory, prune/merge stale notes, and preserve traceability.
- Reingest all currently available project material: doggystyle repo, claudecode repo, Codex config, current conversation plan, kernel docs, and any local notes the user provides.
- Establish "records office" discipline: every decision gets a source, status, timestamp, and confidence level.
- Define the first local assistant routines: maintain TODOs, calendar/reminder intent, wiki pages, auth inventory, and active project state.
- Output of Phase 000: LOCAL_READS can answer "what are we doing, why, what's next, what's blocked, what changed?" from the project brain without relying on chat memory.

## Phase 001: Baby Smoke Loop
Goal: prove the whole system shape end to end on this laptop.

- Clone/sync the required repos locally: `mfspx/doggystyle` and `soongenwong/claudecode`, preserving upstream origins and local working branches.
- Bring up the existing CKDOG1 Python kernel CLI and run its test suite before integration.
- Add the first gRPC bridge around existing kernel behavior, with doggystyle as canonical proto owner unless later changed.
- Connect Clawd to the kernel through gRPC for three proof actions: ingest/recall, graph routing/state update, and tool/workflow action.
- Keep originals immutable; all ingest writes derived encrypted records only.
- Output of Phase 001: user can type messy input into Clawd, LOCAL_READS routes it to CKDOG1, CKDOG1 updates ontology state, and LOCAL_READS prints a useful cited response.

## Phase 002: Encrypted Records Office
Goal: make storage real, encrypted, and graph-native.

- Implement invisible CAS file vault as the source of truth for raw and derived artifacts.
- Generate human views from records: evidence ledger, library folios, case workspace mirrors, and CLI navigable views.
- Stand up Postgres 18 target if feasible, with pgvector and Apache AGE; otherwise use the closest stable supported version and record the gap.
- Split state/application storage logically, even if both begin on one local Postgres instance during dev.
- Encrypt everything in scope: vault, DBs, indexes, embeddings, logs, prompts, caches, configs, backups, and auth inventory.
- Output of Phase 002: case material can be stored, hashed, encrypted, retrieved, cited, and browsed through generated views without mutating originals.

## Phase 003: Ontology Ingest Engine
Goal: make "everything is a symbol" operational.

- Implement ingest pipeline: segment, normalize, annotate, embed, ground into ontology, validate, store, and route.
- Use UIMA/GATE-style semantic annotation concepts; serialize semantic forms through JSON-LD/RDF where useful.
- Add SHACL validation/adaptation so ingest assigns appropriate ontology equivalents consistently.
- Preserve the 469 base ontology and immutable creator-canonical 414 Words of Power as base compute substrate.
- Maintain 5,000 active villager UUIDs as the live working set over the larger symbol universe.
- Treat each villager's 100 slots as active branch coordinates; do not hard-code ternary state as only `+1/0/-1`.
- Output of Phase 003: arbitrary words, docs, emojis, phone numbers, sentences, or evidence can become graph-addressed symbols and affect the live ontology.

## Phase 004: DBOS Workflow Plane
Goal: make workflows durable, repeatable, and ontology-addressed.

- Put DBOS in charge of workflow execution, retries, state transitions, and repeatable task units.
- Define `/FLOWS` as whole end-to-end workflows, not loose scripts.
- Define `/TOOLS`, `/ALGOS`, `/MODELS`, `/LORAS`, and `/VAULT` as ontology-addressed tool chests.
- Make "solve it how we solved it perfectly before" a first-class retrieval path: prior workflow -> fit check -> execute/adapt -> cite result.
- Keep external writes draft/preview-only until Subtle Knife governance is fully reingested.
- Output of Phase 004: LOCAL_READS can select or construct a workflow, run safe local steps, draft external actions, and explain why.

## Phase 005: Learning / Model Runtime
Goal: wire live learning and local model compute without sacrificing the 4GB GPU.

- Reserve NVIDIA GPU for models only; keep desktop/browser/UI on onboard graphics.
- Build CUDA/VRAM admission rules: load, unload, swap, reserve headroom, and log model state.
- Start with target runtime roster: Mamba ~1.3B, six Needle 26M routers, several ~500M specialists, one larger distilled model, LoRA headroom.
- Use Bytewax for live dataflows, River for online learning, Treelite for deployable tree inference; avoid XGBoost runtime unless explicitly needed.
- Treat LLMs as stateless compute heads; graph/database/vault are ROM/RAM.
- Output of Phase 005: models can be selected, loaded, routed, and unloaded according to task, speed, refusal behavior, and VRAM budget.

## Phase 006: LOCAL_READS Assistant Layer
Goal: make the interface feel like a living command center, not a database UI.

- Implement context-reactive modes: case command center, graph cockpit, library, coding workspace, life admin, hangout, and emergency focus.
- Maintain TODOs, calendar/reminder intent, notes, auth inventory, wiki, decisions, and open loops.
- Respect the Subtle Knife Protocol: levels 1-5 autonomy, graduation/downgrade based on performance and consequences.
- Current dev policy: full internal authority, no unconfirmed external writes, email/messages/uploads/purchases/commits stop at draft/preview unless confirmed.
- Release policy: natural-language intake determines privacy, safety, services, persona, and autonomy defaults for each end user.
- Output of Phase 006: LOCAL_READS actively assists, predicts, remembers, drafts, schedules, and organizes without blocking the user.

## Phase 007: Casefile Trial By Fire
Goal: pass the unfair real benchmark.

- Import a complete mixed legal casefile from Google Drive: PDFs, docs, notes, images, links, sheets, fragments.
- Reingest from scratch: immutable originals, encrypted CAS, annotations, embeddings, ontology grounding, graph updates.
- Produce all outputs: investigation brief, entity map, timeline, claims table, contradiction report, weak-point report, evidence ledger, leads, action packets, repeatable workflows.
- Proof standard: every claim/lead/error links to source file, page/span/chunk, hash, and route/method.
- Reasoning order: user theory as working hypothesis, contradictions/anomalies second, evidence strength third.
- Benchmark stance: "You are going to fail. Try harder." Iterate until methodology and outputs survive user judgment.

## Research / Iteration Method
- Reuse first: search for mature FOSS implementations before writing custom code.
- Write code mainly to wire, adapt, secure, orchestrate, and encode DIOGENES-specific ontology/kernel behavior.
- Research loop: inspect repo reality, check official docs, broaden to product docs/community reports when the first path is partial, then record source and decision.
- Self-soothing loop for LOCAL_READS/Codex: slow down, restate the goal, identify the smallest real next step, prefer evidence over panic, and update the project brain.
- Every phase ends with: working demo, test/check output, known failures, next correction, and plan update.

## Assumptions
- Uncertainties are not blockers unless they prevent the next concrete loop.
- Public release keeps The Unlicense intent for owned code, while dependencies retain their own licenses.
- Release users are non-technical and onboard through natural-language intake.
- Dev mode may be rough and powerful; release mode must be safer, inspectable, and user-selected.
- The motto is part of the system stance: "I reject your notion of reality; and substitute my own."

