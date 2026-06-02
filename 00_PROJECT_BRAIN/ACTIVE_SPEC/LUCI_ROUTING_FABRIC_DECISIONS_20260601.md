# LUCI Routing Fabric Decisions — Strict Local, Router-of-Routers

Saved on 2026-06-01 from operator build-session notes. This is implementation-control doctrine, not decorative architecture prose.

## Decision Delta

1. **Local model admission is strict.**
   - Resident/local lanes must pass explicit admission before use: files present, launchers present, ports/health known, VRAM/RAM budget acceptable, receipts written.
   - Local model admission must fail closed. If a local model is not admitted, LUCI must route around it or report a truthful skip.
   - Target local fabric includes:
     - 6 Needle router workers, target throughput expectation: ~2k tokens/sec each where measured.
     - 2 stateless Mamba lanes for fast concept/task motion.
     - DeepSeek R1 Distill Qwen in VRAM as a reasoning/coding lane.
     - Bonsai 4B ternary must be switchable into the same operator fabric as an alternate local lane; if sharing the VRAM slot with DeepSeek, registry/admission controls which one is resident.
     - BGE embedding lane in VRAM only when ingestion quality and memory admission are safe.
     - Indy_READs stays always-nearby as a research/intake assistant surface.

2. **External API lanes are not local admission.**
   - Groq, Vibes, and future API-key lanes are capability/provider lanes, not resident local models.
   - They need secret checks, quota/rate/budget checks, redaction-safe receipts, and graceful skip behavior.
   - Missing API keys must not fail local LUCI startup.
   - External lanes are routed as bounded processors, never confused with local resident model health.

3. **Routing routes to routing.**
   - LUCI should not have one flat router that picks a final model.
   - General routing may emit a packet to another router, workflow, ontology expansion, queue, kernel task, or model lane.
   - Routers are first-class DB/graph nodes with receipts: every route decision should be replayable as `input -> packet -> decision -> target -> result/skip`.

4. **DB and graph are the party floor.**
   - Postgres, graph/ontology, queues, receipts, telemetry, and deterministic algorithms are the control plane.
   - LLMs/API models/local models are bounded processors inside the control plane.
   - Important state, attempts, decisions, and outcomes must be DB/receipt-backed.

5. **ABSURD workflows are executable routing fabric.**
   - LUCI must be able to make new workflows, enqueue them, run them in parallel/async, and complete or retry them with receipts.
   - File-only queues are acceptable transitional scaffolding only; convergence target is canonical DB-backed queue/workflow state.

6. **Kernel/concept traffic should be fast and stateless where possible.**
   - Use Needles/Mambas/deterministic routers for rapid concept/task movement.
   - Preserve state in DB/graph, not hidden model memory.
   - Stateless lanes can be restarted/swapped without losing the system’s mind because the DB/graph owns memory.

7. **Operator metaphor: asymmetric board game.**
   - This is local project shorthand, not an established systems term.
   - It means LUCI development should be played as resource/routing/control optimization: strict admission, quick probes, many small parallel moves, receipts, and constant re-routing based on evidence.

## Acceptance Checks For Next Implementation Pass

- `./luci operate --text ...` still works and writes a Postgres workflow event.
- Local admission distinguishes local resident models from external API lanes.
- Missing Groq/Vibes keys produce `skipped` receipts, not startup failure.
- A route can target a second router/workflow, not only a model.
- Slow-lane queue writes canonical DB state, or reports the remaining file-queue gap truthfully.
- A DeepSeek/Bonsai lane switch is represented in model registry/admission, with only admitted resident lanes exposed as available.
- BGE drain remains blocked unless ingestion quality and memory admission pass.
