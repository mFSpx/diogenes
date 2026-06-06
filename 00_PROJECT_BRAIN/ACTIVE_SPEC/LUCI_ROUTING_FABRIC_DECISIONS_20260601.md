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

8. **RNS is the packet spine; Postgres stays the ledger.**
   - RNS / local-mesh packet fabric is the inter-lane transport: encrypted links, channel streams, resource payloads, and packet receipts.
   - Raw RNS.Packet / RNS.Channel / RNS.Resource are valid transport levels; the system should use the smallest one that carries the message.
   - Needles, encoders, and watchers are packet consumers/producers. They do not own canon and they do not infer cross-lane joins.
   - The Transaction Combiner is required for semantic joins across lanes. It groups packets by transaction_id, part_id, lane_id, and deadline/expected_parts, then emits the render packet or failure packet.
   - Bytewax is demoted to stateful stream work only: replayable windows, aggregation, recovery, River feed, drift counters, and checkpointed state. It is not the primary switchboard if RNS is present.
   - Postgres/PostgREST remains the authority for canon, custody, receipts, graph state, and API/manual visibility. LISTEN/NOTIFY may remain for DB-local signaling, but it is not the main hyperplexer.
   - Minimal packet contract is bounded and encrypted; the hard cap is a compact packet under the project’s 383-byte encrypted envelope target.
   - A proof harness should start with one console destination and bounded mock lane destinations, then measure latency, loss, retries, and resident RAM before any broad rollout.

9. **Route-cost kernel sits above transport, not above truth.**
   - A route-cost kernel may score RNS lane destinations, retry risk, queue pressure, trust, jitter, and link quality, but it does not replace RNS transport and it does not own canon.
   - The kernel belongs between RNS metrics and the lane chooser: `RNS metrics -> route feature vector -> deterministic scorer / Treelite / River -> RNS next hop`.
   - Its output is a small bounded receipt: next hop, route score, fallback, action, TTL. It is a routing economics engine, not a language model and not a semantic narrator.
   - Deterministic EWMA / threshold scoring comes first, Treelite/XGBoost second, River third, and any tiny neural model only earns a slot after it proves better under measured memory, latency, and route-quality receipts.
   - If the cost kernel cannot beat deterministic scoring and a compiled tiny tree model on reproducible data, it gets demoted to experiment or trash; no prestige lanes.
   - The correct placement is hot-path routing beside Treelite/River, not raw-text interpretation.

10. **RNS is transport; Transaction Combiner is semantics; Postgres is canon.**
    - RNS only answers whether a packet, channel message, or resource transfer got to the destination with receipts and link behavior. It does not decide semantic completeness.
    - Transaction Combiner is the explicit semantic join layer. It groups packets by `transaction_id`, `part_id`, `lane_id`, `expected_parts`, `deadline_ms`, and causal parent, then emits a render packet or failure packet.
    - Postgres/PostgREST receives async receipt writes and owns canonical custody, durable truth, graph state, and operator-visible history. It is not the hot switchboard.
    - Bytewax remains optional and narrower: stateful stream windows, joins, replay, recovery, River feed, drift counters, and checkpointed state. It is not the default packet fabric.
    - RNS packets should stay tiny and deterministic; the target envelope is a compact encrypted packet under the project’s 383-byte hard cap, with `fp16`-size vectors or smaller when vectors are required.
    - RNS Channel is for live small-message lane traffic; RNS Resource is for larger artifacts. Raw Packet is fire-and-prove telemetry.
    - The proof harness should first measure latency, loss, retry behavior, and resident RAM for a few bounded destinations before any broader rollout.

## Acceptance Checks For Next Implementation Pass

- `./luci operate --text ...` still works and writes a Postgres workflow event.
- Local admission distinguishes local resident models from external API lanes.
- Missing Groq/Vibes keys produce `skipped` receipts, not startup failure.
- A route can target a second router/workflow, not only a model.
- Slow-lane queue writes canonical DB state, or reports the remaining file-queue gap truthfully.
- A DeepSeek/Bonsai lane switch is represented in model registry/admission, with only admitted resident lanes exposed as available.
- BGE drain remains blocked unless ingestion quality and memory admission pass.
- RNS proof harness has an explicit console destination, bounded packet receipt path, and measured latency/loss/retry behavior before any broader cutover.
- Route-cost kernel receipts exist for packet next-hop selection, and the kernel is provably below truth/canon in the control stack.
- Transaction Combiner exists as the semantic join above RNS transport, and the packet contract fits the encrypted envelope target rather than pretending 512-byte float vectors are free.
