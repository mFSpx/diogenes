# PROJECT 2501 CORE CONTRACT

Status: ACTIVE LAW  
Purpose: turn every operator/model/tool action into measurable board state, not chat theater.

## Core Contract

Every input becomes an EventEnvelope.
Every EventEnvelope is timestamped, hashed, embedded, classified, and routed.
Every route decision is made by explicit gates: deterministic rules first, Treelite second, model fallback last.
Every model invocation is logged as a ModelInvocation with prompt hash, output hash, latency, token counts, and verdict.
Every durable action emits a WorkReceipt.
Every WorkReceipt updates River training rows.
Every River training row improves future routing, cost prediction, and slop detection.
Every graph write is staged, justified, receipt-backed, and replayable.
Every script lives only if it has a caller, a purpose, a receipt contract, and a survival score.
Every dead script is never deleted; it is corpse-manifested and sent to Krampuschewing.
Every UI tile reads from receipts, not claims.
The system wins by reducing entropy, preserving proof, improving routing, and increasing executable board position.

Every ontology statement remains a structured hypothesis ecology, not final truth.
Every selected working reality records evidence, hypothesis, selected action frame, move, result, contradictions, and `record_for_future=true`.

## Big Board Law

`http://127.0.0.1:8765/` is the Project 2501 / LUCIDOTA Big Board.

- The Big Board is command instrumentation, not decoration.
- A trackable metric is a trackable metric: it belongs in PostgreSQL and/or the status ledger before the UI displays it as truth.
- Big Board feature additions/removals are operator-authority changes. The operator and the operator alone requests them.
- Agents may fix broken telemetry plumbing and may add backend receipts/DB rows for already-requested metrics, but must not silently invent new dashboard tiles as completion theater.
- Every tile must be traceable to a receipt path, DB row, or explicit live probe.
- Backend workers may emit `lucidota_control.watch_metric` rows for trackable facts, but UI feature shape changes wait for explicit operator request. Ledger first, screen second, siempre.
- Agents must not suggest UI features, dashboard tiles, or screen changes. Only the operator orders UI changes.

## Board Move Shape

Each move records:

```json
{
  "actor": "operator|codex|groq|local_model|scraper|auditor|worker",
  "position": "current board state",
  "move": "thing attempted",
  "lane": "fast|slow|audit|external|dead_letter",
  "cost": "tokens|time|cpu|vram|risk|graph_mutation",
  "gain": "proof|compression|routing_accuracy|artifact|fixed_code|reduced_entropy",
  "receipt": "path or db row",
  "verdict": "win|loss|stall|poison|retry|promote|kill"
}
```

When a move depends on selected uncertainty, it also records:

```json
{
  "evidence": ["receipt://...", "log://...", "hash://..."],
  "hypothesis": "This subsystem fails because graph writes are blocked by admission policy.",
  "working_reality": "Treat graph writes as blocked until a verified allow-gate exists.",
  "move": "Stage graph candidates instead of claiming canonical write.",
  "result": "PASS|FAIL|CONFLICT",
  "record_for_future": true
}
```

Reality is the board. Evidence is what we can see. Ontology is our map. Hypothesis is our planned route. Working reality is the move we choose. Receipts are how future players know whether we cheated.

## Execution Order

1. Event envelope schema.
2. Route decision schema.
3. Work receipt schema.
4. Bytewax stream skeleton.
5. Treelite gate interface.
6. River training-row writer.
7. Fast lane first-frame responder.
8. Slow lane durable worker.
9. Audit lane script classifier.
10. Watch UI reads from receipts only.

Math is in charge. Models are role players.

BOARD EFFECT TOURNAMENT LAW

Every move affects the board. Krampus is the fair audit/enforcement persona: prefer moves affecting 2 or 4+ board systems, and classify naughty/slop only from evidence. Santa is the glow-finding/exploration persona: one board effect can be enough, but assumptions must be labeled and converted into working realities before authority. Board moves carry an effect_gate receipt. Canon: 00_PROJECT_BRAIN/ACTIVE_SPEC/08_BOARD_EFFECT_TOURNAMENT_LAW.md.
