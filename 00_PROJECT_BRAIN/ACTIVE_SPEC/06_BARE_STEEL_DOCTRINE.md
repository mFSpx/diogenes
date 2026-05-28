# Bare Steel Doctrine

Status: ACTIVE SPEC SOURCE — targeted async DB/Graph state law
Authority: active runtime/data-flow doctrine for LUCIDOTA components, swarm workers, dashboards, model bridges, queue workers, and graph promotion gates.

## Rule 4. DB & Graph as Absolute Truth (Targeted & Async)

State does not exist in ghost variables or temporary in-memory illusions; it persists in the PostgreSQL DB and the Graph.

However, DO NOT bottleneck the system with global synchronous reads. Components must operate asynchronously.

Fetch only the hyper-specific, localized data required for the immediate task. Emit events and write state asynchronously. Read selectively, persist globally, and never block the main loop.

## Execution contract

1. Hot-lane workers may use bounded in-memory working state only as a cache, debounce buffer, or current task frame; it is not system truth.
2. PostgreSQL is the durable control/state/event substrate. Read only the specific rows, receipts, tables, ranges, cursors, or counts required now.
3. The Graph is the durable semantic substrate. Candidate graph changes are staged through graph promotion packets; canonical graph materialization remains gated.
4. Main loops, SSE streams, dashboards, and agents must not perform broad blocking scans. Use timeouts, SKIP LOCKED work claims, small limits, cached snapshots, and background refresh.
5. Persistence is global; reads are local. Emit compact events and receipts, then let async workers consolidate, promote, roll up, or materialize.
6. If a component lacks the exact local data it needs, it should degrade visibly with a receipt/status, not hallucinate state and not freeze the system.

## Operator shorthand

Read selectively. Persist globally. Route asynchronously. Promote through gates. Never block the main loop.
