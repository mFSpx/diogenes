# Receipt Graph Normalization Design

Date: 2026-06-05
Status: Draft for review
Goal: Normalize LUCIDOTA receipt/work accounting into a relational graph that prevents aggregate-count laundering while keeping display surfaces compact and fetchable.

## 1. Problem Statement

The current live proof chain is good enough to prove boot/activity, but it is still flatter than the architecture wants:

- `agent_work_receipt`
- `model_invocation_receipt`
- `provider_call_receipt`
- `work_order`
- prompt surfaces
- workload ledger views

This makes it possible to show compact status, but not yet to express the full anti-lie graph cleanly:

- worker identity
- work order identity
- individual attempts
- model identity
- invocation identity
- provider call identity
- receipt/debt linkage

The result is that the system can prove work happened, but it cannot yet always expose the exact chain in the shape we want without flattening some of the relational truth.

## 2. Design Principles

1. **Display compact.**
   Status surfaces should summarize, not dump full rows.

2. **Fetch deep.**
   The full chain must remain queryable by foreign-key-style refs or stable IDs.

3. **Store relationally.**
   Truth belongs in tables/views, not in manual/root blobs.

4. **Never flatten proof into manual/root.**
   Manual and root should carry refs and summaries only.

5. **No claim without receipt or UNKNOWN debt.**
   Any actor or model claim without proof must be represented as debt, not narrative.

## 3. Target Shape

### 3.1 Visible status row
A compact row suitable for `/current` surfaces:

- `worker`
- `work_order`
- `model_identifier`
- `proof_status`
- `receipt_uuid`
- `timestamp`
- `next_route`

### 3.2 Fetchable trace
A relational chain that can be expanded on demand:

- `worker -> work_order_attempt -> model_invocation_receipt -> provider_call_receipt -> evidence/output refs`

### 3.3 New or normalized entities
The schema should support the following canonical entities:

- `worker`
- `work_order`
- `work_order_attempt`
- `model_identifier`
- `model_invocation_receipt`
- `provider_call_receipt`
- `agent_work_receipt`
- `unproven_work_debt`

The initial implementation may use existing tables plus compatibility views if a full rename would be too risky, but the resulting graph must expose these concepts explicitly.

## 4. Model Identifier Contract

The `model_identifier` concept must be stable enough to distinguish lanes that currently collapse together in `model_id` strings.

Required fields:

- `provider`
- `model_family`
- `model_id`
- `weight_hash`
- `quantization`
- `adapter_id`
- `runtime_backend`
- `lane_id`
- `context_window`
- `kv_cache_policy`

This can be materialized as a table or a view over normalized fields, but it must be addressable as a distinct DB entity.

## 5. Receipts and Attribution Rules

### 5.1 Work attribution
Every model invocation row must be able to answer:

- who invoked it
- from what worker lane
- for which work order
- under what model identifier
- with what receipt
- what proof status applies

### 5.2 Allowed nulls
A background/ambient/probe action may be unbound from a work order, but only if it is explicitly marked with an unbound reason such as:

- `ambient`
- `daemon`
- `probe`

If there is no work order, the row must say so explicitly.

### 5.3 Unknown debt
If an action is claimed but cannot be tied to receipts, it must land in debt surfaces instead of being promoted into compact status.

## 6. Display Surfaces

The following surfaces should remain compact and current:

- `active_operation_mode`
- `manual_current`
- `root_orchestrator_current`
- `workload_audit_current`
- Indy runtime current surfaces
- prompt surfaces

These surfaces should only carry refs/summaries to the receipt graph, not inline raw receipts.

## 7. Runtime Implication for Indy_READs

Indy_READs should not be treated as a Codex subagent. The runtime should be able to emit its own work records when booted through the IronClaw/local-model lane, but the schema must not assume that runtime is always active.

The graph needs to support:

- boot receipts
- Indy-authored self-model/wiki/hunch/system-map rows
- future work-order attempts
- proof and debt

## 8. Implementation Phasing

### Phase 1: Schema normalization
Add or normalize DB objects for:

- worker
- work_order_attempt
- model_identifier
- explicit links on invocation/provider receipts

### Phase 2: Compact current surfaces
Expose current summaries through PostgREST and keep manual/root as ref-only surfaces.

### Phase 3: Backfill/compatibility
Backfill from existing receipts and keep compatibility views so existing routes do not break.

### Phase 4: Gatekeeping
Add tests that assert:

- no claim without receipt
- no manual/root proof flattening
- compact status can fetch deep trace
- unknown debt remains visible

## 9. Success Criteria

The design is complete when:

- the DB can represent the full worker → order → attempt → model/provider receipt chain
- display surfaces remain compact
- manual/root do not flatten proof rows
- Indy and other lanes can prove work through receipts rather than narrative
- unknown debt remains visible when evidence is missing

## 10. Open Questions

- Whether `worker` should be a new physical table or a compatibility view over existing actor/runtime concepts.
- Whether `model_identifier` should be a physical table immediately or a view backed by existing model registry/loadout fields.
- Whether `work_order_attempt` should absorb any current receipt-like rows or remain a separate additive layer.

