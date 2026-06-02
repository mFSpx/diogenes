# LUCI Convergence Mode Prompt

Save This Prompt, Pass on this Handoff:

## Canonical operator intent

Turn Claw into LUCI as the real front door.
LUCI must be launched from the terminal, accept typed commands, route them through fast/slow lanes, log the raw message verbatim, write receipts, and make the system feel like one coherent database-operated machine.

## Co-operator model

- Indy_READs is the greeter voice and active research operator.
- The human operator and Indy_READs are co-operators of the same system.
- Both can use the full suite of models, routers, workflows, and tools.
- Everything important is async, pubsubbed, tracked, and ledgered.

## Hard requirements

- LUCI is the canonical user-facing entrypoint.
- Claw is implementation history only.
- PromptFlow stays sidecar-only and must not block live operator input.
- Local model admission is strict/fail-closed.
- Provider/API lanes are separate from local resident model lanes.
- Ingestion must accept readable material only; archives, PDFs, emails, images, and binary sludge must be routed correctly or quarantined.
- Postgres, ontology, queues, telemetry, receipts, and deterministic work trees own the control plane.
- The visible surface should not leak leftover DBOS-era branding.

## Operator feel

When I type into LUCI, it should feel like:

1. my message was captured exactly,
2. the system decided how to route it,
3. Indy_READs and the operator stack are live,
4. the result is written to the ledger,
5. and the next action is obvious.

## Outcome language

The goal is not “more components.”  
The goal is one runnable product: LUCI.

