# Tri-Algo Conduit Design — Low-Power Signal Recovery

Date: 2026-05-14

## Purpose

Hardwire a Karpathy-simple ingress path for capture nodes:

```text
Monitor / standby -> Hoeffding statistical gate -> Serpentina self-righting recovery
```

This is resource-efficiency logic, not security theatre and not biological metaphor in runtime names.

## Components

- Passive monitor: `ALGOS/thanatosis.py` keeps non-promoted candidates dormant.
- Statistical filter: `ALGOS/hoeffding_tree.py` promotes only when signal score beats uncertainty.
- Stability flip: `ALGOS/serpentina_self_righting.py` scores dangerous payload shapes and returns the node to standby instead of crash-looping.
- Unified primitive: `ALGOS/tri_algo_conduit.py`.
- Durable decision trail: `06_SCHEMA/013_signal_ingress.sql`, table `lucidota_signal.ingress_decision`.

## Body Capture wire

`scripts/lucidota_body_capture.py` now runs the conduit after fetch and before CAS write:

- `burst`: write CAS + capture metadata as normal.
- `standby`: record ingress decision and skip heavy capture.
- `recover`: record ingress decision and return nonzero for dangerous payload shape unless explicitly bypassed.
- `--disable-signal-gate`: operator bypass for diagnosis.

## Why this matters

- Cuts junk capture pressure before CAS/graph writes.
- Produces auditable signal decisions instead of invisible heuristics.
- Keeps DBOS/Postgres truth intact: algorithms rank/gate/explain, not canonicalize.

## Next hardwire candidates

1. Wake Bus: rank outbox delivery by tri-algo admission score.
2. Hop Pivot: run conduit before browser/extractor escalation.
3. Big Board: add signal ingress counts and standby/burst/recover ratio.


## Ruthless audit fixes applied

- Wake Bus batching slop fixed: `lucidota_wake_bus.py` now claims/delivers a batch with `FOR UPDATE SKIP LOCKED` and one SQL CTE round-trip; `pg_notify` happens inside the CTE.
- Body Capture split-brain reduced: `lucidota_body_capture.py` writes `lucidota_body_capture.workflow_event_outbox` in the same graph transaction as capture/delta. Cross-DB state delivery is best-effort, but graph-local pending/failed delivery is durable.
- Survey obfuscation removed: `validate_source_url()` now uses explicit `ipaddress` properties instead of string-fragment getattr tricks.
