# LUCIDOTA Runtime Topology

```text
Operator
  -> Clawd CLI
      -> indy-brief / Indy_Reads runtime tables
      -> lucidota-survey / local CAS + Postgres survey schema
      -> diogenes-smoke / CKDOG1 gRPC bridge
  -> DBOS workflow plane
      -> lucidota_control.workflow_event
      -> event_outbox / Wake Bus
  -> Reflex stack
      -> Bytewax live cursor
      -> River scores
      -> Treelite route advisory
  -> Evidence stack
      -> Body_Capture HTTP/browser captures
      -> CAS manifest / GC
      -> AGE edges
```

## Databases

- `lucidota_state`: control plane, DBOS events, Indy_Reads runtime, reflex cursors.
- `lucidota_graph`: CAS manifest/evidence graph, Survey and Body_Capture evidence rows.

## Safety Boundaries

- No ambient Drive/Gmail/Calendar reads.
- Auth inventory stores status and secret references only, never raw secrets.
- Local-address URL access requires explicit option.
- Dashboard bars reflect audited checklist state, not aspirational 100% values.
