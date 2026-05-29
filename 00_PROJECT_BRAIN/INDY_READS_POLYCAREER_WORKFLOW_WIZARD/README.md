# INDY_READs Polycareer Workflow Wizard

Dedicated subproject for turning INDY_READs from a page-locked reading companion into a lawful, evidence-first workflow router for a one-person investigative force.

Core thesis: **boring custody, weird insight, ruthless routing, beautiful reports.**

Start here:

1. `ARCHITECTURE.md` — full design document.
2. `WORKFLOW_CONTRACT.md` — the non-negotiable operating contract.
3. `ROLE_MODES.json` — machine-readable role/mode registry seed.
4. `GLOW_HUNTER_SEEDLIST.md` — anomaly/operator pattern library seed.

Status: architecture/design subproject. Runtime hook target: `indy-polycareer-workflow-wizard` in `lucidota_control.workflow_registry`.

## Runtime hook

Implemented hook:

```bash
scripts/lucidota_indy_polycareer.py --json watch-once --since-hours 24 --threshold 35
```

Outputs:

- `05_OUTPUTS/indy_polycareer/glow_watch_findings.jsonl`
- `05_OUTPUTS/indy_polycareer/glow_watch_latest.md`
- DBOS workflow events under `indy-polycareer-glow-watch`

This hook lets Indy watch CLAWD/agent/workflow artifacts for both correct workflow routing and emergent "glow" methods worth reviewing.

## Auto-start hook

`./claw` now quietly starts:

```bash
scripts/lucidota_start_indy_polycareer_watch.sh
```

PID/log:

- `04_RUNTIME/indy_polycareer_watch.pid`
- `04_RUNTIME/indy_polycareer_watch.log`

Environment knobs:

- `LUCIDOTA_INDY_POLYCAREER_INTERVAL` default `120`
- `LUCIDOTA_INDY_POLYCAREER_SINCE_HOURS` default `2`
- `LUCIDOTA_INDY_POLYCAREER_GLOW_THRESHOLD` default `35`
