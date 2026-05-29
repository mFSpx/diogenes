# agentSI Current Status

Status: MVP scaffold installed and smoke-tested.

Smoke completed: 2026-05-15T20:16:51-07:00

Live ingestion impact: none intended; this subproject only adds docs, a CLI, DB registry rows, runtime profile JSON, output markdown, and workflow events.

Primary demo agent: `INDY_READs`.

Primary demo profile: `04_RUNTIME/agentsi/agents/indy-reads.json`.

Primary demo output: `05_OUTPUTS/agentsi/indy-reads.job_fair.md`.

Registered workflows:

- `agentsi-profile-init`
- `agentsi-job-match`
- `agentsi-job-fair`
- `agentsi-growth-journal`
- `agentsi-glow-absorb`
- `agentsi-mvp-demo`

Last demo result: profile init / job fair / growth journal all succeeded.

Glow bridge: `scripts/agentsi.py absorb-glow --dry-run` previews Indy Polycareer Glow Watch findings as candidate growth lessons.

Additional smoke: `absorb-glow --dry-run` returned candidate lessons without writing; `match` emitted a workflow event.
