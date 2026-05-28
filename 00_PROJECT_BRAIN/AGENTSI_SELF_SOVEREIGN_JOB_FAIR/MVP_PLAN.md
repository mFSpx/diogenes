# agentSI MVP Plan

## MVP definition

agentSI is MVP-ready when it can:

1. create an agent profile from only the human persona seed;
2. tolerate a skipped 100-question interview;
3. list job booths;
4. match the agent to job booths for a current task;
5. write a job fair board artifact;
6. append growth journal entries;
7. absorb Glow Watch findings as candidate lessons;
8. emit LUCIDOTA workflow events;
8. stay local/deterministic and not disturb live ingestion.

## Done in this pass

- Created project folder under `00_PROJECT_BRAIN/AGENTSI_SELF_SOVEREIGN_JOB_FAIR/`.
- Added persona seed schema.
- Added 100-question fantasy interview.
- Added 17 job booths.
- Added `scripts/agentsi.py` CLI.
- Added `absorb-glow` bridge from Indy Polycareer Glow Watch into candidate growth lessons.
- Added runtime profile/output paths.
- Added operational self-sovereignty charter.
- Added architecture and README docs.

## Next useful upgrades

### V0.2 — Operator review gate

Add a command like:

```bash
scripts/agentsi.py promote --agent INDY_READs --entry <id> --decision approved|denied
```

This will separate candidate growth from canonical doctrine.

### V0.3 — Promotion UI for Glow lessons

The CLI can already absorb Glow Watch findings as candidate lessons. Next: add approval/denial UI so candidates become explicit doctrine only after review.

```text
glow finding → candidate lesson → operator review → skill/job booth update
```

### V0.4 — Mission packets

Generate booth-specific mission packets:

- Intake Clerk routing slip;
- Evidence Vault custody checklist;
- Glow Hunter anomaly card;
- Legal Clerk issue memo;
- News Editor fact-check table.

### V0.5 — Multi-agent job fair

Allow multiple profiles to interview for the same mission and compare fit without overwriting one another.

## Exit criteria for MVP smoke

```bash
python3 -m json.tool 00_PROJECT_BRAIN/AGENTSI_SELF_SOVEREIGN_JOB_FAIR/JOB_BOOTHS.json
python3 -m json.tool 00_PROJECT_BRAIN/AGENTSI_SELF_SOVEREIGN_JOB_FAIR/PERSONA_SEED_SCHEMA.json
.venv/bin/python -m py_compile scripts/agentsi.py
.venv/bin/python scripts/agentsi.py --json status
.venv/bin/python scripts/agentsi.py --json mvp-demo
```
