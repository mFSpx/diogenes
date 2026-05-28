# CURRENT GOAL HANDOFF

"Save This Prompt, Pass on this Handoff:"

- Goal: RESOURCE_GOVERNED_CAPABILITY_BUILD
- Generated: `2026-05-28T07:22:35Z`
- Current step: 4/6
- Status: in_progress
- Objective: Execute capability factory + DIOGENES/system-become under hard resource governance; Codex steers, deterministic/local/Groq workers chew, every PID owned, learn from failures, avoid thrash/OOM, and back up safely.
- Completed: Implemented scripts/resource_governor.py, added 06_SCHEMA/122_resource_governor.sql, wired apply_lucidota_control_schema.sh, fixed spawn dry-run dispatch, and verified .venv/bin/python -m pytest tests/test_resource_governor.py -q => 5 passed.
- Next action: Apply resource-governor schema, run live preflight, adopt active Krampus PID into JSON+DB registry, and write cycle report/backup receipt.
- Resume command: `.venv/bin/python -m pytest tests/test_resource_governor.py -q && .venv/bin/python scripts/resource_governor.py preflight --json`

Technical Summary Review and Dev Notes: The collar now exists and has teeth; next pass puts it on the live chewing beast without spawning more.
