# Auth Surface Report

Generated: 2026-05-14

Scope: tracked local `AUTH_INVENTORY.md` synthesis only. No connector calls, no
network lookup, no token copying, and no credential validation were performed.

## Known Auth Surface

| Surface | Tracked Status | Operational Meaning | Gap |
|---|---|---|---|
| GitLab / `glab` | authenticated for API and git push; remote recorded | repo publishing works locally | release sanitization gate still pending |
| Google Drive connector | available in prior session inventory | targeted retrieval can be requested explicitly | not ambient context; import workflow smoke pending |
| Google Contacts connector | available in prior session inventory | contact reads/searches possible when explicitly needed | not part of product runtime |
| Gmail | not exposed as callable tool | email assistant not operational | needs connector or local CLI adapter |
| Google Calendar | not exposed as callable tool | scheduling/reminders not operational | needs connector or local CLI adapter |
| Local `.codex` auth | known but not copied into repo | local operator state only | hash/inventory policy pending; never commit secrets |
| Drive credential/env archives | category located, values redacted | quarantine-only evidence category | ignored vault handling, rotation status, revoke runbook pending |

## Release/Auth Gaps To Keep Truthful

- No tracked file should contain secret values or private auth artifacts.
- Email/calendar assistant behavior remains design intent until a real adapter is
  implemented and tested.
- Drive writes and broad Drive browsing remain disallowed without explicit target
  authorization.
- Credential imports, if ever requested, must be staged in ignored vault paths,
  scanned, hashed, and tracked by redacted evidence only.

## Check Result

- Secret-like value pattern scan over input inventory and generated report: pass.
- Operational status changed: none; report only.
