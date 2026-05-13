# Auth Inventory

Do not store secrets, tokens, passwords, app passwords, recovery codes, or OAuth client secrets in this file.

## GitLab

- Account: `mfspx`
- Status: authenticated for `glab` API and git push.
- LUCIDOTA remote: `https://gitlab.com/mfspx/LUCIDOTA`

## Google

- Account visible to current Codex connectors: `maroonedpilot@gmail.com`
- Current connector access in this session:
  - Google Drive: available.
  - Google Contacts: available.
- Missing connector access in this session:
  - Gmail: not exposed as a callable tool.
  - Google Calendar: not exposed as a callable tool.

### Google Drive Boundary

Google Drive is not ambient project context. Do not browse, root around, inventory, search broadly, or fetch Drive material unless northern.strike explicitly asks for a specific file, folder, query, or retrieval target. Retrieve only what was requested and do not infer that Drive is current, clean, authoritative, or the right place to look.

## Operational Rule

Email/calendar assistant behavior is authorized as a goal, but not operational until a real Gmail/Calendar connector or local CLI integration exists. Drafting, reading, scheduling, and reminders require explicit tool support, typed contracts, and no hidden token storage in the repo.
