# Emergency Revoke Runbook

Scope: local LUCIDOTA development credentials. No secret values are stored here.

1. Stop external writes and background workers.
2. Rotate provider-side tokens in the provider UI/CLI.
3. Remove local credential files from their owner-approved storage only after replacement is verified.
4. Update `lucidota_indy.auth_inventory` status to `blocked` or `available` with a redacted note.
5. Run `python scripts/lucidota_auth_report.py --json` and `python scripts/lucidota_release_checklist.py --json`.
6. Commit only redacted status changes, never credentials.
