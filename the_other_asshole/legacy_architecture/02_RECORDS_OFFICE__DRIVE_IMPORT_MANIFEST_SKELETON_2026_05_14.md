# Drive Import Manifest Skeleton

Generated: 2026-05-14

Scope: local tracked-record synthesis only. This file does **not** prove that Drive
content is current, complete, safe, or imported. It is a manifest scaffold for
future operator-selected imports.

## Non-Negotiable Gates

1. No Google Drive connector use during skeleton generation.
2. No secret values, OAuth tokens, passwords, app passwords, recovery codes, or
   private keys in tracked output.
3. Credential/env categories are quarantine-only and belong in ignored vault
   paths, not product ingestion.
4. Any future import must record: operator-selected target, source path/ID,
   retrieval time, byte count, SHA-256, local CAS/vault path, scanner result,
   and explicit promotion decision.
5. Existing local records are evidence of prior mapping only; they are not a
   substitute for fresh hash verification at import time.

## Candidate Import Rows

| Candidate | Local Evidence Record | Existing Evidence | Current Truthful Status | Required Next Gate |
|---|---|---|---|---|
| PYPELINE canonical archive | 02_RECORDS_OFFICE/DRIVE_MAP_STATUS.md | C_PYPELINE.zip, source_manifest/gap-copy summary, SHA-256 recorded | candidate; verified archive metadata exists locally, bytes not imported here | operator selects exact archive; fetch to ignored vault/CAS; hash must match recorded SHA-256 before extract |
| PYPELINE Drive folder listing | 02_RECORDS_OFFICE/DRIVE_MAP_STATUS.md | sparse folder listing with .env and legal_authority_system noted | map-only; not canonical for full import | do not import as source of truth unless operator confirms folder over verified zip |
| math-intrinsics folder | 02_RECORDS_OFFICE/DRIVE_MAP_STATUS.md | folder nucleus and children constants/.github/test recorded | candidate; source archive/manifest not correlated in tracked docs | operator selects target; create private granular map; require hash/manifest before ingest |
| scraper inventory | 02_RECORDS_OFFICE/DRIVE_MAP_STATUS.md | guide/coverage/repoint/fleet audit names and likely PYPELINE homes recorded | candidate category; no file bytes imported | select a bounded scraper source path; preserve robots/policy notes; hash before execution |
| document-template generation suite | 02_RECORDS_OFFICE/DRIVE_MAP_STATUS.md | template filenames and PROJ_legal_authority_system candidate recorded | candidate category; private granular details redacted | select exact template suite; import read-only; scan for secrets/client data before use |
| credential/env categories | 02_RECORDS_OFFICE/DRIVE_MAP_STATUS.md and 02_RECORDS_OFFICE/AUTH_INVENTORY.md | category existence only; private metadata intentionally redacted | quarantine only; never product-ingest by default | ignored vault only, hash-only tracked evidence, explicit rotation/revoke plan before operational use |

## Minimal Future Import Record Shape

```yaml
source_label: ""
operator_request: ""
source_kind: drive_file_or_folder_or_archive
source_identifier: "redact_if_sensitive"
retrieved_at_utc: ""
retrieved_by: "manual/operator-approved"
local_staging_path: "03_VAULT/... (ignored)"
sha256: ""
byte_count: null
manifest_path: ""
secret_scan: "pending|pass|fail"
license_policy_scan: "pending|pass|fail"
promotion_decision: "quarantine|cas_only|extract_readonly|product_candidate|reject"
notes: ""
```

## Check Result

- Secret-like value pattern scan over input records and generated output: pass.
- Import status: no bytes imported; skeleton only.
