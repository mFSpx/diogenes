# Drive Map Status

Updated: 2026-05-13

## Rules

- This is a map/inventory pass, not an ingest pass.
- Do not print or commit secret values.
- Full sensitive inventory belongs under `03_VAULT/drive_map/`, not tracked docs.
- Drive files are not to be moved or copied unless explicitly authorized after target confirmation.

## Confirmed current targets

### PYPELINE algorithms

Canonical Drive archive:

- Folder: `NorthernStrike_FINAL_GAP_COPY_FAST_20260512T122155/archives`
- File: `C_PYPELINE.zip`
- Drive ID: `1yykK9Mru8Yk3FU1FZ5pYP5lE8eUVnok8`
- Source: `C:\PYPELINE`
- Source file count: `10,325`
- Source bytes: `2,608,530,663`
- Zip bytes: `2,610,663,939`
- SHA-256: `FA7A64E1138616D3565A52546F3A77674CD488C055AAD9B4C7CE7BBAF6702A75`
- Verified by `gap_copy_summary.csv`: yes

Also visible as a Drive folder:

- Folder: `PYPELINE`
- Drive ID: `1mxuyGIspG6zPFbLUcIFt3lF62rZk-92V`
- Contains: `.env`, `SELF_DOX_2026-05-01`, `legal_authority_system` through the folder listing.
- Note: folder listing appears sparse; the complete version is the verified zip archive plus manifest.

### math / intrinsics algorithms

- Folder: `math-intrinsics`
- Drive ID: `1uB8FG2jRYdGpH-u-G6yym11AuyUghfN6`
- Listed children: `constants`, `.github`, `test`.
- Treat as algorithm target candidate until source archive/manifest path is correlated.

### Credential inventory with redaction

Confirmed categories exist, but tracked docs intentionally omit artifact names and provider IDs.

- Environment/config archives: present; private metadata moved to ignored vault map.
- Local assistant/auth-state archive category: present; private metadata moved to ignored vault map.
- Environment folder category: present; private metadata moved to ignored vault map.
- Non-Google private archive note: present; private metadata moved to ignored vault map.

Private metadata location: `03_VAULT/drive_map/SENSITIVE_CREDENTIAL_INVENTORY_PRIVATE.md` (ignored).
No credential values are printed, committed, or ingested by default.

### Scrapers

Drive search confirms scraper inventory exists, including:

- `SCRAPER_GUIDE.md`
- `scraper_coverage.md`
- `SCRAPER_REPOINT_PLAN.md`
- `SCRAPER_FLEET_AUDIT.md`
- `SCRAPER_PLAN_20260314.md`
- many `scraper_run_*.log`

Likely code homes from manifests/WHEREIS:

- `C:\PYPELINE\legal_authority_system` includes robots-aware fetcher.
- `C:\PYPELINE\scripts\mtg_17lands_pull.py` referenced by `PROJ_17lands_mtg`.
- `C:\ARCHIVE\NORTHERN_STRIKE\archive\ai_exports\claude_projects\...\ssl-limpet\limpet.py` is a scraper/crawler candidate.

### Document template generation suite

Drive search confirms template folders/files, including many `templates` folders and:

- `PITCH_EMAIL_TEMPLATE.md`
- `SARAH_MACLEOD_AFFIDAVIT_TEMPLATE_20260405.md`
- `DEMAND_LETTER_TEMPLATE.md`
- `RR_DM_TEMPLATES_20260305.md`
- `template.md` duplicates
- `ISSUE_TEMPLATE`, `PULL_REQUEST_TEMPLATE`

WHEREIS confirms `PROJ_legal_authority_system` as a likely doc-generation suite:

- Source: `C:\PYPELINE\legal_authority_system\`
- Description: frozen-green legal-doc pipeline; Typer CLI; CiviX manual import; sharded CAS; section-aware citation chunker; robots-aware fetcher.

## Current top-level Drive map nuclei

Tracked docs keep only category-level nuclei. Full granular provider IDs and sensitive folder labels belong in ignored private vault maps.

- final-gap-copy category: present
- archive/report/loose-file categories: present
- credential/env categories: present, private metadata redacted
- PYPELINE category: present
- math-intrinsics category: present
- external memory/import categories: present

## Complete map gap

The Drive connector can list folders and fetch manifests, but it does not provide a single recursive export/pagination artifact. The complete granular map is being assembled from:

1. Top-level Drive listing.
2. Recursive folder listing for high-value folders.
3. Existing source manifests from verified archives (`*.source_manifest.csv`).
4. Existing index docs (`WHEREIS.md`, `gap_copy_summary.csv`, `NOT_COPIED_TO_GOOGLE.md`).

