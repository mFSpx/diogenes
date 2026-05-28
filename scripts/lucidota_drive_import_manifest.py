#!/usr/bin/env python3
"""Build a Drive import manifest skeleton from tracked local records only.

No Google Drive connector, network, or secret material is used. The output is an
operator-facing intake checklist: it identifies known source nuclei and records
what evidence still must exist before any import is allowed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import re
import os
import sys
from pathlib import Path

import psycopg

ROOT = Path(__file__).resolve().parents[1]
DRIVE_MAP = ROOT / "02_RECORDS_OFFICE" / "DRIVE_MAP_STATUS.md"
DRIVE_TARGETS = ROOT / "02_RECORDS_OFFICE" / "DRIVE_TARGETS_ALGORITHMS_INTAKE.md"
DEFAULT_OUTPUT = ROOT / "05_OUTPUTS" / "drive_import_manifest_skeleton.txt"

SECRET_VALUE_PATTERNS = [
    re.compile(r"AIza[0-9A-Za-z_\-]{20,}"),
    re.compile(r"ya29\.[0-9A-Za-z_\-]+"),
    re.compile(r"ghp_[0-9A-Za-z]{20,}"),
    re.compile(r"glpat-[0-9A-Za-z_\-]{20,}"),
    re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----"),
]

TARGETS = [
    {
        "source": "PYPELINE canonical archive",
        "local_record": "02_RECORDS_OFFICE/DRIVE_MAP_STATUS.md",
        "evidence": "C_PYPELINE.zip, source_manifest/gap-copy summary, SHA-256 recorded",
        "status": "candidate; verified archive metadata exists locally, bytes not imported here",
        "next_gate": "operator selects exact archive; fetch to ignored vault/CAS; hash must match recorded SHA-256 before extract",
    },
    {
        "source": "PYPELINE Drive folder listing",
        "local_record": "02_RECORDS_OFFICE/DRIVE_MAP_STATUS.md",
        "evidence": "sparse folder listing with .env and legal_authority_system noted",
        "status": "map-only; not canonical for full import",
        "next_gate": "do not import as source of truth unless operator confirms folder over verified zip",
    },
    {
        "source": "math-intrinsics folder",
        "local_record": "02_RECORDS_OFFICE/DRIVE_MAP_STATUS.md",
        "evidence": "folder nucleus and children constants/.github/test recorded",
        "status": "candidate; source archive/manifest not correlated in tracked docs",
        "next_gate": "operator selects target; create private granular map; require hash/manifest before ingest",
    },
    {
        "source": "scraper inventory",
        "local_record": "02_RECORDS_OFFICE/DRIVE_MAP_STATUS.md",
        "evidence": "guide/coverage/repoint/fleet audit names and likely PYPELINE homes recorded",
        "status": "candidate category; no file bytes imported",
        "next_gate": "select a bounded scraper source path; preserve robots/policy notes; hash before execution",
    },
    {
        "source": "document-template generation suite",
        "local_record": "02_RECORDS_OFFICE/DRIVE_MAP_STATUS.md",
        "evidence": "template filenames and PROJ_legal_authority_system candidate recorded",
        "status": "candidate category; private granular details redacted",
        "next_gate": "select exact template suite; import read-only; scan for secrets/client data before use",
    },
    {
        "source": "credential/env categories",
        "local_record": "02_RECORDS_OFFICE/DRIVE_MAP_STATUS.md and 02_RECORDS_OFFICE/AUTH_INVENTORY.md",
        "evidence": "category existence only; private metadata intentionally redacted",
        "status": "quarantine only; never product-ingest by default",
        "next_gate": "ignored vault only, hash-only tracked evidence, explicit rotation/revoke plan before operational use",
    },
]


def read_records() -> str:
    chunks = []
    missing = []
    for path in (DRIVE_MAP, DRIVE_TARGETS):
        if path.exists():
            chunks.append(path.read_text(encoding="utf-8"))
        else:
            missing.append(str(path.relative_to(ROOT)))
    if missing:
        try:
            dsn = os.environ.get("LUCIDOTA_GO_STORAGE_DSN", "postgresql:///lucidota_storage")
            with psycopg.connect(dsn, connect_timeout=3) as conn:
                rows = conn.execute(
                    "SELECT original_path, excerpt FROM lucidota_indy.markdown_artifact WHERE original_path = ANY(%s)",
                    (missing,),
                ).fetchall()
            chunks.extend(f"# {path}\n{excerpt}" for path, excerpt in rows)
        except Exception as exc:
            print(f"warning: unable to read missing Drive records from Postgres: {exc}", file=sys.stderr)
    return "\n".join(chunks)


def assert_no_secret_values(text: str) -> None:
    hits = [pat.pattern for pat in SECRET_VALUE_PATTERNS if pat.search(text)]
    if hits:
        raise SystemExit(f"secret-like value pattern found; refusing output: {hits}")


def render(today: str) -> str:
    records = read_records()
    assert_no_secret_values(records)
    rows = "\n".join(
        f"| {t['source']} | {t['local_record']} | {t['evidence']} | {t['status']} | {t['next_gate']} |"
        for t in TARGETS
    )
    body = f"""# Drive Import Manifest Skeleton

Generated: {today}

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
{rows}

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
secret_scan: "pending|ok|fail"
license_policy_scan: "pending|ok|fail"
promotion_decision: "quarantine|cas_only|extract_readonly|product_candidate|reject"
notes: ""
```

## Check Result

- Secret-like value pattern scan over input records and generated output: ok.
- Import status: no bytes imported; skeleton only.
"""
    assert_no_secret_values(body)
    return body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--date", default=None, help="YYYY-MM-DD generated date override")
    parser.add_argument("--check", action="store_true", help="validate generated content without writing")
    args = parser.parse_args()

    today = args.date or dt.date.today().isoformat()
    text = render(today)
    if args.check:
        required = ["# Drive Import Manifest Skeleton", "## Candidate Import Rows", "## Minimal Future Import Record Shape"]
        missing = [item for item in required if item not in text]
        if missing:
            raise SystemExit(f"missing required sections: {missing}")
        print("drive import manifest skeleton check: ok")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(args.output.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
