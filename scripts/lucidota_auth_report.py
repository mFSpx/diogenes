#!/usr/bin/env python3
"""Create a redacted local auth surface report from tracked auth inventory only."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import os
from pathlib import Path

import psycopg

ROOT = Path(__file__).resolve().parents[1]
AUTH_INVENTORY = ROOT / "02_RECORDS_OFFICE" / "AUTH_INVENTORY.md"
DEFAULT_OUTPUT = ROOT / "05_OUTPUTS" / "auth_surface_report.txt"

SECRET_VALUE_PATTERNS = [
    re.compile(r"AIza[0-9A-Za-z_\-]{20,}"),
    re.compile(r"ya29\.[0-9A-Za-z_\-]+"),
    re.compile(r"ghp_[0-9A-Za-z]{20,}"),
    re.compile(r"glpat-[0-9A-Za-z_\-]{20,}"),
    re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----"),
]


def assert_no_secret_values(text: str) -> None:
    hits = [pat.pattern for pat in SECRET_VALUE_PATTERNS if pat.search(text)]
    if hits:
        raise SystemExit(f"secret-like value pattern found; refusing output: {hits}")


def read_inventory() -> str:
    if AUTH_INVENTORY.exists():
        return AUTH_INVENTORY.read_text(encoding="utf-8")
    try:
        dsn = os.environ.get("LUCIDOTA_GO_STORAGE_DSN", "postgresql:///lucidota_storage")
        with psycopg.connect(dsn, connect_timeout=3) as conn:
            row = conn.execute(
                "SELECT excerpt FROM lucidota_indy.markdown_artifact WHERE original_path='02_RECORDS_OFFICE/AUTH_INVENTORY.md' LIMIT 1"
            ).fetchone()
        return row[0] if row else ""
    except psycopg.Error:
        return ""

def render(today: str) -> str:
    src = read_inventory()
    assert_no_secret_values(src)
    body = f"""# Auth Surface Report

Generated: {today}

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

- Secret-like value pattern scan over input inventory and generated report: ok.
- Operational status changed: none; report only.
"""
    assert_no_secret_values(body)
    return body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--date", default=None, help="YYYY-MM-DD generated date override")
    parser.add_argument("--check", action="store_true", help="validate generated content without writing")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    text = render(args.date or dt.date.today().isoformat())
    required = ["# Auth Surface Report", "## Known Auth Surface", "## Release/Auth Gaps To Keep Truthful"]
    missing = [item for item in required if item not in text]
    ok = not missing
    if args.json:
        print(json.dumps({"ok": ok, "output": str(args.output.relative_to(ROOT)), "missing": missing, "redaction": "secret-like scan passed"}, sort_keys=True))
        return 0 if ok else 1
    if args.check:
        if missing:
            raise SystemExit(f"missing required sections: {missing}")
        print("auth surface report check: ok")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(args.output.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
