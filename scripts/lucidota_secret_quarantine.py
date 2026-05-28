#!/usr/bin/env python3
"""Plan and create redacted copies for secret-bearing Lucidota files.

Default is non-destructive: report files/rules only. With
--write-redacted-copies, originals are left in place and sanitized copies plus a
manifest are written under 04_RUNTIME/secret_quarantine/redacted/<timestamp>/.
Credential rotation and original-file deletion/quarantine stay human actions.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from lucidota_security_scan import ALLOWLIST, PATTERNS, placeholder, scan_repo, skipped  # type: ignore  # noqa: E402


def jdump(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)


def redact_text(rel: str, text: str) -> tuple[str, list[dict[str, object]]]:
    replacements: list[dict[str, object]] = []
    redacted = text
    for rule, pat in PATTERNS.items():
        if (rel, rule) in ALLOWLIST:
            continue

        def repl(match):
            secret = match.group(0)
            if placeholder(secret):
                return secret
            line = redacted.count("\n", 0, match.start()) + 1
            replacements.append({"line": line, "rule": rule, "bytes": len(secret.encode("utf-8", errors="ignore"))})
            return f"[REDACTED:{rule}]"

        redacted = pat.sub(repl, redacted)
    return redacted, replacements


def write_redacted_copies(files: list[str]) -> dict[str, object]:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_root = ROOT / "04_RUNTIME" / "secret_quarantine" / "redacted" / stamp
    manifest_path = out_root / "manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for rel in files:
        src = ROOT / rel
        if not src.is_file() or skipped(src):
            continue
        text = src.read_text(encoding="utf-8", errors="ignore")
        redacted, replacements = redact_text(rel, text)
        if not replacements:
            continue
        dest = out_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(redacted, encoding="utf-8")
        rows.append({"source": rel, "redacted_copy": dest.relative_to(ROOT).as_posix(), "replacement_count": len(replacements), "replacements": replacements[:200]})
    manifest_path.write_text("".join(jdump(row) + "\n" for row in rows), encoding="utf-8")
    return {"redacted_root": out_root.relative_to(ROOT).as_posix(), "manifest": manifest_path.relative_to(ROOT).as_posix(), "file_count": len(rows)}


def main() -> int:
    ap = argparse.ArgumentParser(prog="lucidota-secret-quarantine")
    ap.add_argument("--write-redacted-copies", action="store_true", help="write sanitized copies; never moves/deletes originals")
    ap.add_argument("--max-findings", type=int, default=0)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    findings = scan_repo(max_findings=args.max_findings or None)
    by_file: dict[str, list[dict]] = defaultdict(list)
    for f in findings:
        by_file[str(f["file"])].append(f)
    rule_counts: dict[str, int] = defaultdict(int)
    for f in findings:
        rule_counts[str(f["rule"])] += 1

    result: dict[str, object] = {
        "ok": not findings,
        "mode": "redacted_copy" if args.write_redacted_copies else "plan",
        "finding_count": len(findings),
        "file_count": len(by_file),
        "rules": dict(sorted(rule_counts.items())),
        "files": [{"file": rel, "finding_count": len(items), "rules": sorted({str(i["rule"]) for i in items})} for rel, items in sorted(by_file.items())[:200]],
    }
    if args.write_redacted_copies and findings:
        result["redaction"] = write_redacted_copies(sorted(by_file))

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(jdump(result))
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
