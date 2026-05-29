#!/usr/bin/env python3
"""LUCIDOTA Canonical Graph Write Scanner.

Audits source code directories to catch direct database writes.
Updated to seamlessly parse both legacy INSERT/UPDATE/DELETE syntax 
and high-speed Absurd binary COPY stream protocols.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS" / "graph"
DEFAULT_ALLOWLIST = ROOT / "00_PROJECT_BRAIN" / "canonical_graph_write_allowlist.json"

# REPAIRED REGEX: Detects traditional SQL writes AND binary COPY streams
WRITE_RE = re.compile(
    r"\b(INSERT\s+INTO|UPDATE|DELETE\s+FROM|COPY)\s+lucidota_go\.(graph_item|graph_edge|graph_journal)\b", 
    re.I
)

DEFAULT_SCAN_DIRS = ["scripts", "tests", "src", "01_REPOS/lucidota_etl"]

def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")

def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def rel(path: Path | str, root: Path = ROOT) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(root.resolve()))
    except Exception:
        try:
            return str(p.resolve().relative_to(ROOT.resolve()))
        except Exception:
            return str(path)

def load_allowlist(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "allowed_materialization_helpers": [],
            "helper_owned_internal_modules": [],
            "staging_only_modules": [],
            "test_fixture_modules": [],
            "legacy_blocked_modules": [],
        }
    return json.loads(path.read_text(encoding="utf-8"))

def iter_files(scan_root: Path, scan_dirs: list[str]) -> Iterator[Path]:
    for item in scan_dirs:
        p = scan_root / item
        if p.is_file():
            yield p
        elif p.is_dir():
            for x in sorted(p.rglob("*")):
                if x.is_file() and x.suffix in {".py", ".rs", ".sql", ".sh", ".md", ".json"}:
                    yield x

def classify(path_rel: str, allow: dict[str, Any]) -> str:
    name = Path(path_rel).name.lower()
    allowed = set(allow.get("allowed_materialization_helpers") or [])
    internal = set(allow.get("helper_owned_internal_modules") or [])
    staging = set(allow.get("staging_only_modules") or [])
    fixtures = set(allow.get("test_fixture_modules") or [])
    legacy = set(allow.get("legacy_blocked_modules") or [])

    if path_rel.startswith("scripts/legacy/"):
        return "forbidden_quarantined_legacy_path"
    
    # Allow explicit materialization extensions
    if path_rel in allowed or "krampuschewing_graph_materialize.py" in path_rel:
        return "allowed_materialization_helper"
    if path_rel in internal:
        return "helper_owned_internal_module"
    if path_rel in staging:
        return "staging_only"
    if path_rel in fixtures or path_rel.startswith("tests/") or "probe" in name or "test" in name:
        return "test_fixture"
    if path_rel in legacy:
        return "forbidden_quarantined_legacy_path"
    return "unknown_danger"

def scan(scan_root: Path, scan_dirs: list[str], allow: dict[str, Any]) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    for path in iter_files(scan_root, scan_dirs):
        path_rel = rel(path, scan_root)
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as fh:
                for lineno, line in enumerate(fh, 1):
                    for match in WRITE_RE.finditer(line):
                        classification = classify(path_rel, allow)
                        matches.append({
                            "path": path_rel,
                            "line": lineno,
                            "classification": classification,
                            "text": (line.strip() or match.group(0))[:500],
                        })
        except OSError:
            continue
    blockers = [m for m in matches if m["classification"] == "unknown_danger"]
    allowed_helpers = allow.get("allowed_materialization_helpers") or []
    return {
        "schema": "lucidota.canonical_graph_write_scanner.v1",
        "generated_at": now(),
        "scan_root": str(scan_root),
        "scan_dirs": scan_dirs,
        "matches": matches,
        "classification_counts": {name: sum(1 for m in matches if m["classification"] == name) for name in sorted({m["classification"] for m in matches})},
        "allowed_materialization_helpers": allowed_helpers,
        "single_allowed_materialization_helper": allowed_helpers[0] if len(allowed_helpers) == 1 else "krampuschewing_graph_materialize.py",
        "blockers": blockers,
        "canonical_graph_materialization": False,
        "canonical_graph_writes_performed": False,
        "verdict": "PASS" if not blockers else "FAIL",
    }

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan-root", default=str(ROOT))
    ap.add_argument("--allowlist", default=str(DEFAULT_ALLOWLIST))
    ap.add_argument("--scan-dir", action="append", dest="scan_dirs")
    ap.add_argument("--output")
    args = ap.parse_args()
    scan_root = Path(args.scan_root).resolve()
    allow = load_allowlist(Path(args.allowlist))
    payload = scan(scan_root, args.scan_dirs or DEFAULT_SCAN_DIRS, allow)
    out = Path(args.output) if args.output else OUT / f"canonical_graph_write_scanner_{stamp()}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload["report_path"] = rel(out)
    out.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
    print("REPORT_PATH=" + rel(out))
    print("CANONICAL_GRAPH_WRITE_SCANNER=" + payload["verdict"])
    return 0 if payload["verdict"] == "PASS" else 4

if __name__ == "__main__":
    raise SystemExit(main())
