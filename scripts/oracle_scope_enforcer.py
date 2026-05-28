#!/usr/bin/env python3
"""Independent filesystem scope oracle for implementation slices."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS/oracle"

def stamp() -> str: return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
def now() -> str: return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
def rel(p: Path | str) -> str:
    try: return str(Path(p).resolve().relative_to(ROOT))
    except Exception: return str(p)
def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""): h.update(chunk)
    return h.hexdigest()
def write_report(name: str, payload: dict[str, Any]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True); p=OUT/f"oracle_scope_enforcer_{name}_{stamp()}.json"; payload.setdefault("generated_at", now()); payload["report_path"]=rel(p); p.write_text(json.dumps(payload, indent=2, sort_keys=False, default=str)); print(f"REPORT_PATH={rel(p)}"); return p


def snapshot(paths: list[str]) -> dict[str, Any]:
    out = {}
    for raw in paths:
        p = (ROOT / raw).resolve() if not Path(raw).is_absolute() else Path(raw)
        if p.exists() and p.is_file():
            st = p.stat()
            out[rel(p)] = {"exists": True, "size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": sha_file(p)}
        else:
            out[rel(p)] = {"exists": False}
    return out


def changed(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    return sorted([k for k in set(before) | set(after) if before.get(k) != after.get(k)])


def main() -> int:
    p = argparse.ArgumentParser(description="Run a command and fail if file changes exceed allowed scope")
    p.add_argument("--watch-file", action="append", required=True)
    p.add_argument("--allowed-file", action="append", default=[])
    p.add_argument("--command", required=True)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    before = snapshot(args.watch_file)
    proc = None
    if args.execute:
        proc = subprocess.run(args.command, cwd=ROOT, shell=True, text=True, capture_output=True, timeout=120)
    after = snapshot(args.watch_file)
    changed_files = changed(before, after)
    allowed = {rel(ROOT / f) if not Path(f).is_absolute() else rel(f) for f in args.allowed_file}
    violations = [f for f in changed_files if f not in allowed]
    ok = not violations and (proc is None or proc.returncode == 0)
    report = {"action": "scope_enforce", "execute_performed": bool(args.execute), "command": args.command, "command_returncode": proc.returncode if proc else None, "stdout_tail": proc.stdout[-1000:] if proc else "", "stderr_tail": proc.stderr[-1000:] if proc else "", "before": before, "after": after, "changed_files": changed_files, "allowed_files": sorted(allowed), "violations": violations, "pass": ok}
    write_report("execute" if args.execute else "dry_run", report)
    print("ORACLE_SCOPE=" + ("PASS" if ok else "FAIL"))
    return 0 if ok else 2

if __name__ == "__main__":
    raise SystemExit(main())
