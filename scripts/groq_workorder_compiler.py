#!/usr/bin/env python3
"""Compile remaining work into Groq-safe batches."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "05_OUTPUTS" / "gaps"


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def latest_gap_workflow_path(source_dir: Path = DEFAULT_SOURCE) -> Path:
    candidates = sorted(source_dir.glob("gap_workflows_*.json"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"no gap workflow receipts in {source_dir}")
    return candidates[-1]


def compile_workorders(path: Path | str, batch_size: int = 8) -> dict[str, Any]:
    source = Path(path)
    if not source.is_absolute():
        source = ROOT / source
    data = json.loads(source.read_text(encoding="utf-8"))
    workflows = list(data.get("workflows") or [])
    batch_size = max(1, int(batch_size))
    batches = [workflows[i : i + batch_size] for i in range(0, len(workflows), batch_size)]
    return {
        "schema": "lucidota.groq.workorder_compiler.v1",
        "generated_at": now(),
        "source_path": rel(source),
        "workflow_count": len(workflows),
        "batch_size": batch_size,
        "batch_count": len(batches),
        "workflows": workflows,
        "batches": batches,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    source = Path(args.source) if args.source else latest_gap_workflow_path()
    report = compile_workorders(source, batch_size=args.batch_size)
    out_dir = ROOT / "05_OUTPUTS" / "goals"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"groq_workorder_compiler_{stamp()}.json"
    report["report_path"] = rel(path)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("REPORT_PATH=" + rel(path))
    print("GROQ_WORKORDER_COMPILER=PASS")
    if args.json:
        print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
