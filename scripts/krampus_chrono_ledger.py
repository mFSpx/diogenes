#!/usr/bin/env python3
"""KORPUS KRAMPII chrono-ledger builder.

Builds the Time Machine ledger:
  absolute_path, iso_timestamp, source, confidence_bps, size_bytes, suffix

Order is deterministic and content-aware:
  explicit frontmatter/date fields -> filename/folder date -> inline text date -> OS time.

No LLM calls.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "05_OUTPUTS" / "korpus_krampii"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from ALGOS.krampus_chrono import chrono_file_date  # type: ignore  # noqa: E402
from korpus_krampii import iter_files  # type: ignore  # noqa: E402


def jdump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)


def row_for(root: Path, path: Path) -> dict[str, Any]:
    chrono = chrono_file_date(path)
    try:
        st = path.stat()
        size = st.st_size
        mtime = dt.datetime.fromtimestamp(st.st_mtime, tz=dt.timezone.utc).isoformat().replace("+00:00", "Z")
    except OSError:
        size = 0
        mtime = ""
    return {
        "absolute_path": str(path.resolve()),
        "root": str(root.resolve()),
        "relative_path": str(path.resolve().relative_to(root.resolve())) if str(path.resolve()).startswith(str(root.resolve())) else path.name,
        "iso_timestamp": chrono["iso"],
        "timestamp_epoch": chrono["timestamp"].timestamp(),
        "source": chrono["source"],
        "confidence_bps": chrono["confidence_bps"],
        "raw": chrono.get("raw", ""),
        "candidate_count": chrono.get("candidate_count", 0),
        "size_bytes": size,
        "suffix": path.suffix.lower(),
        "os_mtime": mtime,
    }


def collect(args: argparse.Namespace) -> list[dict[str, Any]]:
    roots = [Path(p).expanduser().resolve() for p in args.roots]
    rows = [row_for(root, path) for root, path in iter_files(roots, args.include_hidden, args.no_default_excludes)]
    rows.sort(key=lambda r: (float(r["timestamp_epoch"]), r["absolute_path"]))
    if args.limit:
        rows = rows[: args.limit]
    return rows


def cmd_build(args: argparse.Namespace) -> dict[str, Any]:
    rows = collect(args)
    out_path = Path(args.out).expanduser() if args.out else OUTPUT_ROOT / f"{dt.datetime.now().strftime('%Y%m%dT%H%M%S')}.chrono_ledger.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "absolute_path", "root", "relative_path", "iso_timestamp", "timestamp_epoch",
        "source", "confidence_bps", "raw", "candidate_count", "size_bytes", "suffix", "os_mtime",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return {"ok": True, "rows": len(rows), "ledger": str(out_path), "first": rows[0] if rows else None, "last": rows[-1] if rows else None}


def cmd_paths0(args: argparse.Namespace) -> dict[str, Any]:
    rows = collect(args)
    data = b"\0".join(str(r["absolute_path"]).encode("utf-8", errors="surrogateescape") for r in rows)
    if data:
        os.write(sys.stdout.fileno(), data + b"\0")
    return {"ok": True, "rows": len(rows)}


def cmd_jsonl(args: argparse.Namespace) -> dict[str, Any]:
    rows = collect(args)
    for row in rows:
        print(jdump(row))
    return {"ok": True, "rows": len(rows)}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="krampus-chrono-ledger")
    ap.add_argument("--json", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, func in [("build", cmd_build), ("paths0", cmd_paths0), ("jsonl", cmd_jsonl)]:
        p = sub.add_parser(name)
        p.add_argument("roots", nargs="+")
        p.add_argument("--out", default="")
        p.add_argument("--limit", type=int, default=0)
        p.add_argument("--include-hidden", action="store_true")
        p.add_argument("--no-default-excludes", action="store_true")
        p.set_defaults(func=func)
    return ap


def main() -> int:
    args = build_parser().parse_args()
    result = args.func(args)
    if args.json and args.cmd != "paths0":
        print(jdump(result))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
