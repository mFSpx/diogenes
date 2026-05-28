#!/usr/bin/env python3
"""Full-world reingestion manifest for LUCI/KRAMPUS/PONY/KORPUS.

Indexes files into Bitloops/River-compatible JSONL rows. It never deletes,
purges, mutates canonical graph tables, or calls models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS" / "bitloops"
DEFAULT_SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".cache",
    ".venv",
    "node_modules",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_obj(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def source_family(path: Path) -> str:
    text = rel(path).lower()
    if "ponyboy" in text or "/pony/" in text or text.startswith("pony"):
        return "ponyboy"
    if "korpus" in text or "korpus_krampii" in text:
        return "korpus_krampii"
    if "krampus" in text:
        return "krampuschewing"
    if "luci" in text or "lucidota" in text:
        return "lucidota_current"
    return "lucidota_current"


def lane_for(path: Path) -> str:
    text = rel(path).lower()
    suffix = path.suffix.lower()
    if "receipt" in text or "report" in text or suffix in {".json", ".jsonl", ".csv"}:
        return "FILE_ORGANIZATION"
    if "case" in text or "rickshaw" in text or "osint" in text:
        return "INVESTIGATIVE_WORK"
    if "prompt" in text or "instruction" in text or suffix in {".md", ".txt"}:
        return "PROMPTING"
    if suffix in {".py", ".sh", ".rs", ".sql", ".js", ".ts", ".html", ".css"}:
        return "DEV_WORK"
    return "FILE_ORGANIZATION"


def iter_files(roots: Iterable[Path], *, max_files: int = 0, include_hidden: bool = True, exclude: set[Path] | None = None) -> Iterable[Path]:
    seen: set[Path] = set()
    exclude = exclude or set()
    emitted = 0
    for root in roots:
        root = root if root.is_absolute() else ROOT / root
        if not root.exists():
            continue
        if root.is_file():
            candidates = [root]
        else:
            candidates = []
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if d not in DEFAULT_SKIP_DIRS and (include_hidden or not d.startswith("."))]
                base = Path(dirpath)
                for name in filenames:
                    if not include_hidden and name.startswith("."):
                        continue
                    candidates.append(base / name)
        for path in candidates:
            try:
                resolved = path.resolve()
            except Exception:
                resolved = path
            if resolved in exclude:
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            yield path
            emitted += 1
            if max_files and emitted >= max_files:
                return


def row_for(path: Path) -> dict[str, Any]:
    stat = path.stat()
    digest = sha256_file(path)
    relative = rel(path)
    family = source_family(path)
    row = {
        "schema": "lucidota.bitloops.full_reingest_manifest.row.v1",
        "row_id": "bfr_" + sha256_obj({"path": relative, "sha256": digest})[:24],
        "source_family": family,
        "source_path": relative,
        "source_sha256": digest,
        "source_sha256_status": "computed",
        "size_bytes": int(stat.st_size),
        "modified_time_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat().replace("+00:00", "Z"),
        "extension": path.suffix.lower(),
        "lane": lane_for(path),
        "truth_status": "training_candidate_only",
        "not_for_accepted_truth": True,
        "bitloops_context_ready": True,
        "river_training_ready": True,
        "canonical_graph_writes_performed": False,
    }
    return row


def build_manifest(
    *,
    roots: list[Path],
    output: Path | None = None,
    summary_output: Path | None = None,
    max_files: int = 0,
    include_hidden: bool = True,
) -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    out = output or OUT / f"bitloops_full_reingest_manifest_{stamp()}.jsonl"
    summary_path = summary_output or OUT / f"bitloops_full_reingest_manifest_summary_{stamp()}.json"
    if not out.is_absolute():
        out = ROOT / out
    if not summary_path.is_absolute():
        summary_path = ROOT / summary_path
    out.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    failures: list[dict[str, str]] = []
    indexed = 0
    with out.open("w", encoding="utf-8") as fh:
        excludes = {out.resolve(), summary_path.resolve()}
        for path in iter_files(roots, max_files=max_files, include_hidden=include_hidden, exclude=excludes):
            try:
                row = row_for(path)
            except Exception as exc:
                failures.append({"path": rel(path), "error": type(exc).__name__, "message": str(exc)[:240]})
                continue
            fh.write(json.dumps(row, sort_keys=True) + "\n")
            counts[row["source_family"]] += 1
            indexed += 1
    summary = {
        "schema": "lucidota.bitloops.full_reingest_manifest.summary.v1",
        "generated_at": now(),
        "status": "PASS" if indexed else "EMPTY",
        "roots": [rel(r) for r in roots],
        "manifest_path": str(out),
        "summary_path": str(summary_path),
        "files_indexed": indexed,
        "family_counts": dict(sorted(counts.items())),
        "failures": failures[:200],
        "failure_count": len(failures),
        "purged_case_count": 0,
        "destroyed_case_count": 0,
        "db_writes_performed": False,
        "model_calls_performed": False,
        "canonical_graph_writes_performed": False,
        "next_step": "feed manifest JSONL to scripts/bitloops_automation_loop.py --legacy-jsonl",
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def default_roots() -> list[Path]:
    return [p for p in [ROOT, ROOT / "KRAMPUSCHEWING", ROOT / "03_VAULT" / "korpus_krampii"] if p.exists()]


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a no-purge full-world Bitloops/River reingestion manifest.")
    ap.add_argument("--root", action="append", default=[])
    ap.add_argument("--output")
    ap.add_argument("--summary-output")
    ap.add_argument("--max-files", type=int, default=0)
    ap.add_argument("--no-hidden", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    roots = [Path(r) for r in args.root] if args.root else default_roots()
    summary = build_manifest(
        roots=roots,
        output=Path(args.output) if args.output else None,
        summary_output=Path(args.summary_output) if args.summary_output else None,
        max_files=args.max_files,
        include_hidden=not args.no_hidden,
    )
    print(f"MANIFEST_PATH={rel(summary['manifest_path'])}")
    print(f"SUMMARY_PATH={rel(summary['summary_path'])}")
    if args.json:
        print(json.dumps(summary, sort_keys=True))
    return 0 if summary["status"] in {"PASS", "EMPTY"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
