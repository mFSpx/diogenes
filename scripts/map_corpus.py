#!/usr/bin/env python3
"""Deterministically map the remaining corpus into chewable slices."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "09_STORAGE" / "krampuschewing_unpacked"
DEFAULT_OUTDIR = ROOT / "05_OUTPUTS" / "goals"
DEFAULT_START_AFTER = "Luci.zip_1640788353/Luci/Lucidota/LOG/ARCHIVE/SESSION012_HANDOFF.md"

IGNORE_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    "05_OUTPUTS",
    "04_RUNTIME",
    "03_VAULT/cas",
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


def discover_files(root: Path, *, start_after: str = "") -> list[Path]:
    files: list[Path] = []
    cursor = start_after
    marker = f"{root.name}/"
    if marker in cursor:
        cursor = cursor.split(marker, 1)[1]
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        rel_parts = path.relative_to(root).parts
        if any(part in IGNORE_DIRS for part in rel_parts):
            continue
        if path.name.startswith("."):
            continue
        rel_path = path.relative_to(root).as_posix()
        if cursor and rel_path <= cursor:
            continue
        files.append(path)
    return files


def classify(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt", ".log", ".csv", ".json", ".jsonl", ".yaml", ".yml", ".xml", ".ics", ".html", ".htm", ".py"}:
        return "easy_text"
    if suffix in {".pdf"}:
        return "heavy_pdf"
    if suffix in {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".webp", ".svg"}:
        return "heavy_image"
    if suffix in {".mp4", ".mkv", ".mov", ".avi", ".webm"}:
        return "heavy_video"
    if suffix in {".parquet", ".sqlite3", ".db", ".mbox"}:
        return "heavy_database"
    if suffix in {".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar"}:
        return "heavy_archive"
    return "other"


def size_bin(size: int) -> str:
    if size < 100 * 1024:
        return "<100KB"
    if size < 1024 * 1024:
        return "100KB-1MB"
    if size < 10 * 1024 * 1024:
        return "1MB-10MB"
    if size < 100 * 1024 * 1024:
        return "10MB-100MB"
    return "100MB+"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--start-after", default=DEFAULT_START_AFTER)
    ap.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    root = Path(args.root)
    outdir = Path(args.outdir)
    files = discover_files(root, start_after=args.start_after)
    outdir.mkdir(parents=True, exist_ok=True)

    total_bytes = 0
    categories = defaultdict(lambda: {"count": 0, "total_bytes": 0, "extensions": Counter()})
    ext_counts = Counter()
    top_counts = Counter()
    bin_counts = Counter()
    largest: list[dict[str, Any]] = []

    for p in files:
        st = p.stat()
        total_bytes += st.st_size
        category = classify(p)
        categories[category]["count"] += 1
        categories[category]["total_bytes"] += st.st_size
        ext = p.suffix.lower() or "<noext>"
        ext_counts[ext] += 1
        categories[category]["extensions"][ext] += 1
        top_counts[p.relative_to(root).parts[0]] += 1
        bin_counts[size_bin(st.st_size)] += 1
        largest.append({"path": rel(p), "size_bytes": st.st_size, "category": category})

    largest.sort(key=lambda x: x["size_bytes"], reverse=True)

    manifest = {
        "schema": "lucidota.corpus_map.v1",
        "generated_at": now(),
        "root": rel(root),
        "start_after": args.start_after,
        "remaining_files": len(files),
        "remaining_bytes": total_bytes,
        "remaining_mb": round(total_bytes / 1024 / 1024, 2),
        "size_bins": dict(bin_counts),
        "top_level_counts": top_counts.most_common(),
        "extension_counts": ext_counts.most_common(),
        "categories": {
            name: {
                "count": data["count"],
                "total_bytes": data["total_bytes"],
                "extensions": data["extensions"].most_common(),
            }
            for name, data in categories.items()
        },
        "largest_files": largest[:100],
        "recommendations": {
            "easy_text_chunk_size": 200,
            "heavy_layout_chunk_size": 20,
            "huge_file_threshold_bytes": 100 * 1024 * 1024,
            "bypass_or_stage": [
                p["path"]
                for p in largest
                if p["size_bytes"] >= 100 * 1024 * 1024
            ][:50],
        },
    }

    path = outdir / f"corpus_map_{stamp()}.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(f"REPORT_PATH={rel(path)}")
    print("CORPUS_MAP=PASS")
    if args.json:
        print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
