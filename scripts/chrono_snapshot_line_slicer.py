#!/usr/bin/env python3
"""Newline-safe slicer for the Chrono master snapshot.

The slicer is intentionally boring:
- read bytes from the source file,
- accumulate complete lines only,
- rotate output parts before a line would cross the target size,
- never split a JSONL object across files,
- write a manifest with byte counts and SHA256 receipts.

If a single source line is larger than the requested part size, that line is
written as an oversized part and recorded in the manifest.  This preserves JSON
lineage over strict byte ceilings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "05_OUTPUTS" / "CHRONO_MASTER_SNAPSHOT_CURRENT.txt"
DEFAULT_OUT_DIR = ROOT / "05_OUTPUTS" / "CHRONO_PARTS"
DEFAULT_PART_BYTES = 5 * 1024 * 1024


@dataclass
class PartReceipt:
    index: int
    path: str
    bytes: int
    lines: int
    sha256: str
    oversized_line: bool = False


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def cleanup_existing_parts(out_dir: Path, prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in out_dir.glob(f"{prefix}_*.txt"):
        path.unlink()
    manifest = out_dir / "manifest.json"
    if manifest.exists():
        manifest.unlink()


def open_part(out_dir: Path, prefix: str, index: int) -> tuple[Path, BinaryIO]:
    path = out_dir / f"{prefix}_{index:02d}.txt"
    return path, path.open("wb")


def slice_snapshot(source: Path, out_dir: Path, prefix: str, max_part_bytes: int) -> dict:
    if max_part_bytes <= 0:
        raise ValueError("--part-bytes must be positive")
    if not source.exists():
        raise FileNotFoundError(source)

    cleanup_existing_parts(out_dir, prefix)
    source_sha = sha256_file(source)
    source_bytes = source.stat().st_size

    part_index = 1
    part_path, part_f = open_part(out_dir, prefix, part_index)
    part_bytes = 0
    part_lines = 0
    part_sha = hashlib.sha256()
    part_oversized = False
    receipts: list[PartReceipt] = []
    total_lines = 0
    oversized_lines: list[dict] = []

    def finish_part() -> None:
        nonlocal part_f, part_path, part_bytes, part_lines, part_sha, part_oversized
        if part_f.closed:
            return
        part_f.close()
        if part_bytes == 0 and not receipts:
            return
        receipts.append(
            PartReceipt(
                index=len(receipts) + 1,
                path=rel(part_path),
                bytes=part_bytes,
                lines=part_lines,
                sha256=part_sha.hexdigest(),
                oversized_line=part_oversized,
            )
        )

    def start_next_part() -> None:
        nonlocal part_index, part_path, part_f, part_bytes, part_lines, part_sha, part_oversized
        finish_part()
        part_index += 1
        part_path, part_f = open_part(out_dir, prefix, part_index)
        part_bytes = 0
        part_lines = 0
        part_sha = hashlib.sha256()
        part_oversized = False

    with source.open("rb") as src:
        for line_number, line in enumerate(src, start=1):
            if not line.endswith(b"\n"):
                line += b"\n"
            line_len = len(line)
            if part_bytes > 0 and part_bytes + line_len > max_part_bytes:
                start_next_part()
            if line_len > max_part_bytes:
                part_oversized = True
                oversized_lines.append({"line_number": line_number, "bytes": line_len, "part_index": part_index})
            part_f.write(line)
            part_sha.update(line)
            part_bytes += line_len
            part_lines += 1
            total_lines += 1

    finish_part()

    manifest = {
        "schema": "lucidota.chrono_snapshot_line_slicer.v1",
        "generated_at": utc_now(),
        "source_path": rel(source),
        "source_bytes": source_bytes,
        "source_sha256": source_sha,
        "out_dir": rel(out_dir),
        "prefix": prefix,
        "target_part_bytes": max_part_bytes,
        "parts_count": len(receipts),
        "total_lines": total_lines,
        "total_part_bytes": sum(r.bytes for r in receipts),
        "newline_safe": True,
        "jsonl_lineage_preserved": True,
        "oversized_lines": oversized_lines,
        "parts": [asdict(r) for r in receipts],
    }
    manifest_path = out_dir / "manifest.json"
    manifest["manifest_path"] = rel(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Split CHRONO_MASTER_SNAPSHOT_CURRENT.txt on clean newlines.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--prefix", default="CHRONO_PART")
    parser.add_argument("--part-bytes", type=int, default=int(os.environ.get("LUCIDOTA_CHRONO_PART_BYTES", DEFAULT_PART_BYTES)))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    manifest = slice_snapshot(args.source, args.out_dir, args.prefix, args.part_bytes)
    if args.json:
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
    else:
        print(f"CHRONO_SLICER=PASS")
        print(f"SOURCE={manifest['source_path']}")
        print(f"PARTS={manifest['parts_count']}")
        print(f"TOTAL_LINES={manifest['total_lines']}")
        print(f"MANIFEST={manifest['manifest_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
