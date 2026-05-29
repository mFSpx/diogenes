#!/usr/bin/env python3
"""Line-safe Chrono snapshot slicer with per-part header lineage."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "05_OUTPUTS" / "CHRONO_MASTER_SNAPSHOT_CURRENT.txt"
DEFAULT_OUT_DIR = ROOT / "05_OUTPUTS" / "CHRONO_PARTS"
DEFAULT_PART_BYTES = 5 * 1024 * 1024


@dataclass
class PartReceipt:
    index: int
    path: str
    bytes: int
    source_lines: int
    header_bytes: int
    sha256: str
    first_source_line: int | None
    last_source_line: int | None
    oversized_line: bool = False


def now_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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


def cleanup(out_dir: Path, prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob(f"{prefix}_*.txt"):
        old.unlink()
    manifest = out_dir / "manifest.json"
    if manifest.exists():
        manifest.unlink()


def header_block(source: Path) -> bytes:
    lines: list[bytes] = []
    with source.open("rb") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped.startswith(b"{") or stripped.startswith(b"["):
                break
            lines.append(line if line.endswith(b"\n") else line + b"\n")
            if not stripped and len(lines) >= 10:
                break
    if not lines:
        return b""
    return b"".join(lines).rstrip(b"\n") + b"\n"


def part_header(base_header: bytes, index: int, source: Path) -> bytes:
    meta = (
        f"\n[CHRONO_PART_METADATA]\n"
        f"part_index: {index}\n"
        f"source_path: {rel(source)}\n"
        f"split_policy: newline_only_no_json_object_byte_slice\n"
        f"generated_at_utc: {now_z()}\n\n"
    ).encode("utf-8")
    return base_header + meta


def slice_snapshot(source: Path, out_dir: Path, prefix: str, max_part_bytes: int, prepend_header: bool) -> dict:
    if not source.exists():
        raise FileNotFoundError(source)
    if max_part_bytes <= 0:
        raise ValueError("part bytes must be positive")
    cleanup(out_dir, prefix)
    source_sha = sha256_file(source)
    base_header = header_block(source) if prepend_header else b""
    receipts: list[PartReceipt] = []
    oversized: list[dict] = []

    part_index = 0
    fh = None
    part_path: Path | None = None
    part_hash = hashlib.sha256()
    part_bytes = 0
    source_lines = 0
    first_source_line: int | None = None
    last_source_line: int | None = None
    header_bytes = 0
    part_oversized = False

    def close_part() -> None:
        nonlocal fh
        if fh is None:
            return
        fh.close()
        receipts.append(PartReceipt(
            index=part_index,
            path=rel(part_path),
            bytes=part_bytes,
            source_lines=source_lines,
            header_bytes=header_bytes,
            sha256=part_hash.hexdigest(),
            first_source_line=first_source_line,
            last_source_line=last_source_line,
            oversized_line=part_oversized,
        ))
        fh = None

    def open_part() -> None:
        nonlocal part_index, fh, part_path, part_hash, part_bytes, source_lines, first_source_line, last_source_line, header_bytes, part_oversized
        close_part()
        part_index += 1
        part_path = out_dir / f"{prefix}_{part_index:02d}.txt"
        fh = part_path.open("wb")
        part_hash = hashlib.sha256()
        part_bytes = 0
        source_lines = 0
        first_source_line = None
        last_source_line = None
        part_oversized = False
        hdr = part_header(base_header, part_index, source) if prepend_header else b""
        header_bytes = len(hdr)
        if hdr:
            fh.write(hdr)
            part_hash.update(hdr)
            part_bytes += len(hdr)

    open_part()
    total_source_lines = 0
    with source.open("rb") as src:
        for line_no, line in enumerate(src, 1):
            if not line.endswith(b"\n"):
                line += b"\n"
            line_len = len(line)
            if source_lines > 0 and part_bytes + line_len > max_part_bytes:
                open_part()
            if line_len + header_bytes > max_part_bytes:
                part_oversized = True
                oversized.append({"source_line": line_no, "line_bytes": line_len, "part_index": part_index})
            assert fh is not None
            fh.write(line)
            part_hash.update(line)
            part_bytes += line_len
            source_lines += 1
            total_source_lines += 1
            first_source_line = line_no if first_source_line is None else first_source_line
            last_source_line = line_no
    close_part()

    manifest = {
        "schema": "lucidota.snapshot_slicer.v1",
        "generated_at": now_z(),
        "source_path": rel(source),
        "source_bytes": source.stat().st_size,
        "source_sha256": source_sha,
        "out_dir": rel(out_dir),
        "prefix": prefix,
        "target_part_bytes": max_part_bytes,
        "prepend_header_to_each_part": prepend_header,
        "newline_safe": True,
        "json_object_lineage_preserved": True,
        "byte_slice_used": False,
        "parts_count": len(receipts),
        "total_source_lines": total_source_lines,
        "oversized_lines": oversized,
        "parts": [asdict(r) for r in receipts],
    }
    mpath = out_dir / "manifest.json"
    manifest["manifest_path"] = rel(mpath)
    mpath.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--prefix", default="CHRONO_PART")
    ap.add_argument("--part-bytes", type=int, default=int(os.environ.get("LUCIDOTA_CHRONO_PART_BYTES", DEFAULT_PART_BYTES)))
    ap.add_argument("--no-prepend-header", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    manifest = slice_snapshot(args.source, args.out_dir, args.prefix, args.part_bytes, not args.no_prepend_header)
    if args.json:
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
    else:
        print("SNAPSHOT_SLICER=PASS")
        print(f"PARTS={manifest['parts_count']}")
        print(f"MANIFEST={manifest['manifest_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
