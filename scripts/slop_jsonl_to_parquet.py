#!/usr/bin/env python3
"""Transcode large JSONL artifacts to machine-readable compact forms.

- Input: repo-wide *.jsonl (>= threshold, excluding cleanup outputs)
- Output:
  - 05_OUTPUTS/slop_cleanup/jsonl_to_parquet/.../<source>.parquet
  - 05_OUTPUTS/slop_cleanup/jsonl_to_parquet/.../<source>.jsonl.gz

Behavior:
- Never mutates or deletes source files.
- Skips outputs that already exist.
- Uses duckdb/Parquet conversion with JSON auto-detection.
"""

from __future__ import annotations

import argparse
import errno
import gzip
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb


ROOT = Path(__file__).resolve().parents[1]
DUCKDB_JSON_READ_OPTS = "records=true, union_by_name=true, ignore_errors=true, maximum_object_size=536870912"
DUCKDB_JSON_READ_VARIANTS = [
    DUCKDB_JSON_READ_OPTS,
    DUCKDB_JSON_READ_OPTS + ", maximum_depth=1",
    "records=auto, union_by_name=true, ignore_errors=true, maximum_object_size=536870912",
    "records=false, union_by_name=true, ignore_errors=true, maximum_object_size=536870912",
]


def utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def line_count(path: Path) -> int:
    n = 0
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for n, _ in enumerate(fh, start=1):
            pass
    return n


def gzip_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("rb") as fin, gzip.open(dst, "wb") as fout:
        shutil.copyfileobj(fin, fout, length=8 * 1024 * 1024)


def parquet_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for opts in DUCKDB_JSON_READ_VARIANTS:
        con = duckdb.connect(database=":memory:")
        try:
            con.execute(
                f"COPY (SELECT * FROM read_json_auto('{src.as_posix()}', {opts})) TO '{dst.as_posix()}' (FORMAT PARQUET, COMPRESSION 'snappy')"
            )
            return
        except Exception as exc:
            last_error = exc
            try:
                if dst.exists():
                    dst.unlink()
            except Exception:
                pass
        finally:
            con.close()
    if last_error is not None:
        raise last_error


def has_file(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def build_paths(src: Path, out_root: Path) -> tuple[Path, Path]:
    rel = src.relative_to(ROOT)
    base = out_root / rel
    return base.with_suffix(".parquet"), base.with_suffix(".jsonl.gz")


def process_one(src: Path, out_root: Path, force: bool, overwrite: bool = False) -> dict[str, Any]:
    parquet_path, gz_path = build_paths(src, out_root)
    src_size = src.stat().st_size
    out = {
        "path": str(src.relative_to(ROOT)),
        "size_bytes": src_size,
        "parquet_path": str(parquet_path.relative_to(ROOT)),
        "gzip_path": str(gz_path.relative_to(ROOT)),
        "parquet_status": "skip",
        "gzip_status": "skip",
        "error": "",
        "rows": 0,
        "size_before_mb": round(src_size / 1024 / 1024, 2),
        "size_after_mb": None,
        "size_after_gz_mb": None,
        "generated_at_utc": utcnow(),
    }

    try:
        if force or overwrite or not has_file(gz_path):
            gzip_copy(src, gz_path)
            out["gzip_status"] = "ok"
        if force or overwrite or not has_file(parquet_path):
            parquet_copy(src, parquet_path)
            out["parquet_status"] = "ok"

        try:
            out["rows"] = line_count(src)
        except Exception:
            out["rows"] = -1

        if has_file(parquet_path):
            out["size_after_mb"] = round(parquet_path.stat().st_size / 1024 / 1024, 2)
        if has_file(gz_path):
            out["size_after_gz_mb"] = round(gz_path.stat().st_size / 1024 / 1024, 2)
    except Exception as exc:
        out["error"] = str(exc)
        if not has_file(gz_path):
            out["gzip_status"] = "fail"
        if not has_file(parquet_path):
            out["parquet_status"] = "fail"
    return out


def collect_sources(root: Path, min_mb: float) -> list[Path]:
    threshold = int(min_mb * 1024 * 1024)
    out: list[Path] = []
    for p in root.rglob("*.jsonl"):
        if not p.is_file():
            continue
        if "05_OUTPUTS/slop_cleanup" in p.as_posix():
            continue
        if "/05_OUTPUTS/ahoy/" in p.as_posix():
            continue
        try:
            if p.stat().st_size >= threshold:
                out.append(p)
        except OSError:
            continue
    return sorted(out)


def acquire_lock(lock_path: Path) -> None:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            break
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
            try:
                contents = lock_path.read_text(encoding="utf-8", errors="ignore")
                pid = None
                for line in contents.splitlines():
                    if line.startswith("pid="):
                        pid = int(line.split("=", 1)[1].strip())
                        break
                if pid is not None:
                    os.kill(pid, 0)
                    raise SystemExit(f"lock already held: {lock_path.relative_to(ROOT)}")
            except ProcessLookupError:
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass
                continue
            except PermissionError:
                raise SystemExit(f"lock already held: {lock_path.relative_to(ROOT)}") from exc
            except Exception:
                raise SystemExit(f"lock already held: {lock_path.relative_to(ROOT)}") from exc
            raise SystemExit(f"lock already held: {lock_path.relative_to(ROOT)}") from exc
    else:
        raise SystemExit(f"lock already held: {lock_path.relative_to(ROOT)}")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(f"pid={os.getpid()}\nstarted_at={utcnow()}\n")


def release_lock(lock_path: Path) -> None:
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Transcode large JSONL artifacts for machine-readable storage.")
    parser.add_argument("--min-mb", type=float, default=100.0)
    parser.add_argument("--paths", nargs="*", default=None, help="Optional explicit JSONL paths to process instead of auto-discovery.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--output-root", default=str(ROOT / "05_OUTPUTS" / "slop_cleanup" / "jsonl_to_parquet"))
    args = parser.parse_args()

    out_root = Path(args.output_root)
    slop_dir = out_root.parent
    slop_dir.mkdir(parents=True, exist_ok=True)
    state_path = slop_dir / "jsonl_to_parquet_receipt.jsonl"
    lock_path = slop_dir / "jsonl_to_parquet.lock"
    acquire_lock(lock_path)

    try:
        if state_path.exists():
            prior = len(state_path.read_text(encoding="utf-8").strip().splitlines())
        else:
            prior = 0

        if args.paths:
            sources = [Path(p) if Path(p).is_absolute() else ROOT / Path(p) for p in args.paths]
        else:
            sources = collect_sources(ROOT / "05_OUTPUTS", args.min_mb)
        if args.max_files:
            sources = sources[: args.max_files]

        processed = 0
        success = 0

        for src in sources:
            print(f"PROCESSING {src.relative_to(ROOT)}", flush=True)
            result = process_one(src, out_root=out_root, force=args.force, overwrite=args.overwrite)
            processed += 1
            if result.get("error"):
                print(f"  ERROR {result['error'][:140]}", flush=True)
            else:
                success += 1
                print(
                    f"  rows={result['rows']} parquet={result['parquet_status']} "
                    f"gz={result['gzip_status']}"
                , flush=True)

            with state_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(result, ensure_ascii=False) + "\n")

        print(f"DONE processed={processed} success={success} prior_receipts={prior}", flush=True)
        return 0
    finally:
        release_lock(lock_path)


if __name__ == "__main__":
    raise SystemExit(main())
