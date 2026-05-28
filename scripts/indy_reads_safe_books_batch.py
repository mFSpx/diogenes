#!/usr/bin/env python3
"""Safe INDY_READs BOOKS batch wrapper: local extract -> <=500 chunks -> LoRA staging receipts.

This wrapper deliberately reuses the legacy local BOOKS extractors, then feeds
extracted UTF-8 text into scripts.indy_book_learning_pipeline. It performs no
DB writes, no model calls, no network calls, and no canonical graph writes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.indy_book_learning_pipeline import run_pipeline  # noqa: E402
from scripts.legacy.lucidota_indy_library_ingest import extract_text, iter_books, title_author_from_name  # noqa: E402

OUT = ROOT / "05_OUTPUTS" / "indy_reads" / "batches"
EXTRACTED = ROOT / "04_RUNTIME" / "indy_reads_extracted"


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: str | Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_extracted_text(path: Path, text: str, *, title: str) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    name = hashlib.sha256((title + "\n" + text[:4096]).encode("utf-8", errors="replace")).hexdigest()[:16]
    out = path / f"{name}.txt"
    out.write_text(text, encoding="utf-8")
    return out


def pipeline_args(*, source_file: Path, title: str, author: str, max_tokens: int, overlap_tokens: int) -> Namespace:
    return Namespace(
        source_file=str(source_file),
        source_url=None,
        annas_url=None,
        annas_query=None,
        title=title,
        author=author,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        timeout_sec=60.0,
        json=True,
    )


def run_batch(
    *,
    books_root: Path,
    limit: int = 1,
    max_tokens: int = 500,
    overlap_tokens: int = 25,
    out_dir: Path | None = None,
    extracted_root: Path | None = None,
) -> dict[str, Any]:
    if max_tokens > 500:
        raise ValueError("max_tokens must be <= 500")
    out_dir = out_dir or OUT
    extracted_root = extracted_root or EXTRACTED
    selected = iter_books(books_root)[: max(0, limit)]
    entries: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    for book in selected:
        title, author = title_author_from_name(book.name)
        try:
            text, method = extract_text(book)
            extracted_path = write_extracted_text(extracted_root, text, title=title)
            child = run_pipeline(
                pipeline_args(
                    source_file=extracted_path,
                    title=title,
                    author=author,
                    max_tokens=max_tokens,
                    overlap_tokens=overlap_tokens,
                )
            )
            entries.append(
                {
                    "source_book": rel(book),
                    "source_book_sha256": hashlib.sha256(book.read_bytes()).hexdigest(),
                    "title": title,
                    "author": author,
                    "extract_method": method,
                    "extracted_text_path": str(extracted_path),
                    "extracted_text_sha256": sha_text(text),
                    "child_status": child.get("status"),
                    "child_receipt_path": str(ROOT / child["receipt_path"]) if not Path(child["receipt_path"]).is_absolute() else child["receipt_path"],
                    "child_receipt_relpath": child.get("receipt_path"),
                    "chunks_written": child.get("chunks_written", 0),
                    "max_observed_chunk_tokens": child.get("max_observed_chunk_tokens", 0),
                    "ontology_hits": child.get("ontology_hits", []),
                    "jzload_count": len(child.get("jzloads", [])),
                    "lora_manifest": child.get("lora_manifest"),
                }
            )
        except Exception as exc:  # keep batch moving; receipt carries exact blocker
            blockers.append({"source_book": rel(book), "error": f"{type(exc).__name__}:{exc}"})
    status = "PASS" if entries and not blockers else ("PARTIAL" if entries else "BLOCKED")
    receipt = {
        "schema": "lucidota.indy_reads.safe_books_batch_receipt.v1",
        "generated_at": now(),
        "status": status,
        "books_root": rel(books_root),
        "books_considered": len(selected),
        "books_processed": len(entries),
        "max_tokens_per_chunk": max_tokens,
        "overlap_tokens": overlap_tokens,
        "entries": entries,
        "blockers": blockers,
        "network_calls_performed": False,
        "model_calls_performed": False,
        "canonical_graph_writes_performed": False,
        "db_writes_performed": False,
    }
    out_path = out_dir / f"indy_reads_safe_books_batch_{stamp()}.json"
    receipt["receipt_path"] = rel(out_path)
    write_json(out_path, receipt)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run safe local INDY_READs BOOKS ingestion through extractor + bounded learning pipeline.")
    p.add_argument("--books-root", default=str(ROOT / "BOOKS"))
    p.add_argument("--limit", type=int, default=1)
    p.add_argument("--max-tokens", type=int, default=500)
    p.add_argument("--overlap-tokens", type=int, default=25)
    p.add_argument("--json", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    receipt = run_batch(
        books_root=Path(args.books_root),
        limit=args.limit,
        max_tokens=args.max_tokens,
        overlap_tokens=args.overlap_tokens,
    )
    if args.json:
        print(json.dumps(receipt, sort_keys=True, default=str))
    print("REPORT_PATH=" + receipt["receipt_path"])
    print("INDY_READS_SAFE_BOOKS_BATCH=" + receipt["status"])
    return 0 if receipt["status"] in {"PASS", "PARTIAL"} else 4


if __name__ == "__main__":
    raise SystemExit(main())
