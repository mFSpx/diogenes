#!/usr/bin/env python3
"""Status and work-order tool for INDY_READs BOOK_READER_LORA staging.

The tool is a local-only inventory auditor. It counts locally discoverable books
in BOOKS, measures staged LoRA assets (500-token chunks/cards/embeddings), and
emits missing-work orders for any book that is not ready for 3x target staging.
No network calls, no ANNAs archive fetches, no training/model execution.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from scripts import indy_reads

DEFAULT_BOOKS_ROOT = ROOT / "BOOKS"
DEFAULT_RUNTIME_ROOT = ROOT / "04_RUNTIME" / "BOOK_READER_LORA"
DEFAULT_STATUS_RECEIPT = ROOT / "05_OUTPUTS" / "runtime" / "indy_reads_book_inventory_status_latest.json"
TARGETS = ["talkie", "bonsai8b_q1", "bonsai8b_q2"]
TARGET_PATHS = {
    "talkie": "adapter_targets/talkie/adapter_manifest.json",
    "bonsai8b_q1": "adapter_targets/bonsai8b_q1/adapter_manifest.json",
    "bonsai8b_q2": "adapter_targets/bonsai8b_q2/adapter_manifest.json",
}


def now_z() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def load_jsonl_rows(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    parse_errors: list[str] = []
    for i, line in enumerate(read_lines(path), start=1):
        try:
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
            else:
                parse_errors.append(f"{rel(path)}: line {i}: non-object row")
        except json.JSONDecodeError as exc:
            parse_errors.append(f"{rel(path)}: line {i}: {type(exc).__name__}: {exc}")
    return rows, parse_errors


def count_files_by_ext(root: Path) -> dict[str, int]:
    by_ext: Counter[str] = Counter()
    if not root.exists():
        return {}
    for path in root.iterdir():
        if path.is_file():
            by_ext[path.suffix.lower()] += 1
    return dict(by_ext)


def count_nested_assets(paths: list[Path]) -> tuple[dict[str, dict[str, int]], dict[str, int]]:
    by_root: dict[str, dict[str, int]] = {}
    global_counts: Counter[str] = Counter()
    for p in paths:
        counts = count_files_by_ext(p)
        by_root[rel(p)] = counts
        for ext, n in counts.items():
            global_counts[ext] += n
    return by_root, dict(global_counts)


def count_jsonl_by_book(rows: list[dict[str, Any]], *, id_field: str = "book_id") -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        bid = str(row.get(id_field, "")).strip()
        if bid:
            counts[bid] += 1
    return dict(counts)

def build_chunk_book_map(chunk_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_book: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in chunk_rows:
        bid = str(row.get("book_id", "")).strip()
        if bid:
            by_book[bid].append(row)
    return by_book


def derive_embedding_book_counts(chunk_rows: list[dict[str, Any]], embedding_rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    # Embedding rows are produced aligned to chunk rows for deterministic local staging.
    # Keep mapping by positional row when possible, while still being resilient.
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"rows_seen": 0, "embedded": 0, "failed_or_missing": 0, "pending": 0})
    for i, chunk in enumerate(chunk_rows):
        bid = str(chunk.get("book_id", "")).strip()
        if not bid:
            continue
        data = counts[bid]
        data["rows_seen"] += 1
        if i >= len(embedding_rows):
            data["pending"] += 1
            continue
        status = str(embedding_rows[i].get("status", "")).upper()
        if status == "EMBEDDED":
            data["embedded"] += 1
        else:
            data["failed_or_missing"] += 1
    return counts


def target_manifest_status(runtime_root: Path, target: str) -> str:
    info = read_json(runtime_root / TARGET_PATHS[target])
    if not info:
        return "MANIFEST_MISSING"
    return str(info.get("status", "STATUS_UNKNOWN"))


def build_missing_orders_for_book(
    *,
    bid: str,
    chunk_count: int,
    card_count: int,
    train_count: int,
    target_status: dict[str, str],
    embedding_metrics: dict[str, int],
) -> list[dict[str, Any]]:
    orders: list[dict[str, Any]] = []

    if chunk_count == 0:
        orders.append(
            {
                "kind": "CHUNKS_500_TOKEN_MISSING",
                "severity": "HIGH",
                "action": "run book_reader_lora_stage --max-pages-per-book 1 --cards-per-page 5 --chunk-tokens 500",
                "notes": "No 500-token chunk rows found for this book in runtime chunks_500tok.jsonl",
            }
        )
    if card_count == 0:
        orders.append(
            {
                "kind": "CARDS_MISSING",
                "severity": "HIGH",
                "action": "run book_reader_lora_stage for this book and rebuild cards",
                "notes": "Reading-card JSONL rows are missing for this book.",
            }
        )
    if train_count == 0:
        orders.append(
            {
                "kind": "TRAINING_SPLIT_MISSING",
                "severity": "MEDIUM",
                "action": "re-run stage with stable cards per chunk >0",
                "notes": "No train-card rows were produced for this book.",
            }
        )

    emb_rows = int(embedding_metrics.get("rows_seen", 0))
    if emb_rows == 0 and chunk_count > 0:
        orders.append(
            {
                "kind": "EMBEDDINGS_MISSING",
                "severity": "LOW",
                "action": "embed chunks via --embed in book_reader_lora_stage or schedule embed worker",
                "notes": "No embedding rows mapped to this book.",
            }
        )

    for target, status in target_status.items():
        if status == "MANIFEST_MISSING":
            orders.append(
                {
                    "kind": "TARGET_MANIFEST_MISSING",
                    "severity": "HIGH",
                    "action": f"build adapter target manifest for {target}",
                    "notes": f"adapter_targets/{target}/adapter_manifest.json missing.",
                }
            )

    return orders


def book_inventory_status(
    *,
    books_root: Path,
    runtime_root: Path,
    chunk_file: Path,
    cards_train_file: Path,
    cards_val_file: Path,
    embeddings_file: Path,
    output_path: Path,
    work_orders_path: Path,
) -> dict[str, Any]:
    # Local inventory snapshot from INDY_READS.
    books = indy_reads.library()
    if books_root != indy_reads.BOOKS:
        books = [b for b in books if Path(b.path).parent == books_root]

    books_by_id = {b.id: b for b in books}
    by_ext = Counter()
    for b in books:
        by_ext[b.ext] += 1

    # Staging artifacts.
    chunk_rows, chunk_parse_errors = load_jsonl_rows(chunk_file)
    train_rows, train_parse_errors = load_jsonl_rows(cards_train_file)
    val_rows, val_parse_errors = load_jsonl_rows(cards_val_file)
    emb_rows, emb_parse_errors = load_jsonl_rows(embeddings_file)

    chunks_by_book = count_jsonl_by_book(chunk_rows, id_field="book_id")
    train_by_book = count_jsonl_by_book(train_rows, id_field="book_id")
    val_by_book = count_jsonl_by_book(val_rows, id_field="book_id")
    emb_by_book = derive_embedding_book_counts(chunk_rows, emb_rows)

    target_status = {target: target_manifest_status(runtime_root, target) for target in TARGETS}

    book_summaries = []
    missing_orders: list[dict[str, Any]] = []
    ready_count = 0
    for b in books:
        chunk_count = int(chunks_by_book.get(b.id, 0))
        train_count = int(train_by_book.get(b.id, 0))
        val_count = int(val_by_book.get(b.id, 0))
        card_count = train_count + val_count
        embedding_metrics = emb_by_book.get(b.id, {"rows_seen": 0, "embedded": 0, "failed_or_missing": 0, "pending": 0})
        target_statuses = {
            target: {
                "status": target_status[target],
                "has_manifest": target_status[target] != "MANIFEST_MISSING",
            }
            for target in TARGETS
        }

        book_orders = build_missing_orders_for_book(
            bid=b.id,
            chunk_count=chunk_count,
            card_count=card_count,
            train_count=train_count,
            target_status=target_status,
            embedding_metrics=embedding_metrics,
        )
        for order in book_orders:
            missing_orders.append({"book_id": b.id, "book_name": b.name, **order})

        ready_for_targets = (
            chunk_count > 0
            and card_count > 0
            and all(ts["has_manifest"] for ts in target_statuses.values())
        )
        if ready_for_targets:
            ready_count += 1

        book_summaries.append(
            {
                "book_id": b.id,
                "book_name": b.name,
                "book_path": rel(b.path),
                "ext": b.ext,
                "size_bytes": int(b.size_bytes),
                "chunk_counts": {
                    "chunks": chunk_count,
                    "source": rel(chunk_file),
                },
                "card_counts": {
                    "train": train_count,
                    "val": val_count,
                    "total": card_count,
                },
                "embedding_counts": {
                    "rows_seen": int(embedding_metrics.get("rows_seen", 0)),
                    "embedded": int(embedding_metrics.get("embedded", 0)),
                    "failed_or_missing": int(embedding_metrics.get("failed_or_missing", 0)),
                    "pending": int(embedding_metrics.get("pending", 0)),
                },
                "eligible_for_3x_targets": ready_for_targets,
                "targets": [
                    {
                        "target": target,
                        "status": status["status"],
                        "has_manifest": status["has_manifest"],
                    }
                    for target, status in target_statuses.items()
                ],
                "missing_work_orders": book_orders,
            }
        )

    by_root, global_file_counts = count_nested_assets([
        ROOT / "03_VAULT",
        ROOT / "05_OUTPUTS/indy_reads",
        ROOT / "05_OUTPUTS/runpod/talkie_book_lora/talkie_book_lora_runpod_pack/book_lora",
        books_root,
        runtime_root,
    ])

    line_counts = {
        rel(cards_train_file): len(train_rows),
        rel(cards_val_file): len(val_rows),
        rel(chunk_file): len(chunk_rows),
        rel(embeddings_file): len(emb_rows),
        rel(runtime_root / "cards" / "reading_cards.train.jsonl"): len(train_rows),
        rel(runtime_root / "cards" / "reading_cards.val.jsonl"): len(val_rows),
        rel(runtime_root / "chunks" / "chunks_500tok.jsonl"): len(chunk_rows),
    }

    blockers = chunk_parse_errors + train_parse_errors + val_parse_errors + emb_parse_errors
    blockers.extend(
        [
            f"book_reader_lora_stage chunks file missing: {rel(chunk_file)}"
            if not chunk_file.exists()
            else "",
            f"book_reader_lora_stage train cards file missing: {rel(cards_train_file)}"
            if not cards_train_file.exists()
            else "",
            f"book_reader_lora_stage val cards file missing: {rel(cards_val_file)}"
            if not cards_val_file.exists()
            else "",
        ]
    )
    blockers = [b for b in blockers if b]

    all_work_orders_have_three_targets = all(len(s["targets"]) == 3 for s in book_summaries) and all(
        all(t["has_manifest"] for t in s["targets"]) for s in book_summaries
    )

    status = "PASS_READY" if not blockers and not missing_orders else "PASS_STAGED_NOT_TRAINED"
    staging_summary = {
        "actual_book_file_count": sum(1 for e in by_ext.elements() if e != ".md"),
        "context_pack_count": by_ext.get(".md", 0),
        "book_count": len(books),
        "chunk_tokens": 500,
        "targets": TARGETS,
        "work_order_books": len(book_summaries),
        "all_work_orders_have_three_targets": all_work_orders_have_three_targets,
        "line_counts": line_counts,
        "eligible_for_three_targets": ready_count,
        "note": "Books are eligible for 3x LoRA target training when all targets have manifests and chunks/cards exist.",
    }

    work_order_receipt = {
        "schema": "lucidota.book_reader_lora.work_order_manifest.v1",
        "scope": "missing_lora_tasks_from_inventory",
        "generated_at": now_z(),
        "receipts": [rel(runtime_root / "receipts" / "stage_receipt.json")],
        "global_chunk_source": rel(chunk_file),
        "books": [],
        "missing_work_orders": missing_orders,
        "summary": {
            "missing_work_order_count": len(missing_orders),
            "eligible_for_three_targets": ready_count,
            "books_scanned": len(book_summaries),
        },
    }
    for entry in book_summaries:
        missing_for_book = bool(entry["missing_work_orders"])
        work_order_receipt["books"].append(
            {
                "book_id": entry["book_id"],
                "book_name": entry["book_name"],
                "book_path": entry["book_path"],
                "chunk_counts": {
                    "chunks": entry["chunk_counts"]["chunks"],
                    "source": entry["chunk_counts"]["source"],
                },
                "card_counts": entry["card_counts"],
                "embedding_counts": entry["embedding_counts"],
                "targets": [
                    {
                        "target": t["target"],
                        "status": t["status"],
                    }
                    for t in entry["targets"]
                ],
                "missing_work_orders": entry["missing_work_orders"],
                "ready_for_three_targets": not missing_for_book,
            }
        )

    status_payload: dict[str, Any] = {
        "schema": "lucidota.indy_reads.book_inventory_status.v1",
        "generated_at": now_z(),
        "status": status,
        "training_status": "DATASET_READY; LoRA training not run in this receipt" if status == "PASS_READY" else "DATASET_INCOMPLETE",
        "book_like_file_counts_by_root": by_root,
        "book_like_file_counts_global": global_file_counts,
        "staged_book_lora": staging_summary,
        "books": book_summaries,
        "missing_work_orders": missing_orders,
        "inventory": {
            "book_count": len(books),
            "actual_book_file_count": staging_summary["actual_book_file_count"],
            "context_pack_count": staging_summary["context_pack_count"],
            "by_extension": dict(by_ext),
            "books_root": rel(books_root),
            "source_root_counts": by_ext,
        },
        "line_counts": line_counts,
        "next_actions": [
            "build missing 500-token chunks/cards for books with missing_work_orders",
            "run book_reader_lora_stage with --embed for incomplete embedding coverage",
            "generate missing target manifests so 3x target set is complete",
            "refresh work order receipt after remediation",
        ],
        "canonical_graph_writes_performed": False,
        "db_writes_performed": False,
        "model_calls_performed": False,
        "blockers": blockers,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(status_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    work_orders_path.parent.mkdir(parents=True, exist_ok=True)
    work_orders_path.write_text(json.dumps(work_order_receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return status_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Report INDY_READs book inventory and LoRA staging status.")
    parser.add_argument("--books-root", default=str(DEFAULT_BOOKS_ROOT))
    parser.add_argument("--runtime-root", default=str(DEFAULT_RUNTIME_ROOT))
    parser.add_argument("--output", default=str(DEFAULT_STATUS_RECEIPT))
    parser.add_argument("--work-orders-output", default=str(DEFAULT_RUNTIME_ROOT / "book_lora_work_orders.json"))
    parser.add_argument("--json", action="store_true", help="Print JSON receipt to stdout")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    books_root = Path(args.books_root)
    runtime_root = Path(args.runtime_root)
    chunk_file = runtime_root / "chunks" / "chunks_500tok.jsonl"
    cards_train_file = runtime_root / "cards" / "reading_cards.train.jsonl"
    cards_val_file = runtime_root / "cards" / "reading_cards.val.jsonl"
    embedding_file = runtime_root / "embeddings" / "chunk_embeddings.jsonl"

    payload = book_inventory_status(
        books_root=books_root,
        runtime_root=runtime_root,
        chunk_file=chunk_file,
        cards_train_file=cards_train_file,
        cards_val_file=cards_val_file,
        embeddings_file=embedding_file,
        output_path=Path(args.output),
        work_orders_path=Path(args.work_orders_output),
    )

    if args.json:
        print(json.dumps(payload, sort_keys=True))
    print("BOOK_INVENTORY_STATUS=" + payload["status"])
    print("INDY_READS_BOOK_INVENTORY_STATUS_PATH=" + rel(payload_path := Path(args.output)))
    return 0 if payload["status"].startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
