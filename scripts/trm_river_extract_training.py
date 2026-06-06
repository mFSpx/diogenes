#!/usr/bin/env python3
"""Extract River ML training candidates into TRM training pairs.

Finds corpus_river_rows JSONL files from 05_OUTPUTS/corpus_ingest/
and extracts 13 binary features + 3 bucket features -> lane classification pairs.

Output: 05_OUTPUTS/trm_training/river/train.jsonl + receipt
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "05_OUTPUTS" / "trm_training" / "river"
RECEIPT_DIR = PROJECT_ROOT / "05_OUTPUTS" / "trm_training" / "receipts"

# Known locations for river ML training data
RIVER_DATA_DIRS = [
    PROJECT_ROOT / "05_OUTPUTS" / "corpus_ingest",
]

BINARY_FEATURES = [
    "has_diff_terms",
    "has_error_terms",
    "has_fix_terms",
    "has_receipt_terms",
    "has_case_terms",
    "has_graph_terms",
    "has_prompt_terms",
    "has_success_terms",
    "has_failure_terms",
]

BUCKET_FEATURES = [
    "file_size_bucket",
    "extension",
    "path_depth",
]

LANE_MAP = {
    "DEV_WORK": "DEV_WORK",
    "INVESTIGATIVE_WORK": "INVESTIGATIVE_WORK",
    "FILE_ORGANIZATION": "FILE_ORGANIZATION",
    "PROMPTING": "PROMPTING",
    "PROMPT_NOTE": "PROMPTING",
}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def find_river_rows() -> list[Path]:
    """Find all corpus_river_rows JSONL files."""
    found: list[Path] = []
    for d in RIVER_DATA_DIRS:
        if d.exists():
            for f in sorted(d.glob("corpus_river_rows_*.jsonl")):
                found.append(f)
    return found


def features_to_text(features: dict[str, Any]) -> str:
    """Convert river features to compact text representation."""
    parts = []
    # Binary features
    bin_parts = []
    for feat in BINARY_FEATURES:
        val = features.get(feat, False)
        bin_parts.append(f"{feat}={int(val)}")
    parts.append(f"[Binary] {' '.join(bin_parts)}")

    # Bucket features
    buck_parts = []
    for feat in BUCKET_FEATURES:
        val = features.get(feat, "unknown")
        buck_parts.append(f"{feat}={val}")
    parts.append(f"[Buckets] {' '.join(buck_parts)}")

    # Extra features if present
    extra = ["large_file_class", "recommended_next_action"]
    extra_parts = []
    for feat in extra:
        val = features.get(feat)
        if val:
            extra_parts.append(f"{feat}={val}")
    if extra_parts:
        parts.append(f"[Extra] {' '.join(extra_parts)}")

    return "\n".join(parts)


def row_to_training_pair(row: dict[str, Any]) -> dict[str, Any] | None:
    """Convert a river ML candidate to a training pair."""
    features = row.get("features", {})
    lane = row.get("lane", "")
    normalized_lane = row.get("normalized_lane_guess", "")

    if not lane:
        return None

    mapped_lane = LANE_MAP.get(lane, lane)

    text = features_to_text(features)
    row_id = row.get("row_id", f"river_{_sha256_text(text)[:12]}")

    return {
        "id": row_id,
        "source": "river_ml",
        "text": text,
        "lane": mapped_lane,
        "lane_raw": lane,
        "normalized_guess": normalized_lane,
        "features": features,
        "source_path": row.get("source_path", ""),
        "source_sha256": row.get("source_sha256", ""),
    }


def iter_river_jsonl(paths: list[Path]) -> list[dict[str, Any]]:
    """Read all river rows from JSONL files, deduplicating by row_id."""
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            print(f"  WARNING: not found: {path}", file=sys.stderr)
            continue
        print(f"  Reading {path}...", file=sys.stderr)
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rid = row.get("row_id", "")
                if rid in seen:
                    continue
                seen.add(rid)
                rows.append(row)
    return rows


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Extract River ML training data for TRM"
    )
    ap.add_argument("--source", nargs="+", default=[],
                    help="Path(s) to corpus_river_rows JSONL files")
    ap.add_argument("--limit", type=int, default=0,
                    help="Max total rows to process (0 = all)")
    ap.add_argument("--dry-run", action="store_true", default=True,
                    help="Print discovery info without writing (default)")
    ap.add_argument("--execute", action="store_true",
                    help="Actually write training files")
    ap.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = ap.parse_args()

    now_utc = datetime.now(timezone.utc).isoformat()

    # Find or use provided sources
    if args.source:
        source_paths = [Path(s) for s in args.source]
    else:
        source_paths = find_river_rows()

    if not source_paths:
        print("ERROR: No river ML source files found.", file=sys.stderr)
        print("  Searched in:", file=sys.stderr)
        for d in RIVER_DATA_DIRS:
            print(f"    {d}/corpus_river_rows_*.jsonl", file=sys.stderr)
        sys.exit(1)

    print(f"TRM River Extraction — {len(source_paths)} source(s)")

    # Read
    raw_rows = iter_river_jsonl(source_paths)
    print(f"  Loaded {len(raw_rows)} unique rows")

    if not raw_rows:
        print("  ERROR: No rows loaded.", file=sys.stderr)
        sys.exit(1)

    if args.limit:
        raw_rows = raw_rows[:args.limit]
        print(f"  Limited to {len(raw_rows)} rows")

    # Convert
    training_pairs = []
    for row in raw_rows:
        pair = row_to_training_pair(row)
        if pair:
            training_pairs.append(pair)

    print(f"  Converted {len(training_pairs)} training pairs")

    # Stats
    lane_counts: Counter = Counter()
    for pair in training_pairs:
        lane_counts[pair["lane"]] += 1

    if args.dry_run and not args.execute:
        print(f"\n  DRY RUN — use --execute to write {len(training_pairs)} pairs")
        print(f"  Lane distribution: {dict(lane_counts)}")

        receipt = {
            "schema": "lucidota.trm.river_extraction_receipt.v1",
            "verdict": "DRY_RUN",
            "sources": [str(s) for s in source_paths],
            "total_rows_loaded": len(raw_rows),
            "training_pairs_prepared": len(training_pairs),
            "lane_distribution": dict(lane_counts),
            "binary_features": BINARY_FEATURES,
            "bucket_features": BUCKET_FEATURES,
            "created_at_utc": now_utc,
            "command": " ".join(sys.argv),
        }
        receipt_path = RECEIPT_DIR / "river_extract_receipt.json"
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = receipt_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        tmp.replace(receipt_path)
        print(f"  Receipt (dry-run): {receipt_path}")
        return

    # Write
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "train.jsonl"

    written = 0
    with open(output_path, "w") as f:
        for pair in training_pairs:
            f.write(json.dumps(pair, sort_keys=True) + "\n")
            written += 1

    print(f"  Wrote {written} training pairs to {output_path}")

    # Source file hashes
    source_hashes = {}
    for s in source_paths:
        source_hashes[str(s)] = _sha256_file(s)

    # Receipt
    receipt = {
        "schema": "lucidota.trm.river_extraction_receipt.v1",
        "verdict": "PASS" if written > 0 else "FAIL",
        "sources": [str(s) for s in source_paths],
        "source_sha256": source_hashes,
        "total_rows_loaded": len(raw_rows),
        "training_pairs_written": written,
        "lane_distribution": dict(lane_counts),
        "binary_features": BINARY_FEATURES,
        "bucket_features": BUCKET_FEATURES,
        "files_written": [str(output_path)],
        "created_at_utc": now_utc,
        "processed_at_utc": now_utc,
        "verified_at_utc": now_utc,
        "output_sha256": _sha256_file(output_path) if written > 0 else None,
        "command": " ".join(sys.argv),
    }
    receipt_path = RECEIPT_DIR / "river_extract_receipt.json"
    tmp = receipt_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    tmp.replace(receipt_path)

    print(f"\n  Receipt: {receipt_path}")
    print(f"  Done. {written} River ML training pairs extracted.")


if __name__ == "__main__":
    main()
