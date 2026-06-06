#!/usr/bin/env python3
"""Extract MTG 17lands draft data into TRM training pairs.

Searches for 17lands Parquet files on disk (AFR, MID, VOW, WOE draft data).
If Parquet files are found, extracts draft sequences as:
  Pack N, Pick M -> card choices + pool state -> selected card -> outcome

If no Parquet files are found, this script operates as a stub documenting
where to obtain the data.

Output: 05_OUTPUTS/trm_training/mtg/train.jsonl + receipt
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "05_OUTPUTS" / "trm_training" / "mtg"
RECEIPT_DIR = PROJECT_ROOT / "05_OUTPUTS" / "trm_training" / "receipts"

# Search paths for 17lands Parquet files
POTENTIAL_PARQUET_PATHS = [
    Path("/home/mfspx/BOARDGAMES"),
    Path("/home/mfspx/BOARD_GAMES"),
    Path("/home/mfspx/LUCIDOTA/09_STORAGE"),
    Path("/home/mfspx/LUCIDOTA/03_VAULT"),
]

# Canonical download URLs for 17lands data
SEVENTEEN_LANDS_DOWNLOAD_URLS = {
    "afr": "https://www.17lands.com/data/draft/afr",
    "mid": "https://www.17lands.com/data/draft/mid",
    "vow": "https://www.17lands.com/data/draft/vow",
    "woe": "https://www.17lands.com/data/draft/woe",
}

SEVENTEEN_LANDS_DOCS_URL = (
    "https://www.17lands.com/data_documentation"
)

REQUIRED_SETS = ["AFR", "MID", "VOW", "WOE"]


def _sha256_file(path: Path) -> str:
    """Compute sha256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def find_parquet_files() -> list[Path]:
    """Search known paths for 17lands Parquet files."""
    found: list[Path] = []
    for base in POTENTIAL_PARQUET_PATHS:
        if not base.exists():
            continue
        # Look for files matching 17lands parquet naming patterns
        for pattern in [
            "**/*17lands*.parquet",
            "**/*draft*.parquet",
            "**/*.parquet",
        ]:
            candidates = sorted(base.glob(pattern))
            for c in candidates:
                name = c.name.lower()
                # Filter to likely 17lands/mtg related files
                if any(kw in name for kw in ["17lands", "draft", "mtg", "afr", "mid", "vow", "woe"]):
                    found.append(c)
    return found


def extract_mtg_draft_row(row: dict[str, Any]) -> dict[str, Any] | None:
    """Convert a single draft pick row into a TRM training pair.

    Expected input format (17lands Parquet -> dict):
    {
        "draft_id": str,
        "pack_number": int,
        "pick_number": int,
        "card_name": str,
        "card_colors": str,
        "card_rarity": str,
        "pool": [card_names...],
        "picked": bool,
        "won_match": bool (outcome),  # optional
        ...
    }
    """
    draft_id = row.get("draft_id")
    pack = row.get("pack_number")
    pick = row.get("pick_number")
    card = row.get("card_name")
    picked = row.get("picked", False)
    pool = row.get("pool", [])
    colors = row.get("card_colors", "unknown")
    rarity = row.get("card_rarity", "unknown")
    outcome = row.get("won_match", None)

    if not draft_id or pack is None or pick is None or not card:
        return None

    # Build text representation
    pool_text = f"Pool: {', '.join(pool[:20])}" if pool else "Pool: (empty)"
    if pool and len(pool) > 20:
        pool_text += f" ... (+{len(pool)-20} more)"

    text = (
        f"Draft: {draft_id}\n"
        f"Pack {pack}, Pick {pick}\n"
        f"Colors: {colors} | Rarity: {rarity}\n"
        f"{pool_text}\n"
        f"Selected: {card}"
    )

    labels = {
        "card_selected": card if picked else None,
        "card_passed": card if not picked else None,
        "won_match": outcome,
    }

    row_id = f"mtg_{draft_id}_{pack}_{pick}"

    return {
        "id": row_id,
        "source": "mtg_17lands",
        "text": text,
        "labels": {k: v for k, v in labels.items() if v is not None},
        "draft_id": draft_id,
        "pack": pack,
        "pick": pick,
        "card": card,
        "picked": bool(picked),
        "pool_size": len(pool),
    }


def iter_parquet_rows(parquet_paths: list[Path], limit: int = 0) -> list[dict[str, Any]]:
    """Read Parquet files and yield draft pick rows."""
    rows: list[dict[str, Any]] = []
    for ppath in parquet_paths:
        if not ppath.exists():
            print(f"  WARNING: Path not found: {ppath}", file=sys.stderr)
            continue
        print(f"  Reading {ppath}...", file=sys.stderr)
        try:
            import pandas as pd
            df = pd.read_parquet(ppath)
            records = df.to_dict(orient="records")
            for rec in records:
                rows.append(rec)
                if limit and len(rows) >= limit:
                    return rows
        except ImportError:
            print("  ERROR: pandas not installed. Cannot read Parquet.", file=sys.stderr)
            continue
        except Exception as e:
            print(f"  ERROR reading Parquet: {e}", file=sys.stderr)
            continue
    return rows


def write_jsonl(rows: list[dict], path: Path) -> int:
    """Write training pairs to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(path, "w") as f:
        for row in rows:
            pair = extract_mtg_draft_row(row)
            if pair:
                f.write(json.dumps(pair, sort_keys=True) + "\n")
                written += 1
    return written


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Extract MTG 17lands draft data for TRM training"
    )
    ap.add_argument("--source", nargs="+", default=[],
                    help="Path(s) to 17lands Parquet files")
    ap.add_argument("--limit", type=int, default=0,
                    help="Max total rows to process (0 = all)")
    ap.add_argument("--dry-run", action="store_true", default=True,
                    help="Print discovery info without writing (default)")
    ap.add_argument("--execute", action="store_true",
                    help="Actually write training files")
    ap.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = ap.parse_args()

    now_utc = datetime.now(timezone.utc).isoformat()
    parquet_sources = [Path(s) for s in args.source] if args.source else []
    search_msg = ""

    if not parquet_sources:
        # Auto-discover
        found = find_parquet_files()
        if found:
            print(f"TRM MTG Extraction — found {len(found)} potential Parquet files:")
            for f in found:
                print(f"  {f} ({f.stat().st_size / 1e6:.1f} MB)")
            parquet_sources = found
        else:
            search_msg = (
                "NO 17LANDS PARQUET FILES FOUND ON DISK.\n\n"
                "To obtain MTG draft data:\n"
                "1. Download draft pick data from 17lands.com:\n"
            )
            for set_code, url in SEVENTEEN_LANDS_DOWNLOAD_URLS.items():
                search_msg += f"   - {set_code.upper()}: {url}\n"
            search_msg += (
                f"\n2. Documentation: {SEVENTEEN_LANDS_DOCS_URL}\n"
                "3. Each set produces ~2-3M draft picks as Parquet.\n"
                "4. Place files in /home/mfspx/BOARDGAMES/ or LUCIDOTA storage paths.\n"
            )
            print(f"TRM MTG Extraction — {search_msg}")

    if not parquet_sources:
        # Write stub receipt
        receipt_dir = RECEIPT_DIR
        receipt_dir.mkdir(parents=True, exist_ok=True)
        receipt = {
            "schema": "lucidota.trm.mtg_extraction_receipt.v1",
            "verdict": "SKIP_NO_DATA",
            "search_paths_checked": [str(p) for p in POTENTIAL_PARQUET_PATHS],
            "required_sets": REQUIRED_SETS,
            "download_urls": SEVENTEEN_LANDS_DOWNLOAD_URLS,
            "documentation_url": SEVENTEEN_LANDS_DOCS_URL,
            "download_instructions": search_msg,
            "created_at_utc": now_utc,
            "processed_at_utc": now_utc,
            "verified_at_utc": now_utc,
            "command": " ".join(sys.argv),
        }
        receipt_path = receipt_dir / "mtg_extract_receipt.json"
        tmp = receipt_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        tmp.replace(receipt_path)
        print(f"\n  Receipt (stub): {receipt_path}")
        print("  Done. No MTG data available — stub receipt written.")
        return

    print(f"\n  Reading {len(parquet_sources)} Parquet file(s)...")

    # Read raw rows
    raw_rows = iter_parquet_rows(parquet_sources, limit=args.limit)
    print(f"  Loaded {len(raw_rows)} raw rows")

    if not raw_rows:
        print("  ERROR: No rows loaded.", file=sys.stderr)
        sys.exit(1)

    # Convert to training pairs
    training_pairs = []
    for row in raw_rows:
        pair = extract_mtg_draft_row(row)
        if pair:
            training_pairs.append(pair)

    print(f"  Converted {len(training_pairs)} training pairs")

    if args.dry_run and not args.execute:
        print(f"\n  DRY RUN — use --execute to write {len(training_pairs)} pairs")
        # Still write a receipt documenting the discovery
        receipt = {
            "schema": "lucidota.trm.mtg_extraction_receipt.v1",
            "verdict": "DRY_RUN",
            "sources": [str(s) for s in parquet_sources],
            "total_rows_loaded": len(raw_rows),
            "training_pairs_prepared": len(training_pairs),
            "created_at_utc": now_utc,
            "command": " ".join(sys.argv),
        }
        receipt_path = RECEIPT_DIR / "mtg_extract_receipt.json"
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = receipt_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        tmp.replace(receipt_path)
        print(f"  Receipt (dry-run): {receipt_path}")
        return

    # Write
    out_dir = Path(args.output_dir)
    n_written = write_jsonl(training_pairs, out_dir / "train.jsonl")
    print(f"  Wrote {n_written} training pairs to {out_dir / 'train.jsonl'}")

    # Compute source file hashes
    source_hashes = {}
    for s in parquet_sources:
        source_hashes[str(s)] = _sha256_file(s)

    # Write receipt
    receipt = {
        "schema": "lucidota.trm.mtg_extraction_receipt.v1",
        "verdict": "PASS" if n_written > 0 else "FAIL",
        "sources": [str(s) for s in parquet_sources],
        "source_sha256": source_hashes,
        "total_rows_loaded": len(raw_rows),
        "training_pairs_written": n_written,
        "files_written": [str(out_dir / "train.jsonl")],
        "created_at_utc": now_utc,
        "processed_at_utc": now_utc,
        "verified_at_utc": now_utc,
        "output_sha256": _sha256_file(out_dir / "train.jsonl") if n_written > 0 else None,
        "command": " ".join(sys.argv),
    }
    receipt_path = RECEIPT_DIR / "mtg_extract_receipt.json"
    tmp = receipt_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    tmp.replace(receipt_path)

    print(f"\n  Receipt: {receipt_path}")
    print(f"  Done. {n_written} MTG training pairs extracted.")


if __name__ == "__main__":
    main()
