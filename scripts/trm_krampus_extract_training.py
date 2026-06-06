#!/usr/bin/env python3
"""Extract Krampus ChatML training data into TRM training pairs.

Aggregates all gen_*.jsonl files (gen_codebase, gen_internal_docs, gen_ornament,
gen_style_perb, gen_mtg) and splits (train/val/test) from KRAMPUSCHEWING,
deduplicates by message content hash, and writes a unified training set.

Output: 05_OUTPUTS/trm_training/krampus/train.jsonl + receipt
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
OUTPUT_DIR = PROJECT_ROOT / "05_OUTPUTS" / "trm_training" / "krampus"
RECEIPT_DIR = PROJECT_ROOT / "05_OUTPUTS" / "trm_training" / "receipts"

# Known Krampus training data locations
KRAMPUS_TRAINING_DIR = (
    PROJECT_ROOT
    / "KRAMPUSCHEWING"
    / "Lucidota"
    / "Lucidota"
    / "PROJECTS"
    / "KRAMPUS_EXPRESS"
    / "runtime"
    / "training"
)

GEN_FILE_NAMES = [
    "gen_codebase.jsonl",
    "gen_internal_docs.jsonl",
    "gen_ornament.jsonl",
    "gen_style_perb.jsonl",
    "gen_mtg.jsonl",
]

SPLIT_FILE_NAMES = {
    "train": "splits/train.jsonl",
    "val": "splits/val.jsonl",
    "test": "splits/test.jsonl",
}

# Special tokens for conversation format
SPECIAL_TOKENS = {
    "system": "<|system|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _messages_to_text(messages: list[dict[str, str]]) -> str:
    """Convert ChatML messages list to text with special tokens."""
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        token = SPECIAL_TOKENS.get(role, f"<|{role}|>")
        parts.append(f"{token}\n{content}")
    parts.append("<|end|>")
    return "\n\n".join(parts)


def _row_to_chatml(row: dict[str, Any]) -> dict[str, Any] | None:
    """Convert a ChatML row to a standardized training pair."""
    messages = row.get("messages", [])
    if not messages:
        return None

    # Calculate content hash for dedup
    all_content = "".join(m.get("content", "") for m in messages)
    content_hash = _sha256_text(all_content)

    # Determine source type from content patterns
    text = _messages_to_text(messages)

    return {
        "id": row.get("id", f"krampus_{content_hash[:12]}"),
        "source": "krampus_chatml",
        "content_hash": content_hash,
        "text": text,
        "messages": messages,
        "n_messages": len(messages),
        "roles": [m.get("role", "unknown") for m in messages],
    }


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read all JSON objects from a JSONL file."""
    rows: list[dict[str, Any]] = []
    if not path.exists():
        print(f"  WARNING: not found: {path}", file=sys.stderr)
        return rows
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"  ERROR reading {path}: {e}", file=sys.stderr)
    return rows


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Extract Krampus ChatML training data for TRM"
    )
    ap.add_argument("--gen-dir", default=str(KRAMPUS_TRAINING_DIR),
                    help="Path to Krampus training directory")
    ap.add_argument("--dry-run", action="store_true", default=True,
                    help="Print discovery info without writing (default)")
    ap.add_argument("--execute", action="store_true",
                    help="Actually write training files")
    ap.add_argument("--output-dir", default=str(OUTPUT_DIR))
    ap.add_argument("--dedup", action="store_true", default=True,
                    help="Deduplicate by content hash (default: True)")
    args = ap.parse_args()

    now_utc = datetime.now(timezone.utc).isoformat()
    gen_dir = Path(args.gen_dir)

    if not gen_dir.exists():
        print(f"ERROR: Krampus training dir not found: {gen_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"TRM Krampus Extraction — source: {gen_dir}")

    # Load all gen_*.jsonl files
    all_raw: list[dict[str, Any]] = []
    source_files: list[Path] = []
    for fname in GEN_FILE_NAMES:
        fpath = gen_dir / fname
        if fpath.exists():
            rows = iter_jsonl(fpath)
            print(f"  {fname}: {len(rows)} rows")
            all_raw.extend(rows)
            source_files.append(fpath)
        else:
            print(f"  {fname}: NOT FOUND (skipping)")

    # Load splits
    split_data: dict[str, list[dict[str, Any]]] = {}
    for split_name, rel_path in SPLIT_FILE_NAMES.items():
        fpath = gen_dir / rel_path
        if fpath.exists():
            rows = iter_jsonl(fpath)
            print(f"  splits/{split_name}.jsonl: {len(rows)} rows")
            split_data[split_name] = rows
            all_raw.extend(rows)
            source_files.append(fpath)
        else:
            print(f"  splits/{split_name}.jsonl: NOT FOUND (skipping)")

    print(f"\n  Total raw rows (all sources): {len(all_raw)}")

    if not all_raw:
        print("  ERROR: No data found.", file=sys.stderr)
        sys.exit(1)

    # Convert to training pairs
    training_pairs = []
    for row in all_raw:
        pair = _row_to_chatml(row)
        if pair:
            training_pairs.append(pair)

    print(f"  Converted to {len(training_pairs)} training pairs")

    # Deduplicate by content hash
    if args.dedup:
        seen_hashes: set[str] = set()
        deduped = []
        for pair in training_pairs:
            ch = pair["content_hash"]
            if ch not in seen_hashes:
                seen_hashes.add(ch)
                deduped.append(pair)
        n_dupes = len(training_pairs) - len(deduped)
        training_pairs = deduped
        print(f"  Deduplicated: removed {n_dupes} duplicates, {len(training_pairs)} remaining")

    # Count stats
    role_counter: Counter = Counter()
    for pair in training_pairs:
        for role in pair["roles"]:
            role_counter[role] += 1

    message_count = sum(pair["n_messages"] for pair in training_pairs)
    avg_messages = message_count / len(training_pairs) if training_pairs else 0

    total_characters = sum(len(pair["text"]) for pair in training_pairs)
    avg_chars = total_characters / len(training_pairs) if training_pairs else 0

    if args.dry_run and not args.execute:
        print(f"\n  DRY RUN — use --execute to write {len(training_pairs)} pairs")
        print(f"  Stats: {message_count} total messages, {avg_messages:.1f} avg/pair")
        print(f"  Role distribution: {dict(role_counter)}")
        print(f"  Avg text length: {avg_chars:.0f} chars")

        # Write dry-run receipt
        receipt = {
            "schema": "lucidota.trm.krampus_extraction_receipt.v1",
            "verdict": "DRY_RUN",
            "sources": [str(s) for s in source_files],
            "total_raw_rows": len(all_raw),
            "training_pairs_prepared": len(training_pairs),
            "n_deduped": n_dupes if args.dedup else 0,
            "total_messages": message_count,
            "role_distribution": dict(role_counter),
            "avg_messages_per_pair": round(avg_messages, 1),
            "avg_chars_per_pair": round(avg_chars, 0),
            "created_at_utc": now_utc,
            "command": " ".join(sys.argv),
        }
        receipt_path = RECEIPT_DIR / "krampus_extract_receipt.json"
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = receipt_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        tmp.replace(receipt_path)
        print(f"  Receipt (dry-run): {receipt_path}")
        return

    # Write output
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
    for s in source_files:
        source_hashes[str(s)] = _sha256_file(s)

    # Write receipt
    receipt = {
        "schema": "lucidota.trm.krampus_extraction_receipt.v1",
        "verdict": "PASS" if written > 0 else "FAIL",
        "sources": [str(s) for s in source_files],
        "source_sha256": source_hashes,
        "total_raw_rows": len(all_raw),
        "training_pairs_written": written,
        "n_deduped": n_dupes if args.dedup else 0,
        "total_messages": message_count,
        "role_distribution": dict(role_counter),
        "avg_messages_per_pair": round(avg_messages, 1),
        "avg_chars_per_pair": round(avg_chars, 0),
        "files_written": [str(output_path)],
        "created_at_utc": now_utc,
        "processed_at_utc": now_utc,
        "verified_at_utc": now_utc,
        "output_sha256": _sha256_file(output_path) if written > 0 else None,
        "command": " ".join(sys.argv),
    }
    receipt_path = RECEIPT_DIR / "krampus_extract_receipt.json"
    tmp = receipt_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    tmp.replace(receipt_path)

    print(f"\n  Receipt: {receipt_path}")
    print(f"  Done. {written} Krampus training pairs extracted.")


if __name__ == "__main__":
    main()
