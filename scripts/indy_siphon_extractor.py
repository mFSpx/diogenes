#!/usr/bin/env python3
"""
INDY_READS → MALKOVICH SIPHON bridge.

Takes Indy_READs book chunks (500-token cleaned text) and feeds them through
the ElasticOntologyCompressor. Every chunk produces a shape_vector that:
  - Labels the chunk by its geometric signature
  - Feeds the XGBoost router (which book/passage type is this?)
  - Flows into River ML for online drift detection
  - Flows into Bytewax for cross-book shape correlation

Usage:
  python3 scripts/indy_siphon_extractor.py --limit 500
  python3 scripts/indy_siphon_extractor.py --all  # all 1,128 chunks
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ELASTIC_BIN = ROOT / "01_REPOS" / "lucidota_resonance" / "target" / "release" / "lucidota_elastic_shape"
CHUNK_SOURCE = ROOT / "04_RUNTIME" / "BOOK_READER_LORA" / "chunks" / "chunks_500tok.jsonl"
WORK_ORDERS = ROOT / "04_RUNTIME" / "BOOK_READER_LORA" / "book_lora_work_orders.json"
OUTPUT_DIR = ROOT / "05_OUTPUTS" / "malkovich_siphon" / "indy_reads"
RECEIPT_DIR = OUTPUT_DIR / "receipts"


def now_z() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def text_to_signals(text: str, max_tokens: int = 128) -> list[tuple[str, float]]:
    """Convert text into token:intensity pairs for the compressor.

    Uses word-level tokenization with tf-based intensity. Only the top
    max_tokens by frequency are kept to avoid flooding the compressor.
    """
    words = text.lower().split()
    if not words:
        return [("EMPTY_CHUNK", 1.0)]

    word_counts: Counter[str] = Counter()
    for w in words:
        # Clean punctuation but keep meaningful tokens
        w = w.strip(".,;:!?\"'()[]{}*_#@$%^&-+=<>/\\|~`")
        if len(w) >= 3:  # skip very short tokens
            word_counts[w] += 1

    total = sum(word_counts.values()) or 1
    # Keep top N tokens as signals, intensity proportional to frequency
    signals = [(word, count / total) for word, count in word_counts.most_common(max_tokens)]
    return signals


def book_id_to_label(book_id: str) -> str:
    """Map book IDs to short readable labels for router training."""
    label_map = {
        "a_big_boy_did_it_and_ran_away": "fiction_thriller",
        "a_death_in_malta": "nonfiction_investigation",
        "blood_in_the_machine": "nonfiction_history_tech",
        "one_day_everyone": "nonfiction_political",
        "out_of_darkness": "nonfiction_corporate_power",
        "the_small_and_the_mighty": "nonfiction_biography",
    }
    for key, label in label_map.items():
        if key in book_id:
            return label
    return "unknown_genre"


def process_chunk(chunk: dict[str, Any], fixed_dims: int = 64) -> dict[str, Any] | None:
    """Run one chunk through the ElasticOntologyCompressor."""
    text = chunk.get("text", "")
    if not text or len(text) < 20:
        return None

    signals = text_to_signals(text)
    if not signals:
        return None

    artifact_uuid = chunk.get("chunk_ref", str(uuid.uuid4()))
    book_id = chunk.get("book_id", "unknown")
    book_label = book_id_to_label(book_id)

    try:
        receipt = json.loads(
            subprocess.run(
                [
                    str(ELASTIC_BIN),
                    "--artifact-uuid", artifact_uuid,
                    "--source", f"IndyReads:{book_label}",
                    "--min-dims", str(fixed_dims),
                    "--max-dims", str(fixed_dims),
                    "--entropy-hint", str(len(signals) / 256.0),
                    "--threshold", "0.05",
                ],
                capture_output=True, text=True, check=True,
            ).stdout
        )
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        return None

    return {
        "chunk_ref": chunk.get("chunk_ref", ""),
        "book_id": book_id,
        "book_name": chunk.get("book_name", "")[:80],
        "book_label": book_label,
        "token_count": chunk.get("token_count", 0),
        "text_preview": text[:120],
        "shape_vector": receipt["shape_vector"],
        "residual_vector": receipt["residual_vector"],
        "fidelity": receipt["fidelity"],
        "collision": receipt["collision"],
        "dimensions": receipt["dimensions"],
        "active_resonances": receipt["active_resonances"][:10],
        "source": f"IndyReads:{book_label}",
    }


def main():
    ap = argparse.ArgumentParser(description="Indy_READs → Malkovich Siphon bridge")
    ap.add_argument("--limit", type=int, default=0, help="Max chunks to process (0=all)")
    ap.add_argument("--all", action="store_true", default=True, help="Process all chunks")
    ap.add_argument("--fixed-dims", type=int, default=64)
    args = ap.parse_args()

    if not ELASTIC_BIN.exists():
        print(f"ERROR: {ELASTIC_BIN} not built. Run: cd 01_REPOS/lucidota_resonance && cargo build --release")
        sys.exit(1)

    if not CHUNK_SOURCE.exists():
        print(f"ERROR: {CHUNK_SOURCE} not found. Indy_READs may not have run yet.")
        sys.exit(1)

    print(f"INDY_READS SIPHON — {CHUNK_SOURCE}")
    print(f"  Elastic Shape binary: {ELASTIC_BIN}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RECEIPT_DIR.mkdir(parents=True, exist_ok=True)

    shape_out = OUTPUT_DIR / "indy_shape_vectors.jsonl"
    collision_out = OUTPUT_DIR / "indy_collisions.jsonl"

    total = 0
    collisions = 0
    fidelity_sum = 0.0
    book_labels: Counter[str] = Counter()
    label_fidelity: dict[str, list[float]] = {}

    with open(CHUNK_SOURCE) as f_in, open(shape_out, "w") as f_shape, open(collision_out, "w") as f_coll:
        for i, line in enumerate(f_in):
            if not line.strip():
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue

            result = process_chunk(chunk, fixed_dims=args.fixed_dims)
            if result is None:
                continue

            total += 1
            fidelity_sum += result["fidelity"]
            book_labels[result["book_label"]] += 1
            label_fidelity.setdefault(result["book_label"], []).append(result["fidelity"])

            f_shape.write(json.dumps(result, sort_keys=True) + "\n")

            if result["collision"]:
                collisions += 1
                f_coll.write(json.dumps({
                    "chunk_ref": result["chunk_ref"],
                    "book_label": result["book_label"],
                    "fidelity": result["fidelity"],
                    "text_preview": result["text_preview"],
                }, sort_keys=True) + "\n")

            if i % 100 == 0 and i > 0:
                print(f"  ... {i} chunks processed, {collisions} collisions")

            if args.limit and total >= args.limit:
                break

    # Receipt
    label_stats = {}
    for label, fids in label_fidelity.items():
        label_stats[label] = {
            "count": len(fids),
            "avg_fidelity": sum(fids) / len(fids),
            "min_fidelity": min(fids),
            "max_fidelity": max(fids),
        }

    receipt = {
        "schema": "lucidota.malkovich.indy_siphon_receipt.v1",
        "verdict": "PASS" if total > 0 else "FAIL",
        "source": str(CHUNK_SOURCE),
        "total_chunks_processed": total,
        "collisions": collisions,
        "collision_rate": collisions / total if total else 0,
        "avg_fidelity": fidelity_sum / total if total else 0,
        "book_label_distribution": dict(book_labels),
        "per_label_fidelity": label_stats,
        "shape_vectors_written": str(shape_out),
        "collisions_written": str(collision_out),
        "timestamp": now_z(),
    }

    receipt_path = RECEIPT_DIR / f"indy_siphon_{now_z().replace(':', '')}.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = receipt_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    tmp.replace(receipt_path)

    print(f"\n  INDY_READS SIPHON COMPLETE")
    print(f"  Chunks processed: {total}")
    print(f"  Collisions: {collisions} ({100*collisions/total:.1f}%)" if total else "  No data")
    print(f"  Avg fidelity: {fidelity_sum/total:.4f}" if total else "")
    for label, count in book_labels.most_common():
        info = label_stats.get(label, {})
        print(f"    {label}: {count} chunks, avg_fid={info.get('avg_fidelity', 0):.4f}")
    print(f"  Receipt: {receipt_path}")
    print(f"  Shape vectors: {shape_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
