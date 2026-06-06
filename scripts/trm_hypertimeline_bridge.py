#!/usr/bin/env python3
"""
trm_hypertimeline_bridge.py  --  Training Hypertimeline Bridge.

Reads training receipts from 05_OUTPUTS/trm_training/receipts/ and
consolidates them into a hypertimeline JSON at 05_OUTPUTS/hypertimeline/.

ETL: Extract -> Verify hash -> Transform -> Validate schema ->
      Triple-timestamp -> Write receipt -> Stage output.

Schema: lucidota.trm.hypertimeline_bridge.v1
Mutation class: receipt_only
"""

import argparse
import glob
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_RECEIPT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "05_OUTPUTS",
    "trm_training",
    "receipts",
)
DEFAULT_OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "05_OUTPUTS",
    "hypertimeline",
    "training_timeline.json",
)
BRIDGE_RECEIPT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "05_OUTPUTS",
    "trm_training",
    "receipts",
)

REQUIRED_RECEIPT_FIELDS = [
    "source_file", "staging_file", "triple_hashes",
    "triple_timestamps", "row_count", "verdict",
]

HYPERTIMELINE_ENTRY_FIELDS = [
    "event_type", "event_ts", "source", "metrics",
    "receipt_hash", "triple_timestamps", "triple_hashes",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Consolidate training receipts into a hypertimeline."
    )
    parser.add_argument(
        "--input-dir",
        default=DEFAULT_RECEIPT_DIR,
        help="Receipt directory (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help="Hypertimeline output path (default: %(default)s)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually write output and receipts (dry-run by default)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Hashing helpers
# ---------------------------------------------------------------------------

def file_sha256(filepath: str) -> str:
    """Compute SHA256 hex digest of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def content_sha256(content: str) -> str:
    """Compute SHA256 hex digest of a string."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Receipt reader
# ---------------------------------------------------------------------------

def read_receipts(receipt_dir: str) -> list[dict]:
    """Read all receipt JSON files from directory."""
    pattern = os.path.join(receipt_dir, "*.json")
    receipt_files = sorted(glob.glob(pattern))
    receipts = []
    for rp in receipt_files:
        with open(rp, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"  [WARN] Skipping unparseable receipt: {rp}", file=sys.stderr)
                continue
        receipts.append((rp, data))
    return receipts


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_receipt(receipt: dict) -> tuple[bool, list[str]]:
    """Validate receipt has required fields."""
    errors = []
    for field in REQUIRED_RECEIPT_FIELDS:
        if field not in receipt:
            errors.append(f"Missing required field: {field}")
    if "triple_hashes" in receipt:
        for hf in ("source_hash", "staging_hash", "receipt_hash"):
            if hf not in receipt["triple_hashes"]:
                errors.append(f"Missing triple_hash.{hf}")
    if "triple_timestamps" in receipt:
        for tf in ("created_at", "processed_at", "verified_at"):
            if tf not in receipt["triple_timestamps"]:
                errors.append(f"Missing triple_timestamp.{tf}")
    return len(errors) == 0, errors


def validate_entry(entry: dict) -> tuple[bool, list[str]]:
    """Validate hypertimeline entry has required fields."""
    errors = []
    for field in HYPERTIMELINE_ENTRY_FIELDS:
        if field not in entry:
            errors.append(f"Missing entry field: {field}")
    return len(errors) == 0, errors


# ---------------------------------------------------------------------------
# Entry builder
# ---------------------------------------------------------------------------

def build_hypertimeline_entry(
    receipt_filepath: str,
    receipt: dict,
    input_hash: str,
    processed_at: str,
    verified_at: str,
) -> dict:
    """Build a hypertimeline entry from a receipt."""
    triple_ts = receipt.get("triple_timestamps", {})
    triple_h = receipt.get("triple_hashes", {})
    metrics = {
        "row_count": receipt.get("row_count", 0),
        "feature_count": receipt.get("feature_count", 0),
    }
    # Also grab feature_count from staging metadata if present
    if "feature_vector" in receipt:
        metrics["feature_count"] = len(receipt.get("feature_vector", {}))

    # Compute bridge receipt hash for this entry
    entry_content = json.dumps({
        "event_type": "training_data_extraction",
        "event_ts": triple_ts.get("created_at", ""),
        "source": os.path.basename(receipt_filepath),
        "metrics": metrics,
        "receipt_hash": triple_h.get("receipt_hash", ""),
        "triple_timestamps": {
            "created_at": triple_ts.get("created_at", ""),
            "processed_at": processed_at,
            "verified_at": verified_at,
        },
        "triple_hashes": {
            "input_hash": input_hash,
            "output_hash": "",  # filled after output is written
            "bridge_receipt_hash": "",
        },
    }, sort_keys=True)

    bridge_receipt_hash = content_sha256(entry_content)

    return {
        "event_type": "training_data_extraction",
        "event_ts": triple_ts.get("created_at", ""),
        "source": os.path.basename(receipt_filepath),
        "metrics": metrics,
        "receipt_hash": triple_h.get("receipt_hash", ""),
        "triple_timestamps": {
            "created_at": triple_ts.get("created_at", ""),
            "processed_at": processed_at,
            "verified_at": verified_at,
        },
        "triple_hashes": {
            "input_hash": input_hash,
            "output_hash": "",  # filled after file writing
            "bridge_receipt_hash": bridge_receipt_hash,
        },
    }


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------

def load_existing_entries(output_path: str) -> list[dict]:
    """Load existing hypertimeline entries, returning empty list if missing."""
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                # Also handle dict with "entries" key
                if isinstance(data, dict) and "entries" in data:
                    return data["entries"]
            except json.JSONDecodeError:
                pass
    return []


def dedup_entries(new_entries: list[dict], existing: list[dict]) -> list[dict]:
    """Merge new entries with existing, dedup by receipt_hash."""
    seen = set()
    merged = []

    for e in existing:
        rh = e.get("receipt_hash", "")
        if rh not in seen:
            seen.add(rh)
            merged.append(e)

    for e in new_entries:
        rh = e.get("receipt_hash", "")
        if rh not in seen:
            seen.add(rh)
            merged.append(e)

    return merged


# ---------------------------------------------------------------------------
# Bridge run receipt
# ---------------------------------------------------------------------------

def write_bridge_receipt(
    receipt_dir: str,
    input_dir: str,
    output_path: str,
    total_entries: int,
    new_entries_count: int,
    input_hashes: list[str],
    output_hash: str,
    bridge_receipt_hash: str,
    created_at: str,
    processed_at: str,
    verified_at: str,
    verdict: str,
    dry_run: bool,
) -> str:
    """Write a receipt for this bridge run. Returns receipt path."""
    os.makedirs(receipt_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    receipt_path = os.path.join(receipt_dir, f"hypertimeline_bridge_{timestamp}.json")

    receipt = {
        "schema": "lucidota.trm.hypertimeline_bridge_receipt.v1",
        "generated_by": "scripts/trm_hypertimeline_bridge.py",
        "input_dir": input_dir,
        "output_path": output_path,
        "triple_hashes": {
            "input_hashes": input_hashes,
            "output_hash": output_hash,
            "bridge_receipt_hash": bridge_receipt_hash,
        },
        "triple_timestamps": {
            "created_at": created_at,
            "processed_at": processed_at,
            "verified_at": verified_at,
        },
        "total_entries": total_entries,
        "new_entries": new_entries_count,
        "dry_run": dry_run,
        "verdict": verdict,
    }

    if not dry_run:
        with open(receipt_path, "w") as f:
            json.dump(receipt, f, indent=2)
    else:
        print(f"  [DRY-RUN] Would write bridge receipt to {receipt_path}")
        receipt_path = None

    return receipt_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    dry_run = not args.execute

    created_at = datetime.now(timezone.utc).isoformat()

    # =====================================================================
    # Step 1: Extract
    # =====================================================================
    print("ETL Step: Extract")
    print(f"  Reading receipts from: {args.input_dir}")
    if not os.path.isdir(args.input_dir):
        print(f"  [ERROR] Receipt directory not found: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    raw_receipts = read_receipts(args.input_dir)
    if not raw_receipts:
        print("  No receipts found. Nothing to do.")
        sys.exit(0)
    print(f"  Found {len(raw_receipts)} receipt file(s)")

    # =====================================================================
    # Step 2: Verify hash
    # =====================================================================
    print("ETL Step: Verify hash")
    verified_receipts = []
    input_hashes = []
    for rp, data in raw_receipts:
        actual_hash = file_sha256(rp)
        expected_hash = data.get("triple_hashes", {}).get("receipt_hash", "")
        if expected_hash and actual_hash != expected_hash:
            print(f"  [WARN] Hash mismatch for {os.path.basename(rp)}: "
                  f"expected={expected_hash}, actual={actual_hash}")
        else:
            print(f"  Hash OK: {os.path.basename(rp)} -> {actual_hash[:16]}...")
        input_hashes.append(actual_hash)
        verified_receipts.append((rp, data))
    print(f"  Verified {len(verified_receipts)} receipt(s)")

    # =====================================================================
    # Step 3: Transform
    # =====================================================================
    print("ETL Step: Transform")
    processed_at = datetime.now(timezone.utc).isoformat()
    if args.execute:
        time.sleep(1)
    verified_at = datetime.now(timezone.utc).isoformat() if args.execute else processed_at

    new_entries = []
    for rp, data in verified_receipts:
        input_hash = file_sha256(rp)
        entry = build_hypertimeline_entry(
            receipt_filepath=rp,
            receipt=data,
            input_hash=input_hash,
            processed_at=processed_at,
            verified_at=verified_at,
        )
        new_entries.append(entry)

    print(f"  Built {len(new_entries)} hypertimeline entry(ies)")

    # =====================================================================
    # Step 4: Validate schema
    # =====================================================================
    print("ETL Step: Validate schema")
    all_valid = True
    for entry in new_entries:
        is_valid, errors = validate_entry(entry)
        if not is_valid:
            all_valid = False
            for e in errors:
                print(f"  [VALIDATION ERROR] {e}")
    if all_valid:
        print("  All entries passed schema validation")

    # =====================================================================
    # Step 5: Triple-timestamp
    # =====================================================================
    print("ETL Step: Triple-timestamp")
    entries_stamped = []
    for entry in new_entries:
        entry["triple_timestamps"] = {
            "created_at": entry.get("triple_timestamps", {}).get("created_at", created_at),
            "processed_at": processed_at,
            "verified_at": verified_at,
        }
        entries_stamped.append(entry)
    print(f"  created_at: {created_at}")
    print(f"  processed_at: {processed_at}")
    print(f"  verified_at: {verified_at}")

    # =====================================================================
    # Step 6: Write receipt (bridge run receipt)
    # =====================================================================
    print("ETL Step: Write receipt")

    # Compute output file hash later, after writing
    output_hash_placeholder = "" if dry_run else ""

    # Pre-compute bridge receipt hash
    bridge_rc_json = json.dumps({
        "input_dir": args.input_dir,
        "output_path": args.output,
        "entries_count": len(entries_stamped),
        "processed_at": processed_at,
    }, sort_keys=True)
    bridge_receipt_hash = content_sha256(bridge_rc_json)

    # Load existing entries and dedup
    existing_entries = load_existing_entries(args.output)
    merged_entries = dedup_entries(entries_stamped, existing_entries)
    new_count = len(merged_entries) - len(existing_entries)
    print(f"  Existing entries: {len(existing_entries)}")
    print(f"  New unique entries: {new_count}")
    print(f"  Total merged entries: {len(merged_entries)}")

    # Write bridge run receipt
    verdict = "PASS" if all_valid else "FAIL"
    receipt_path = write_bridge_receipt(
        receipt_dir=BRIDGE_RECEIPT_DIR,
        input_dir=args.input_dir,
        output_path=args.output,
        total_entries=len(merged_entries),
        new_entries_count=new_count,
        input_hashes=input_hashes,
        output_hash=output_hash_placeholder,
        bridge_receipt_hash=bridge_receipt_hash,
        created_at=created_at,
        processed_at=processed_at,
        verified_at=verified_at,
        verdict=verdict,
        dry_run=dry_run,
    )
    if receipt_path:
        print(f"  Bridge receipt: {receipt_path}")

    # =====================================================================
    # Step 7: Stage output
    # =====================================================================
    print("ETL Step: Stage output")
    if dry_run:
        print(f"  [DRY-RUN] Would write hypertimeline to: {args.output}")
        print(f"  [DRY-RUN] Would write bridge receipt to: {receipt_path or '(not created)'}")
        print("  Use --execute to write output.")
        sys.exit(0)

    # Write the hypertimeline JSON
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(merged_entries, f, indent=2)
    print(f"  Hypertimeline written: {args.output}")

    # Compute output hash and update entries
    output_hash = file_sha256(args.output)
    print(f"  Output hash: {output_hash}")

    # Update the bridge receipt with the actual output hash
    if receipt_path and os.path.exists(receipt_path):
        with open(receipt_path, "r") as f:
            rcpt = json.load(f)
        rcpt["triple_hashes"]["output_hash"] = output_hash
        # Recompute bridge receipt hash
        rcpt["triple_hashes"]["bridge_receipt_hash"] = content_sha256(
            json.dumps(rcpt, sort_keys=True)
        )
        with open(receipt_path, "w") as f:
            json.dump(rcpt, f, indent=2)
        print(f"  Bridge receipt updated with output hash: {receipt_path}")

    # Update entry triple_hashes with output_hash
    for entry in merged_entries:
        entry["triple_hashes"]["output_hash"] = output_hash

    # Re-write with updated hashes
    with open(args.output, "w") as f:
        json.dump(merged_entries, f, indent=2)
    print(f"  Hypertimeline re-written with output hashes: {args.output}")

    print(f"\nDone. {new_count} new entry(ies) merged into hypertimeline.")


if __name__ == "__main__":
    main()
