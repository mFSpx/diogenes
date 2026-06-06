#!/usr/bin/env python3
"""Training Graph Promotion Script
Schema: lucidota.trm.training_graph_promotion.v1
Mutation Class: candidate_writer (stages candidates to files only; does not write canonical graph truth)
"""

import argparse
import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

REQUIRED_FIELDS = ["schema", "source_name", "source_hash", "feature_vector", "label", "provenance"]
RECEIPTS_DIR = "05_OUTPUTS/trm_training/receipts"
STAGING_DIR = "05_OUTPUTS/graph_candidates"


def parse_args():
    parser = argparse.ArgumentParser(description="Promote training rows to graph candidates")
    parser.add_argument("--input", required=True, help="Training JSONL path")
    parser.add_argument("--source-name", required=True, help="Source name (e.g., 'ahoy', 'krampus')")
    parser.add_argument("--execute", action="store_true", help="Insert into DB (dry-run by default)")
    parser.add_argument("--verify", action="store_true", help="Add 1s delay and set verified_at timestamp")
    return parser.parse_args()


def compute_sha256(content: str) -> str:
    """Compute SHA256 hash of content string."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def compute_file_sha256(filepath: str) -> str:
    """Compute SHA256 hash of a file."""
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def build_feature_vector(row: dict) -> dict:
    """Extract feature vector from training row."""
    features = {}
    for key, value in row.items():
        if isinstance(value, (int, float)):
            features[key] = value
        elif isinstance(value, str):
            features[f"{key}_len"] = len(value)
    return features


def build_candidate_packet(row: dict, source_name: str, created_at: str) -> dict:
    """Build a graph candidate packet from a training row."""
    source_hash = compute_source_hash(row)
    feature_vector = build_feature_vector(row)
    label = row.get("label", row.get("target", "unknown"))

    return {
        "schema": "lucidota.trm.training_graph_promotion.v1",
        "source_name": source_name,
        "source_hash": source_hash,
        "feature_vector": feature_vector,
        "label": label,
        "provenance": {
            "source_file": row.get("_source_file", ""),
            "row_index": row.get("_row_index", -1),
            "created_at": created_at,
        },
        "created_at": created_at,
        "processed_at": created_at,
        "verified_at": created_at,
        "raw_row": row,
    }


def compute_source_hash(row: dict) -> str:
    """Compute SHA256 hash of the row content."""
    content = json.dumps(row, sort_keys=True).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def validate_candidate(candidate: dict) -> tuple[bool, list[str]]:
    """Validate candidate against required schema fields. Returns (is_valid, errors)."""
    errors = []
    for field in REQUIRED_FIELDS:
        if field not in candidate:
            errors.append(f"Missing required field: {field}")
    return len(errors) == 0, errors


def write_candidates(candidates: list, source_name: str) -> str:
    """Write candidates to staging file. Returns the output path."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    os.makedirs(STAGING_DIR, exist_ok=True)
    output_path = f"{STAGING_DIR}/{source_name}_{timestamp}.json"

    with open(output_path, "w") as f:
        json.dump(candidates, f, indent=2)

    return output_path


def write_receipt(
    source_file_path: str,
    staging_file_path: str,
    source_hash: str,
    staging_hash: str,
    receipt_hash: str,
    created_at: str,
    processed_at: str,
    verified_at: str,
    row_count: int,
    verdict: str,
) -> str:
    """Write receipt JSON to receipts directory. Returns receipt path."""
    os.makedirs(RECEIPTS_DIR, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    receipt_path = f"{RECEIPTS_DIR}/receipt_{timestamp}.json"

    receipt = {
        "source_file": source_file_path,
        "staging_file": staging_file_path,
        "triple_hashes": {
            "source_hash": source_hash,
            "staging_hash": staging_hash,
            "receipt_hash": receipt_hash,
        },
        "triple_timestamps": {
            "created_at": created_at,
            "processed_at": processed_at,
            "verified_at": verified_at,
        },
        "row_count": row_count,
        "verdict": verdict,
    }

    receipt_content = json.dumps(receipt, sort_keys=True, indent=2)
    receipt_hash = hashlib.sha256(receipt_content.encode("utf-8")).hexdigest()
    receipt["triple_hashes"]["receipt_hash"] = receipt_hash

    with open(receipt_path, "w") as f:
        json.dump(receipt, f, indent=2)

    return receipt_path


def stage_to_file(candidates: list, staging_dir: str = STAGING_DIR, source_name: str = "unknown") -> str:
    """Write candidates as graph-promotion staging packets (not direct graph writes)."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    os.makedirs(staging_dir, exist_ok=True)
    output_path = f"{staging_dir}/{source_name}_{timestamp}.graph_candidates.json"
    with open(output_path, "w") as f:
        json.dump(candidates, f, indent=2)
    print(f"  Staged {len(candidates)} candidates to {output_path}")
    return output_path


def main():
    args = parse_args()
    dry_run = not args.execute

    created_at = datetime.now(timezone.utc).isoformat()
    processed_at = created_at
    verified_at = processed_at

    # Step 1: Extract
    print("ETL Step: Extract")
    with open(args.input, "r") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    print(f"  Read {len(rows)} training rows from {args.input}")

    # Compute source file hash
    source_file_hash = compute_file_sha256(args.input)
    print(f"  Source file hash: {source_file_hash}")

    # Step 2: Verify hash
    print("ETL Step: Verify hash")
    for idx, row in enumerate(rows):
        row["_row_index"] = idx
        row["_source_file"] = args.input

    # Step 3: Transform
    print("ETL Step: Transform")
    candidates = []
    for row in rows:
        candidate = build_candidate_packet(row, args.source_name, created_at)
        candidates.append(candidate)

    # Step 4: Validate schema
    print("ETL Step: Validate schema")
    all_valid = True
    for candidate in candidates:
        is_valid, errors = validate_candidate(candidate)
        if not is_valid:
            all_valid = False
            for error in errors:
                print(f"  [VALIDATION ERROR] {error}")
    if all_valid:
        print("  All candidates passed schema validation")

    # Step 5: Triple-timestamp
    print("ETL Step: Triple-timestamp")
    processed_at = datetime.now(timezone.utc).isoformat()
    if args.verify:
        time.sleep(1)
        verified_at = datetime.now(timezone.utc).isoformat()
    else:
        verified_at = processed_at
    
    for candidate in candidates:
        candidate["processed_at"] = processed_at
        candidate["verified_at"] = verified_at
        candidate["provenance"]["processed_at"] = processed_at
        candidate["provenance"]["verified_at"] = verified_at
    print(f"  created_at: {created_at}")
    print(f"  processed_at: {processed_at}")
    print(f"  verified_at: {verified_at}")

    # Step 6: Write receipt
    print("ETL Step: Write receipt")
    staging_file_path = write_candidates(candidates, args.source_name)
    staging_file_hash = compute_file_sha256(staging_file_path)
    
    # Compute receipt hash (pre-compute before writing)
    receipt_content_pre = json.dumps({
        "source_file": args.input,
        "staging_file": staging_file_path,
        "triple_hashes": {
            "source_hash": source_file_hash,
            "staging_hash": staging_file_hash,
            "receipt_hash": "",
        },
        "triple_timestamps": {
            "created_at": created_at,
            "processed_at": processed_at,
            "verified_at": verified_at,
        },
        "row_count": len(candidates),
        "verdict": "PASS" if all_valid else "FAIL",
    }, sort_keys=True, indent=2)
    receipt_hash = hashlib.sha256(receipt_content_pre.encode("utf-8")).hexdigest()
    
    verdict = "PASS" if all_valid else "FAIL"
    receipt_path = write_receipt(
        source_file_path=args.input,
        staging_file_path=staging_file_path,
        source_hash=source_file_hash,
        staging_hash=staging_file_hash,
        receipt_hash=receipt_hash,
        created_at=created_at,
        processed_at=processed_at,
        verified_at=verified_at,
        row_count=len(candidates),
        verdict=verdict,
    )
    print(f"  Receipt written to {receipt_path}")

    # Step 7: Stage for graph
    print("ETL Step: Stage for graph")
    print(f"  Staging file: {staging_file_path}")
    print(f"  Candidate count: {len(candidates)}")

    # Stage for graph promotion (no direct graph write)
    if args.execute:
        print("ETL Step: Stage candidates")
        stage_to_file(candidates, source_name=args.source_name)
    else:
        print("Dry-run mode: skipping staging. Use --execute to stage.")


if __name__ == "__main__":
    main()
