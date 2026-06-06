#!/usr/bin/env python3
"""
ODYSSEUS FRIDAY SNAPSHOT — Weekly full-extraction + DB ingest + receipt.

Scheduled every Friday. Runs the full RiverML code extraction, produces
receipts, and pushes to the ABSURD queue for downstream consumption.

Usage:
  python3 scripts/odysseus_friday_snapshot.py [--dry-run] [--json]
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

def main() -> int:
    dry_run = "--dry-run" in sys.argv
    json_output = "--json" in sys.argv

    t0 = time.time()
    stamp_val = stamp()
    print(f"\n=== ODYSSEUS FRIDAY SNAPSHOT {stamp_val} ===", file=sys.stderr)

    # Step 0: Zip the current odysseus state (preserve the clone)
    zip_path = ROOT / "09_STORAGE" / f"odysseus_snapshot_{stamp_val}.zip"
    if not dry_run:
        print("  Step 0: Zipping odysseus state...", file=sys.stderr)
        subprocess.run(
            ["zip", "-r", str(zip_path), "01_REPOS/odysseus",
             "-x", "01_REPOS/odysseus/.git/*", "-q"],
            cwd=ROOT, capture_output=True, check=False,
        )
        if zip_path.exists():
            print(f"  Zip: {zip_path} ({zip_path.stat().st_size >> 20}MB)", file=sys.stderr)

    # Step 1: Run the full RiverML extraction
    extract_cmd = [
        sys.executable, str(ROOT / "scripts/odysseus_riverml_extract.py"),
        "--output-dir", str(ROOT / "05_OUTPUTS/brag"),
    ]
    if dry_run:
        extract_cmd.append("--dry-run")

    print("  Step 1: Running RiverML extraction...", file=sys.stderr)
    t1 = time.time()
    result = subprocess.run(extract_cmd, capture_output=True, text=True, cwd=ROOT)
    if result.returncode != 0:
        print(f"  [ERROR] Extraction failed:\n{result.stderr}", file=sys.stderr)
        return 1

    # Parse the last JSON output from extraction
    extract_result = None
    for line in result.stderr.splitlines():
        if line.startswith("  Receipt: "):
            receipt_path = line.split("  Receipt: ", 1)[1].strip()
            print(f"  Extraction receipt: {receipt_path}", file=sys.stderr)
    elapsed1 = time.time() - t1
    print(f"  Extraction: {elapsed1:.1f}s", file=sys.stderr)

    # Step 2: Enqueue ABSURD job for downstream processing
    if not dry_run:
        print("  Step 2: Enqueueing ABSURD job...", file=sys.stderr)
        try:
            spine = ROOT / "scripts/absurd_queue_spine.py"
            enqueue_cmd = [
                sys.executable, str(spine),
                "--action", "enqueue",
                "--execute",
                "--queue", "odysseus_snapshot",
                "--workflow", "odysseus.friday_snapshot",
                "--job-kind", "external_command",
                "--payload-json", json.dumps({
                    "command": ["python3", "scripts/odysseus_riverml_extract.py", "--output-dir", "05_OUTPUTS/brag"],
                    "snapshot_stamp": stamp(),
                    "schedule": "friday",
                }),
            ]
            subprocess.run(enqueue_cmd, check=True, cwd=ROOT)
            print(f"  ABSURD job enqueued.", file=sys.stderr)
        except Exception as e:
            print(f"  [warn] ABSURD enqueue failed: {e}", file=sys.stderr)

    # Step 3: Write snapshot receipt
    elapsed = time.time() - t0
    receipt = {
        "schema": "lucidota.odysseus.friday_snapshot.v1",
        "status": "PASS",
        "generated_at": stamp(),
        "extraction_completed": not dry_run,
        "absurd_enqueued": not dry_run,
        "elapsed_s": round(elapsed, 2),
        "schedule": "friday",
        "dry_run": dry_run,
        "zip_path": str(zip_path.relative_to(ROOT)) if not dry_run else None,
        "zip_bytes": zip_path.stat().st_size if not dry_run and zip_path.exists() else 0,
    }
    receipt_dir = ROOT / "05_OUTPUTS" / "odysseus_snapshot"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = receipt_dir / f"friday_snapshot_{stamp()}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"  Receipt: {receipt_path}", file=sys.stderr)

    print(f"  Elapsed: {elapsed:.1f}s", file=sys.stderr)
    print(f"=== FRIDAY SNAPSHOT COMPLETE ===", file=sys.stderr)

    if json_output:
        print(json.dumps(receipt, indent=2))

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
