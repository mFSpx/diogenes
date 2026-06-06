#!/usr/bin/env python3
"""
Krampus Hypertimeline Scanner — scans KRAMPUSCHEWING and ingests all files
into lucidota_korpus.krampus_hypertimeline as deprecated timestamped lore.

Closes the Ouroboros loop. Every artifact accounted for on the timeline.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RECEIPT_DIR = ROOT / "05_OUTPUTS" / "receipts"

KRAMPUS = ROOT / "KRAMPUSCHEWING"
BATCH_SIZE = 500

def scan_file(filepath: Path) -> dict[str, Any]:
    """Scan a single file and return metadata."""
    try:
        stat = filepath.stat()
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                sha256.update(chunk)
        digest = sha256.hexdigest()
        return {
            "file_path": str(filepath.relative_to(ROOT)),
            "file_name": filepath.name,
            "file_ext": filepath.suffix.lower(),
            "file_size_bytes": stat.st_size,
            "sha256": digest,
            "file_mtime": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(stat.st_mtime)),
            "status": "hashed",
            "timeline_bucket": "deprecated",
        }
    except Exception as e:
        return {
            "file_path": str(filepath.relative_to(ROOT)),
            "file_name": filepath.name,
            "file_ext": filepath.suffix.lower() if filepath.suffix else "",
            "file_size_bytes": 0,
            "sha256": "",
            "file_mtime": "",
            "status": "error",
            "error": str(e),
        }

def ingest_batch(entries: list[dict], dsn: str) -> int:
    """Ingest a batch into Postgres."""
    import psycopg2
    conn = psycopg2.connect(dsn)
    cur = conn.cursor()
    count = 0
    for entry in entries:
        try:
            cur.execute("""
                INSERT INTO lucidota_korpus.krampus_hypertimeline
                    (file_path, file_name, file_ext, file_size_bytes, sha256, file_mtime, status, timeline_bucket)
                VALUES (%s, %s, %s, %s, %s, %s::timestamptz, %s, %s)
                ON CONFLICT (sha256) DO UPDATE SET
                    file_path = EXCLUDED.file_path,
                    file_size_bytes = EXCLUDED.file_size_bytes,
                    status = EXCLUDED.status
                    WHERE lucidota_korpus.krampus_hypertimeline.sha256 = ''
            """, (
                entry["file_path"], entry["file_name"], entry["file_ext"],
                entry["file_size_bytes"], entry["sha256"],
                entry.get("file_mtime") or None,
                entry["status"], entry["timeline_bucket"],
            ))
            count += 1
        except Exception:
            pass
    conn.commit()
    cur.close()
    conn.close()
    return count

def main():
    parser = argparse.ArgumentParser(description="Krampus Hypertimeline Scanner")
    parser.add_argument("--dsn", default=os.environ.get("LUCIDOTA_GO_STATE_DSN", "postgresql:///lucidota_state"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    print(f"Scanning {KRAMPUS}...", file=sys.stderr)
    t0 = time.time()
    all_files = list(KRAMPUS.rglob("*"))
    all_files = [f for f in all_files if f.is_file()]
    total = len(all_files)
    print(f"Found {total} files ({sum(f.stat().st_size for f in all_files)/1e9:.1f} GB)", file=sys.stderr)

    scanned = 0
    ingested = 0
    errors = 0
    batch: list[dict] = []

    for i, f in enumerate(all_files):
        entry = scan_file(f)
        if entry["status"] == "error":
            errors += 1
        scanned += 1
        batch.append(entry)

        if len(batch) >= args.batch:
            if not args.dry_run:
                ingested += ingest_batch(batch, args.dsn)
            batch = []

        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (total - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{total}] {scanned} scanned, {errors} errors, {rate:.0f} files/s, ETA {eta:.0f}s", file=sys.stderr)

    # Final batch
    if batch and not args.dry_run:
        ingested += ingest_batch(batch, args.dsn)

    elapsed = time.time() - t0
    result = {
        "schema": "lucidota.krampus_hypertimeline_scan.v1",
        "total_files": total,
        "total_bytes": sum(f.stat().st_size for f in all_files),
        "scanned": scanned,
        "ingested": ingested,
        "errors": errors,
        "elapsed_s": round(elapsed, 2),
        "rate": round(total / elapsed, 2) if elapsed > 0 else 0,
    }

    RECEIPT_DIR.mkdir(parents=True, exist_ok=True)
    receipt_path = RECEIPT_DIR / f"krampus_scan_{time.strftime('%Y%m%dT%H%M%S')}.json"
    receipt_path.write_text(json.dumps(result, indent=2))

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"\n=== Krampus Hypertimeline Scan ===")
        print(f"  Total: {total} files ({result['total_bytes']/1e9:.1f} GB)")
        print(f"  Scanned: {scanned} | Ingested: {ingested} | Errors: {errors}")
        print(f"  Time: {elapsed:.1f}s ({result['rate']:.0f} files/s)")
        print(f"  Receipt: {receipt_path.relative_to(ROOT)}")

if __name__ == "__main__":
    main()
