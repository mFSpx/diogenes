#!/usr/bin/env python3
"""Archive duplicate script directories to KRAMPUSCHEWING/Script_Corpses/.

Archives:
  - scripts/lucidota/ (3 files)
  - scripts/ironclaw_host_os/ (5 files)

Pattern: SHA256 + UUID + timestamp + creator_id corpse format.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CORPSES = ROOT / "KRAMPUSCHEWING" / "Script_Corpses"
CREATOR_ID = "harden_pass1_duplicate_archive_v1"

DIRECTORIES = [
    ("scripts/lucidota", "duplicate_of_root_scripts_bonsai_asserts"),
    ("scripts/ironclaw_host_os", "duplicate_ironclaw_integration_scripts"),
]


def now_z() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def corpse_path(original_rel: str, sha: str) -> Path:
    """Generate a Script_Corpses path: <path_with_underscores>.<sha[:16]>.corpse"""
    # scripts/lucidota/foo.sh -> scripts__lucidota__foo.sh
    name = original_rel.replace("/", "__").replace(" ", "_")
    return CORPSES / f"{name}.{sha[:16]}.corpse"


def archive_file(file_path: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    """Archive a single file into Script_Corpses with full corpse metadata."""
    content = file_path.read_bytes()
    file_sha = sha256_file(file_path)
    file_uuid = str(uuid.uuid4())

    # Compute relative path from ROOT
    rel = str(file_path.relative_to(ROOT))
    dest = corpse_path(rel, file_sha)

    dest.parent.mkdir(parents=True, exist_ok=True)

    archive_record = {
        "uuid": file_uuid,
        "creator_id": CREATOR_ID,
        "original_path": rel,
        "sha256": file_sha,
        "sha256_short": file_sha[:16],
        "atomic_datestamp": now_z(),
        "processed_timestamp": iso_now(),
        "content_size_bytes": len(content),
        "superseded_by": metadata.get("superseded_by", ""),
        "reason": metadata.get("reason", "duplicate_directory_archive"),
        "status": "archived",
        "archive_path": str(dest.relative_to(ROOT)),
        "schema": "lucidota.script_corpse.duplicate_archive.v1",
    }

    # Write corpse file (content)
    dest.write_bytes(content)

    # Write manifest alongside
    manifest_path = dest.with_suffix(".corpse.json")
    manifest_path.write_text(
        json.dumps(archive_record, indent=2, sort_keys=True), encoding="utf-8"
    )

    return archive_record


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Archive duplicate script directories to Script_Corpses")
    ap.add_argument("--dry-run", action="store_true", help="Show what would be archived without doing it")
    ap.add_argument("--json", action="store_true", help="Output results as JSON")
    args = ap.parse_args()

    archived: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for dir_rel, reason in DIRECTORIES:
        dir_path = ROOT / dir_rel
        if not dir_path.is_dir():
            errors.append({"path": dir_rel, "error": "directory_not_found"})
            continue

        files = sorted(dir_path.iterdir())
        for file_path in files:
            if file_path.name.startswith("."):
                continue
            if file_path.is_dir():
                continue

            if args.dry_run:
                rel = str(file_path.relative_to(ROOT))
                size = file_path.stat().st_size
                sha = sha256_file(file_path)
                print(f"  WOULD ARCHIVE: {rel} ({size}B) sha256={sha[:16]}", file=sys.stderr)
                continue

            metadata = {
                "reason": reason,
                "superseded_by": "scripts/ root level equivalents",
            }
            try:
                record = archive_file(file_path, metadata)
                archived.append(record)
                print(
                    f"  ARCHIVED: {record['original_path']} ({record['content_size_bytes']}B) "
                    f"sha256={record['sha256_short']} uuid={record['uuid'][:8]}",
                    file=sys.stderr,
                )
            except Exception as e:
                errors.append({"path": str(file_path), "error": str(e)})
                print(f"  ERROR: {file_path}: {e}", file=sys.stderr)

    # Write batch receipt
    summary = {
        "schema": "lucidota.script_corpse.duplicate_archive_batch.v1",
        "status": "PASS" if not errors else "PARTIAL",
        "generated_at": iso_now(),
        "creator_id": CREATOR_ID,
        "archived": len(archived),
        "errors": len(errors),
        "archive_dir": str(CORPSES.relative_to(ROOT)),
        "receipts": archived,
        "error_details": errors,
    }

    receipt_filename = f"duplicate_archive_batch_receipt_{now_z()}.json"
    receipt_path = CORPSES / receipt_filename
    CORPSES.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    print(f"\n  Batch receipt: {receipt_path.relative_to(ROOT)}", file=sys.stderr)
    print(f"  Archived: {len(archived)} | Errors: {len(errors)}", file=sys.stderr)

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))

    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
