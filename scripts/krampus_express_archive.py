#!/usr/bin/env python3
"""KRAMPUS EXPRESS ARCHIVE — Hash, timestamp, UUID stale brain docs into KRAMPUSCHEWING.

Doctrine:
  - Nothing deleted. Ever.
  - Stale = KRAMPUSCHEWING. Superseded = KRAMPUSCHEWING.
  - Every file gets: SHA256 hash, UUID, creator ID, atomic datestamp, processed timestamp.
  - Only deleted if emergency gigs of runoff. Eventually cold storage.

Usage:
  python3 scripts/krampus_express_archive.py --recover-deleted
  python3 scripts/krampus_express_archive.py --status
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
KRAMPUS = ROOT / "KRAMPUSCHEWING"
ARCHIVE_DIR = KRAMPUS / "00_PROJECT_BRAIN_archive"
CORPSES = KRAMPUS / "Script_Corpses"

CREATOR_ID = "krampus_express_archive_v1"


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


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def krampus_path(original_path: str, sha: str) -> Path:
    """Generate a KRAMPUSCHEWING archive path: <original_name>.<sha[:16]>.archive"""
    name = original_path.replace("/", "__").replace(" ", "_")
    return ARCHIVE_DIR / f"{name}.{sha[:16]}.archive"


def recover_from_git(relative_path: str, commit: str = "HEAD") -> str | None:
    """Recover a file from git history. Returns content or None."""
    try:
        result = subprocess.run(
            ["git", "show", f"{commit}:{relative_path}"],
            capture_output=True, text=True, cwd=ROOT,
        )
        if result.returncode == 0 and result.stdout:
            return result.stdout
    except Exception:
        pass
    return None


def archive_file(content: str, original_path: str, metadata: dict[str, Any]) -> dict[str, Any]:
    """Hash, timestamp, UUID a single file into KRAMPUSCHEWING."""
    file_sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
    file_uuid = str(uuid.uuid4())
    dest = krampus_path(original_path, file_sha)

    dest.parent.mkdir(parents=True, exist_ok=True)

    archive_record = {
        # File identity
        "uuid": file_uuid,
        "creator_id": CREATOR_ID,
        "original_path": original_path,
        "sha256": file_sha,
        "sha256_short": file_sha[:16],

        # Timestamps — atomic at processing AND at processed
        "atomic_datestamp": now_z(),
        "processed_timestamp": iso_now(),
        "content_size_bytes": len(content.encode("utf-8")),

        # Version tracking
        "git_commit_at_archive": _git_head(),
        "superseded_by": metadata.get("superseded_by", ""),
        "reason": metadata.get("reason", "stale/superseded"),
        "status": "archived",

        # KRAMPUS EXPRESS metadata
        "archive_path": str(dest.relative_to(ROOT)),
        "schema": "lucidota.krampus_express.archive_entry.v1",
    }

    # Write content file
    dest.write_text(content, encoding="utf-8")

    # Write archive manifest alongside
    manifest_path = dest.with_suffix(".archive.json")
    manifest_path.write_text(json.dumps(archive_record, indent=2, sort_keys=True), encoding="utf-8")

    return archive_record


def _git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    except Exception:
        return "unknown"


def find_deleted_brain_docs() -> list[str]:
    """Find all brain docs deleted from working tree (git status D)."""
    result = subprocess.run(
        ["git", "diff", "--name-status", "HEAD", "--", "00_PROJECT_BRAIN/"],
        capture_output=True, text=True, cwd=ROOT,
    )
    deleted = []
    for line in result.stdout.strip().splitlines():
        if line.startswith("D\t"):
            path = line[2:].strip()
            deleted.append(path)
    return deleted


def archive_deleted_docs(dry_run: bool = False) -> dict[str, Any]:
    """Recover and archive all deleted brain docs."""
    deleted = find_deleted_brain_docs()
    print(f"\n=== KRAMPUS EXPRESS ARCHIVE ===", file=sys.stderr)
    print(f"  Deleted brain docs found: {len(deleted)}", file=sys.stderr)

    archived: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    skipped: list[str] = []

    for i, rel_path in enumerate(deleted):
        # Skip AGENTSI (already archived)
        if "AGENTSI" in rel_path:
            skipped.append(f"{rel_path} (already archived)")
            continue
        # Skip KNOWLEDGE_LIBRARY (already archived)
        if "KNOWLEDGE_LIBRARY" in rel_path:
            skipped.append(f"{rel_path} (already archived)")
            continue

        content = recover_from_git(rel_path)
        if content is None:
            errors.append({"path": rel_path, "error": "could not recover from git"})
            continue

        if dry_run:
            print(f"  [{i+1}/{len(deleted)}] WOULD ARCHIVE: {rel_path} ({len(content)}B)", file=sys.stderr)
            continue

        metadata = {
            "reason": "deleted_from_working_tree_stale_superseded",
            "superseded_by": "ODYSSEUS_*_MANUAL.md / ACTIVE_SPEC/",
        }
        record = archive_file(content, rel_path, metadata)
        archived.append(record)
        print(f"  [{i+1}/{len(deleted)}] ARCHIVED: {rel_path} ({len(content)}B) sha256={record['sha256_short']}", file=sys.stderr)

    # Write archive receipt
    summary = {
        "schema": "lucidota.krampus_express.archive_batch.v1",
        "status": "PASS" if not errors else "PARTIAL",
        "generated_at": iso_now(),
        "creator_id": CREATOR_ID,
        "found_deleted": len(deleted),
        "archived": len(archived),
        "errors": len(errors),
        "skipped_already_archived": len(skipped),
        "archive_dir": str(ARCHIVE_DIR.relative_to(ROOT)),
        "receipts": archived,
        "error_details": errors,
        "skipped": skipped,
    }
    receipt_path = ARCHIVE_DIR / f"archive_batch_receipt_{stamp()}.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"\n  Receipt: {receipt_path.relative_to(ROOT)}", file=sys.stderr)
    print(f"  Archived: {len(archived)} | Errors: {len(errors)} | Skipped: {len(skipped)}", file=sys.stderr)

    return summary


def status() -> dict[str, Any]:
    """Print KRAMPUSCHEWING archive status."""
    deleted = find_deleted_brain_docs()
    existing_archives = list(ARCHIVE_DIR.rglob("*.archive"))
    existing_corpses = list(CORPSES.rglob("*"))

    status_data = {
        "schema": "lucidota.krampus_express.status.v1",
        "generated_at": iso_now(),
        "deleted_brain_docs_pending_archive": len(deleted),
        "existing_archives": len(existing_archives),
        "existing_corpses": len(existing_corpses),
        "pending_files": deleted[:20],
    }
    print(f"\n=== KRAMPUS EXPRESS STATUS ===", file=sys.stderr)
    print(f"  Deleted brain docs (pending archive): {len(deleted)}", file=sys.stderr)
    print(f"  Existing archive entries: {len(existing_archives)}", file=sys.stderr)
    print(f"  Script corpses: {len(existing_corpses)}", file=sys.stderr)
    for d in deleted[:10]:
        print(f"    PENDING: {d}", file=sys.stderr)
    return status_data


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="KRAMPUS EXPRESS ARCHIVE")
    ap.add_argument("--recover-deleted", action="store_true", help="Recover and archive all deleted brain docs")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.status:
        data = status()
        if args.json:
            print(json.dumps(data, indent=2))
        return 0

    if args.recover_deleted:
        data = archive_deleted_docs(dry_run=args.dry_run)
        if args.json:
            print(json.dumps(data, indent=2))
        return 0

    # Default: show status
    status()
    print(f"\n  Run with --recover-deleted to archive all pending files.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
