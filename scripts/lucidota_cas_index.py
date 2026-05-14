#!/usr/bin/env python3
"""Index and verify the local LUCIDOTA CAS vault."""
from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
from pathlib import Path

import psycopg

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "06_SCHEMA" / "005_cas_manifest.sql"
DEFAULT_DB = "postgresql://mfspx@/lucidota_graph"
DEFAULT_VAULT = ROOT / "03_VAULT" / "cas"


def iter_cas_files(vault: Path):
    if not vault.exists():
        return
    for path in vault.rglob("*"):
        if path.is_file() and not path.name.endswith(".tmp"):
            yield path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(prog="lucidota-cas-index")
    ap.add_argument("--vault", type=Path, default=Path(os.environ.get("LUCIDOTA_CAS_VAULT", DEFAULT_VAULT)))
    ap.add_argument("--db-url", default=os.environ.get("LUCIDOTA_GRAPH_DATABASE_URL", DEFAULT_DB))
    ap.add_argument("--source", default="local_cas")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    total = indexed = corrupt = 0
    corrupt_detail = []
    with psycopg.connect(args.db_url) as conn:
        conn.execute(SCHEMA.read_text())
        for path in iter_cas_files(args.vault) or []:
            total += 1
            digest = sha256_file(path)
            expected = path.name
            if len(expected) == 64 and expected != digest:
                corrupt += 1
                corrupt_detail.append({"path": str(path), "expected": expected, "actual": digest})
                continue
            rel = path.relative_to(args.vault).as_posix()
            mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
            conn.execute(
                """
                INSERT INTO lucidota_vault.cas_manifest
                  (sha256, cas_uri, relative_path, size_bytes, mime, source, last_seen_at)
                VALUES (%s,%s,%s,%s,%s,%s,now())
                ON CONFLICT (sha256) DO UPDATE SET
                  relative_path=EXCLUDED.relative_path,
                  size_bytes=EXCLUDED.size_bytes,
                  mime=EXCLUDED.mime,
                  source=EXCLUDED.source,
                  last_seen_at=now()
                """,
                (digest, f"cas://sha256/{digest}", rel, path.stat().st_size, mime, args.source),
            )
            indexed += 1
        conn.execute(
            """
            INSERT INTO lucidota_vault.cas_integrity_check
              (total_files, indexed_files, missing_files, corrupt_files, detail)
            VALUES (%s,%s,0,%s,%s::jsonb)
            """,
            (total, indexed, corrupt, json.dumps({"corrupt": corrupt_detail[:50]})),
        )
        conn.commit()
    report = {"ok": corrupt == 0, "total_files": total, "indexed_files": indexed, "corrupt_files": corrupt}
    print(json.dumps(report, sort_keys=True) if args.json else report)
    return 0 if corrupt == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
