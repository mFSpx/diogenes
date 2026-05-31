#!/usr/bin/env python3
"""Apply the Percyphon GIN index migration against lucidota_storage."""
import os
import sys
import psycopg2
from pathlib import Path

SCHEMA_FILE = Path(__file__).resolve().parents[1] / "06_SCHEMA" / "128_percyphon_gin_index.sql"
DSN = os.environ.get("LUCIDOTA_GO_STORAGE_DSN", "postgresql:///lucidota_storage")


def main() -> None:
    sql = SCHEMA_FILE.read_text()
    try:
        conn = psycopg2.connect(DSN)
        conn.autocommit = True  # CREATE INDEX CONCURRENTLY requires autocommit
        cur = conn.cursor()
        cur.execute(sql)
        print("GIN index applied (or already existed).")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
