#!/usr/bin/env python3
from __future__ import annotations

import os
import threading
import time

import psycopg2
import psycopg2.errors

DSN = os.environ.get("LUCIDOTA_GO_STORAGE_DSN") or os.environ.get("DATABASE_URL") or "postgresql:///lucidota_storage"


def setup() -> None:
    conn = psycopg2.connect(DSN)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE SCHEMA IF NOT EXISTS lucidota_spine")
            cur.execute("CREATE TABLE IF NOT EXISTS lucidota_spine.deadlock_probe (id integer PRIMARY KEY, note text NOT NULL)")
            cur.execute("INSERT INTO lucidota_spine.deadlock_probe(id,note) VALUES (1,'left'),(2,'right') ON CONFLICT (id) DO NOTHING")
    finally:
        conn.close()


def worker(first: int, second: int, ready: threading.Barrier, result: list[str]) -> None:
    conn = psycopg2.connect(DSN)
    try:
        conn.autocommit = False
        with conn.cursor() as cur:
            cur.execute("SET deadlock_timeout = '100ms'")
            cur.execute("UPDATE lucidota_spine.deadlock_probe SET note = note WHERE id = %s", (first,))
            ready.wait()
            time.sleep(0.25)
            cur.execute("UPDATE lucidota_spine.deadlock_probe SET note = note WHERE id = %s", (second,))
            conn.commit()
            result.append("committed")
    except psycopg2.errors.DeadlockDetected as exc:
        conn.rollback()
        result.append(f"deadlock_detected:{exc.__class__.__name__}")
    except Exception as exc:
        conn.rollback()
        result.append(f"other_error:{exc.__class__.__name__}:{exc}")
    finally:
        conn.close()


def main() -> int:
    setup()
    ready = threading.Barrier(2)
    result: list[str] = []
    left = threading.Thread(target=worker, args=(1, 2, ready, result))
    right = threading.Thread(target=worker, args=(2, 1, ready, result))
    left.start()
    right.start()
    left.join()
    right.join()
    print("\n".join(result))
    return 0 if any(item.startswith("deadlock_detected") for item in result) else 1


if __name__ == "__main__":
    raise SystemExit(main())
