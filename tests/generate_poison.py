#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import os
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "tests" / "poison_drop"


def write_utf8_nul_file(path: Path, lines: int = 50_000, seed: int = 414) -> None:
    rng = random.Random(seed)
    with path.open("wb") as fh:
        for i in range(lines):
            date = f"2026-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}"
            text = f"{date} poison-line index={i} operator=KRAMPUSCHEWING deterministic=utf8 payload={hashlib.sha256(str(i).encode()).hexdigest()}"
            data = text.encode("utf-8")
            if rng.randrange(0, 11) == 0:
                pivot = rng.randrange(0, len(data) + 1)
                data = data[:pivot] + b"\x00" + data[pivot:]
            fh.write(data + b"\n")


def write_binary_copies(out_dir: Path) -> None:
    payload = bytearray()
    for i in range(4096):
        payload.extend(hashlib.sha256(f"binary-copy-seed-{i}".encode()).digest())
    first = out_dir / "02_binary_copy_a.bin"
    first.write_bytes(bytes(payload))
    shutil.copyfile(first, out_dir / "03_binary_copy_b.bin")
    shutil.copyfile(first, out_dir / "04_binary_copy_c.bin")


def write_deadlock_script(path: Path) -> None:
    path.write_text(
        r'''#!/usr/bin/env python3
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
''',
        encoding="utf-8",
    )
    path.chmod(0o755)


def main() -> int:
    out_dir = Path(sys.argv[1]).expanduser().resolve() if len(sys.argv) > 1 else DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    write_utf8_nul_file(out_dir / "01_utf8_with_random_nul.txt")
    write_binary_copies(out_dir)
    write_deadlock_script(out_dir / "05_force_deadlock.py")
    manifest = {
        "drop_dir": str(out_dir),
        "files": sorted(p.name for p in out_dir.iterdir() if p.is_file()),
    }
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
