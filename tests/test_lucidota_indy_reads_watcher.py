from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.legacy import lucidota_indy_reads_watcher as watcher


class _Cursor:
    def __init__(self):
        self.executed: list[tuple[str, tuple | None]] = []

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _Conn:
    def __init__(self):
        self.cursor_obj = _Cursor()
        self.committed = False

    def cursor(self):
        return self.cursor_obj

    def commit(self):
        self.committed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_record_heartbeat_writes_ironclaw_daemon_heartbeats(tmp_path: Path, monkeypatch) -> None:
    conn = _Conn()
    monkeypatch.setattr(watcher.psycopg, "connect", lambda dsn: conn)
    watcher.record_heartbeat(books_root=tmp_path, result={"ok": True, "changed": 3, "results": []})
    assert conn.committed is True
    assert any("ironclaw.daemon_heartbeats" in sql for sql, _ in conn.cursor_obj.executed)
