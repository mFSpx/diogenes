from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.krampuschewing_quarantine_triage as triage


class _Cursor:
    def __init__(self):
        self.executed: list[tuple[str, tuple | None]] = []
        self.fetchone_result = ("job-uuid-1", True)

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def fetchone(self):
        return self.fetchone_result

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


def test_quarantine_triage_moves_candidates_and_queues_reingest(tmp_path: Path, monkeypatch) -> None:
    source_root = tmp_path / "source"
    quarantine_root = tmp_path / "KRAMPUSCHEWING" / "quarantine"
    source_root.mkdir(parents=True)
    candidate = source_root / "bad.zip"
    candidate.write_bytes(b"zip-ish bytes")
    conn = _Conn()
    monkeypatch.setattr(triage.psycopg, "connect", lambda dsn: conn)

    report = triage.triage(
        source_root=source_root,
        quarantine_root=quarantine_root,
        execute=True,
        reason_class=None,
        max_files=10,
        database_url="postgresql:///lucidota_state",
    )

    assert report["moved_count"] == 1
    assert report["queued_count"] == 1
    assert report["moved"][0]["reason_class"] == "ARCHIVE_EXTRACT_OUTPUT"
    assert report["moved"][0]["reingest_status"] == "queued"
    assert conn.committed is True
    assert any("absurd_queue_job" in sql for sql, _ in conn.cursor_obj.executed)
    assert any("absurd_queue_event" in sql for sql, _ in conn.cursor_obj.executed)
    assert not candidate.exists()
    assert any(p.name == "bad.zip" for p in quarantine_root.rglob("bad.zip"))
    receipt = Path(report["receipt_path"])
    assert receipt.exists()
    payload = json.loads(receipt.read_text())
    assert payload["operator_delete_performed"] is False
