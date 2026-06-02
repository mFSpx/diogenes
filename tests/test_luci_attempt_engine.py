from __future__ import annotations

import json
from pathlib import Path

import scripts.luci_attempt_engine as engine


def test_classify_score_and_decide():
    job = {"job_kind": "probe", "payload": {"probe_sql": "SELECT 1", "priority": "low"}}
    attempt = engine.classify_job(job)
    assert attempt["attempt_kind"] == "safe_probe"
    obs = {"ok": True, "latency_ms": 3.2, "row_count": 1}
    score = engine.score_attempt(attempt, obs)
    assert score["verdict"] == "win"
    assert score["score"] > 0.0
    assert engine.next_status({"attempt_count": 0, "max_attempts": 2}, score)[0] == "succeeded"


class FakeCursor:
    def __init__(self):
        self.description = None
        self.calls: list[str] = []
        self._fetchone = None
        self._fetchall = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params=None):
        self.calls.append(sql.strip().split()[0].upper())
        text = sql.lower()
        if "from lucidota_control.absurd_queue_job" in text:
            self._fetchone = None
        elif "select 1 as ok" in text:
            self.description = [object()]
            self._fetchall = [(1,)]
        elif "returning raw_artifact_uuid" in text:
            self._fetchone = ["11111111-1111-1111-1111-111111111111"]
        elif "returning event_id" in text:
            self._fetchone = ["22222222-2222-2222-2222-222222222222"]
        elif "returning work_order_uuid" in text:
            self._fetchone = ["33333333-3333-3333-3333-333333333333"]
        elif "returning work_receipt_uuid" in text:
            self._fetchone = ["44444444-4444-4444-4444-444444444444"]
        else:
            self._fetchone = None
        return self

    def fetchone(self):
        return self._fetchone

    def fetchall(self):
        return self._fetchall


class FakeConn:
    def __init__(self):
        self.cursor_obj = FakeCursor()
        self.committed = False

    def cursor(self, row_factory=None):
        return self.cursor_obj

    def commit(self):
        self.committed = True


def test_run_once_writes_receipt(tmp_path: Path):
    conn = FakeConn()
    receipt = engine.run_once(conn, queue_name="control", synthetic=True, receipt_dir=tmp_path)
    assert receipt["status"] == "PASS"
    path = tmp_path / Path(receipt["receipt_path"]).name
    assert path.exists()
    payload = json.loads(path.read_text())
    assert payload["attempt"]["attempt_kind"] == "safe_probe"
    assert payload["score"]["verdict"] == "win"
    assert conn.committed is True
