from __future__ import annotations

import sys


def test_run_job_ignores_legacy_offset_and_fetches_next_null_rows(monkeypatch):
    import scripts.corpus_embed_fill_worker as worker

    captured: dict[str, object] = {}

    class FakeCursor:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def execute(self, query, params):
            captured["query"] = query
            captured["params"] = params

        def fetchall(self):
            return [
                ("chunk-1", "This is readable human text with enough words to be safely embedded.", "text/plain", "ok.md"),
                ("chunk-2", "Content-Transfer-Encoding: quoted-printable\r\n<div class=3D\"x\">bad=C2=A0mail</div>", "text/eml", "bad.eml"),
            ]

    class FakeConnection:
        def cursor(self):
            return FakeCursor()

        def close(self):
            pass

    monkeypatch.setattr(worker.psycopg2, "connect", lambda dsn: FakeConnection())

    result = worker.run_job(
        {"payload": {"job_kind": "embed_fill_batch", "offset": 999999, "limit": 50}},
        concurrency=1,
        http_batch=1,
        dry_run=True,
        max_chunks=7,
    )

    assert "OFFSET" not in str(captured["query"]).upper()
    assert captured["params"] == (140,)
    assert result["selection"] == "next_null_chunks"
    assert result["legacy_offset_ignored"] == 999999
    assert result["limit"] == 7
    assert result["rows_found"] == 1
    assert result["quality_skipped"] == 1


def test_worker_cli_has_safe_default_job_cap():
    from pathlib import Path

    source = Path("scripts/corpus_embed_fill_worker.py").read_text(encoding="utf-8")

    assert 'default=500' in source
    assert "safe per-job cap" in source


def test_worker_selection_excludes_audit_blocked_rows():
    from pathlib import Path

    source = Path("scripts/corpus_embed_fill_worker.py").read_text(encoding="utf-8")

    assert "embedding_quality_sql_where" in source
    assert "quality_skipped" in source


def test_worker_requeues_no_progress_embedding_errors(monkeypatch):
    import scripts.corpus_embed_fill_worker as worker

    calls: list[object] = []
    jobs = [
        {
            "job_uuid": "00000000-0000-0000-0000-000000000001",
            "attempt_count": 1,
            "max_attempts": 3,
            "payload": {"limit": 24},
        }
    ]

    class FakeConnection:
        def rollback(self):
            calls.append("rollback")

        def close(self):
            calls.append("close")

    monkeypatch.setattr(worker.psycopg2, "connect", lambda dsn: FakeConnection())
    monkeypatch.setattr(worker, "dequeue_one", lambda conn: jobs.pop(0) if jobs else None)
    monkeypatch.setattr(worker, "run_job", lambda *args, **kwargs: {"filled": 0, "errors": 24, "limit": 24})
    monkeypatch.setattr(worker, "mark_done", lambda *args, **kwargs: calls.append("done"))
    monkeypatch.setattr(worker, "mark_failed", lambda conn, job_uuid, error, attempt, max_attempts: calls.append(("failed", error, attempt, max_attempts)))
    monkeypatch.setattr(sys, "argv", ["corpus_embed_fill_worker.py"])

    assert worker.main() == 0
    assert "done" not in calls
    failed = [call for call in calls if isinstance(call, tuple) and call[0] == "failed"]
    assert failed
    assert "no_embedding_progress" in failed[0][1]
