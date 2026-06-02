from __future__ import annotations

from types import SimpleNamespace

from scripts import luci_response_composer as composer


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def execute(self, *_args, **_kwargs):
        return self

    def fetchall(self):
        return self._rows

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeConn:
    def __init__(self, rows):
        self._rows = rows

    def cursor(self):
        return _FakeCursor(self._rows)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_compose_response_prefers_db_recent_receipts(monkeypatch):
    rows = [
        {
            "work_receipt_uuid": "wr:test-1",
            "receipt_path": "05_OUTPUTS/model_invocations/groq_chat_execute_test.json",
            "verdict": "promote",
            "receipt_detail": {"summary": "Groq provider wrote the ledger cleanly."},
            "work_order_uuid": "wo:test-1",
            "work_kind": "groq_chat_provider",
            "work_status": "succeeded",
            "work_payload": {"summary": "ignored"},
        }
    ]

    monkeypatch.setattr(
        composer.psycopg,
        "connect",
        lambda *_args, **_kwargs: _FakeConn(rows),
    )

    out = composer.compose_response(
        {
            "text": "make my system work and get my shit ingested",
            "intent": "ops",
            "lane": "FASTLANE",
            "text_chars": 43,
            "word_count": 9,
            "ontology_terms": ["TIME", "EVENT", "TOOL", "MODE"],
            "work_order_id": "wo:test-1",
            "attempt_id": "wo:test-1",
            "work_receipt_id": "wr:test-1",
            "receipt_path": "05_OUTPUTS/luci/test.json",
            "database_url": "postgresql:///lucidota_state",
        }
    )

    improve = next(seg for seg in out["segments"] if seg["lane"] == "improve")
    assert "db:groq_chat_provider" in improve["values"][0]["source"]
    assert "Groq provider wrote the ledger cleanly." in improve["text"]
    assert out["visible_response"]["summary"].startswith("Indy_READs:")
    assert out["composition"]["has_improve_lane"] is True
