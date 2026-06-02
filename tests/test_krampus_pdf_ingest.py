from __future__ import annotations

import json
import sys
from pathlib import Path


def test_main_skips_existing_source_path_variants_and_records_receipt(monkeypatch, tmp_path):
    import scripts.lucidota_krampus_pdf_ingest as ingest

    krampus_dir = tmp_path / "KRAMPUSCHEWING"
    krampus_dir.mkdir()
    sample = krampus_dir / "Board Resolution.pdf"
    sample.write_bytes(b"%PDF-1.4 fake pdf bytes")

    receipt_dir = tmp_path / "receipts"

    class FakeCursor:
        def __init__(self):
            self.executed = []

        def execute(self, sql, params=None):
            self.executed.append((sql, params))

        def fetchall(self):
            return [("KRAMPUSCHEWING/Board Resolution.pdf",)]

        def close(self):
            pass

    class FakeConn:
        def __init__(self):
            self.cursor_obj = FakeCursor()
            self.closed = False

        def cursor(self):
            return self.cursor_obj

        def commit(self):
            pass

        def close(self):
            self.closed = True

    fake_conn = FakeConn()

    monkeypatch.setattr(ingest, "KRAMPUS_DIR", krampus_dir)
    monkeypatch.setattr(ingest, "RECEIPT_DIR", receipt_dir)
    monkeypatch.setattr(ingest.psycopg, "connect", lambda dsn: fake_conn)
    monkeypatch.setattr(ingest, "extract_content", lambda path: (_ for _ in ()).throw(AssertionError("skip should avoid extraction")))
    monkeypatch.setattr(sys, "argv", ["lucidota_krampus_pdf_ingest.py"])

    ingest.main()

    insert_sql = [sql for sql, _ in fake_conn.cursor_obj.executed if sql.startswith("INSERT INTO lucidota_korpus.corpus_chunk")]
    assert insert_sql == []

    receipt_files = sorted(receipt_dir.glob("krampus_pdf_*.json"))
    assert len(receipt_files) == 1

    receipt = json.loads(receipt_files[0].read_text(encoding="utf-8"))
    assert receipt["files"] == [
        {
            "path": str(sample),
            "source_path": "KRAMPUSCHEWING/Board Resolution.pdf",
            "status": "skipped",
            "reason": "already_ingested",
        }
    ]


def test_main_supports_bounded_json_ingest(monkeypatch, tmp_path, capsys):
    import scripts.lucidota_krampus_pdf_ingest as ingest

    krampus_dir = tmp_path / "KRAMPUSCHEWING"
    krampus_dir.mkdir()
    first = krampus_dir / "a-first.md"
    second = krampus_dir / "b-second.md"
    first.write_text("alpha evidence text", encoding="utf-8")
    second.write_text("beta evidence text", encoding="utf-8")

    receipt_dir = tmp_path / "receipts"

    class FakeCursor:
        def __init__(self):
            self.executed = []

        def execute(self, sql, params=None):
            self.executed.append((sql, params))

        def fetchall(self):
            return []

        def close(self):
            pass

    class FakeConn:
        def __init__(self):
            self.cursor_obj = FakeCursor()
            self.commits = 0
            self.closed = False

        def cursor(self):
            return self.cursor_obj

        def commit(self):
            self.commits += 1

        def close(self):
            self.closed = True

    fake_conn = FakeConn()

    monkeypatch.setattr(ingest, "KRAMPUS_DIR", krampus_dir)
    monkeypatch.setattr(ingest, "RECEIPT_DIR", receipt_dir)
    monkeypatch.setattr(ingest.psycopg, "connect", lambda dsn: fake_conn)
    monkeypatch.setattr(sys, "argv", ["lucidota_krampus_pdf_ingest.py", "--max-files", "1", "--json"])

    assert ingest.main() == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["schema"] == "lucidota.krampus_top_level_document_ingest.v1"
    assert payload["status"] == "PASS"
    assert payload["max_files"] == 1
    assert payload["files_seen"] == 1
    assert payload["chunks_inserted"] == 1
    assert payload["receipt_path"].endswith(".json")
    assert Path(payload["receipt_path"]).exists()
    assert len(payload["files"]) == 1
    assert payload["files"][0]["path"].endswith("a-first.md")
    assert fake_conn.closed is True


def test_main_queries_only_candidate_source_paths_for_bounded_runs(monkeypatch, tmp_path, capsys):
    import scripts.lucidota_krampus_pdf_ingest as ingest

    krampus_dir = tmp_path / "KRAMPUSCHEWING"
    krampus_dir.mkdir()
    sample = krampus_dir / "single.md"
    sample.write_text("single file", encoding="utf-8")

    class FakeCursor:
        def __init__(self):
            self.executed = []

        def execute(self, sql, params=None):
            self.executed.append((sql, params))

        def fetchall(self):
            return []

        def close(self):
            pass

    class FakeConn:
        def __init__(self):
            self.cursor_obj = FakeCursor()

        def cursor(self):
            return self.cursor_obj

        def commit(self):
            pass

        def close(self):
            pass

    fake_conn = FakeConn()

    monkeypatch.setattr(ingest, "KRAMPUS_DIR", krampus_dir)
    monkeypatch.setattr(ingest, "RECEIPT_DIR", tmp_path / "receipts")
    monkeypatch.setattr(ingest.psycopg, "connect", lambda dsn: fake_conn)
    monkeypatch.setattr(sys, "argv", ["lucidota_krampus_pdf_ingest.py", "--file", "single.md", "--max-files", "1", "--json"])

    assert ingest.main() == 0
    capsys.readouterr()

    select_sql, select_params = fake_conn.cursor_obj.executed[0]
    assert "WHERE source_path = ANY" in select_sql
    assert select_params
    assert "KRAMPUSCHEWING/single.md" in select_params[0]


def test_main_accepts_already_prefixed_file_path(monkeypatch, tmp_path, capsys):
    import scripts.lucidota_krampus_pdf_ingest as ingest

    krampus_dir = tmp_path / "KRAMPUSCHEWING"
    krampus_dir.mkdir()
    sample = krampus_dir / "INVESTORS.docx"
    sample.write_text("investor evidence", encoding="utf-8")

    class FakeCursor:
        def __init__(self):
            self.executed = []

        def execute(self, sql, params=None):
            self.executed.append((sql, params))

        def fetchall(self):
            return []

        def close(self):
            pass

    class FakeConn:
        def __init__(self):
            self.cursor_obj = FakeCursor()

        def cursor(self):
            return self.cursor_obj

        def commit(self):
            pass

        def close(self):
            pass

    fake_conn = FakeConn()

    monkeypatch.setattr(ingest, "KRAMPUS_DIR", krampus_dir)
    monkeypatch.setattr(ingest, "RECEIPT_DIR", tmp_path / "receipts")
    monkeypatch.setattr(ingest.psycopg, "connect", lambda dsn: fake_conn)
    monkeypatch.setattr(sys, "argv", ["lucidota_krampus_pdf_ingest.py", "--file", "KRAMPUSCHEWING/INVESTORS.docx", "--json"])

    assert ingest.main() == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["status"] == "PASS"
    assert payload["files_seen"] == 1
    assert payload["files_processed"] == 1
    assert payload["files"][0]["source_path"] == "KRAMPUSCHEWING/INVESTORS.docx"


def test_main_skips_duplicate_content_files_and_continues(monkeypatch, tmp_path, capsys):
    import scripts.lucidota_krampus_pdf_ingest as ingest

    krampus_dir = tmp_path / "KRAMPUSCHEWING"
    krampus_dir.mkdir()
    first = krampus_dir / "a-first.md"
    dup = krampus_dir / "b-dup.md"
    third = krampus_dir / "c-third.md"
    first.write_text("first file text", encoding="utf-8")
    dup.write_text("duplicate file text", encoding="utf-8")
    third.write_text("third file text", encoding="utf-8")

    receipt_dir = tmp_path / "receipts"

    class FakeCursor:
        def __init__(self):
            self.executed = []
            self._current_path = None

        def execute(self, sql, params=None):
            self.executed.append((sql, params))
            if params and len(params) > 2:
                self._current_path = params[2]
                if self._current_path == "KRAMPUSCHEWING/b-dup.md":
                    self.rowcount = 0
                else:
                    self.rowcount = 1

        def fetchall(self):
            return []

        def close(self):
            pass

    class FakeConn:
        def __init__(self):
            self.cursor_obj = FakeCursor()
            self.commits = 0
            self.closed = False

        def cursor(self):
            return self.cursor_obj

        def commit(self):
            self.commits += 1

        def close(self):
            self.closed = True

    fake_conn = FakeConn()

    monkeypatch.setattr(ingest, "KRAMPUS_DIR", krampus_dir)
    monkeypatch.setattr(ingest, "RECEIPT_DIR", receipt_dir)
    monkeypatch.setattr(ingest.psycopg, "connect", lambda dsn: fake_conn)
    monkeypatch.setattr(sys, "argv", ["lucidota_krampus_pdf_ingest.py", "--max-files", "3", "--json"])

    assert ingest.main() == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["status"] == "PASS"
    assert payload["files_seen"] == 3
    assert payload["files_processed"] == 2
    assert payload["files_skipped"] == 1
    assert payload["chunks_inserted"] == 2
    assert fake_conn.commits == 2
    assert fake_conn.closed is True
    assert [row["status"] for row in payload["files"]] == ["success", "skipped", "success"]
