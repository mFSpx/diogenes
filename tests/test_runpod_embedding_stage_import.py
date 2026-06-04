#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from contextlib import redirect_stdout
from io import StringIO
from importlib import import_module
from pathlib import Path

mod = import_module("scripts.runpod_embedding_stage_import")


class FakeCursor:
    def __init__(self):
        self.executemany_calls: list[tuple[str, list[tuple]]] = []
        self.commits = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def executemany(self, sql: str, params: list[tuple]):
        self.executemany_calls.append((sql, params))


class FakeConnection:
    def __init__(self):
        self.cur = FakeCursor()
        self.committed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def cursor(self):
        return self.cur

    def commit(self):
        self.committed = True


def _write_rows(path: Path, rows: list[dict]) -> str:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    return _sha256(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row(payload: str, *, dimensions: int = 64, model: str = "sha256-placeholder") -> dict[str, object]:
    return {
        "chunk_id": payload,
        "text_sha256": "aa" * 16,
        "status": "EMBEDDED",
        "provider": "deterministic",
        "model": model,
        "dimensions": dimensions,
        "embedding": [0.1] * dimensions,
        "error": None,
        "source_path": "src",
        "chunk_text_preview": "preview",
    }


def test_read_and_validate_jsonl_enforces_hash_and_row_count(tmp_path: Path):
    input_path = tmp_path / "chunk_embeddings.jsonl"
    rows = [_row("c1"), _row("c2")]
    expected = _write_rows(input_path, rows)

    parsed, report, ok = mod.read_and_validate_jsonl(
        input_path,
        expected_rows=2,
        expected_sha256=expected,
    )

    assert ok is True
    assert report["valid_rows"] == 2
    assert report["row_count_ok"] is True
    assert report["sha256_ok"] is True
    assert parsed[0]["chunk_id"] == "c1"


def test_read_and_validate_jsonl_reports_invalid_rows(tmp_path: Path):
    input_path = tmp_path / "chunk_embeddings.jsonl"
    bad = _row("bad", dimensions=2)
    bad["model"] = ""
    bad["embedding"] = [0.1]
    _write_rows(input_path, [_row("good"), bad])

    parsed, report, ok = mod.read_and_validate_jsonl(input_path)

    assert ok is False
    assert report["valid_rows"] == 1
    assert report["invalid_rows"] == 1
    assert report["row_errors"][0]["line"] == 2


def test_build_import_plan_uses_copy_and_upsert_shape():
    rows = [_row("c1"), _row("c2")]
    plan = mod.build_import_plan(rows, import_table="lucidota_projection.runpod_chunk_embeddings_stage")

    assert plan["ok"] is True
    assert "\\copy" in plan["copy_stmt"]
    assert "ON CONFLICT (chunk_id)" in plan["upsert_stmt"]
    assert "INSERT INTO" in plan["upsert_stmt"]
    assert "embedding_json" in plan["upsert_stmt"]
    assert "::vector" not in plan["upsert_stmt"]


def test_dry_run_mode_prints_receipt_without_db_calls(tmp_path: Path, monkeypatch):
    input_path = tmp_path / "chunk_embeddings.jsonl"
    rows = [_row("c1"), _row("c2")]
    expected = _write_rows(input_path, rows)
    receipt = tmp_path / "runpod_embedding_stage_import_receipt.json"

    monkeypatch.setattr(mod, "psycopg", None)
    stdout = StringIO()
    with redirect_stdout(stdout):
        code = mod.main(
            [
                "--input",
                str(input_path),
                "--expected-rows",
                "2",
                "--expected-sha256",
                expected,
                "--receipt",
                str(receipt),
                "--json",
            ],
        )

    payload = json.loads(stdout.getvalue())
    assert code == 0
    assert payload["status"] == "PASS"
    assert payload["dry_run"] is True
    assert payload["execution_status"] is None
    assert payload["validation"]["row_count_ok"] is True
    assert payload["validation"]["sha256_ok"] is True


def test_execute_requires_database_url(tmp_path: Path, monkeypatch):
    input_path = tmp_path / "chunk_embeddings.jsonl"
    _write_rows(input_path, [_row("c1")])
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("ABSURD_SYSTEM_DATABASE_URL", raising=False)
    stdout = StringIO()

    with redirect_stdout(stdout):
        code = mod.main(
            [
                "--input",
                str(input_path),
                "--execute",
                "--json",
            ],
        )

    payload = json.loads(stdout.getvalue())
    assert code == 1
    assert payload["status"] == "FAIL"
    assert payload["error"] == "Missing DATABASE_URL for --execute"


def test_dry_run_fails_when_row_count_mismatch(tmp_path: Path):
    input_path = tmp_path / "chunk_embeddings.jsonl"
    _write_rows(input_path, [_row("c1"), _row("c2")])
    stdout = StringIO()

    with redirect_stdout(stdout):
        code = mod.main(
            [
                "--input",
                str(input_path),
                "--expected-rows",
                "3",
                "--json",
            ],
        )

    payload = json.loads(stdout.getvalue())
    assert code == 1
    assert payload["status"] == "FAIL"
    assert payload["validation"]["row_count_ok"] is False


def test_execute_import_runs_bounded_batches_with_upsert_sql():
    conn = FakeConnection()

    def _connect(dsn: str):
        return conn

    rows = [_row(f"c{i}", dimensions=4) for i in range(1, 6)]
    report = mod.execute_import(
        database_url="postgresql:///memory",
        rows=rows,
        import_table="lucidota_projection.runpod_chunk_embeddings_stage",
        columns=mod.DEFAULT_COLUMNS,
        batch_size=2,
        connect_fn=_connect,
    )

    assert report["executed_rows"] == 5
    assert report["executed_batches"] == 3
    assert len(conn.cur.executemany_calls) == 3
    for sql, batch in conn.cur.executemany_calls:
        assert "ON CONFLICT (chunk_id)" in sql
        assert "embedding_json" in sql
        assert "::vector" not in sql
        assert isinstance(batch, list)
        assert batch and len(batch) <= 2
