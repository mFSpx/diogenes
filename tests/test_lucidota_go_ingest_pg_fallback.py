from unittest.mock import Mock, patch

import pytest

import scripts.lucidota_go_ingest as ingest


class FakeCursor:
    def __init__(self, rows=None):
        self.rows = rows or []
        self.executed = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, sql):
        self.executed.append(sql)

    def fetchall(self):
        return self.rows


class FakeConn:
    def __init__(self, rows=None):
        self.cursor_obj = FakeCursor(rows)
        self.committed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def cursor(self):
        return self.cursor_obj

    def commit(self):
        self.committed = True


def test_run_sql_falls_back_to_psycopg_when_psql_missing():
    conn = FakeConn()
    with patch("scripts.lucidota_go_ingest.subprocess.run", side_effect=FileNotFoundError), patch(
        "scripts.lucidota_go_ingest.psycopg.connect", return_value=conn
    ) as connect:
        out = ingest.run_sql("SELECT 1;", dsn="postgresql:///lucidota_storage")

    assert out == ""
    connect.assert_called_once_with("postgresql:///lucidota_storage")
    assert conn.cursor_obj.executed == ["SELECT 1;"]
    assert conn.committed is True


def test_query_tsv_falls_back_to_psycopg_rows_when_psql_missing():
    conn = FakeConn(rows=[("a", 1), ("b", None)])
    with patch("scripts.lucidota_go_ingest.subprocess.run", side_effect=FileNotFoundError), patch(
        "scripts.lucidota_go_ingest.psycopg.connect", return_value=conn
    ):
        rows = ingest.query_tsv("SELECT label, count FROM demo", dsn="postgresql:///lucidota_storage")

    assert rows == [["a", "1"], ["b", ""]]


def test_run_sql_reports_missing_driver_when_psql_and_psycopg_unavailable():
    with patch("scripts.lucidota_go_ingest.subprocess.run", side_effect=FileNotFoundError), patch.object(
        ingest, "psycopg", None
    ):
        with pytest.raises(SystemExit, match="psql not found and psycopg is not installed"):
            ingest.run_sql("SELECT 1;")
