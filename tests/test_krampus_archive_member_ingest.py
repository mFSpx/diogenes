from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path


def test_archive_member_ingest_opens_zip_and_retires_source_after_success(monkeypatch, tmp_path, capsys):
    import scripts.lucidota_krampus_archive_member_ingest as ingest

    krampus_dir = tmp_path / "KRAMPUSCHEWING"
    krampus_dir.mkdir()
    archive = krampus_dir / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../evil.md", "must not escape")
        zf.writestr("inside.md", "alpha archive evidence")

    receipt_dir = tmp_path / "receipts"

    class FakeCursor:
        rowcount = 1

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

    monkeypatch.setattr(ingest.psycopg, "connect", lambda dsn: fake_conn)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lucidota_krampus_archive_member_ingest.py",
            "--krampus-dir",
            str(krampus_dir),
            "--receipt-dir",
            str(receipt_dir),
            "--archive",
            "bundle.zip",
            "--max-members",
            "1",
            "--json",
        ],
    )

    assert ingest.main() == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["schema"] == "lucidota.krampus_archive_member_ingest.v1"
    assert payload["status"] == "PASS"
    assert payload["archives_seen"] == 1
    assert payload["members_seen"] == 1
    assert payload["unsafe_members_skipped"] == 1
    assert payload["chunks_inserted"] == 1
    assert payload["source_files_deleted"] is True
    assert not archive.exists()
    assert payload["files"][0]["source_path"] == "KRAMPUSCHEWING/bundle.zip!inside.md"
    assert Path(payload["receipt_path"]).exists()
    assert fake_conn.closed is True


def test_luci_ingest_archive_routes_to_archive_member_ingest(tmp_path):
    root = Path(__file__).resolve().parents[1]
    krampus_dir = tmp_path / "KRAMPUSCHEWING"
    krampus_dir.mkdir()
    with zipfile.ZipFile(krampus_dir / "bundle.zip", "w") as zf:
        zf.writestr("inside.md", "dry run archive evidence")

    proc = __import__("subprocess").run(
        [
            str(root / "luci"),
            "ingest",
            "archive",
            "--krampus-dir",
            str(krampus_dir),
            "--receipt-dir",
            str(tmp_path / "receipts"),
            "--archive",
            "bundle.zip",
            "--max-members",
            "1",
            "--dry-run",
            "--json",
        ],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
    )

    payload = json.loads(proc.stdout)
    assert payload["schema"] == "lucidota.krampus_archive_member_ingest.v1"
    assert payload["dry_run"] is True
    assert payload["chunks_inserted"] == 1
    assert payload["source_files_deleted"] is False


def test_archive_member_ingest_recurses_into_nested_zip(monkeypatch, tmp_path, capsys):
    import io
    import scripts.lucidota_krampus_archive_member_ingest as ingest

    krampus_dir = tmp_path / "KRAMPUSCHEWING"
    krampus_dir.mkdir()
    nested_bytes = io.BytesIO()
    with zipfile.ZipFile(nested_bytes, "w") as zf:
        zf.writestr("deep.md", "nested archive evidence")
    archive = krampus_dir / "outer.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("nested/inner.zip", nested_bytes.getvalue())

    class FakeCursor:
        rowcount = 1

        def execute(self, sql, params=None):
            pass

        def fetchall(self):
            return []

        def close(self):
            pass

    class FakeConn:
        def cursor(self):
            return FakeCursor()

        def commit(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(ingest.psycopg, "connect", lambda dsn: FakeConn())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lucidota_krampus_archive_member_ingest.py",
            "--krampus-dir",
            str(krampus_dir),
            "--receipt-dir",
            str(tmp_path / "receipts"),
            "--archive",
            "outer.zip",
            "--max-members",
            "1",
            "--json",
        ],
    )

    assert ingest.main() == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["nested_archives_opened"] == 1
    assert payload["members_seen"] == 1
    assert payload["chunks_inserted"] == 1
    assert payload["files"][0]["source_path"] == "KRAMPUSCHEWING/outer.zip!nested/inner.zip!deep.md"
    assert payload["source_files_deleted"] is True
    assert not archive.exists()


def test_archive_member_ingest_opens_7z_with_system_tool(monkeypatch, tmp_path, capsys):
    import scripts.lucidota_krampus_archive_member_ingest as ingest

    krampus_dir = tmp_path / "KRAMPUSCHEWING"
    krampus_dir.mkdir()
    archive = krampus_dir / "pack.7z"
    archive.write_bytes(b"fake 7z bytes; subprocess is mocked")

    class FakeCursor:
        rowcount = 1

        def execute(self, sql, params=None):
            pass

        def fetchall(self):
            return []

        def close(self):
            pass

    class FakeConn:
        def cursor(self):
            return FakeCursor()

        def commit(self):
            pass

        def close(self):
            pass

    def fake_run(cmd, capture_output=True, check=True):
        if cmd[1] == "l":
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout=b"Path = inner.md\nSize = 22\nAttributes = A\n\n",
                stderr=b"",
            )
        if cmd[1] == "x":
            assert cmd[-1] == "inner.md"
            return subprocess.CompletedProcess(cmd, 0, stdout=b"seven archive evidence", stderr=b"")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(ingest.psycopg, "connect", lambda dsn: FakeConn())
    monkeypatch.setattr(ingest.shutil, "which", lambda name: "/usr/bin/7z" if name in {"7z", "7zz"} else None)
    monkeypatch.setattr(ingest.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lucidota_krampus_archive_member_ingest.py",
            "--krampus-dir",
            str(krampus_dir),
            "--receipt-dir",
            str(tmp_path / "receipts"),
            "--archive",
            "pack.7z",
            "--max-members",
            "1",
            "--json",
        ],
    )

    assert ingest.main() == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["archives_seen"] == 1
    assert payload["archives_opened"] == 1
    assert payload["members_seen"] == 1
    assert payload["chunks_inserted"] == 1
    assert payload["files"][0]["source_path"] == "KRAMPUSCHEWING/pack.7z!inner.md"
    assert payload["source_files_deleted"] is True
    assert not archive.exists()


def test_insert_chunks_uses_conflict_agnostic_idempotency():
    import scripts.lucidota_krampus_archive_member_ingest as ingest

    class FakeCursor:
        rowcount = 1

        def __init__(self):
            self.sql = ""

        def execute(self, sql, params=None):
            self.sql = sql

    cur = FakeCursor()
    assert ingest.insert_chunks(cur, "KRAMPUSCHEWING/a.zip!same.md", "same.md", "duplicate text", dry_run=False) == 1

    normalized = " ".join(cur.sql.split())
    assert "ON CONFLICT DO NOTHING" in normalized
    assert "ON CONFLICT (chunk_uuid)" not in normalized


def test_batch_archive_ingest_skips_opened_archives_before_max_archives(monkeypatch, tmp_path, capsys):
    import scripts.lucidota_krampus_archive_member_ingest as ingest

    krampus_dir = tmp_path / "KRAMPUSCHEWING"
    krampus_dir.mkdir()
    old_archive = krampus_dir / "00_old.zip"
    new_archive = krampus_dir / "01_new.zip"
    with zipfile.ZipFile(old_archive, "w") as zf:
        zf.writestr("old.md", "already opened archive evidence")
    with zipfile.ZipFile(new_archive, "w") as zf:
        zf.writestr("new.md", "next pending archive evidence")

    class FakeCursor:
        rowcount = 1

        def __init__(self):
            self.last_params = None

        def execute(self, sql, params=None):
            self.last_params = params

        def fetchall(self):
            params = self.last_params or ()
            flat = " ".join(str(item) for group in params for item in (group if isinstance(group, (list, tuple)) else [group]))
            if "00_old.zip!%" in flat:
                return [("KRAMPUSCHEWING/00_old.zip!old.md",)]
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

    monkeypatch.setattr(ingest.psycopg, "connect", lambda dsn: FakeConn())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lucidota_krampus_archive_member_ingest.py",
            "--krampus-dir",
            str(krampus_dir),
            "--receipt-dir",
            str(tmp_path / "receipts"),
            "--max-archives",
            "1",
            "--max-members",
            "1",
            "--json",
        ],
    )

    assert ingest.main() == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["archive_selection"] == "pending_unopened"
    assert payload["archives_available"] == 2
    assert payload["archives_skipped_opened"] == 1
    assert payload["archives_seen"] == 1
    assert payload["files"][0]["source_path"] == "KRAMPUSCHEWING/01_new.zip!new.md"
    assert not new_archive.exists()
