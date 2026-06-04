from __future__ import annotations

import json
import sys
from pathlib import Path
from subprocess import CompletedProcess

import pytest


def _make_command_runner(returncode: int = 0, stdout: bytes = b"", stderr: bytes = b""):
    calls: list[list[str]] = []

    def _runner(command: list[str], **kwargs):
        calls.append(command)
        return CompletedProcess(command, returncode, stdout=stdout, stderr=stderr)

    _runner.calls = calls  # type: ignore[attr-defined]
    return _runner


def test_same_command_signature_skips_after_pass(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from scripts import test_receipt_gate as gate

    watched = tmp_path / "watched.py"
    watched.write_text("print('a')\n", encoding="utf-8")

    store = gate.InMemoryReceiptStore()
    runner = _make_command_runner()
    first = gate.run_gate(
        [
            "run",
            "--scope",
            "indy",
            "--cwd",
            str(tmp_path),
            "--watch",
            str(watched),
            "--",
            sys.executable,
            "-c",
            "print('hello')",
        ],
        store=store,
        command_runner=runner,
        openapi_fetcher=lambda: {"included": False, "hash": None},
        git_probe=lambda watched_paths: gate.GitProbe(
            branch="main",
            commit="abc1234",
            diff_hash="d" * 64,
            status_hash="e" * 64,
            dirty=False,
        ),
    )
    second = gate.run_gate(
        [
            "run",
            "--scope",
            "indy",
            "--cwd",
            str(tmp_path),
            "--watch",
            str(watched),
            "--",
            sys.executable,
            "-c",
            "print('hello')",
        ],
        store=store,
        command_runner=runner,
        openapi_fetcher=lambda: {"included": False, "hash": None},
        git_probe=lambda watched_paths: gate.GitProbe(
            branch="main",
            commit="abc1234",
            diff_hash="d" * 64,
            status_hash="e" * 64,
            dirty=False,
        ),
    )

    out = capsys.readouterr().out
    assert first.exit_code == 0
    assert first.status == "passed"
    assert second.exit_code == 0
    assert second.status == "skipped"
    assert "SKIPPED_ALREADY_VERIFIED" in out
    assert runner.calls == [[sys.executable, "-c", "print('hello')"]]
    assert store.count(status="passed") == 1
    assert store.count(status="skipped") == 1


def test_changed_file_hash_forces_rerun(tmp_path: Path) -> None:
    from scripts import test_receipt_gate as gate

    watched = tmp_path / "watched.py"
    watched.write_text("print('a')\n", encoding="utf-8")
    store = gate.InMemoryReceiptStore()

    runner = _make_command_runner(returncode=0, stdout=b"hello\n", stderr=b"")
    first = gate.run_gate(
        [
            "run",
            "--scope",
            "indy",
            "--cwd",
            str(tmp_path),
            "--watch",
            str(watched),
            "--",
            sys.executable,
            "-c",
            "print('hello')",
        ],
        store=store,
        command_runner=runner,
        openapi_fetcher=lambda: {"included": False, "hash": None},
        git_probe=lambda watched_paths: gate.GitProbe(
            branch="main",
            commit="abc1234",
            diff_hash="d" * 64,
            status_hash="e" * 64,
            dirty=False,
        ),
    )
    watched.write_text("print('b')\n", encoding="utf-8")
    second = gate.run_gate(
        [
            "run",
            "--scope",
            "indy",
            "--cwd",
            str(tmp_path),
            "--watch",
            str(watched),
            "--",
            sys.executable,
            "-c",
            "print('hello')",
        ],
        store=store,
        command_runner=runner,
        openapi_fetcher=lambda: {"included": False, "hash": None},
        git_probe=lambda watched_paths: gate.GitProbe(
            branch="main",
            commit="abc1234",
            diff_hash="d" * 64,
            status_hash="e" * 64,
            dirty=False,
        ),
    )

    assert first.exit_code == 0
    assert first.status == "passed"
    assert second.exit_code == 0
    assert second.status == "passed"
    assert len(runner.calls) == 2
    assert runner.calls[0] == [sys.executable, "-c", "print('hello')"]
    assert runner.calls[1] == [sys.executable, "-c", "print('hello')"]
    assert store.count(status="passed") == 2


def test_failed_test_does_not_create_passing_skip(tmp_path: Path) -> None:
    from scripts import test_receipt_gate as gate

    watched = tmp_path / "watched.py"
    watched.write_text("print('a')\n", encoding="utf-8")
    runner = _make_command_runner(returncode=2, stdout=b"", stderr=b"boom\n")
    store = gate.InMemoryReceiptStore()

    first = gate.run_gate(
        [
            "run",
            "--scope",
            "indy",
            "--cwd",
            str(tmp_path),
            "--watch",
            str(watched),
            "--",
            sys.executable,
            "-c",
            "raise SystemExit(2)",
        ],
        store=store,
        command_runner=runner,
        openapi_fetcher=lambda: {"included": False, "hash": None},
        git_probe=lambda watched_paths: gate.GitProbe(
            branch="main",
            commit="abc1234",
            diff_hash="d" * 64,
            status_hash="e" * 64,
            dirty=False,
        ),
    )

    second = gate.run_gate(
        [
            "run",
            "--scope",
            "indy",
            "--cwd",
            str(tmp_path),
            "--watch",
            str(watched),
            "--",
            sys.executable,
            "-c",
            "raise SystemExit(2)",
        ],
        store=store,
        command_runner=runner,
        openapi_fetcher=lambda: {"included": False, "hash": None},
        git_probe=lambda watched_paths: gate.GitProbe(
            branch="main",
            commit="abc1234",
            diff_hash="d" * 64,
            status_hash="e" * 64,
            dirty=False,
        ),
    )

    assert first.status == "failed"
    assert second.status == "failed"
    assert runner.calls and len(runner.calls) == 2
    assert store.count(status="passed") == 0
    assert store.count(status="failed") == 2


def test_db_unavailable_stops_no_fallback_truth(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    from scripts import test_receipt_gate as gate

    watched = tmp_path / "watched.py"
    watched.write_text("print('a')\n", encoding="utf-8")

    monkeypatch.setattr(gate, "create_store", lambda *args, **kwargs: (_ for _ in ()).throw(gate.DBBlocked("db down")))
    runner = _make_command_runner()
    result = gate.run_gate(
        [
            "run",
            "--scope",
            "indy",
            "--cwd",
            str(tmp_path),
            "--watch",
            str(watched),
            "--",
            sys.executable,
            "-c",
            "print('hello')",
        ],
        store=None,
        command_runner=runner,
        openapi_fetcher=lambda: {"included": False, "hash": None},
        git_probe=lambda watched_paths: gate.GitProbe(
            branch="main",
            commit="abc1234",
            diff_hash="d" * 64,
            status_hash="e" * 64,
            dirty=False,
        ),
    )

    out = capsys.readouterr().out
    assert result.exit_code == 3
    assert "DB_BLOCKED" in out
    assert runner.calls == []


def test_duplicate_same_signature_calls_do_not_create_fake_green(tmp_path: Path) -> None:
    from scripts import test_receipt_gate as gate

    watched = tmp_path / "watched.py"
    watched.write_text("print('a')\n", encoding="utf-8")
    runner = _make_command_runner(returncode=0, stdout=b"ok\n", stderr=b"")
    store = gate.InMemoryReceiptStore()

    args = [
        "run",
        "--scope",
        "indy",
        "--cwd",
        str(tmp_path),
        "--watch",
        str(watched),
        "--",
        sys.executable,
        "-c",
        "print('hello')",
    ]
    first = gate.run_gate(
        args,
        store=store,
        command_runner=runner,
        openapi_fetcher=lambda: {"included": False, "hash": None},
        git_probe=lambda watched_paths: gate.GitProbe(
            branch="main",
            commit="abc1234",
            diff_hash="d" * 64,
            status_hash="e" * 64,
            dirty=False,
        ),
    )
    second = gate.run_gate(
        args,
        store=store,
        command_runner=runner,
        openapi_fetcher=lambda: {"included": False, "hash": None},
        git_probe=lambda watched_paths: gate.GitProbe(
            branch="main",
            commit="abc1234",
            diff_hash="d" * 64,
            status_hash="e" * 64,
            dirty=False,
        ),
    )

    assert first.status == "passed"
    assert second.status == "skipped"
    assert runner.calls == [[sys.executable, "-c", "print('hello')"]]
    assert store.count(status="passed") == 1
    assert store.count(status="skipped") == 1
