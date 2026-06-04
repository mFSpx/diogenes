from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path
import re

import pytest


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts" / "cli_process_authority.py"
LIVE_BASE_URL = "http://127.0.0.1:3000"


def test_cli_process_authority_injects_auth_token_and_records_receipt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_cli = tmp_path / "fake_cli_auth.py"
    fake_cli.write_text(
        """#!/usr/bin/env python3\n"""
        """import os, sys\n"""
        """print('AUTH REQUIRED', flush=True)\n"""
        """token = sys.stdin.readline().strip()\n"""
        """if token == os.environ['TEST_AUTH_TOKEN']:\n"""
        """    print('AUTH OK', flush=True)\n"""
        """    sys.exit(0)\n"""
        """print('AUTH BAD', flush=True)\n"""
        """sys.exit(2)\n""",
        encoding="utf-8",
    )
    fake_cli.chmod(0o755)

    monkeypatch.setenv("TEST_AUTH_TOKEN", "s3cr3t-token")
    receipt_path = tmp_path / "cli_receipt.json"
    result = subprocess.run(
        [
            sys.executable,
            str(WRAPPER),
            "--auth-env-var",
            "TEST_AUTH_TOKEN",
            "--receipt-path",
            str(receipt_path),
            "--timeout-seconds",
            "5",
            "--max-restarts",
            "0",
            "--",
            sys.executable,
            str(fake_cli),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "AUTH REQUIRED" in result.stdout
    assert "AUTH OK" in result.stdout
    assert receipt_path.exists()

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["auth_prompt_seen"] is True
    assert receipt["auth_injected"] is True
    assert receipt["exit_code"] == 0


def test_cli_process_authority_restarts_after_timeout(tmp_path: Path) -> None:
    fake_cli = tmp_path / "fake_cli_timeout.py"
    marker = tmp_path / "attempt.marker"
    fake_cli.write_text(
        """#!/usr/bin/env python3\n"""
        """import os, sys, time\n"""
        f"""marker = {str(marker)!r}\n"""
        """if not os.path.exists(marker):\n"""
        """    open(marker, 'w').write('first')\n"""
        """    print('HANGING FIRST ATTEMPT', flush=True)\n"""
        """    time.sleep(3)\n"""
        """    sys.exit(0)\n"""
        """print('RECOVERED ON RESTART', flush=True)\n"""
        """sys.exit(0)\n""",
        encoding="utf-8",
    )
    fake_cli.chmod(0o755)

    receipt_path = tmp_path / "cli_timeout_receipt.json"
    result = subprocess.run(
        [
            sys.executable,
            str(WRAPPER),
            "--receipt-path",
            str(receipt_path),
            "--timeout-seconds",
            "1",
            "--max-restarts",
            "1",
            "--",
            sys.executable,
            str(fake_cli),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "RECOVERED ON RESTART" in result.stdout
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["restart_count"] >= 1
    assert receipt["status"] == "succeeded"


def test_cli_process_authority_starts_new_session_and_kills_process_group_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.cli_process_authority as cli_process_authority

    captured: dict[str, object] = {}

    class FakeStream:
        def readline(self):
            return ""

    class DummyProc:
        pid = 4321
        returncode = None

        def __init__(self) -> None:
            self.stdin = None
            self.stdout = FakeStream()
            self.stderr = FakeStream()

        def poll(self):
            return None

        def wait(self, timeout=None):  # noqa: D401
            return None

        def kill(self):
            captured["fallback_kill"] = True

    def fake_popen(*args, **kwargs):
        captured["start_new_session"] = kwargs.get("start_new_session")
        return DummyProc()

    monkeypatch.setattr(cli_process_authority.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(cli_process_authority.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(cli_process_authority.os, "killpg", lambda pgid, sig: captured.update({"killpg": (pgid, sig)}))
    timeline = iter([0.0, 0.2])
    monkeypatch.setattr(cli_process_authority.time, "monotonic", lambda: next(timeline))
    monkeypatch.setattr(cli_process_authority, "pick_auth_token", lambda env_names: (None, None))

    result = cli_process_authority.run_attempt(
        ["fake-cli"],
        timeout_seconds=0.01,
        auth_patterns=[re.compile(r"AUTH")],
        auth_env_vars=["TEST_AUTH_TOKEN"],
    )

    assert captured["start_new_session"] is True
    assert captured["killpg"] == (4321, signal.SIGKILL)
    assert result.status == "timeout"
