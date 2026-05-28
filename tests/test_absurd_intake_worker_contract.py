#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import absurd_intake_worker


def test_pending_count_is_bounded(tmp_path):
    for index in range(3):
        (tmp_path / f"drop-{index}.txt").write_text("x", encoding="utf-8")
    assert absurd_intake_worker.pending_count(tmp_path, max_entries=2) == 2
    assert absurd_intake_worker.pending_count(tmp_path, max_entries=10) == 3


def test_service_active_timeout_or_error_routes_false(monkeypatch):
    def raise_timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd=["systemctl"], timeout=1)

    monkeypatch.setattr(absurd_intake_worker.subprocess, "run", raise_timeout)
    assert absurd_intake_worker.service_active() is False
