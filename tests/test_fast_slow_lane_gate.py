#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "fast_slow_lane_gate.py"


def json_payload(stdout: str) -> dict:
    for line in stdout.splitlines():
        if line.startswith("{"):
            return json.loads(line)
    raise AssertionError(stdout)


def run_gate(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            *args,
            "--cache-dir",
            str(tmp_path / "cache"),
            "--receipt-root",
            str(tmp_path / "receipts"),
            "--json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=5,
    )


def test_fast_packet_caches_without_model_or_graph_writes(tmp_path: Path) -> None:
    proc = run_gate(
        tmp_path,
        "route",
        "--text",
        "cli status probe metadata receipt",
        "--cache-key",
        "pytest",
        "--importance",
        "0.25",
    )
    assert proc.returncode == 0, proc.stderr
    payload = json_payload(proc.stdout)
    assert payload["base_lane"] == "FASTLANE"
    assert payload["model_calls_performed"] is False
    assert payload["canonical_graph_writes_performed"] is False
    assert payload["flush"]["flushed"] is False
    assert payload["status"]["fastlane_count"] == 1
    assert "USER_CLI_INPUT" in json.dumps(payload["flow"])


def test_fast_cache_flushes_to_slowlane_bundle_at_count_threshold(tmp_path: Path) -> None:
    for idx in range(2):
        proc = run_gate(
            tmp_path,
            "route",
            "--text",
            f"fastlane cache receipt bit {idx}",
            "--cache-key",
            "pytest-flush",
            "--importance",
            "0.2",
            "--flush-count",
            "2",
            "--flush-importance",
            "9",
        )
        assert proc.returncode == 0, proc.stderr
    payload = json_payload(proc.stdout)
    assert payload["base_lane"] == "FASTLANE"
    assert payload["flush"]["flushed"] is True
    assert "count>=2" in payload["flush"]["reason"]
    bundle = ROOT / payload["flush"]["bundle_path"] if not Path(payload["flush"]["bundle_path"]).is_absolute() else Path(payload["flush"]["bundle_path"])
    data = json.loads(bundle.read_text(encoding="utf-8"))
    assert data["target_queue"] == "SLOWLANE_ANALYSIS_QUEUE"
    assert data["bit_count"] == 2
    assert payload["status"]["fastlane_count"] == 0


def test_slow_metadata_goes_to_slowlane_inbox_not_fast_cache(tmp_path: Path) -> None:
    proc = run_gate(
        tmp_path,
        "route",
        "--text",
        "deep research analysis for model integration",
        "--metadata-json",
        '{"deep_analysis": true}',
        "--cache-key",
        "pytest-slow",
    )
    assert proc.returncode == 0, proc.stderr
    payload = json_payload(proc.stdout)
    assert payload["base_lane"] == "SLOWLANE"
    assert payload["status"]["fastlane_count"] == 0
    assert payload["status"]["slowlane_inbox_count"] == 1
    assert any("metadata:deep_analysis=true" in r for r in payload["decision"]["route_reason"])


def test_status_and_help_are_runnable(tmp_path: Path) -> None:
    help_proc = subprocess.run([sys.executable, str(SCRIPT), "--help"], cwd=ROOT, text=True, capture_output=True, timeout=5)
    assert help_proc.returncode == 0
    assert "fastlane/slowlane" in help_proc.stdout
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "status", "--cache-key", "empty", "--cache-dir", str(tmp_path / "cache"), "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=5,
    )
    assert proc.returncode == 0
    payload = json_payload(proc.stdout)
    assert payload["fastlane_count"] == 0
