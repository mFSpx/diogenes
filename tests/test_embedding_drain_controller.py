from __future__ import annotations

import json


def test_embedding_drain_controller_builds_memory_snapshot_from_proc_when_env_missing(monkeypatch):
    from scripts import lucidota_embedding_drain_controller as ctl

    monkeypatch.delenv("LUCIDOTA_AVAILABLE_MB", raising=False)
    monkeypatch.setattr(ctl, "read_mem_available_mb", lambda: 3210)

    snapshot = ctl.build_snapshot()

    assert snapshot["memory"]["available_mb"] == 3210
    assert snapshot["memory"]["available_mb_source"] == "proc_meminfo"


def test_embedding_drain_controller_skips_under_memory_pressure(monkeypatch, tmp_path):
    from scripts import lucidota_embedding_drain_controller as ctl

    snapshot = {
        "cpu": {"count": 4, "loadavg_1m": 1.0},
        "memory": {"available_mb": 800, "swap_used_pct": 35},
        "disk": {"used_pct": 50},
        "vram": {"gpus": []},
    }
    calls = []
    monkeypatch.setattr(ctl.subprocess, "run", lambda *a, **k: calls.append((a, k)))

    report = ctl.run_controller(
        execute=True,
        snapshot=snapshot,
        governor_rung=3,
        receipt_root=tmp_path,
        requested_jobs=1,
    )

    assert report["status"] == "SKIPPED"
    assert report["decision"]["admit"] is False
    assert "mem_available_below_floor" in report["decision"]["reasons"]
    assert calls == []


def test_embedding_drain_controller_rung3_caps_to_tiny_safe_slice(monkeypatch, tmp_path):
    from scripts import lucidota_embedding_drain_controller as ctl

    snapshot = {
        "cpu": {"count": 4, "loadavg_1m": 1.0},
        "memory": {"available_mb": 4096, "swap_used_pct": 10},
        "disk": {"used_pct": 50},
        "vram": {"gpus": []},
    }

    class Proc:
        returncode = 0
        stdout = "[embed_worker] done job=abc filled=24 errors=0 receipt=embed_fill.json\n"
        stderr = ""

    commands = []

    def fake_run(cmd, **kwargs):
        commands.append(cmd)
        return Proc()

    monkeypatch.setattr(ctl.subprocess, "run", fake_run)
    report = ctl.run_controller(
        execute=True,
        snapshot=snapshot,
        governor_rung=3,
        receipt_root=tmp_path,
        requested_jobs=2,
        requested_max_chunks=500,
        requested_concurrency=3,
        requested_http_batch=32,
    )

    assert report["status"] == "PASSED"
    assert report["decision"]["safe_jobs"] == 1
    assert report["decision"]["worker_args"]["max_chunks"] == 24
    assert report["decision"]["worker_args"]["concurrency"] == 1
    assert report["decision"]["worker_args"]["http_batch"] == 8
    assert len(commands) == 1
    assert "--max-chunks" in commands[0]
    assert "24" in commands[0]
    assert "receipt_path" in report
    receipt_path = tmp_path / report["receipt_path"].split("/")[-1]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "PASSED"
    assert receipt["worker_results"][0]["filled"] == 24


def test_embedding_drain_controller_marks_noop_worker_as_skipped(monkeypatch, tmp_path):
    from scripts import lucidota_embedding_drain_controller as ctl

    snapshot = {
        "cpu": {"count": 4, "loadavg_1m": 1.0},
        "memory": {"available_mb": 4096, "swap_used_pct": 10},
        "disk": {"used_pct": 50},
        "vram": {"gpus": []},
    }

    class Proc:
        returncode = 0
        stdout = "[embed_worker] done job=abc filled=0 errors=0 receipt=embed_fill.json\n"
        stderr = ""

    commands = []

    def fake_run(cmd, **kwargs):
        commands.append((cmd, kwargs))
        return Proc()

    monkeypatch.setattr(ctl.subprocess, "run", fake_run)
    report = ctl.run_controller(
        execute=True,
        snapshot=snapshot,
        governor_rung=3,
        receipt_root=tmp_path,
        requested_jobs=1,
        requested_max_chunks=500,
        requested_concurrency=3,
        requested_http_batch=32,
    )

    assert report["status"] == "SKIPPED"
    assert "no_embedding_progress" in report["decision"]["reasons"]
    assert len(commands) == 1
    assert report["worker_results"][0]["filled"] == 0
