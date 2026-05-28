import json
import subprocess
import sys

from scripts import resource_governor as rg


def test_default_dials_file_is_created(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(rg, "ROOT", root)
    dials = rg.load_dials(root)
    assert dials["GLOBAL_MODE"] == "BALANCED"
    assert dials["MAX_CLOUD_WORKERS"] >= 10
    assert dials["MAX_DB_CONNECTIONS"] >= 1
    assert (root / "05_OUTPUTS" / "runtime" / "governor_dials.json").exists()


def test_cloud_decision_ignores_local_cpu_and_ramps_aggressively():
    telemetry = {
        "cpu": {"loadavg_1m": 99.0},
        "memory": {"available_mb": 128.0},
        "postgres": {"available": True, "connection_count": 2},
        "cloud": {"http_429_rate": 0.0, "latency_ms": 200},
    }
    dials = {
        "GLOBAL_MODE": "AGGRESSIVE",
        "MAX_CLOUD_WORKERS": 50,
        "TARGET_API_LATENCY_MS": 1200,
    }
    decision = rg.decide_cloud_workers(telemetry, dials)
    assert decision["safe_workers"] > 1
    assert decision["throttle"] is False


def test_kill_switch_halts_cloud_workers():
    decision = rg.decide_cloud_workers({}, {"KILL_SWITCH": True, "MAX_CLOUD_WORKERS": 50})
    assert decision["safe_workers"] == 0
    assert decision["throttle"] is True
    assert "kill_switch" in decision["reasons"]
