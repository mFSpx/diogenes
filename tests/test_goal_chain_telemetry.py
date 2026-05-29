import json
from pathlib import Path

from scripts import goal_chain, goal_system_index, goal_telemetry


def test_goal_chain_seeds_next_goal_packet(tmp_path):
    (tmp_path / "GOALS").mkdir()
    pkt = goal_chain.seed(
        tmp_path,
        title="GOAL 5",
        objective="Slim system without losing features.",
        minutes=30,
    )
    queue = json.loads((tmp_path / "GOALS" / "NEXT_GOAL_QUEUE.json").read_text())
    assert pkt["schema"] == "lucidota.goals.next_goal_packet.v1"
    assert pkt["title"] == "GOAL 5"
    assert queue["queue"][0]["objective"] == "Slim system without losing features."
    assert (tmp_path / "GOALS" / "NEXT_GOAL.md").exists()


def test_system_index_finds_python_functions(tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "ALGOS").mkdir()
    (tmp_path / "scripts" / "a.py").write_text("def one():\n pass\nclass C:\n def two(self):\n  pass\n")
    idx = goal_system_index.build_index(tmp_path, dirs=["scripts", "ALGOS"])
    assert idx["function_count"] == 2
    assert idx["systems"][0]["path"] == "scripts"
    assert "scripts/a.py:one" in idx["functions"]


def test_telemetry_snapshot_has_resource_keys():
    snap = goal_telemetry.snapshot()
    assert snap["schema"] == "lucidota.goals.telemetry.sample.v1"
    for key in ["loadavg", "mem", "cpu", "gpu", "temps"]:
        assert key in snap


def test_telemetry_summarize_reads_loadavg_list():
    rows = [
        {"loadavg": ["0.50", "0.30", "0.10"], "mem": {"MemAvailable": 10}, "gpu": {"parsed": {"mem_used_mb": "1", "temp_c": "40"}}},
        {"loadavg": ["2.50", "1.20", "0.80"], "mem": {"MemAvailable": 5}, "gpu": {"parsed": {"mem_used_mb": "2", "temp_c": "41"}}},
    ]
    assert goal_telemetry.summarize(rows)["max_load1"] == 2.5


def test_telemetry_snapshot_uses_one_cpu_read(monkeypatch):
    calls = {"cpu": 0}

    def fake_cpu():
        calls["cpu"] += 1
        return {"stat": [1], "loadavg": ["1.00", "0.50", "0.25"]}

    monkeypatch.setattr(goal_telemetry, "cpu", fake_cpu)
    monkeypatch.setattr(goal_telemetry, "mem", lambda: {})
    monkeypatch.setattr(goal_telemetry, "gpu", lambda: {})
    monkeypatch.setattr(goal_telemetry, "temps", lambda: {})
    snap = goal_telemetry.snapshot()
    assert snap["loadavg"] == ["1.00", "0.50", "0.25"]
    assert calls["cpu"] == 1


def test_new_goal_helpers_stay_under_100_loc():
    for module in [goal_chain, goal_system_index, goal_telemetry]:
        source = Path(module.__file__).read_text().splitlines()
        code_lines = [line for line in source if line.strip() and not line.lstrip().startswith("#")]
        assert len(code_lines) <= 100, module.__file__


def test_chain_telemetry_helpers_are_in_recovery_matrix():
    text = Path("scripts/recovery_matrix.py").read_text()
    assert "goal_chain_check" in text
    assert "goal_telemetry_snapshot" in text
    assert "goal_system_index" in text


def test_recovery_matrix_has_model_fabric_stop_path():
    text = Path("scripts/recovery_matrix.py").read_text()
    assert "goal_model_fabric_stop_heavy" in text
    assert "goal_model_fabric_control.py stop" in text
