import subprocess
import time
import os
from pathlib import Path

from scripts import goal_model_fabric_control


def test_model_fabric_control_status_and_stop_fake_sleep(tmp_path):
    runtime = tmp_path / "04_RUNTIME" / "inference_os"
    runtime.mkdir(parents=True)
    proc = subprocess.Popen(["sleep", "60"])
    try:
        (runtime / "deepseek_q4.pid").write_text(str(proc.pid))
        status = goal_model_fabric_control.build_status(tmp_path)
        assert status["lanes"]["deepseek"]["pid"] == proc.pid
        assert status["lanes"]["deepseek"]["pid_alive"] is True
        stopped = goal_model_fabric_control.stop_lanes(tmp_path, ["deepseek"])
        assert stopped["stopped"][0]["lane"] == "deepseek"
        time.sleep(0.2)
        assert proc.poll() is not None
    finally:
        if proc.poll() is None:
            proc.kill()


def test_model_fabric_control_start_uses_lane_script_and_writes_pid(tmp_path, monkeypatch):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    starter = scripts / "fake_start.sh"
    starter.write_text("#!/usr/bin/env bash\nsleep 60\n")
    starter.chmod(0o755)
    old = goal_model_fabric_control.START["deepseek"]
    goal_model_fabric_control.START["deepseek"] = ("scripts/fake_start.sh", "04_RUNTIME/inference_os/fake.log")
    monkeypatch.setattr(goal_model_fabric_control, "health", lambda port: {"ok": False, "test": True})
    monkeypatch.setattr(goal_model_fabric_control, "gpu_free", lambda: 4096)
    try:
        started = goal_model_fabric_control.start_lanes(tmp_path, ["deepseek"], wait=0)
        pid = started["started"][0]["pid_started"]
        assert (tmp_path / "04_RUNTIME/inference_os/deepseek_q4.pid").read_text() == str(pid)
        stopped = goal_model_fabric_control.stop_lanes(tmp_path, ["deepseek"])
        assert stopped["stopped"][0]["signal_sent"] is True
    finally:
        goal_model_fabric_control.START["deepseek"] = old


def test_model_fabric_control_helper_stays_tiny_and_wired():
    source = Path(goal_model_fabric_control.__file__).read_text().splitlines()
    code_lines = [line for line in source if line.strip() and not line.lstrip().startswith("#")]
    assert len(code_lines) <= 100
    manifest = Path("GOALS/plugin_build_mode_bootstrap.json").read_text()
    recovery = Path("scripts/recovery_matrix.py").read_text()
    assert "scripts/goal_model_fabric_control.py" in manifest
    assert "goal_model_fabric_status" in recovery


def test_model_fabric_control_knows_mamba_gpu_and_keeps_cpu_lanes_off_cuda():
    assert "mamba_gpu" in goal_model_fabric_control.START
    source = Path(goal_model_fabric_control.__file__).read_text()
    assert "CUDA_VISIBLE_DEVICES" in source
    assert "mamba_cpu" in source


def test_model_fabric_control_status_runs_without_cloud_env_keys(tmp_path):
    env = {k: v for k, v in os.environ.items() if "GROQ_API_KEY" not in k and "COHERE_API_KEY" not in k and "CO_API_KEY" not in k}
    proc = subprocess.run(
        ["python3", "scripts/goal_model_fabric_control.py", "status", "--json"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        timeout=10,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    assert "GOAL_MODEL_FABRIC_CONTROL=PASS" in proc.stdout
