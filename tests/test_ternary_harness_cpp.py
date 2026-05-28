from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "TOOLS/RUNTIME/ternary_harness/lucidota_ternary_harness.cpp"


def compile_harness(tmp_path: Path) -> Path:
    binary = tmp_path / "lucidota_ternary_harness"
    subprocess.run(
        ["g++", "-std=c++20", "-O2", "-Wall", "-Wextra", "-pedantic", str(SRC), "-o", str(binary)],
        cwd=ROOT,
        check=True,
    )
    return binary


def run_json(binary: Path, *args: str) -> dict:
    cp = subprocess.run([str(binary), *args], cwd=ROOT, text=True, capture_output=True, check=True)
    return json.loads(cp.stdout)


def test_ternary_harness_compiles_and_declares_requested_slots(tmp_path):
    binary = compile_harness(tmp_path)
    plan = run_json(binary, "--print-plan")
    lanes = {lane["id"]: lane for lane in plan["lanes"]}

    assert plan["schema"] == "lucidota.ternary_harness.plan.v1"
    assert lanes["mamba7b_ram"]["slot"] == "ram_always"
    assert lanes["bonsai4b_ram_a"]["slot"] == "ram_always"
    assert lanes["bonsai4b_ram_b"]["slot"] == "ram_always"
    assert lanes["mamba7b_vram_always"]["slot"] == "vram_always"
    assert lanes["bonsai4b_vram_switch"]["slot"] == "vram_switchable"
    assert lanes["deepseek_r1_int8_switch"]["slot"] == "vram_switchable"
    assert plan["slot_policy"]["vram_switchable"]["exclusive"] is True


def test_harness_builds_hard_capped_llama_argv_for_ram_and_vram_lanes(tmp_path):
    binary = compile_harness(tmp_path)

    mamba_argv = run_json(binary, "--lane-argv", "mamba7b_ram")
    mamba_args = mamba_argv["argv"]
    assert "03_VAULT/models/tensorblock/Falcon3-Mamba-7B-Instruct-GGUF/Falcon3-Mamba-7B-Instruct-Q2_K.gguf" in mamba_args
    assert "-ngl" in mamba_args and mamba_args[mamba_args.index("-ngl") + 1] == "0"
    assert "--parallel" in mamba_args and mamba_args[mamba_args.index("--parallel") + 1] == "1"
    assert "-c" in mamba_args and int(mamba_args[mamba_args.index("-c") + 1]) <= 512
    assert mamba_argv["caps"]["cgroup_memory_max_mb"] > 0
    assert mamba_argv["caps"]["kv_ctx_tokens"] <= 512

    vram_argv = run_json(binary, "--lane-argv", "deepseek_r1_int8_switch")
    vram_args = vram_argv["argv"]
    assert "-ngl" in vram_args and int(vram_args[vram_args.index("-ngl") + 1]) > 0
    assert vram_argv["slot"] == "vram_switchable"


def test_harness_rejects_unknown_lanes_and_missing_hard_caps(tmp_path):
    binary = compile_harness(tmp_path)
    unknown = subprocess.run([str(binary), "--lane-argv", "fake_lane"], cwd=ROOT, text=True, capture_output=True)
    assert unknown.returncode != 0
    assert "unknown_lane" in unknown.stderr

    bad_caps = subprocess.run([str(binary), "--check-caps", "--ctx", "0"], cwd=ROOT, text=True, capture_output=True)
    assert bad_caps.returncode != 0
    assert "invalid_ctx" in bad_caps.stderr


def test_harness_swap_plan_is_drain_stop_verify_start_health_receipt(tmp_path):
    binary = compile_harness(tmp_path)
    swap = run_json(binary, "--swap-plan", "deepseek_r1_int8_switch")
    steps = [step["action"] for step in swap["steps"]]
    assert steps == [
        "reject_new_requests",
        "drain_active_requests",
        "stop_current_switchable_pid",
        "verify_pid_dead",
        "verify_port_dead",
        "verify_vram_released",
        "start_target_lane_under_caps",
        "health_check_target_lane",
        "write_swap_receipt",
    ]
    assert swap["target_lane"] == "deepseek_r1_int8_switch"
