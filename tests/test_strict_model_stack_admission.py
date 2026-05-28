#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from scripts.lucidota_strict_model_stack_admission import (
    build_strict_stack_plan,
    evaluate_admission,
    run_admission,
)


def passing_diogenes_receipt(tmp_path: Path) -> Path:
    path = tmp_path / "diogenes_pass.json"
    path.write_text(json.dumps({"passed": True, "ceiling_kb": 51200, "cases": []}), encoding="utf-8")
    return path


def failing_diogenes_receipt(tmp_path: Path) -> Path:
    path = tmp_path / "diogenes_fail.json"
    path.write_text(json.dumps({"passed": False, "ceiling_kb": 51200, "cases": [{"name": "bad", "blockers": ["rss_over_ceiling"]}]}), encoding="utf-8")
    return path


def test_strict_stack_plan_names_required_model_lanes() -> None:
    plan = build_strict_stack_plan(env={})
    names = {service["name"] for service in plan["services"]}
    assert {
        "deepseek_r1_qwen_1p5b_gpu",
        "mamba7b_ram",
        "bonsai4b_ram",
        "mamba7b_gpu_partial",
        "needle_swarm_6x",
        "indy_reads_watcher",
    } <= names
    deepseek = next(service for service in plan["services"] if service["name"] == "deepseek_r1_qwen_1p5b_gpu")
    assert deepseek["device_lane"] == "nvidia_vram_model_only"
    assert plan["display_policy"]["effective"]["DRI_PRIME"] == "0"
    assert plan["display_policy"]["effective"]["__NV_PRIME_RENDER_OFFLOAD"] == "0"


def test_admission_requires_passing_diogenes_memory_receipt(tmp_path: Path) -> None:
    plan = build_strict_stack_plan(env={})
    failed = evaluate_admission(
        plan,
        diogenes_receipt=json.loads(failing_diogenes_receipt(tmp_path).read_text()),
        check_files=False,
    )
    assert failed["passed"] is False
    assert "diogenes_memory_gate_failed" in failed["blockers"]

    passed = evaluate_admission(
        plan,
        diogenes_receipt=json.loads(passing_diogenes_receipt(tmp_path).read_text()),
        check_files=False,
    )
    assert passed["passed"] is True
    assert passed["vram"]["requested_mb"] <= passed["vram"]["usable_mb"]


def test_run_admission_writes_receipt_and_env_file(tmp_path: Path) -> None:
    receipt = run_admission(
        diogenes_receipt_path=passing_diogenes_receipt(tmp_path),
        receipt_dir=tmp_path,
        check_files=False,
        env={},
    )
    assert receipt["passed"] is True
    assert Path(receipt["receipt_path"]).exists()
    env_path = Path(receipt["env_path"])
    assert env_path.exists()
    env_text = env_path.read_text(encoding="utf-8")
    assert "LUCIDOTA_STRICT_STACK_ADMITTED='1'" in env_text
    assert "LUCIDOTA_ENABLE_MAMBA_GPU_PARTIAL='1'" in env_text


def test_admission_blocks_if_display_is_on_discrete_gpu(tmp_path: Path) -> None:
    plan = build_strict_stack_plan(env={"DRI_PRIME": "1", "__NV_PRIME_RENDER_OFFLOAD": "1"})
    receipt = evaluate_admission(
        plan,
        diogenes_receipt=json.loads(passing_diogenes_receipt(tmp_path).read_text()),
        check_files=False,
    )
    assert receipt["passed"] is False
    assert "display_not_forced_to_onboard_gpu" in receipt["blockers"]


def test_admission_allows_onboard_intel_pci_display_selector(tmp_path: Path) -> None:
    plan = build_strict_stack_plan(env={"DRI_PRIME": "pci-0000_00_02_0", "__NV_PRIME_RENDER_OFFLOAD": "0"})
    receipt = evaluate_admission(
        plan,
        diogenes_receipt=json.loads(passing_diogenes_receipt(tmp_path).read_text()),
        check_files=False,
    )
    assert receipt["passed"] is True
    assert "display_not_forced_to_onboard_gpu" not in receipt["blockers"]


def test_strict_stack_shell_calls_admission_before_starting_models() -> None:
    shell = Path("scripts/lucidota_start_strict_model_stack.sh").read_text(encoding="utf-8")
    admission_idx = shell.index("lucidota_strict_model_stack_admission.py")
    first_start_idx = shell.index("start_server deepseek")
    assert admission_idx < first_start_idx
    assert "lucidota_start_bonsai_ternary_llama.sh" in shell


def test_strict_stack_uses_canonical_mamba_gpu_partial_launcher() -> None:
    plan = build_strict_stack_plan(env={})
    mamba_gpu = next(service for service in plan["services"] if service["name"] == "mamba7b_gpu_partial")
    shell = Path("scripts/lucidota_start_strict_model_stack.sh").read_text(encoding="utf-8")
    assert mamba_gpu["start_script"] == "scripts/lucidota_start_mamba_gpu_partial.sh"
    assert "lucidota_start_mamba_gpu_partial.sh" in shell
    assert "scripts/lucidota_start_strict_model_stack.sh:inline:mamba7b_gpu_partial" not in str(mamba_gpu)


def test_cpu_ram_start_scripts_hide_cuda_for_zero_gpu_layers() -> None:
    mamba = Path("scripts/lucidota_start_mamba_llama.sh").read_text(encoding="utf-8")
    bonsai = Path("scripts/lucidota_start_bonsai_ternary_llama.sh").read_text(encoding="utf-8")
    fabric = Path("scripts/goal_model_fabric_control.py").read_text(encoding="utf-8")
    safe_env = Path("scripts/lucidota_safe_ops_env.sh").read_text(encoding="utf-8")
    assert "CUDA_VISIBLE_DEVICES=()" not in mamba  # guard against shell-array typo
    assert "export CUDA_VISIBLE_DEVICES=\"\"" in mamba
    assert "export CUDA_VISIBLE_DEVICES=\"\"" in bonsai
    assert 'NGL="${LUCIDOTA_BONSAI_NGL:-0}"' in bonsai
    assert "LUCIDOTA_BONSAI_CUDA_VISIBLE_DEVICES" not in fabric
    assert 'LUCIDOTA_BONSAI_NGL="${LUCIDOTA_BONSAI_NGL:-0}"' in safe_env
    assert 'LUCIDOTA_BONSAI_NGL="${LUCIDOTA_BONSAI_NGL:-99}"' not in safe_env
