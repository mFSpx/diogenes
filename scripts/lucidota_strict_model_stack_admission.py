#!/usr/bin/env python3
"""Admission gate for the strict local model stack.

This does not start models. It proves the local DIOGENES/CKDOG1 hot path is
under the RAM ceiling, checks the display/offload policy keeps graphics off the
NVIDIA model card, verifies model/launcher files, and writes a tiny env file for
`lucidota_start_strict_model_stack.sh`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from diogenes_memory_gate import MEMORY_CEILING_KB, run_gate as run_diogenes_gate  # noqa: E402

DEFAULT_RECEIPT_DIR = ROOT / "05_OUTPUTS" / "model_runtime"
DEFAULT_ENV_PATH = ROOT / "04_RUNTIME" / "inference_os" / "strict_model_stack_admission.env"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _env_int(env: dict[str, str], key: str, default: int) -> int:
    try:
        return int(env.get(key, str(default)))
    except Exception:
        return default


def _path_info(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(ROOT)) if path.is_absolute() and path.exists() else str(path),
        "exists": path.exists(),
        "executable": os.access(path, os.X_OK) if path.exists() else False,
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else 0,
    }


def _display_on_onboard_gpu(value: Any) -> bool:
    normalized = str(value or "").strip()
    return normalized in {"", "0", "pci-0000_00_02_0"}


def _nvidia_render_offload_disabled(value: Any) -> bool:
    return str(value or "").strip() in {"", "0"}


def build_strict_stack_plan(*, root: Path = ROOT, env: dict[str, str] | None = None) -> dict[str, Any]:
    env = dict(os.environ if env is None else env)
    root = Path(root)
    llama_cuda = root / "01_REPOS" / "llama.cpp" / "build-cuda" / "bin" / "llama-server"
    prism_llama = root / "01_REPOS" / "prismml_llama.cpp" / "build-cuda" / "bin" / "llama-server"
    services = [
        {
            "name": "deepseek_r1_qwen_1p5b_gpu",
            "role": "coder/reasoning additive lane",
            "port": 8080,
            "start_script": "scripts/lucidota_start_deepseek_llama.sh",
            "launcher": str(llama_cuda),
            "model_path": "03_VAULT/models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf",
            "device_lane": "nvidia_vram_model_only",
            "requested_vram_mb": _env_int(env, "LUCIDOTA_DEEPSEEK_VRAM_MB", 1500),
            "required": True,
        },
        {
            "name": "mamba7b_ram",
            "role": "CPU/RAM SSM long-context/db-watch lane",
            "port": 8081,
            "start_script": "scripts/lucidota_start_mamba_llama.sh",
            "launcher": str(llama_cuda),
            "model_path": "03_VAULT/models/tensorblock/Falcon3-Mamba-7B-Instruct-GGUF/Falcon3-Mamba-7B-Instruct-Q2_K.gguf",
            "device_lane": "system_ram_cpu",
            "requested_vram_mb": 0,
            "required": True,
        },
        {
            "name": "bonsai4b_ram",
            "role": "CPU/RAM 4B ternary Bonsai lane",
            "port": 8082,
            "start_script": "scripts/lucidota_start_bonsai_ternary_llama.sh",
            "launcher": str(prism_llama),
            "model_path": "03_VAULT/models/prism-ml/Ternary-Bonsai-4B-gguf/Ternary-Bonsai-4B-Q2_0.gguf",
            "device_lane": "system_ram_cpu",
            "requested_vram_mb": 0,
            "required": True,
        },
        {
            "name": "mamba7b_gpu_partial",
            "role": "preemptible partial-VRAM planner lane",
            "port": 8083,
            "start_script": "scripts/lucidota_start_mamba_gpu_partial.sh",
            "launcher": str(llama_cuda),
            "model_path": "03_VAULT/models/tensorblock/Falcon3-Mamba-7B-Instruct-GGUF/Falcon3-Mamba-7B-Instruct-Q2_K.gguf",
            "device_lane": "nvidia_vram_model_only_preemptible",
            "requested_vram_mb": _env_int(env, "LUCIDOTA_MAMBA_GPU_PARTIAL_VRAM_MB", 1200),
            "required": False,
        },
        {
            "name": "needle_swarm_6x",
            "role": "six tiny deterministic/tool-call router workers",
            "port_range": [8090, 8095],
            "start_script": "scripts/lucidota_start_needle_swarm.sh",
            "launcher": str(root / ".venv" / "bin" / "python"),
            "model_path": "03_VAULT/models/needle/needle.pkl",
            "device_lane": "system_ram_cpu",
            "requested_vram_mb": 0,
            "required": True,
            "count": 6,
        },
        {
            "name": "indy_reads_watcher",
            "role": "Indy_READs book/LoRA intake watcher",
            "start_script": "scripts/lucidota_start_indy_reads_watcher.sh",
            "launcher": str(root / "scripts" / "lucidota_indy_reads_watcher.py"),
            "model_path": "BOOKS/.indy_reads",
            "device_lane": "system_ram_cpu",
            "requested_vram_mb": 0,
            "required": True,
        },
    ]
    return {
        "schema": "lucidota.strict_model_stack.plan.v1",
        "generated_at": utc_now(),
        "vram_budget_mb": _env_int(env, "LUCIDOTA_VRAM_BUDGET_MB", 4096),
        "vram_reserve_mb": _env_int(env, "LUCIDOTA_VRAM_RESERVE_MB", 768),
        "floating_vram_min_mb": _env_int(env, "LUCIDOTA_FLOATING_VRAM_MIN_MB", 450),
        "display_policy": {
            "rule": "non-model graphics stay on onboard GPU; NVIDIA VRAM is model-only",
            "effective": {
                "DRI_PRIME": env.get("DRI_PRIME", "0"),
                "__NV_PRIME_RENDER_OFFLOAD": env.get("__NV_PRIME_RENDER_OFFLOAD", "0"),
            },
        },
        "diogenes": {
            "required": True,
            "memory_ceiling_kb": MEMORY_CEILING_KB,
            "gate_script": "scripts/diogenes_memory_gate.py",
        },
        "services": services,
        "external_rule": "LLMs are bounded processing lanes; deterministic graph/DB/math remain the controller.",
    }


def load_receipt(path: Path | str | None) -> dict[str, Any] | None:
    if not path:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        return {"passed": False, "blockers": ["diogenes_receipt_missing"], "receipt_path": str(p)}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        data.setdefault("receipt_path", str(p))
        return data
    except Exception as exc:
        return {"passed": False, "blockers": [f"diogenes_receipt_unreadable:{exc}"], "receipt_path": str(p)}


def latest_diogenes_receipt() -> Path | None:
    candidates = sorted((ROOT / "05_OUTPUTS" / "diogenes").glob("diogenes_memory_gate_*.json"), reverse=True)
    return candidates[0] if candidates else None


def _file_checks(plan: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for service in plan.get("services", []):
        for key in ("start_script", "launcher", "model_path"):
            value = service.get(key)
            if not value or ":inline:" in str(value):
                continue
            path = Path(value)
            if not path.is_absolute():
                path = ROOT / path
            required = bool(service.get("required", True))
            executable_expected = key in {"start_script", "launcher"} and path.suffix != ".py"
            info = _path_info(path)
            info.update({
                "service": service.get("name"),
                "field": key,
                "required": required,
                "executable_expected": executable_expected,
                "passed": (not required or info["exists"]) and (not executable_expected or info["executable"]),
            })
            checks.append(info)
    return checks


def evaluate_admission(
    plan: dict[str, Any],
    *,
    diogenes_receipt: dict[str, Any] | None,
    check_files: bool = True,
) -> dict[str, Any]:
    blockers: list[str] = []
    display = plan.get("display_policy", {}).get("effective", {})
    if not _display_on_onboard_gpu(display.get("DRI_PRIME", "0")) or not _nvidia_render_offload_disabled(display.get("__NV_PRIME_RENDER_OFFLOAD", "0")):
        blockers.append("display_not_forced_to_onboard_gpu")

    if not diogenes_receipt:
        blockers.append("diogenes_memory_gate_receipt_missing")
    elif not diogenes_receipt.get("passed"):
        blockers.append("diogenes_memory_gate_failed")

    requested_vram = sum(int(s.get("requested_vram_mb") or 0) for s in plan.get("services", []))
    budget = int(plan.get("vram_budget_mb") or 4096)
    reserve = max(int(plan.get("vram_reserve_mb") or 0), int(plan.get("floating_vram_min_mb") or 0))
    usable = max(0, budget - reserve)
    if requested_vram > usable:
        blockers.append("strict_stack_vram_budget_exceeded")

    checks = _file_checks(plan) if check_files else []
    missing_required = [c for c in checks if c.get("required") and not c.get("passed")]
    if missing_required:
        blockers.append("required_model_stack_file_missing")

    optional_gpu_requested = [s for s in plan.get("services", []) if str(s.get("device_lane", "")).endswith("preemptible") and int(s.get("requested_vram_mb") or 0) > 0]
    enable_mamba_gpu_partial = "1" if not any(b in blockers for b in ["strict_stack_vram_budget_exceeded", "display_not_forced_to_onboard_gpu"]) and optional_gpu_requested else "0"

    return {
        "schema": "lucidota.strict_model_stack.admission.v1",
        "generated_at": utc_now(),
        "passed": not blockers,
        "blockers": blockers,
        "plan": plan,
        "diogenes_receipt": diogenes_receipt,
        "file_checks": checks,
        "missing_required": missing_required,
        "vram": {
            "budget_mb": budget,
            "reserve_mb": reserve,
            "requested_mb": requested_vram,
            "usable_mb": usable,
            "floating_vram_min_mb": int(plan.get("floating_vram_min_mb") or 0),
        },
        "env": {
            "LUCIDOTA_STRICT_STACK_ADMITTED": "1" if not blockers else "0",
            "LUCIDOTA_ENABLE_MAMBA_GPU_PARTIAL": enable_mamba_gpu_partial,
            "DRI_PRIME": "0",
            "__NV_PRIME_RENDER_OFFLOAD": "0",
        },
    }


def write_env_file(path: Path, env: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Generated by scripts/lucidota_strict_model_stack_admission.py"]
    for key in sorted(env):
        value = str(env[key]).replace("'", "'\\''")
        lines.append(f"export {key}='{value}'")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_admission(
    *,
    diogenes_receipt_path: Path | str | None = None,
    run_diogenes: bool = False,
    receipt_dir: Path = DEFAULT_RECEIPT_DIR,
    env_path: Path = DEFAULT_ENV_PATH,
    check_files: bool = True,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    plan = build_strict_stack_plan(env=env)
    if run_diogenes:
        diogenes = run_diogenes_gate()
    else:
        receipt_path = Path(diogenes_receipt_path) if diogenes_receipt_path else latest_diogenes_receipt()
        diogenes = load_receipt(receipt_path)
    admission = evaluate_admission(plan, diogenes_receipt=diogenes, check_files=check_files)
    receipt_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = receipt_dir / f"strict_model_stack_admission_{stamp()}.json"
    admission["receipt_path"] = str(receipt_path)
    admission["env_path"] = str(env_path)
    receipt_path.write_text(json.dumps(admission, indent=2, sort_keys=True), encoding="utf-8")
    latest = receipt_dir / "strict_model_stack_admission_latest.json"
    latest.write_text(json.dumps(admission, indent=2, sort_keys=True), encoding="utf-8")
    write_env_file(env_path, admission["env"])
    return admission


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preflight/admission gate for LUCIDOTA strict local model stack.")
    parser.add_argument("--diogenes-receipt", type=Path)
    parser.add_argument("--run-diogenes-gate", action="store_true")
    parser.add_argument("--receipt-dir", type=Path, default=DEFAULT_RECEIPT_DIR)
    parser.add_argument("--env-path", type=Path, default=DEFAULT_ENV_PATH)
    parser.add_argument("--skip-file-checks", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    receipt = run_admission(
        diogenes_receipt_path=args.diogenes_receipt,
        run_diogenes=args.run_diogenes_gate,
        receipt_dir=args.receipt_dir,
        env_path=args.env_path,
        check_files=not args.skip_file_checks,
    )
    if args.json:
        print(json.dumps(receipt, sort_keys=True))
    print("STRICT_MODEL_STACK_ADMISSION=" + ("PASS" if receipt["passed"] else "FAIL"))
    print("RECEIPT_PATH=" + receipt["receipt_path"])
    print("ENV_PATH=" + receipt["env_path"])
    return 0 if receipt["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
