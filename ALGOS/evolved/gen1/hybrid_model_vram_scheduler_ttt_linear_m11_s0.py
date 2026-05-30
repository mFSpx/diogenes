# DARWIN HAMMER — match 11, survivor 0
# gen: 1
# parent_a: model_vram_scheduler.py (gen0)
# parent_b: ttt_linear.py (gen0)
# born: 2026-05-29T23:17:54Z

from __future__ import annotations
import numpy as np
import os
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
import math
import random
import sys

"""
This module fuses the mathematical structures of the model_vram_scheduler and ttt_linear algorithms.
The model_vram_scheduler is used for advising VRAM and LoRA preemption planning, while ttt_linear is a 
test-time training algorithm with linear hidden states. The mathematical bridge between these two 
algorithms lies in the use of matrix operations. In ttt_linear, the weight matrix W is updated 
recurrently using gradient descent, while in model_vram_scheduler, the advisory planning involves 
estimating the resident GPU memory and making decisions based on the available VRAM. This fusion 
module integrates these two concepts by using the ttt_linear weight matrix updates as a 
representation of the dynamic changes in the VRAM usage, and incorporating the model_vram_scheduler 
decisions into the ttt_linear update rules.
"""

ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = ROOT / "04_RUNTIME" / "inference_os"
DEFAULT_BUDGET_MB = 4096
DEFAULT_RESERVE_MB = 768

@dataclass(frozen=True)
class VramSlotPlan:
    artifact_id: str
    artifact_kind: str
    action: str
    estimated_mb: int
    reason: str
    detail: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

def gpu_memory() -> dict[str, Any]:
    if not shutil.which("nvidia-smi"):
        return {"status": "missing", "message": "nvidia-smi not found"}
    cp = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,memory.free,driver_version,pstate",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=10,
    )
    if cp.returncode != 0 or not cp.stdout.strip():
        return {"status": "error", "stderr": cp.stderr[-500:]}
    gpus: list[dict[str, Any]] = []
    for line in cp.stdout.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 7:
            continue
        idx, name, total, used, free, driver, pstate = parts[:7]
        gpus.append({"index": int(idx), "name": name, "total_mb": int(total), "used_mb": int(used), "free_mb": int(free), "driver_version": driver, "pstate": pstate})
    return {"status": "ok", "selected_index": gpus[0]["index"], **gpus[0], "gpus": gpus} if gpus else {"status": "error", "stdout": cp.stdout[-500:], "stderr": cp.stderr[-500:]}

def init_ttt(d_in, d_out=None, scale=0.01, seed=0):
    rng = np.random.default_rng(seed)
    if d_out is None:
        d_out = d_in
    return rng.standard_normal((d_out, d_in)) * scale

def ttt_loss(W, x, target=None):
    pred = W @ x
    t = x if target is None else target
    residual = pred - t
    return float(residual @ residual)

def ttt_grad(W, x, target=None):
    pred = W @ x
    t = x if target is None else target
    residual = pred - t
    return 2.0 * np.outer(residual, x)

def ttt_step(W, x, eta=0.01, target=None):
    g = ttt_grad(W, x, target=target)
    return W - eta * g

def ttt_forward(W, x, eta=0.01):
    W_new = ttt_step(W, x, eta=eta)
    h = W_new @ x
    return h, W_new

def plan_dual_engine_residency(W, payload=None, state=None, include_gpu=True):
    payload = payload or {}
    state = state or {}
    gpu = gpu_memory() if include_gpu else {"status": "skipped"}
    observed_total = int(gpu.get("total_mb") or DEFAULT_BUDGET_MB) if isinstance(gpu, dict) else DEFAULT_BUDGET_MB
    budget = min(DEFAULT_BUDGET_MB, observed_total) if observed_total else DEFAULT_BUDGET_MB
    resident_gpu_mb = 1250 + 1200
    requested_adapters = 128
    adapter_headroom_mb = max(0, budget - resident_gpu_mb - 512)
    decision = "allow" if resident_gpu_mb <= budget else "defer"
    if decision == "allow":
        _, W_new = ttt_forward(W, np.array([requested_adapters]), eta=0.01)
        return W_new
    else:
        return W

def update_vram(W, d_in, d_out=None, scale=0.01, seed=0):
    W_new = init_ttt(d_in, d_out=d_out, scale=scale, seed=seed)
    _, W_new = ttt_forward(W_new, np.array([W.shape[0], W.shape[1]]), eta=0.01)
    return W_new

def hybrid_fusion(W, x, target=None):
    h, W_new = ttt_forward(W, x, eta=0.01)
    vram_plan = VramSlotPlan(
        artifact_id="fusion",
        artifact_kind="hybrid",
        action="update",
        estimated_mb=W_new.shape[0] * W_new.shape[1],
        reason="ttt_linear update",
        detail={},
    )
    return h, W_new, vram_plan

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    d_in = 8
    d_out = 8
    W = init_ttt(d_in, d_out=d_out)
    x = np.random.rand(d_in)
    h, W_new, vram_plan = hybrid_fusion(W, x)
    print(f"Initial W norm: {np.linalg.norm(W)}")
    print(f"Updated W norm: {np.linalg.norm(W_new)}")
    print(f"Hybrid fusion output: {h}")
    print(f"VRAM plan: {vram_plan.as_dict()}")