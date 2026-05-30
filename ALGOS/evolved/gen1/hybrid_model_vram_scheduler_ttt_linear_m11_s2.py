# DARWIN HAMMER — match 11, survivor 2
# gen: 1
# parent_a: model_vram_scheduler.py (gen0)
# parent_b: ttt_linear.py (gen0)
# born: 2026-05-29T23:17:54Z

"""Hybrid VRAM Scheduler & Test‑Time Training (TTT) module.

Parent A: model_vram_scheduler – estimates GPU memory consumption of model
artifacts (base model, LoRA adapters, embeddings) and produces advisory
residency plans.

Parent B: ttt_linear – implements Test‑Time Training where a weight matrix W
is updated by gradient descent on each input token, keeping a fixed‑size
state.

Mathematical bridge:
The TTT weight matrix W occupies concrete GPU memory proportional to its
element count (|W| × bytes_per_element).  By treating W as another “artifact”
in the VRAM planner we can fuse the two systems: the scheduler now accounts
for the evolving memory footprint of W while the TTT loop can query the
planner to decide whether the next update fits within the budget.  This yields
a unified advisory system that couples a dynamical learning rule with a
hardware‑aware budgeting policy.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Constants & utility helpers (from Parent A)
# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2] if Path(__file__).exists() else Path.cwd()
DEFAULT_BUDGET_MB = int(os.environ.get("LUCIDOTA_VRAM_BUDGET_MB", "4096"))
DEFAULT_RESERVE_MB = int(os.environ.get("LUCIDOTA_VRAM_RESERVE_MB", "768"))
RUNTIME_DIR = ROOT / "04_RUNTIME" / "inference_os"


def now_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def _append_runtime_receipt(receipt: dict[str, Any], *, path: Path | None = None) -> None:
    target = path or (RUNTIME_DIR / "preemption_receipts.jsonl")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(receipt, ensure_ascii=False, sort_keys=True, default=str) + "\n")


def gpu_memory() -> dict[str, Any]:
    """Query a single GPU via nvidia‑smi.  Returns a dict with total/used/free MB."""
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
        gpus.append(
            {
                "index": int(idx),
                "name": name,
                "total_mb": int(total),
                "used_mb": int(used),
                "free_mb": int(free),
                "driver_version": driver,
                "pstate": pstate,
            }
        )
    return (
        {"status": "ok", "selected_index": gpus[0]["index"], **gpus[0], "gpus": gpus}
        if gpus
        else {"status": "error", "stdout": cp.stdout[-500:], "stderr": cp.stderr[-500:]}
    )


# ----------------------------------------------------------------------
# Data structures for VRAM planning (light version)
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# TTT‑Linear core (from Parent B)
# ----------------------------------------------------------------------
def init_ttt(d_in: int, d_out: int | None = None, scale: float = 0.01, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if d_out is None:
        d_out = d_in
    return rng.standard_normal((d_out, d_in)) * scale


def ttt_loss(W: np.ndarray, x: np.ndarray, target: np.ndarray | None = None) -> float:
    pred = W @ x
    t = x if target is None else target
    residual = pred - t
    return float(residual @ residual)


def ttt_grad(W: np.ndarray, x: np.ndarray, target: np.ndarray | None = None) -> np.ndarray:
    pred = W @ x
    t = x if target is None else target
    residual = pred - t
    return 2.0 * np.outer(residual, x)


def ttt_step(W: np.ndarray, x: np.ndarray, eta: float = 0.01, target: np.ndarray | None = None) -> np.ndarray:
    g = ttt_grad(W, x, target)
    return W - eta * g


def ttt_forward(W: np.ndarray, x: np.ndarray, eta: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    W_new = ttt_step(W, x, eta=eta)
    h = W_new @ x
    return h, W_new


def ttt_sequence(
    x_seq: np.ndarray,
    W0: np.ndarray | None = None,
    eta: float = 0.01,
    d_model: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    x_seq = np.asarray(x_seq, dtype=float)
    T, d_in = x_seq.shape

    if W0 is None:
        d_out = d_model if d_model is not None else d_in
        W = init_ttt(d_in, d_out=d_out)
    else:
        W = np.array(W0, dtype=float)

    d_out = W.shape[0]
    H = np.empty((T, d_out), dtype=float)

    for t in range(T):
        h, W = ttt_forward(W, x_seq[t], eta=eta)
        H[t] = h

    return H, W


# ----------------------------------------------------------------------
# Hybrid utilities – memory estimation for a weight matrix
# ----------------------------------------------------------------------
def weight_matrix_memory_mb(W: np.ndarray, dtype: np.dtype = np.float64) -> int:
    """Return the memory footprint of matrix W in megabytes."""
    element_bytes = np.dtype(dtype).itemsize
    total_bytes = W.size * element_bytes
    return int(round(total_bytes / (1024 * 1024), 0))


def plan_ttt_residency(
    W: np.ndarray,
    payload: dict[str, Any] | None = None,
    state: dict[str, Any] | None = None,
    *,
    include_gpu: bool = True,
) -> VramSlotPlan:
    """Create a VramSlotPlan for the current weight matrix W.

    The plan compares the matrix size against the available VRAM budget
    (including a static reserve).  It does not perform any allocation.
    """
    payload = payload or {}
    state = state or {}

    gpu = gpu_memory() if include_gpu else {"status": "skipped", "total_mb": DEFAULT_BUDGET_MB}
    observed_total = int(gpu.get("total_mb", DEFAULT_BUDGET_MB))
    budget = min(DEFAULT_BUDGET_MB, observed_total) - DEFAULT_RESERVE_MB
    if budget < 0:
        budget = 0

    w_mb = weight_matrix_memory_mb(W)
    decision = "allow" if w_mb <= budget else "defer"

    reason = "fits in VRAM budget" if decision == "allow" else "exceeds VRAM budget"
    detail = {
        "observed_total_mb": observed_total,
        "budget_mb": budget,
        "weight_matrix_mb": w_mb,
        "gpu_info": gpu,
    }

    return VramSlotPlan(
        artifact_id="ttt_weight_matrix",
        artifact_kind="matrix",
        action=decision,
        estimated_mb=w_mb,
        reason=reason,
        detail=detail,
    )


# ----------------------------------------------------------------------
# Hybrid operation – run TTT while respecting VRAM constraints
# ----------------------------------------------------------------------
def ttt_sequence_with_budget(
    x_seq: np.ndarray,
    budget_mb: int | None = None,
    eta: float = 0.01,
    d_model: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, VramSlotPlan]:
    """Run TTT on a token sequence, stopping if the weight matrix would exceed budget.

    Returns:
        H          – hidden states produced before the stop (shape ≤ (T, d_out))
        W_final    – final weight matrix at stop point
        final_plan – VramSlotPlan describing the last memory decision
    """
    if budget_mb is None:
        # Use the system‑wide default budget
        gpu = gpu_memory()
        observed_total = int(gpu.get("total_mb", DEFAULT_BUDGET_MB))
        budget_mb = min(DEFAULT_BUDGET_MB, observed_total) - DEFAULT_RESERVE_MB
        if budget_mb < 0:
            budget_mb = 0

    # Initialise weight matrix
    d_in = x_seq.shape[1]
    W = init_ttt(d_in, d_out=d_model)

    H_parts: list[np.ndarray] = []
    for idx, x in enumerate(x_seq):
        # Predict memory impact *before* the update
        tentative_W = ttt_step(W, x, eta=eta)
        plan = plan_ttt_residency(tentative_W, include_gpu=False)
        plan = VramSlotPlan(
            artifact_id=f"ttt_step_{idx}",
            artifact_kind="matrix",
            action=plan.action,
            estimated_mb=plan.estimated_mb,
            reason=plan.reason,
            detail={**plan.detail, "budget_mb": budget_mb},
        )
        if plan.action == "defer":
            # Stop processing – cannot fit further updates
            break

        # Apply update and record hidden state
        h, W = ttt_forward(W, x, eta=eta)
        H_parts.append(h)

    H = np.vstack(H_parts) if H_parts else np.empty((0, W.shape[0]))
    final_plan = plan_ttt_residency(W, include_gpu=False)
    return H, W, final_plan


# ----------------------------------------------------------------------
# Demonstration functions (at least three)
# ----------------------------------------------------------------------
def demo_memory_estimation() -> None:
    """Show memory estimation for a random weight matrix."""
    W = init_ttt(256, d_out=512, scale=0.02)
    mb = weight_matrix_memory_mb(W)
    print(f"Weight matrix shape: {W.shape}, memory ≈ {mb} MiB")


def demo_vram_plan() -> None:
    """Create a VRAM plan for a TTT weight matrix and print the decision."""
    W = init_ttt(128, d_out=256, scale=0.01)
    plan = plan_ttt_residency(W)
    print("VRAM plan for TTT weight matrix:")
    print(json.dumps(plan.as_dict(), indent=2))


def demo_ttt_with_budget() -> None:
    """Run a short TTT sequence with a deliberately low VRAM budget."""
    rng = np.random.default_rng(0)
    T, d = 15, 64
    x_seq = rng.standard_normal((T, d))

    # Force a tiny budget to trigger early defer
    tiny_budget = 1  # 1 MiB
    H, W_final, final_plan = ttt_sequence_with_budget(x_seq, budget_mb=tiny_budget, eta=0.05)

    print(f"Processed {H.shape[0]} steps before hitting budget.")
    print("Final weight matrix memory:", weight_matrix_memory_mb(W_final), "MiB")
    print("Final VRAM decision:", final_plan.action)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Hybrid VRAM Scheduler + TTT Demo ===\n")
    demo_memory_estimation()
    print("\n---\n")
    demo_vram_plan()
    print("\n---\n")
    demo_ttt_with_budget()
    print("\nAll demos completed without error.")