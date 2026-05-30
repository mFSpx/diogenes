# DARWIN HAMMER — match 11, survivor 3
# gen: 1
# parent_a: model_vram_scheduler.py (gen0)
# parent_b: ttt_linear.py (gen0)
# born: 2026-05-29T23:17:54Z

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
# Constants & utility helpers
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
# Data structures for VRAM planning
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
# TTT‑Linear core
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
    include_gpu: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Iterable[VramSlotPlan]]:
    x_seq = np.asarray(x_seq, dtype=float)
    T, d_in = x_seq.shape

    W0 = init_ttt(d_in, d_out=d_model) if d_model is not None else init_ttt(d_in)
    W = W0

    H = np.empty((T, W.shape[0]), dtype=float)
    plans = []

    for t in range(T):
        w_mb = weight_matrix_memory_mb(W)
        plan = plan_ttt_residency(W, include_gpu=include_gpu)
        plans.append(plan)

        if plan.action == "defer":
            raise MemoryError(f"VRAM budget exceeded at step {t}: {plan.detail['weight_matrix_mb']}MB > {plan.detail['budget_mb']}MB")

        h, W = ttt_forward(W, x_seq[t], eta=eta)
        H[t] = h

    return H, W, plans


def main():
    T = 100
    d_in = 512
    d_model = 2048
    x_seq = np.random.randn(T, d_in)

    try:
        H, W, plans = ttt_sequence_with_budget(x_seq, d_model=d_model)
        print(f"Final weight matrix size: {W.shape}")
        print(f"VRAM plans: {len(plans)}")
    except MemoryError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()