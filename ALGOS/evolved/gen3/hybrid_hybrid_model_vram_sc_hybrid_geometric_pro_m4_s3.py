# DARWIN HAMMER — match 4, survivor 3
# gen: 3
# parent_a: hybrid_model_vram_scheduler_ttt_linear_m11_s3.py (gen1)
# parent_b: hybrid_geometric_product_hybrid_model_vram_sc_m22_s2.py (gen2)
# born: 2026-05-29T23:25:08Z

"""Hybrid GA‑TTT VRAM Scheduler

Parents:
- `model_vram_scheduler.py` (VRAM budgeting, runtime receipts, GPU query)
- `ttt_linear.py` (linear weight matrix update, TTT‑style learning)
- `geometric_product.py` (Clifford/Geometric product, rotor mechanics)

Mathematical bridge:
The TTT linear map `y = W @ x` is a bilinear operation.  In a 3‑D
Clifford algebra the geometric product restricted to grade‑1 vectors is
exactly a rotation implemented by a rotor `R`.  Using the quaternion
isomorphism we embed the weight matrix `W` in a rotor `R` and replace the
linear map by the sandwich product  

    y = R * x * ~R   ⇔   y = quat_mul(R, quat_mul([0, *x], quat_conj(R)))[1:]

The rotor itself is updated by an infinitesimal rotation generated from
the bivector `x ∧ (y−x)`, which in 3‑D coincides with the cross product
`x × (y−x)`.  The VRAM scheduler from the first parent decides whether
the full learning rates `(η_w, η_r)` or a reduced pair are applied
depending on the current free GPU memory.

The module therefore provides:
- VRAM utilities (`gpu_memory`, `_append_runtime_receipt`, `budgeted_lr`)
- Quaternion‑based GA rotor utilities (`quat_mul`, `quat_conj`,
  `apply_rotor`, `rotor_from_axis_angle`)
- Hybrid update step (`ttt_ga_forward`)
- Sequence‑level processing with VRAM awareness (`hybrid_ttt_ga_vram`)
"""

from __future__ import annotations

import json
import math
import os
import random
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Tuple

import numpy as np

# ----------------------------------------------------------------------
# VRAM‑related helpers (derived from parent A)
# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2] if Path(__file__).exists() else Path.cwd()
RUNTIME_DIR = ROOT / "04_RUNTIME" / "inference_os"
DEFAULT_BUDGET_MB = int(os.environ.get("LUCIDOTA_VRAM_BUDGET_MB", "4096"))
DEFAULT_RESERVE_MB = int(os.environ.get("LUCIDOTA_VRAM_RESERVE_MB", "768"))


def now_z() -> str:
    """Current UTC timestamp in ISO‑8601 with trailing “Z”."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _rel(path: Path | str) -> str:
    """Return a path relative to the project root, falling back to `str`."""
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def _append_runtime_receipt(receipt: dict[str, Any], *, path: Path | None = None) -> None:
    """Append a JSON‑L line to the runtime receipt file."""
    target = path or (RUNTIME_DIR / "preemption_receipts.jsonl")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(receipt, ensure_ascii=False, sort_keys=True, default=str) + "\n")


def gpu_memory() -> dict[str, Any]:
    """Query a single GPU via nvidia‑smi.  Returns total/used/free in MB."""
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
    # Use only the first GPU (most common in single‑GPU workloads)
    line = cp.stdout.strip().splitlines()[0]
    idx, name, total, used, free, driver, pstate = [x.strip() for x in line.split(",")][:7]
    return {
        "status": "ok",
        "index": int(idx),
        "name": name,
        "total_mb": int(total),
        "used_mb": int(used),
        "free_mb": int(free),
        "driver_version": driver,
        "pstate": pstate,
    }


def budgeted_lr(
    full_lr: Tuple[float, float],
    reduced_lr: Tuple[float, float],
    budget_mb: int = DEFAULT_BUDGET_MB,
    reserve_mb: int = DEFAULT_RESERVE_MB,
) -> Tuple[float, float]:
    """Choose learning rates based on current free GPU memory.

    If free memory < (budget - reserve) we fall back to the reduced rates.
    """
    mem = gpu_memory()
    if mem.get("status") != "ok":
        # If we cannot query the GPU, assume we have enough memory.
        return full_lr
    free = mem["free_mb"]
    if free < (budget_mb - reserve_mb):
        return reduced_lr
    return full_lr


# ----------------------------------------------------------------------
# Quaternion / GA rotor utilities (derived from parent B)
# ----------------------------------------------------------------------


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two quaternions (w, x, y, z)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def quat_conj(q: np.ndarray) -> np.ndarray:
    """Conjugate of a quaternion (w, -x, -y, -z)."""
    w, x, y, z = q
    return np.array([w, -x, -y, -z], dtype=np.float64)


def quat_norm(q: np.ndarray) -> float:
    return float(np.linalg.norm(q))


def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = quat_norm(q)
    if n == 0:
        raise ValueError("Zero‑norm quaternion cannot be normalized")
    return q / n


def rotor_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Create a unit rotor (quaternion) representing rotation about `axis`."""
    axis = np.asarray(axis, dtype=np.float64)
    if np.linalg.norm(axis) == 0:
        return np.array([1.0, 0.0, 0.0, 0.0])  # identity
    axis = axis / np.linalg.norm(axis)
    half = angle / 2.0
    return quat_normalize(np.concatenate([[math.cos(half)], math.sin(half) * axis]))


def apply_rotor(R: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Rotate a Euclidean 3‑vector `x` by rotor `R` via the sandwich product."""
    x_quat = np.concatenate([[0.0], x.astype(np.float64)])
    return quat_mul(R, quat_mul(x_quat, quat_conj(R)))[1:]


# ----------------------------------------------------------------------
# Hybrid TTT‑GA update (core of the fusion)
# ----------------------------------------------------------------------


def ttt_ga_forward(
    W: np.ndarray,
    R: np.ndarray,
    x: np.ndarray,
    eta_w: float,
    eta_r: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform a single hybrid update step.

    1. Rotate the input vector `x` using the current rotor `R`.
    2. Apply the linear map `W` to the rotated vector → `y_lin = W @ x_rot`.
    3. Treat `y_lin` as the GA‑rotated output `y = R * x * ~R`.
       (Both are mathematically equivalent for a correctly‑trained rotor.)
    4. Compute an error `e = y_lin - x` (identity‑mapping target).
    5. Update the weight matrix with a gradient descent step.
    6. Update the rotor using the bivector generator `b = x_rot × e`.

    Returns the updated `(W, R)`.
    """
    # 1. Rotate input
    x_rot = apply_rotor(R, x)

    # 2. Linear map
    y_lin = W @ x_rot

    # 3. Error (we aim for the identity map)
    e = y_lin - x

    # 4. Matrix gradient (outer product of error and rotated input)
    grad_W = np.outer(e, x_rot)
    W_new = W - eta_w * grad_W

    # 5. Rotor gradient: bivector generator = x_rot ∧ e  (cross product in 3‑D)
    biv = np.cross(x_rot, e)
    biv_norm = np.linalg.norm(biv)
    if biv_norm > 0:
        # infinitesimal rotation angle proportional to learning rate
        delta_angle = eta_r * biv_norm
        delta_axis = biv / biv_norm
        delta_R = rotor_from_axis_angle(delta_axis, delta_angle)
        R_new = quat_mul(delta_R, R)  # left‑multiply to apply new rotation
        R_new = quat_normalize(R_new)
    else:
        R_new = R.copy()

    return W_new, R_new


def hybrid_ttt_ga_vram(
    x_seq: Iterable[np.ndarray],
    dim: int,
    eta_w_full: float = 1e-3,
    eta_r_full: float = 1e-4,
    eta_w_red: float = 5e-4,
    eta_r_red: float = 5e-5,
    budget_mb: int = DEFAULT_BUDGET_MB,
    reserve_mb: int = DEFAULT_RESERVE_MB,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process a sequence of input vectors under VRAM‑aware learning rates.

    Parameters
    ----------
    x_seq : iterable of shape (dim,)
        Input vectors.
    dim : int
        Dimensionality of the vectors (must be 3 for the quaternion bridge).
    eta_*_full / eta_*_red : float
        Full and reduced learning rates for matrix and rotor.
    budget_mb / reserve_mb : int
        VRAM budget and safety reserve used by `budgeted_lr`.

    Returns
    -------
    (W, R) : final weight matrix and rotor.
    """
    if dim != 3:
        raise ValueError("Current quaternion implementation only supports 3‑D vectors")

    # Initialise a near‑identity linear map and rotor.
    W = np.eye(dim, dtype=np.float64)
    R = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)  # identity quaternion

    for step, x in enumerate(x_seq):
        # Choose learning rates based on VRAM availability.
        eta_w, eta_r = budgeted_lr(
            (eta_w_full, eta_r_full),
            (eta_w_red, eta_r_red),
            budget_mb,
            reserve_mb,
        )

        # Hybrid update.
        W, R = ttt_ga_forward(W, R, np.asarray(x, dtype=np.float64), eta_w, eta_r)

        # Log a minimal receipt for debugging / auditing.
        receipt = {
            "timestamp": now_z(),
            "step": step,
            "free_vram_mb": gpu_memory().get("free_mb", None),
            "eta_w": eta_w,
            "eta_r": eta_r,
            "rotor_norm": float(quat_norm(R)),
        }
        _append_runtime_receipt(receipt)

    return W, R


# ----------------------------------------------------------------------
# Minimal public API
# ----------------------------------------------------------------------


def apply_rotor_to_batch(R: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Rotate a batch of vectors `X` (shape (N,3)) with the same rotor `R`.
    Returns an array of the same shape.
    """
    return np.vstack([apply_rotor(R, x) for x in X])


def vram_aware_step(
    W: np.ndarray,
    R: np.ndarray,
    x: np.ndarray,
    full_lrs: Tuple[float, float],
    reduced_lrs: Tuple[float, float],
    budget_mb: int = DEFAULT_BUDGET_MB,
    reserve_mb: int = DEFAULT_RESERVE_MB,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Single step wrapper that decides the learning rates via VRAM budget,
    then calls :func:`ttt_ga_forward`.
    """
    eta_w, eta_r = budgeted_lr(full_lrs, reduced_lrs, budget_mb, reserve_mb)
    return ttt_ga_forward(W, R, x, eta_w, eta_r)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Generate a random walk of 50 3‑D vectors.
    np.random.seed(42)
    seq_len = 50
    dim = 3
    xs = [np.random.randn(dim) for _ in range(seq_len)]

    # Run the hybrid scheduler.
    final_W, final_R = hybrid_ttt_ga_vram(
        xs,
        dim,
        eta_w_full=1e-3,
        eta_r_full=2e-4,
        eta_w_red=5e-4,
        eta_r_red=1e-4,
        budget_mb=DEFAULT_BUDGET_MB,
        reserve_mb=DEFAULT_RESERVE_MB,
    )

    # Simple sanity checks.
    assert final_W.shape == (dim, dim)
    assert final_R.shape == (4,)

    # Rotate the last input with the learned rotor and compare to linear map.
    x_last = xs[-1]
    rotated = apply_rotor(final_R, x_last)
    linear = final_W @ apply_rotor(final_R, x_last)  # both use rotated input
    print("Final rotor norm:", np.linalg.norm(final_R))
    print("Sample rotated vector:", rotated)
    print("Sample linear‑map result:", linear)
    print("Smoke test completed successfully.")