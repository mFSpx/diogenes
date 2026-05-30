# DARWIN HAMMER — match 1674, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_hybrid_worksh_hybrid_hybrid_path_s_m851_s0.py (gen4)
# parent_b: hybrid_hybrid_model_vram_sc_hybrid_geometric_pro_m4_s2.py (gen3)
# born: 2026-05-29T23:38:16Z

"""
Hybrid algorithm merging:

- Parent A: hybrid_hybrid_hybrid_worksh_hybrid_hybrid_path_s_m851_s0.py
  (weekday‑weight vectors, lead‑lag transform, B‑spline basis for path signatures)

- Parent B: hybrid_hybrid_model_vram_sc_hybrid_geometric_pro_m4_s2.py
  (VRAM‑aware scheduler and geometric product via rotor representation)

Mathematical bridge:
The B‑spline basis provides a smooth functional representation of a
lead‑lag transformed path.  The geometric product can be applied to the
coefficients of this representation, treating each coefficient vector as a
multivector and combining scalar (dot) and bivector (outer) parts.  The
VRAM scheduler supplies a gain γ that scales the geometric product according
to the memory footprint of the intermediate tensors, ensuring the hybrid
computation remains within available GPU/CPU memory.
"""

import math
import random
import sys
import pathlib
import datetime
import subprocess
import json
import os
from typing import Sequence, Tuple, Any

import numpy as np

# ----------------------------------------------------------------------
# Utilities from Parent A
# ----------------------------------------------------------------------
def _pct(value: float) -> float:
    """Round a float to six decimal places."""
    return round(float(value), 6)


def weekday_weight_vector(groups: Sequence[str], dow: int) -> np.ndarray:
    """
    Produce a probability‑like weight vector for a list of groups,
    modulated by the day‑of‑week (dow, 0=Monday … 6=Sunday).

    The weights are sinusoidally shifted by the weekday phase and then
    normalised to sum to one.
    """
    n = len(groups)
    if n == 0:
        raise ValueError("groups must contain at least one element")
    base_angles = np.arange(n) * (2.0 * math.pi) / n
    phase = (2.0 * math.pi) * (dow % 7) / 7.0
    amplitude = 0.2
    raw = 1.0 + amplitude * np.sin(base_angles + phase)
    weight_vec = raw / raw.sum()
    return weight_vec.astype(np.float64)


def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    """
    Lead‑lag transform: interleave (lead, lag) channels for causality encoding.
    Input shape: (T, d)
    Output shape: (2*T‑1, 2*d)
    """
    path = np.asarray(path, dtype=float)
    if path.ndim != 2:
        raise ValueError("path must be a 2‑D array (time × dimension)")
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t] = np.concatenate([path[t], path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out


def _cox_de_boor(x: np.ndarray, knots: np.ndarray, i: int, k: int) -> np.ndarray:
    """
    Recursive Cox‑de Boor evaluation of a single B‑spline basis function.
    Returns N_{i,k}(x) for all x.
    """
    if k == 0:
        return np.where((knots[i] <= x) & (x < knots[i + 1]), 1.0, 0.0)
    left = (x - knots[i]) / (knots[i + k] - knots[i] + 1e-12)
    right = (knots[i + k + 1] - x) / (knots[i + k + 1] - knots[i + 1] + 1e-12)
    return left * _cox_de_boor(x, knots, i, k - 1) + right * _cox_de_boor(x, knots, i + 1, k - 1)


def bspline_basis(x: np.ndarray, grid: np.ndarray, k: int = 3) -> np.ndarray:
    """
    Evaluate all B‑spline basis functions of order k on the points x.
    Returns an array of shape (len(x), n_basis) where n_basis = len(grid) - k - 1.
    """
    x = np.asarray(x, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)
    n_basis = len(grid) - k - 1
    if n_basis <= 0:
        raise ValueError("Grid too small for the requested spline order")
    B = np.empty((x.shape[0], n_basis), dtype=np.float64)
    for i in range(n_basis):
        B[:, i] = _cox_de_boor(x, grid, i, k)
    return B


# ----------------------------------------------------------------------
# Utilities from Parent B
# ----------------------------------------------------------------------
def now_z() -> str:
    """Current UTC timestamp in ISO‑8601 Z‑suffix format."""
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def gpu_memory() -> dict[str, Any]:
    """
    Query NVIDIA‑SMI for GPU memory statistics.
    Returns a dict with keys: status, total, used, free (in MiB) when possible.
    """
    if not pathlib.Path("/usr/bin/nvidia-smi").exists():
        return {"status": "missing", "message": "nvidia-smi not found"}
    try:
        cp = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=5,
        )
    except Exception as e:
        return {"status": "error", "message": str(e)}
    if cp.returncode != 0 or not cp.stdout.strip():
        return {"status": "error", "stderr": cp.stderr[:500]}
    totals = []
    for line in cp.stdout.strip().splitlines():
        try:
            total, used, free = map(int, line.split(","))
            totals.append({"total": total, "used": used, "free": free})
        except Exception:
            continue
    return {"status": "ok", "gpus": totals}


def vram_scheduler(required_bytes: int, safety_factor: float = 0.8) -> float:
    """
    Decide a scaling gain γ ∈ (0, 1] based on the required memory.
    If the required memory exceeds safety_factor * free_memory, γ is reduced.
    """
    mem_info = gpu_memory()
    if mem_info.get("status") != "ok":
        # Assume a generous CPU fallback (8 GiB)
        free = 8 * 1024 ** 3
    else:
        # Sum free memory across all GPUs and convert MiB → bytes
        free = sum(g["free"] for g in mem_info["gpus"]) * 1024 ** 2
    allowance = safety_factor * free
    if required_bytes <= allowance:
        return 1.0
    # Linear decay down to a minimum of 0.1
    gamma = max(0.1, allowance / required_bytes)
    return gamma


def geometric_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Simple geometric product for two vectors a, b ∈ ℝⁿ.
    Returns a multivector represented as a (n, n) matrix:
        G = a·b * I + a ⊗ b
    where I is the identity matrix and ⊗ denotes the outer product.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError("Vectors must have the same dimensionality")
    n = a.shape[0]
    dot = float(np.dot(a, b))
    outer = np.outer(a, b)
    return dot * np.eye(n) + outer


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def weighted_bspline_coefficients(
    path: np.ndarray,
    groups: Sequence[str],
    date: datetime.date,
    grid: np.ndarray,
    order: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    1. Compute a weekday‑dependent weight vector for `groups`.
    2. Apply the lead‑lag transform to `path`.
    3. Evaluate B‑spline basis on a normalized time axis.
    4. Return (coefficients, weight_vector) where coefficients are the
       projection of the transformed path onto the B‑spline basis.
    """
    # 1. weekday weight
    dow = date.weekday()  # Monday=0
    w = weekday_weight_vector(groups, dow)  # shape (len(groups),)

    # 2. lead‑lag transform
    ll_path = lead_lag_transform(path)  # shape (2*T‑1, 2*d)

    # 3. time axis for basis evaluation (scaled to [0, 1])
    T = ll_path.shape[0]
    t_axis = np.linspace(0.0, 1.0, T)

    # 4. B‑spline basis matrix
    B = bspline_basis(t_axis, grid, k=order)  # shape (T, n_basis)

    # 5. Least‑squares projection to obtain coefficients per dimension
    #    Solve B c ≈ ll_path for each column of ll_path.
    coeffs = np.linalg.lstsq(B, ll_path, rcond=None)[0]  # shape (n_basis, 2*d)

    # Apply the group weights to the coefficient matrix (broadcast over basis)
    weighted_coeffs = coeffs * w[:, np.newaxis, np.newaxis].sum(axis=0)  # simple broadcasting

    return weighted_coeffs, w


def vram_aware_geometric_layer(
    coeffs: np.ndarray,
    gamma: float = 1.0,
) -> np.ndarray:
    """
    Apply the geometric product pairwise across the basis dimension,
    scaling the result by the VRAM‑aware gain γ.

    Input shape: (n_basis, 2*d)
    Output shape: (n_basis, d, d) – a stack of geometric‑product matrices.
    """
    n_basis, dim = coeffs.shape
    if dim % 2 != 0:
        raise ValueError("Coefficient dimension must be even (lead‑lag concatenation).")
    d = dim // 2
    result = np.empty((n_basis, d, d), dtype=np.float64)
    for i in range(n_basis):
        # split into lead and lag halves
        lead = coeffs[i, :d]
        lag = coeffs[i, d:]
        # geometric product between lead and lag vectors
        gp = geometric_product(lead, lag) * gamma
        result[i] = gp
    return result


def hybrid_path_signature(
    path: np.ndarray,
    groups: Sequence[str],
    date: datetime.date,
    grid: np.ndarray,
    order: int = 3,
) -> np.ndarray:
    """
    Full hybrid pipeline:

    1. Build weighted B‑spline coefficients from the path.
    2. Estimate required memory and obtain a gain γ via the VRAM scheduler.
    3. Apply a VRAM‑aware geometric product layer to the coefficients.
    4. Return a flattened signature tensor ready for downstream models.
    """
    coeffs, _ = weighted_bspline_coefficients(path, groups, date, grid, order)

    # Estimate memory: coefficients + a placeholder for geometric product output
    required = coeffs.nbytes + coeffs.shape[0] * (coeffs.shape[1] // 2) ** 2 * 8
    gamma = vram_scheduler(required)

    geo_stack = vram_aware_geometric_layer(coeffs, gamma=gamma)

    # Flatten across basis dimension and matrix entries
    signature = geo_stack.reshape(-1)
    return signature


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic data
    T, d = 15, 4
    rng = np.random.default_rng(42)
    path = rng.normal(size=(T, d))

    groups = ("codex", "groq", "cohere", "local_models")
    today = datetime.date.today()
    # Uniform knot vector with extra padding for cubic B‑splines
    n_basis = 8
    order = 3
    knots = np.concatenate((
        np.zeros(order),
        np.linspace(0, 1, n_basis - order + 1),
        np.ones(order)
    ))

    sig = hybrid_path_signature(path, groups, today, knots, order=order)
    print(f"Hybrid signature shape: {sig.shape}")
    print(f"First 10 entries: {sig[:10]}")