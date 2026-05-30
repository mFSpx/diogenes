# DARWIN HAMMER — match 22, survivor 2
# gen: 2
# parent_a: geometric_product.py (gen0)
# parent_b: hybrid_model_vram_scheduler_ttt_linear_m11_s1.py (gen1)
# born: 2026-05-29T23:22:54Z

"""Hybrid GA‑TTT VRAM Scheduler.

Parents:
- geometric_product.py (Clifford geometric product on multivectors)
- hybrid_model_vram_scheduler_ttt_linear_m11_s1.py (TTT‑Linear model + VRAM scheduler)

Mathematical bridge:
Both parents rely on a bilinear map.  In Clifford algebra the geometric
product `a * b` is bilinear and, when restricted to grade‑1 blades, is
exactly the matrix‑vector multiplication `A @ x`.  We therefore embed the
TTT‑Linear weight matrix `W` in a GA‑rotor `R`.  Each input vector `x`
is first rotated by `R` via the sandwich product `R * x * ~R` (where `~R`
is the reverse).  The rotated vector is fed to the usual TTT update.
Simultaneously the rotor itself is updated by a gradient step derived from
the same loss, using the bivector `x ∧ (Wx−x)` as a generator of an
infinitesimal rotation.  The VRAM scheduler from the second parent
decides whether the full learning rate or a reduced one is applied.

The module provides:
- `apply_rotor(R, x)` – rotate a Euclidean vector with a rotor.
- `ttt_ga_forward(W, R, x, eta_w, eta_r)` – one‑step hybrid update.
- `hybrid_ttt_ga_vram(x_seq, ...)` – sequence‑level processing with VRAM
  budgeting.

All code is pure Python, NumPy‑based and conforms to the required imports.
"""

from __future__ import annotations
import math
import random
import sys
import pathlib
import json
import os
import numpy as np

# ---------------------------------------------------------------------------
# Clifford algebra utilities (excerpt from geometric_product.py)
# ---------------------------------------------------------------------------

def _blade_sign(indices):
    """Return (sorted_blade, sign) after bubble‑sorting index list.

    Duplicates cancel because e_i*e_i = 1.
    """
    lst = list(indices)
    sign = 1
    n = len(lst)
    i = 0
    while i < n:
        j = 0
        while j < n - 1 - i:
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                # remove the pair
                del lst[j:j + 2]
                n -= 2
                i = -1  # restart outer loop
                break
            j += 1
        i += 1
    return lst, sign


def _multiply_blades(blade_a, blade_b):
    """Multiply two basis blades (frozenset of indices)."""
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


class Multivector:
    """Clifford multivector represented as a dict {blade: coeff}."""

    def __init__(self, components: dict[frozenset[int], float], n: int):
        self.components = {k: float(v) for k, v in components.items() if abs(v) > 1e-15}
        self.n = int(n)

    # ------------------------------------------------------------------
    # Grade extraction helpers
    # ------------------------------------------------------------------

    def grade(self, k: int) -> "Multivector":
        return Multivector(
            {b: c for b, c in self.components.items() if len(b) == k},
            self.n,
        )

    def scalar_part(self) -> float:
        return self.components.get(frozenset(), 0.0)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        if not self.components:
            return "Multivector(0)"
        terms = []
        for blade, coef in sorted(self.components.items(),
                                  key=lambda x: (len(x[0]), sorted(x[0]))):
            if blade:
                label = "e" + "".join(str(i) for i in sorted(blade))
            else:
                label = "1"
            terms.append(f"{coef:+.4g}*{label}")
        return "Multivector(" + " ".join(terms) + ")"


    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------

    def __add__(self, other: "Multivector") -> "Multivector":
        result = dict(self.components)
        for b, c in other.components.items():
            result[b] = result.get(b, 0.0) + c
        return Multivector(result, self.n)

    def __sub__(self, other: "Multivector") -> "Multivector":
        return self + (-other)

    def __neg__(self) -> "Multivector":
        return Multivector({b: -c for b, c in self.components.items()}, self.n)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Multivector({b: c * other for b, c in self.components.items()}, self.n)
        if isinstance(other, Multivector):
            return geometric_product(self, other)
        raise TypeError("Unsupported multiplication")

    __rmul__ = __mul__


def geometric_product(a: Multivector, b: Multivector) -> Multivector:
    """Full Clifford (geometric) product."""
    result: dict[frozenset[int], float] = {}
    for ba, ca in a.components.items():
        for bb, cb in b.components.items():
            bout, s = _multiply_blades(ba, bb)
            result[bout] = result.get(bout, 0.0) + s * ca * cb
    n = max(a.n, b.n)
    return Multivector(result, n)


def reverse(a: Multivector) -> Multivector:
    """Reverse (grade involution)."""
    res = {}
    for b, c in a.components.items():
        k = len(b)
        sign = -1 if (k % 4) in (2, 3) else 1
        res[b] = c * sign
    return Multivector(res, a.n)


def outer_product(a: Multivector, b: Multivector) -> Multivector:
    """Wedge product a ∧ b = (ab - ba)/2."""
    return (geometric_product(a, b) - geometric_product(b, a)) * 0.5


# ---------------------------------------------------------------------------
# Helper to embed a NumPy vector as a grade‑1 multivector
# ---------------------------------------------------------------------------

def vector_to_mv(v: np.ndarray, n: int) -> Multivector:
    """Convert a Euclidean vector to a grade‑1 multivector."""
    comps = {}
    for idx, val in enumerate(v):
        if abs(val) > 1e-15:
            comps[frozenset({idx})] = float(val)
    return Multivector(comps, n)


def mv_to_vector(mv: Multivector) -> np.ndarray:
    """Extract the grade‑1 part of a multivector as a NumPy array."""
    vec = np.zeros(mv.n)
    for blade, coef in mv.components.items():
        if len(blade) == 1:
            idx = next(iter(blade))
            vec[idx] = coef
    return vec


# ---------------------------------------------------------------------------
# VRAM scheduler (unchanged core from parent B)
# ---------------------------------------------------------------------------

def plan_dual_engine_residency(payload=None, state=None, include_gpu=True):
    """Simple VRAM budget decision."""
    payload = payload or {}
    state = state or {}
    gpu = gpu_memory() if include_gpu else {"status": "skipped"}
    observed_total = int(gpu.get("total_mb") or 4096) if isinstance(gpu, dict) else 4096
    budget = min(4096, observed_total) if observed_total else 4096
    resident_gpu_mb = 1250 + 1200
    decision = "allow" if resident_gpu_mb <= budget else "defer"
    headroom = max(0, budget - resident_gpu_mb - 512)
    return decision, headroom


def gpu_memory():
    """Placeholder for GPU memory query."""
    return {"total_mb": 4096, "used_mb": 1024, "free_mb": 3072}


# ---------------------------------------------------------------------------
# Hybrid operations
# ---------------------------------------------------------------------------

def apply_rotor(R: Multivector, x: np.ndarray) -> np.ndarray:
    """Rotate Euclidean vector `x` by rotor `R` using the sandwich product.

    Returns the rotated vector as a NumPy array.
    """
    n = len(x)
    x_mv = vector_to_mv(x, n)
    rotated_mv = geometric_product(geometric_product(R, x_mv), reverse(R))
    return mv_to_vector(rotated_mv)


def ttt_ga_forward(W: np.ndarray,
                   R: Multivector,
                   x: np.ndarray,
                   eta_w: float = 0.01,
                   eta_r: float = 0.001) -> tuple[np.ndarray, Multivector]:
    """One hybrid step.

    1. Rotate input: x' = R * x * ~R
    2. TTT update on weight matrix using x'
    3. Rotor update driven by the bivector x' ∧ (W x' - x')
    """
    # 1. Rotate
    x_rot = apply_rotor(R, x)                     # shape (d_in,)

    # 2. TTT update (same as parent B)
    pred = W @ x_rot
    residual = pred - x_rot                       # shape (d_out,)
    grad_W = 2.0 * np.outer(residual, x_rot)      # (d_out, d_in)
    W_new = W - eta_w * grad_W

    # 3. Rotor gradient (bivector generator)
    # Build multivectors for residual and rotated input
    res_mv = vector_to_mv(residual, W.shape[0])   # grade‑1 in output space
    # Embed residual back into the input space by truncating/padding if needed
    # For simplicity we treat both spaces as same dimension (common case)
    if W.shape[0] != W.shape[1]:
        # Pad/rescale to input dimension
        dim = max(W.shape[0], W.shape[1])
        pad_res = np.zeros(dim)
        pad_res[:len(residual)] = residual
        res_mv = vector_to_mv(pad_res, dim)
        x_mv = vector_to_mv(np.pad(x_rot, (0, dim - len(x_rot))), dim)
    else:
        x_mv = vector_to_mv(x_rot, W.shape[0])

    biv = outer_product(x_mv, res_mv)            # bivector (grade‑2)
    # Infinitesimal rotation: R_new ≈ R - eta_r * (biv * R)
    R_new = R - eta_r * geometric_product(biv, R)

    # Re‑normalize rotor to keep it close to a unit rotor
    # Normalization via magnitude sqrt(R * ~R).scalar_part()
    norm_sq = (geometric_product(R_new, reverse(R_new)).scalar_part())
    if norm_sq > 0:
        R_new = R_new * (1.0 / math.sqrt(norm_sq))

    return W_new, R_new


def hybrid_ttt_ga_vram(x_seq: np.ndarray,
                       W0: np.ndarray | None = None,
                       R0: Multivector | None = None,
                       eta_w: float = 0.01,
                       eta_r: float = 0.001,
                       d_model: int | None = None):
    """Process a token sequence with the hybrid GA‑TTT + VRAM scheduler.

    Returns:
        H  – hidden states after each step (shape (T, d_out))
        Wf – final weight matrix
        Rf – final rotor
    """
    x_seq = np.asarray(x_seq, dtype=float)
    T, d_in = x_seq.shape

    # Initialise weight matrix
    if W0 is None:
        d_out = d_model if d_model is not None else d_in
        rng = np.random.default_rng(0)
        W = rng.standard_normal((d_out, d_in)) * 0.01
    else:
        W = np.array(W0, dtype=float)

    # Initialise rotor as identity (R = 1)
    if R0 is None:
        R = Multivector({frozenset(): 1.0}, n=d_in)
    else:
        R = R0

    H = np.empty((T, W.shape[0]), dtype=float)

    for t in range(T):
        decision, _ = plan_dual_engine_residency()
        if decision == "allow":
            W, R = ttt_ga_forward(W, R, x_seq[t], eta_w=eta_w, eta_r=eta_r)
        else:
            # Reduced learning rates under VRAM pressure
            W, R = ttt_ga_forward(W, R, x_seq[t],
                                  eta_w=eta_w * 0.5,
                                  eta_r=eta_r * 0.5)
        # Produce hidden state after the update
        H[t] = W @ apply_rotor(R, x_seq[t])

    return H, W, R


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    T = 15
    d = 6
    # Generate a slowly varying sinusoidal sequence
    t_vals = np.linspace(0, 2 * np.pi, T)
    x_seq = np.stack([np.sin(t_vals + k * 0.3) for k in range(d)], axis=1)
    x_seq += rng.standard_normal(x_seq.shape) * 0.02

    H, W_final, R_final = hybrid_ttt_ga_vram(
        x_seq,
        eta_w=0.02,
        eta_r=0.001,
        d_model=d
    )

    print("Hybrid GA‑TTT VRAM scheduler smoke test")
    print(f"Hidden states shape: {H.shape}")
    print(f"Final weight matrix shape: {W_final.shape}")
    print(f"Final rotor (scalar part ≈ {R_final.scalar_part():.4f})")
    print("First hidden state:", H[0])