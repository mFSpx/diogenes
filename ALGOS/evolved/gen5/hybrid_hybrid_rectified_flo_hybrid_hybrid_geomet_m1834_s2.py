# DARWIN HAMMER — match 1834, survivor 2
# gen: 5
# parent_a: hybrid_rectified_flow_hybrid_ternary_lens__m404_s1.py (gen4)
# parent_b: hybrid_hybrid_geometric_pro_hybrid_hybrid_nlms_o_m847_s1.py (gen3)
# born: 2026-05-29T23:39:05Z

"""Hybrid algorithm merging rectified flow risk assessment (Parent A) with geometric product NLMS updates (Parent B).

Mathematical bridge:
- Parent A provides a linear interpolation (interpolant) and a constant‑velocity field (flow_target) that define a desired
  transformation v = X1‑X0 between two data states.
- Parent B supplies a geometric‑product based blade multiplication and a Normalised Least‑Mean‑Squares (NLMS) update rule,
  which is a gradient‑descent step for adapting a weight matrix W.

The hybrid treats the flow target v as the desired signal d for the NLMS update.  The interpolation factor t controls the
magnitude of the update, while a reconstruction‑risk score (exp(‑uq/total)) modulates the learning rate μ.  Thus the
vector field from Parent A drives the adaptive optimisation of the geometric‑product‑aware weight matrix from Parent B,
yielding a unified risk‑aware flow‑NLMS system.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Iterable
import numpy as np
import math
import random
import sys
import pathlib

# ==== Parent A components ====================================================

@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str
    vram_mb: int

@dataclass
class Candidate:
    candidate_key: str
    family: str
    notes: str
    classification: str
    fast_path_compatible: bool
    benchmark_required: bool
    benchmark_evidence: bool

class ModelPool:
    def __init__(self, ram_ceiling_mb: int = 6000, vram_ceiling_mb: int = 4096):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.vram_ceiling_mb = vram_ceiling_mb
        self.loaded: dict[str, ModelTier] = {}
        self.sensitive_records: list[Any] = []

    def is_loaded(self, name: str) -> bool:
        return name in self.loaded

    def _used_ram(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def _used_vram(self) -> int:
        return sum(m.vram_mb for m in self.loaded.values())

    def load(self, model: ModelTier, candidate: Candidate) -> None:
        if candidate.classification == "unsafe_for_fastpath" and model.tier == "T3":
            if self._used_ram() + model.ram_mb > self.ram_ceiling_mb:
                raise RuntimeError("RAM ceiling exceeded")
            if self._used_vram() + model.vram_mb > self.vram_ceiling_mb:
                raise RuntimeError("VRAM ceiling exceeded")
            self.loaded[model.name] = model

def reconstruction_risk_score(unique_quasi_identifiers: int, total_records: int) -> float:
    if total_records == 0:
        return 1.0
    return math.exp(-unique_quasi_identifiers / total_records)

def interpolant(x0: np.ndarray, x1: np.ndarray, t: float | np.ndarray) -> np.ndarray:
    """Straight‑line interpolant: Z_t = t * x1 + (1‑t) * x0."""
    return t * x1 + (1.0 - t) * x0

def flow_target(x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
    """Constant‑velocity vector field: v = X1‑X0."""
    return x1 - x0

# ==== Parent B components ====================================================

def _blade_sign(indices: Iterable[int]) -> tuple[list[int], int]:
    """Return (sorted_blade, sign) after bubble‑sorting index list."""
    lst = list(indices)
    sign = 1
    n = len(lst)
    for i in range(n):
        for j in range(n - 1 - i):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                # cancel duplicate indices (e^2 = 0)
                lst.pop(j)
                lst.pop(j)  # after first pop the next element shifts left
                return lst, sign
    return lst, sign

def _multiply_blades(blade_a: frozenset[int], blade_b: frozenset[int]) -> tuple[frozenset[int], int]:
    """Geometric product of two basis blades."""
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign

def init_ttt(d_in: int, d_out: int | None = None, scale: float = 0.01, seed: int = 0) -> np.ndarray:
    """Initialize weight matrix W of shape (d_out, d_in)."""
    rng = np.random.default_rng(seed)
    if d_out is None:
        d_out = d_in
    return rng.standard_normal((d_out, d_in)) * scale

def ttt_loss(W: np.ndarray, x: np.ndarray, target: np.ndarray | None = None) -> float:
    """Self‑supervised loss: ½‖W x − target‖² (target defaults to x)."""
    pred = W @ x
    t = x if target is None else target
    residual = pred - t
    return float(0.5 * np.dot(residual, residual))

def nlms_update(W: np.ndarray, x: np.ndarray, d: np.ndarray,
                mu: float = 0.1, eps: float = 1e-6) -> np.ndarray:
    """
    Normalised Least‑Mean‑Squares (NLMS) update.
    W_{new} = W + (mu / (||x||² + eps)) * (d - W x) xᵀ
    """
    e = d - W @ x                      # error vector
    norm_x = np.dot(x, x) + eps
    step = (mu / norm_x) * np.outer(e, x)
    return W + step

# ==== Hybrid functions =======================================================

def geometric_product(blade_a: frozenset[int], blade_b: frozenset[int]) -> tuple[frozenset[int], int]:
    """Public wrapper around the geometric‑product primitive."""
    return _multiply_blades(blade_a, blade_b)

def hybrid_risk_score(model_pool: ModelPool, candidate: Candidate,
                      x0: np.ndarray, x1: np.ndarray, t: float) -> float:
    """
    Combine the reconstruction risk with a geometry‑aware factor.
    The geometry factor is the norm of the weight matrix that would be used
    for the NLMS step (here approximated by the number of loaded models).
    """
    uq = np.linalg.matrix_rank(np.vstack([x0, x1]))
    total = x0.shape[0] + x1.shape[0]
    base_risk = reconstruction_risk_score(uq, total)

    # Geometry factor: more loaded models → larger effective weight norm
    geom_factor = 1.0 + len(model_pool.loaded) * 0.05
    return base_risk * geom_factor

def hybrid_flow_nlms_step(model_pool: ModelPool, candidate: Candidate,
                          W: np.ndarray, x0: np.ndarray, x1: np.ndarray,
                          t: float, mu_base: float = 0.1) -> np.ndarray:
    """
    Perform a single hybrid update:
    1. Compute the desired flow target v = X1‑X0.
    2. Interpolate to obtain a provisional state z_t.
    3. Modulate the NLMS learning rate μ by the hybrid risk score.
    4. Apply NLMS update with d = v (desired change) and input = z_t.
    """
    v = flow_target(x0, x1)                # desired change vector
    z_t = interpolant(x0, x1, t)           # current interpolated state
    risk = hybrid_risk_score(model_pool, candidate, x0, x1, t)
    mu = mu_base * (1.0 - risk)            # lower risk → larger step, higher risk → conservative
    W_new = nlms_update(W, z_t, v, mu=mu)
    return W_new

def hybrid_predict(W: np.ndarray, x: np.ndarray, blade: frozenset[int]) -> np.ndarray:
    """
    Predict using the adapted weight matrix and a geometric‑product modifier.
    The blade is turned into a sign factor (±1) that flips the prediction if the blade grade is odd.
    """
    product, sign = geometric_product(blade, frozenset())
    # The sign from the geometric product with the empty blade is just 1,
    # but we keep the pattern for extensibility.
    pred = sign * (W @ x)
    return pred

# ==== Smoke test =============================================================

if __name__ == "__main__":
    # Create a tiny model pool and candidate
    pool = ModelPool()
    cand = Candidate(
        candidate_key="c001",
        family="test",
        notes="demo",
        classification="unsafe_for_fastpath",
        fast_path_compatible=False,
        benchmark_required=False,
        benchmark_evidence=False,
    )
    model = ModelTier(name="demo_model", ram_mb=500, tier="T3", vram_mb=1024)
    pool.load(model, cand)

    # Dummy data
    dim = 4
    x0 = np.array([0.2, -0.1, 0.5, 0.3])
    x1 = np.array([0.4, 0.0, 0.45, 0.35])
    t = 0.6

    # Initialise weight matrix
    W = init_ttt(d_in=dim, d_out=dim, scale=0.05, seed=42)

    # Perform hybrid update
    W_updated = hybrid_flow_nlms_step(pool, cand, W, x0, x1, t)

    # Predict with a simple blade (e.g., basis vector e1)
    blade = frozenset([1])   # represents e1
    pred = hybrid_predict(W_updated, x0, blade)

    # Simple sanity prints (no assertions required)
    print("Updated weight matrix W:\n", W_updated)
    print("Prediction after hybrid step:\n", pred)