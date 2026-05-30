# DARWIN HAMMER — match 5663, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_sparse_wta_hy_m626_s0.py (gen5)
# parent_b: hybrid_hybrid_bandit_router_workshare_allocator_m60_s1.py (gen2)
# born: 2026-05-30T00:03:59Z

"""
Hybrid algorithm merging hybrid_hybrid_hybrid_hybrid_hybrid_sparse_wta_hy_m626_s0.py and hybrid_hybrid_bandit_router_workshare_allocator_m60_s1.py.
The mathematical bridge between the two structures is the use of the Structural Similarity Index (SSIM) to inform the selection of sparse expansions in the second parent,
which in turn modulates the deterministic target percentage in the workshare allocation using a StoreState instance.
"""

import numpy as np
import math
import random
import sys
from pathlib import Path
import hashlib

# Constants
PROTOTYPE_VECTOR = np.array([0.2, 0.5, 0.3, 0.7, 0.1], dtype=np.float64)

# SSIM implementation
def compute_ssim(
    x: list[float],
    y: list[float],
    dynamic_range: float = 1.0,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    if len(x) != len(y):
        raise ValueError("samples must have equal length")
    if not x:
        raise ValueError("samples must not be empty")

    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    mx = x_arr.mean()
    my = y_arr.mean()
    vx = ((x_arr - mx) ** 2).mean()
    vy = ((y_arr - my) ** 2).mean()
    cov = ((x_arr - mx) * (y_arr - my)).mean()

    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2

    numerator = (2 * mx * my + c1) * (2 * cov + c2)
    denominator = (mx * mx + my * my + c1) * (vx + vy + c2)
    return float(numerator / denominator)

# Hybrid sparse expansion utilities
def hybrid_sparse_expansion(values: list[float], m: int, salt: str = "") -> list[float]:
    """Hash‑based sparse expansion of `values` into a vector of length `m`."""
    if m <= 0:
        raise ValueError("m must be positive")
    out = [0.0] * m
    for i, v in enumerate(values):
        for r in range(3):
            h = hashlib.sha256(f"{salt}:{i}:{r}".encode()).digest()
            j = int.from_bytes(h[:8], "big") % m
            sign = 1.0 if r % 2 == 0 else -1.0
            out[j] += sign * v
    return out

# StoreState class
class StoreState:
    """Encapsulates the honeybee‑style store and its derived control signal."""

    def __init__(self, level: float = 0.0, alpha: float = 1.0, beta: float = 1.0, dt: float = 1.0, base: float = 1.0, gain: float = 1.0, limit: float = 10.0):
        self.level = level
        self.alpha = alpha
        self.beta = beta
        self.dt = dt
        self.base = base
        self.gain = gain
        self.limit = limit
        self._last_delta = 0.0

    def update(self, inflow: list[float], outflow: list[float]) -> tuple[float, float]:
        """
        Apply the store equation and recompute the dance duration.

        Returns
        -------
        new_level, delta
        """
        delta = self.alpha * sum(inflow) - self.beta * sum(outflow)
        self.level = max(0.0, self.level + self.dt * delta)
        self._last_delta = delta
        return self.level, delta

    @property
    def dance(self) -> float:
        """Bounded control signal derived from the last Δ (computed lazily)."""
        return max(0.0, min(self.limit, self.base + self.gain * self._last_delta))

# Hybrid algorithm functions
def hybrid_algorithm(values: list[float], m: int, salt: str = "", store_state: StoreState = None) -> list[float]:
    if store_state is None:
        store_state = StoreState()
    sparse_expansion = hybrid_sparse_expansion(values, m, salt)
    inflow = [v for v in sparse_expansion if v > 0]
    outflow = [v for v in sparse_expansion if v < 0]
    store_state.update(inflow, outflow)
    return [store_state.dance]

def hybrid_ssim(values1: list[float], values2: list[float]) -> float:
    ssim = compute_ssim(values1, values2)
    return ssim

def hybrid_allocation(values: list[float], m: int, salt: str = "", store_state: StoreState = None) -> list[float]:
    if store_state is None:
        store_state = StoreState()
    allocation = hybrid_sparse_expansion(values, m, salt)
    inflow = [v for v in allocation if v > 0]
    outflow = [v for v in allocation if v < 0]
    store_state.update(inflow, outflow)
    return [store_state.dance]

if __name__ == "__main__":
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    m = 10
    salt = "example"
    store_state = StoreState()
    hybrid_algorithm(values, m, salt, store_state)
    hybrid_ssim(values, values)
    hybrid_allocation(values, m, salt, store_state)