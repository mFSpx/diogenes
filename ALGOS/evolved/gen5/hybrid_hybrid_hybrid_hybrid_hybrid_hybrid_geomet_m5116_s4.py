# DARWIN HAMMER — match 5116, survivor 4
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_caputo_fracti_m149_s5.py (gen4)
# parent_b: hybrid_hybrid_geometric_pro_hybrid_hybrid_model__m176_s0.py (gen3)
# born: 2026-05-29T23:59:53Z

"""
Hybrid Algorithm: hybrid_pheromone_geometric_curvature.py

Parents:
- hybrid_hybrid_hybrid_hybrid_hybrid_caputo_fracti_m149_s5.py (StoreState + fractional pheromone decay)
- hybrid_hybrid_geometric_pro_hybrid_hybrid_model__m176_s0.py (Geometric product + Krampus Ollivier‑Ricci curvature)

Mathematical Bridge:
The bridge is the *store level* `S(t)` from the StoreState dynamics.  
It is used as the “time” argument of a power‑law fractional decay kernel `κ(S) = S^{α-1}`.  

* The pheromone signal decays with a factor `κ(S)` and feeds back into the StoreState.
* The same kernel scales the weight matrix `W` that participates in the geometric‑product
  and curvature computations.  
* The Ollivier‑Ricci curvature (computed on the scaled matrix) then modulates the
  pheromone outflow, closing a feedback loop that couples the two parent topologies
  into a unified hybrid system.
"""

import math
import random
import sys
import pathlib
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Set, FrozenSet

# ----------------------------------------------------------------------
# Parent A – StoreState and fractional decay
# ----------------------------------------------------------------------
@dataclass
class StoreState:
    level: float = 0.0          # current store level (acts as “time” for decay)
    alpha: float = 1.0          # inflow scaling
    beta: float = 1.0           # outflow scaling
    dt: float = 1.0             # integration step
    base: float = 1.0           # baseline for dance signal
    gain: float = 1.0           # gain for dance signal
    limit: float = 10.0         # upper bound for dance
    _last_delta: float = 0.0    # internal bookkeeping

    def update(self, inflow: List[float], outflow: List[float]) -> Tuple[float, float]:
        """Euler integration of the store level."""
        delta = self.alpha * sum(inflow) - self.beta * sum(outflow)
        self.level = max(0.0, self.level + self.dt * delta)
        self._last_delta = delta
        return self.level, delta

    @property
    def dance(self) -> float:
        """Bounded signal derived from the last delta."""
        return max(0.0, min(self.limit, self.base + self.gain * self._last_delta))


def fractional_decay(alpha: float, t: float) -> float:
    """
    Simple power‑law fractional decay kernel.
    For Caputo‑type kernels the weight is proportional to (t)^{α-1}.
    """
    if t <= 0.0:
        return 0.0
    return t ** (alpha - 1.0)


# ----------------------------------------------------------------------
# Parent B – Geometric product & Krampus curvature
# ----------------------------------------------------------------------
def _blade_sign(indices: List[int]) -> Tuple[List[int], int]:
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
                # cancel duplicate indices (e_i ∧ e_i = 0)
                lst.pop(j)
                lst.pop(j)  # after pop the next element shifts left
                return lst, sign
    return lst, sign


def _multiply_blades(blade_a: FrozenSet[int], blade_b: FrozenSet[int]) -> Tuple[FrozenSet[int], int]:
    """Multiply two basis blades (each a frozenset of indices)."""
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


def geometric_product(multivector_a: List[Tuple[FrozenSet[int], float]],
                      multivector_b: List[Tuple[FrozenSet[int], float]]) -> List[Tuple[FrozenSet[int], float]]:
    """
    Compute the geometric product of two multivectors.
    Each multivector is a list of (blade, coefficient) pairs.
    """
    result: dict[FrozenSet[int], float] = {}
    for blade_a, coeff_a in multivector_a:
        for blade_b, coeff_b in multivector_b:
            blade_res, sign = _multiply_blades(blade_a, blade_b)
            result[blade_res] = result.get(blade_res, 0.0) + sign * coeff_a * coeff_b
    # Convert dict back to list, discarding near‑zero coefficients
    return [(b, c) for b, c in result.items() if abs(c) > 1e-12]


def ttt_grad(W: np.ndarray, x: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Gradient of the squared‑error loss for a linear model W @ x ≈ target.
    dL/dW = 2 (W@x - target) x^T
    """
    residual = W @ x - target
    return 2.0 * np.outer(residual, x)


def krampus_ollivier_ricci_curvature(W: np.ndarray, x: np.ndarray, target: np.ndarray) -> float:
    """
    Simple curvature proxy: squared L2 norm of the residual.
    """
    residual = W @ x - target
    return float(residual @ residual) + 1e-12  # avoid division by zero


def krampus_update(W: np.ndarray, x: np.ndarray, target: np.ndarray, lr: float = 0.01) -> np.ndarray:
    """
    Weight update using the TTT‑Linear rule scaled by curvature.
    """
    grad = ttt_grad(W, x, target)
    curvature = krampus_ollivier_ricci_curvature(W, x, target)
    W += lr * grad / curvature
    return W


# ----------------------------------------------------------------------
# Hybrid Functions (demonstrating the fused dynamics)
# ----------------------------------------------------------------------
def hybrid_pheromone_update(store: StoreState,
                            base_signal: float,
                            half_life: float,
                            alpha_decay: float) -> float:
    """
    Compute a pheromone signal that decays both exponentially (half‑life) and
    with the fractional kernel driven by the current store level.
    The signal is then fed back as inflow to the store.
    Returns the computed pheromone value.
    """
    # Exponential decay part
    exp_decay = 0.5 ** (store.dt / half_life)

    # Fractional decay part using the store level as “time”
    frac_decay = fractional_decay(alpha_decay, store.level + 1e-9)  # +ε to avoid zero

    pheromone = base_signal * exp_decay * frac_decay
    # Update store: pheromone acts as inflow, dance signal acts as outflow
    store.update(inflow=[pheromone], outflow=[store.dance])
    return pheromone


def scale_weight_matrix(W: np.ndarray, store: StoreState, alpha_decay: float) -> np.ndarray:
    """
    Apply the same fractional decay kernel to every element of the weight matrix,
    using the store level as the time variable.
    """
    factor = fractional_decay(alpha_decay, store.level + 1e-9)
    return W * factor


def hybrid_geometric_curvature_step(blades_a: List[Tuple[FrozenSet[int], float]],
                                   blades_b: List[Tuple[FrozenSet[int], float]],
                                   W: np.ndarray,
                                   x: np.ndarray,
                                   target: np.ndarray,
                                   store: StoreState,
                                   alpha_decay: float) -> Tuple[np.ndarray, List[Tuple[FrozenSet[int], float]]]:
    """
    1. Scale the weight matrix with the fractional kernel (store level).
    2. Perform a Krampus weight update using curvature computed on the scaled matrix.
    3. Compute the geometric product of two multivectors using the updated matrix
       (the matrix is only conceptually linked; we treat it as a modifier of the
       product coefficients via a simple dot‑product with the store level).
    Returns the updated matrix and the resulting multivector.
    """
    # 1 – scale matrix
    W_scaled = scale_weight_matrix(W, store, alpha_decay)

    # 2 – curvature‑guided update
    W_updated = krampus_update(W_scaled, x, target)

    # 3 – geometric product (coefficients are modulated by the norm of the updated matrix)
    product = geometric_product(blades_a, blades_b)

    # Modulate each coefficient by a factor derived from the updated matrix norm
    norm_factor = np.linalg.norm(W_updated)
    modulated_product = [(blade, coeff * norm_factor) for blade, coeff in product]

    return W_updated, modulated_product


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Initialise store
    store = StoreState(level=1.0, alpha=0.8, beta=0.5, dt=1.0, base=0.5, gain=2.0, limit=5.0)

    # Pheromone dynamics test
    pher = hybrid_pheromone_update(store, base_signal=10.0, half_life=5.0, alpha_decay=0.7)
    print(f"Pheromone signal: {pher:.4f}, Store level: {store.level:.4f}")

    # Weight matrix test
    W = np.array([[0.2, -0.1],
                  [0.5,  0.3]])
    x = np.array([1.0, 2.0])
    target = np.array([0.0, 1.0])

    # Multivector test (simple 2‑blade example)
    blade_e1 = frozenset({1})
    blade_e2 = frozenset({2})
    mv_a = [(blade_e1, 1.0), (blade_e2, 0.5)]
    mv_b = [(blade_e1, 0.3), (blade_e2, -0.2)]

    W_upd, mv_res = hybrid_geometric_curvature_step(
        mv_a, mv_b, W, x, target, store, alpha_decay=0.6
    )
    print("Updated weight matrix:\n", W_upd)
    print("Resulting multivector (blade, coeff):")
    for blade, coeff in mv_res:
        print(f"  Blade {sorted(blade)} : {coeff:.4f}")

    # Final store state sanity check
    print(f"Final Store level: {store.level:.4f}, Dance signal: {store.dance:.4f}")