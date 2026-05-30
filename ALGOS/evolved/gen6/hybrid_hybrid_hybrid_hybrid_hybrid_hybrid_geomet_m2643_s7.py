# DARWIN HAMMER — match 2643, survivor 7
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m560_s4.py (gen5)
# parent_b: hybrid_hybrid_geometric_pro_hybrid_hybrid_model__m176_s0.py (gen3)
# born: 2026-05-29T23:43:17Z

"""HybridPheromoneGeometric
Combines the pheromone decay model from `hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m560_s4.py`
with the geometric product and Ollivier‑Ricci curvature update from
`hybrid_hybrid_geometric_pro_hybrid_hybrid_model__m176_s0.py`.

Mathematical bridge:
- A pheromone entry carries a scalar `signal_value`.  We embed each entry as a
  **multivector** whose basis blade is derived from a hash of the entry's
  `surface_key`.  The scalar coefficient of the blade is the pheromone's
  `signal_value`.
- The geometric product (blade multiplication with sign) provides a natural
  way to *combine* two pheromone entries, producing a new multivector whose
  coefficients are the products of the original scalars.
- The curvature‑driven TTT‑Linear update (`krampus_update`) is used to adapt a
  weight matrix `W` that governs the decay‑adjusted scaling of the multivector.
  After each geometric product we (i) decay the scalar coefficients according
  to the exponential half‑life rule of the original pheromone model, and (ii)
  apply a curvature‑based correction to the coefficients via the updated `W`.

The module therefore fuses the two parent topologies into a single unified
system that can be used for adaptive, geometry‑aware pheromone propagation.
"""

import uuid
import math
import random
import sys
import pathlib
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Dict, FrozenSet, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Pheromone entry (trimmed to essentials)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Span:
    start: int
    end: int
    text: str
    label: str
    score: float
    backend: str = "literal_fallback"


class PheromoneEntry:
    """Lightweight pheromone record with exponential decay."""
    __slots__ = (
        "uuid",
        "surface_key",
        "signal_kind",
        "signal_value",
        "half_life_seconds",
        "created_at",
        "last_decay",
    )

    def __init__(self, surface_key: str, signal_kind: str,
                 signal_value: float, half_life_seconds: int):
        self.uuid = str(uuid.uuid4())
        self.surface_key = surface_key
        self.signal_kind = signal_kind
        self.signal_value = signal_value
        self.half_life_seconds = max(1, half_life_seconds)
        now = datetime.now(timezone.utc)
        self.created_at = now
        self.last_decay = now

    def age_seconds(self) -> float:
        return (datetime.now(timezone.utc) - self.last_decay).total_seconds()

    def decay_factor(self) -> float:
        """Multiplicative factor since the last decay event."""
        return 0.5 ** (self.age_seconds() / self.half_life_seconds)

    def apply_decay(self) -> None:
        self.signal_value *= self.decay_factor()
        self.last_decay = datetime.now(timezone.utc)


# ----------------------------------------------------------------------
# Parent B – Geometric product & curvature utilities (self‑contained)
# ----------------------------------------------------------------------
def _blade_sign(indices: Tuple[int, ...]) -> Tuple[Tuple[int, ...], int]:
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
                # cancel duplicate index (e_i ^ e_i = 0)
                lst.pop(j)
                lst.pop(j)  # second element now occupies position j
                return tuple(lst), sign
    return tuple(lst), sign


def _multiply_blades(blade_a: FrozenSet[int],
                    blade_b: FrozenSet[int]) -> Tuple[FrozenSet[int], int]:
    """Multiply two basis blades (each a frozenset of indices)."""
    combined = list(blade_a) + list(blade_b)
    sorted_blade, sign = _blade_sign(tuple(combined))
    return frozenset(sorted_blade), sign


def geometric_product_multivector(a: Dict[FrozenSet[int], float],
                                 b: Dict[FrozenSet[int], float]) -> Dict[FrozenSet[int], float]:
    """Geometric product of two multivectors represented as dicts {blade: coeff}."""
    result: Dict[FrozenSet[int], float] = {}
    for blade_a, coeff_a in a.items():
        for blade_b, coeff_b in b.items():
            blade_res, sign = _multiply_blades(blade_a, blade_b)
            coeff_res = coeff_a * coeff_b * sign
            result[blade_res] = result.get(blade_res, 0.0) + coeff_res
    return result


# ----------------------------------------------------------------------
# Simple TTT‑Linear model utilities (stand‑ins for parent imports)
# ----------------------------------------------------------------------
def init_ttt(in_dim: int, out_dim: int) -> np.ndarray:
    """Initialize a weight matrix for the TTT‑Linear model."""
    rng = np.random.default_rng()
    return rng.normal(loc=0.0, scale=1.0, size=(out_dim, in_dim))


def ttt_grad(W: np.ndarray, x: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Gradient of the squared error ½‖Wx‑target‖² w.r.t. W.
    Returns the same shape as W.
    """
    residual = (W @ x) - target
    # d/dW ½‖r‖² = residual ⊗ xᵀ
    return np.outer(residual, x)


def krampus_ollivier_ricci_curvature(W: np.ndarray, x: np.ndarray,
                                     target: np.ndarray = None) -> float:
    """
    Compute a scalar curvature measure using the TTT‑Linear residual.
    If `target` is None we use the zero vector as a dummy target.
    """
    if target is None:
        target = np.zeros_like(x)
    pred = W @ x
    residual = pred - target
    return float(residual @ residual + 1e-12)  # avoid division by zero


def krampus_update(W: np.ndarray, x: np.ndarray,
                   target: np.ndarray = None, lr: float = 0.01) -> np.ndarray:
    """
    Perform one curvature‑aware TTT‑Linear update.
    """
    grad = ttt_grad(W, x, target if target is not None else np.zeros_like(x))
    curvature = krampus_ollivier_ricci_curvature(W, x, target)
    W += lr * grad / curvature
    return W


# ----------------------------------------------------------------------
# Hybrid layer: embed pheromone entries as multivectors
# ----------------------------------------------------------------------
def _hash_to_blade(surface_key: str, max_dim: int = 64) -> FrozenSet[int]:
    """
    Deterministically map a surface key to a basis blade.
    The hash is reduced modulo `max_dim` and the resulting bits are interpreted
    as a set of basis indices.
    """
    h = hash(surface_key)
    indices = {i for i in range(max_dim) if (h >> i) & 1}
    # An empty set would correspond to the scalar (grade‑0) blade.
    return frozenset(indices)


def pheromone_to_multivector(entry: PheromoneEntry) -> Dict[FrozenSet[int], float]:
    """
    Convert a single pheromone entry into a multivector with one blade.
    The blade is derived from the entry's `surface_key`; the scalar coefficient
    is the entry's current `signal_value`.
    """
    blade = _hash_to_blade(entry.surface_key)
    return {blade: entry.signal_value}


def decay_multivector(multivector: Dict[FrozenSet[int], float],
                      entry: PheromoneEntry) -> None:
    """
    Apply exponential decay (from Parent A) to all coefficients of the
    multivector, using the entry's half‑life.
    The function mutates the input dict in place.
    """
    factor = entry.decay_factor()
    for blade in list(multivector.keys()):
        multivector[blade] *= factor
    entry.apply_decay()  # keep the original entry consistent


def curvature_adjust_multivector(multivector: Dict[FrozenSet[int], float],
                                 W: np.ndarray,
                                 dim: int = 8) -> Tuple[Dict[FrozenSet[int], float], np.ndarray]:
    """
    Use the curvature‑aware TTT‑Linear update to modify the weight matrix `W`
    and then scale the multivector coefficients by the norm of the updated `W`.
    Returns the adjusted multivector and the new weight matrix.
    """
    # Build a dense vector representation of the multivector.
    # We embed each blade as a one‑hot vector of length `dim`.
    vec = np.zeros(dim)
    for blade, coeff in multivector.items():
        # Simple embedding: sum the indices modulo dim, weighted by coeff.
        idx = sum(blade) % dim
        vec[idx] += coeff

    # Perform curvature‑aware update.
    W = krampus_update(W, vec)

    # Rescale coefficients by the Frobenius norm of the updated W.
    scale = np.linalg.norm(W, ord='fro')
    for blade in multivector:
        multivector[blade] *= scale
    return multivector, W


# ----------------------------------------------------------------------
# Public API – three demonstrative hybrid functions
# ----------------------------------------------------------------------
def hybrid_combine_entries(entry_a: PheromoneEntry,
                           entry_b: PheromoneEntry,
                           W: np.ndarray) -> Tuple[Dict[FrozenSet[int], float], np.ndarray]:
    """
    Combine two pheromone entries via geometric product, apply decay,
    then curvature‑adjust the result.
    Returns the resulting multivector and the updated weight matrix.
    """
    # 1. Convert to multivectors.
    mv_a = pheromone_to_multivector(entry_a)
    mv_b = pheromone_to_multivector(entry_b)

    # 2. Geometric product.
    combined = geometric_product_multivector(mv_a, mv_b)

    # 3. Decay according to the *older* entry (conservative choice).
    older = entry_a if entry_a.created_at <= entry_b.created_at else entry_b
    decay_multivector(combined, older)

    # 4. Curvature‑driven adjustment.
    adjusted, W_new = curvature_adjust_multivector(combined, W)

    return adjusted, W_new


def hybrid_propagate(entry: PheromoneEntry,
                     W: np.ndarray,
                     steps: int = 3) -> Tuple[Dict[FrozenSet[int], float], np.ndarray]:
    """
    Propagate a single pheromone entry through `steps` self‑interactions.
    Each step multiplies the multivector by itself, decays, and applies curvature.
    """
    mv = pheromone_to_multivector(entry)
    for _ in range(steps):
        mv = geometric_product_multivector(mv, mv)
        decay_multivector(mv, entry)
        mv, W = curvature_adjust_multivector(mv, W)
    return mv, W


def hybrid_batch_process(entries: Tuple[PheromoneEntry, ...],
                         dim: int = 8) -> Tuple[Dict[FrozenSet[int], float], np.ndarray]:
    """
    Process a batch of pheromone entries:
    - Initialise a weight matrix `W`.
    - Iteratively combine each entry with the accumulated multivector.
    - Return the final multivector and weight matrix.
    """
    W = init_ttt(dim, dim)
    accumulator: Dict[FrozenSet[int], float] = {frozenset(): 1.0}  # scalar 1

    for entry in entries:
        entry_mv = pheromone_to_multivector(entry)
        accumulator = geometric_product_multivector(accumulator, entry_mv)
        decay_multivector(accumulator, entry)
        accumulator, W = curvature_adjust_multivector(accumulator, W)

    return accumulator, W


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create two sample pheromone entries.
    e1 = PheromoneEntry(surface_key="alpha", signal_kind="typeA",
                       signal_value=1.2, half_life_seconds=30)
    e2 = PheromoneEntry(surface_key="beta", signal_kind="typeB",
                       signal_value=0.8, half_life_seconds=45)

    # Initialise a weight matrix.
    W0 = init_ttt(in_dim=8, out_dim=8)

    # Test hybrid_combine_entries
    combined_mv, W1 = hybrid_combine_entries(e1, e2, W0)
    print("Combined multivector (sample):", list(combined_mv.items())[:5])

    # Test hybrid_propagate
    propagated_mv, W2 = hybrid_propagate(e1, W1, steps=2)
    print("Propagated multivector (sample):", list(propagated_mv.items())[:5])

    # Test batch processing
    batch_mv, W_final = hybrid_batch_process((e1, e2), dim=8)
    print("Batch multivector (sample):", list(batch_mv.items())[:5])
    print("Final weight matrix Frobenius norm:", np.linalg.norm(W_final, ord='fro'))