# DARWIN HAMMER — match 2643, survivor 6
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m560_s4.py (gen5)
# parent_b: hybrid_hybrid_geometric_pro_hybrid_hybrid_model__m176_s0.py (gen3)
# born: 2026-05-29T23:43:17Z

"""Hybrid algorithm merging Pheromone decay (Parent A) with Geometric Product & Ollivier‑Ricci curvature (Parent B).

Mathematical bridge:
- Parent A models scalar decay of a pheromone value via an exponential factor.
- Parent B provides a curvature scalar `C = ||W·x - t||²` derived from a linear model and a geometric product on basis blades.
- In this hybrid we reinterpret the decay factor as a curvature‑scaled exponential:
      new_signal = old_signal * 0.5**(age / half_life) * exp(-α * C)
  where `C` is the Ollivier‑Ricci curvature computed from a weight matrix `W`
  and a feature vector `x` (the pheromone’s surface key encoded as one‑hot indices).
- The geometric product supplies the coefficient `α` through blade multiplication,
  allowing the algebraic structure of the basis to modulate decay.

The module implements:
* Blade utilities (`_blade_sign`, `_multiply_blades`).
* `geometric_product(blade_a, blade_b, coeff)` – returns (blade, coeff) after multiplication.
* `ollivier_ricci_curvature(W, x, target=None)` – curvature from Parent B.
* `apply_pheromone_decay(pheromones, W, target_vector, alpha=0.1)` – hybrid update.
* `random_weight_matrix(dim)` – helper to generate `W`.
"""

import uuid
import math
import random
import sys
import pathlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Tuple, Dict, Any, FrozenSet

import numpy as np

# ----------------------------------------------------------------------
# Parent A – data structures (trimmed to essentials)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Span:
    """Immutable representation of a text span."""
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
        self.half_life_seconds = max(1, half_life_seconds)  # avoid division by zero
        now = datetime.now(timezone.utc)
        self.created_at = now
        self.last_decay = now

    def age_seconds(self) -> float:
        return (datetime.now(timezone.utc) - self.last_decay).total_seconds()

    def decay_factor(self) -> float:
        """Multiplicative factor since the last decay event (pure exponential)."""
        return 0.5 ** (self.age_seconds() / self.half_life_seconds)

    def apply_decay(self) -> None:
        self.signal_value *= self.decay_factor()
        self.last_decay = datetime.now(timezone.utc)


# ----------------------------------------------------------------------
# Parent B – geometric algebra utilities
# ----------------------------------------------------------------------
def _blade_sign(indices: List[int]) -> Tuple[List[int], int]:
    """Return (sorted_blade, sign) after bubble‑sorting index list.
    Duplicates cancel (Grassmann algebra)."""
    lst = list(indices)
    sign = 1
    n = len(lst)
    for i in range(n):
        for j in range(n - 1 - i):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                # cancel identical indices
                lst.pop(j)
                lst.pop(j)  # second element shifts to position j
                return lst, sign
    return lst, sign


def _multiply_blades(blade_a: FrozenSet[int],
                    blade_b: FrozenSet[int]) -> Tuple[FrozenSet[int], int]:
    """Multiply two basis blades (each a frozenset of indices)."""
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


def geometric_product(blade_a: FrozenSet[int],
                     blade_b: FrozenSet[int],
                     coeff: float = 1.0) -> Tuple[FrozenSet[int], float]:
    """Full geometric product returning (blade, new_coeff)."""
    new_blade, sign = _multiply_blades(blade_a, blade_b)
    return new_blade, coeff * sign


# ----------------------------------------------------------------------
# Curvature (Parent B) – TTT‑Linear model bridge
# ----------------------------------------------------------------------
def ollivier_ricci_curvature(W: np.ndarray,
                             x: np.ndarray,
                             target: np.ndarray | None = None) -> float:
    """Curvature = squared L2 norm of the residual of a linear model."""
    pred = W @ x
    t = x if target is None else target
    residual = pred - t
    return float(residual @ residual)


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def apply_pheromone_decay(pheromones: List[PheromoneEntry],
                          W: np.ndarray,
                          target_vector: np.ndarray,
                          alpha: float = 0.1) -> None:
    """
    Update each pheromone entry by:
      1. Computing its one‑hot feature vector from the surface_key.
      2. Evaluating curvature C = ||W·x - target||².
      3. Multiplying C by a geometric‑product coefficient derived from the
         key’s character indices (treated as basis blades).
      4. Scaling the exponential decay with exp(-α * coeff * C).
    The function mutates the pheromone objects in place.
    """
    dim = W.shape[1]

    for p in pheromones:
        # ----- Step 1: one‑hot encoding of the surface_key -----------------
        # Use the hash of the key to pick an index in [0, dim)
        idx = abs(hash(p.surface_key)) % dim
        x = np.zeros(dim, dtype=float)
        x[idx] = 1.0

        # ----- Step 2: curvature ------------------------------------------------
        C = ollivier_ricci_curvature(W, x, target_vector)

        # ----- Step 3: geometric‑product coefficient ----------------------------
        # Build two trivial blades from the key’s character codes.
        # Blade A: set of even‑code points, Blade B: set of odd‑code points.
        even_codes = frozenset(i for i, ch in enumerate(p.surface_key) if ord(ch) % 2 == 0)
        odd_codes = frozenset(i for i, ch in enumerate(p.surface_key) if ord(ch) % 2 == 1)
        _, coeff = geometric_product(even_codes, odd_codes, coeff=1.0)

        # ----- Step 4: hybrid decay ------------------------------------------------
        curvature_factor = math.exp(-alpha * coeff * C)
        # Apply pure exponential decay first, then curvature scaling.
        p.apply_decay()
        p.signal_value *= curvature_factor


def random_weight_matrix(dim: int, seed: int | None = None) -> np.ndarray:
    """Generate a random square weight matrix with modest spectral norm."""
    rng = np.random.default_rng(seed)
    W = rng.normal(loc=0.0, scale=0.5, size=(dim, dim))
    # Optional: scale to keep eigenvalues < 1 for numerical stability.
    max_ev = max(abs(np.linalg.eigvals(W)))
    if max_ev > 0:
        W /= (2 * max_ev)
    return W


def summarize_pheromones(pheromones: List[PheromoneEntry]) -> Dict[str, Any]:
    """Return a dictionary with basic statistics useful for debugging."""
    values = [p.signal_value for p in pheromones]
    ages = [p.age_seconds() for p in pheromones]
    return {
        "count": len(pheromones),
        "mean_value": float(np.mean(values)) if values else 0.0,
        "max_value": float(np.max(values)) if values else 0.0,
        "min_value": float(np.min(values)) if values else 0.0,
        "mean_age": float(np.mean(ages)) if ages else 0.0,
    }


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a small set of pheromones.
    demo_pheromones = [
        PheromoneEntry(surface_key="alpha", signal_kind="typeA", signal_value=1.0, half_life_seconds=30),
        PheromoneEntry(surface_key="beta",  signal_kind="typeB", signal_value=0.8, half_life_seconds=45),
        PheromoneEntry(surface_key="gamma", signal_kind="typeA", signal_value=1.2, half_life_seconds=60),
    ]

    # Dimension for the linear model (must be >0).
    DIM = 16
    W = random_weight_matrix(DIM, seed=42)

    # Target vector – choose a unit vector for simplicity.
    target = np.zeros(DIM, dtype=float)
    target[0] = 1.0

    print("Before hybrid update:", summarize_pheromones(demo_pheromones))

    # Run the hybrid decay/update.
    apply_pheromone_decay(demo_pheromones, W, target_vector=target, alpha=0.05)

    print("After hybrid update :", summarize_pheromones(demo_pheromones))
    # Verify that no exception is raised and values remain finite.
    for p in demo_pheromones:
        assert math.isfinite(p.signal_value), "Signal value became non‑finite"
    print("Smoke test completed successfully.")