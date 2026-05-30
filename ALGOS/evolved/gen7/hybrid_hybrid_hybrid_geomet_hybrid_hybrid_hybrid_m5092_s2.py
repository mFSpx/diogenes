# DARWIN HAMMER — match 5092, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_geometric_pro_hybrid_hybrid_hard_t_m2229_s1.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_tropical_maxp_m1632_s1.py (gen6)
# born: 2026-05-29T23:59:45Z

"""Hybrid Geometric‑Tropical Algebra (HGTA)

This module fuses the core mathematics of two parent algorithms:

* **Parent A** – a geometric algebra engine that defines a `Multivector` with the
  geometric product (`*`).  The product combines basis blades by concatenation,
  removes duplicate indices (e_i·e_i = 1) and introduces a sign given by the
  parity of swaps required to sort the indices.

* **Parent B** – a tropical max‑plus framework that treats coefficients as
  “energies” in the log‑domain.  Tropical addition is `max`, tropical
  multiplication is ordinary `+`.  The algorithm also provides a variational
  free‑energy computation where the log‑partition function is approximated by a
  tropical sum (max) of the exponentiated coefficients.

**Mathematical bridge**

If a multivector `M = Σ c_B B` (B = basis blade, c_B ∈ ℝ) is mapped to a tropical
polynomial by the log‑transform  


τ(M) :  B ↦ w_B = log(|c_B|)            (w_B = -inf if c_B = 0)


then:

* **Geometric product** `M₁ * M₂` ↔ **tropical multiplication** of τ(M₁) and τ(M₂):
  the blade union corresponds to the tropical monomial product, and the
  coefficient addition `log|c₁c₂| = log|c₁| + log|c₂|` is exactly tropical
  multiplication.

* **Geometric addition** `M₁ + M₂` ↔ **tropical addition** (max) of the
  transformed weights, because in the log‑domain the dominant term dominates the
  sum.

Consequently a hybrid algorithm can operate on both representations, convert
between them, and use the tropical free‑energy (`F = -max_B w_B`) as a scalar
energy that feeds a dynamical `StoreState` (the variational‑free‑energy engine
from Parent B).

The functions below demonstrate this unified system.
"""

import math
import random
import sys
import pathlib
from typing import Dict, Tuple, FrozenSet, List

import numpy as np

# ----------------------------------------------------------------------
# Geometric Algebra core (Parent A)
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Edge = Tuple[str, str]

class Multivector:
    """Element of Cl(n,0) represented as a sum of basis blades.

    components: dict mapping frozenset(basis_indices) -> float coefficient.
                frozenset() is the scalar (grade‑0) blade.
    n: dimension of the base vector space.
    """

    def __init__(self, components: Dict[FrozenSet[int], float], n: int):
        self.components = {
            k: float(v) for k, v in components.items() if abs(v) > 1e-15
        }
        self.n = int(n)

    # ------------------------------------------------------------------
    # Algebraic operations
    # ------------------------------------------------------------------
    def __add__(self, other: "Multivector") -> "Multivector":
        result = self.components.copy()
        for k, v in other.components.items():
            result[k] = result.get(k, 0.0) + v
        return Multivector(result, self.n)

    def __mul__(self, other: "Multivector") -> "Multivector":
        """Geometric product."""
        result: Dict[FrozenSet[int], float] = {}
        for k1, v1 in self.components.items():
            for k2, v2 in other.components.items():
                combined, sign = self._multiply_blades(k1, k2)
                result[combined] = result.get(combined, 0.0) + sign * v1 * v2
        return Multivector(result, self.n)

    def scalar_part(self) -> float:
        """Return the grade‑0 (scalar) coefficient."""
        return self.components.get(frozenset(), 0.0)

    # ------------------------------------------------------------------
    # Internal blade handling
    # ------------------------------------------------------------------
    @staticmethod
    def _blade_sign(indices: List[int]) -> Tuple[List[int], int]:
        """Sort indices, applying sign changes for swaps and cancelling pairs.

        Returns the sorted list of remaining indices after removing any
        duplicate pairs (e_i e_i = 1) and the overall sign (+1 or -1) coming from
        the number of swaps needed to achieve the sorted order.
        """
        # Cancel duplicate pairs first
        counts: Dict[int, int] = {}
        for i in indices:
            counts[i] = counts.get(i, 0) + 1
        # Keep only indices with odd multiplicity
        remaining = [i for i, c in counts.items() if c % 2 == 1]

        # Determine sign by counting swaps needed to sort
        sign = 1
        # Simple bubble‑sort counting
        for i in range(len(remaining)):
            for j in range(i + 1, len(remaining)):
                if remaining[i] > remaining[j]:
                    sign *= -1
        remaining.sort()
        return remaining, sign

    def _multiply_blades(
        self, blade_a: FrozenSet[int], blade_b: FrozenSet[int]
    ) -> Tuple[FrozenSet[int], int]:
        combined = list(blade_a) + list(blade_b)
        result, sign = self._blade_sign(combined)
        return frozenset(result), sign

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        terms = []
        for blade, coeff in sorted(self.components.items(), key=lambda x: (len(x[0]), sorted(x[0]))):
            if blade:
                blade_str = "e" + "".join(str(i) for i in sorted(blade))
            else:
                blade_str = "1"
            terms.append(f"{coeff:.3g}{blade_str}")
        return " + ".join(terms) if terms else "0"

# ----------------------------------------------------------------------
# Tropical max‑plus core (Parent B)
# ----------------------------------------------------------------------
def tropical_add(poly1: Dict[FrozenSet[int], float],
                 poly2: Dict[FrozenSet[int], float]) -> Dict[FrozenSet[int], float]:
    """Tropical addition (max) of two tropical polynomials."""
    result = poly1.copy()
    for blade, w in poly2.items():
        result[blade] = max(result.get(blade, -math.inf), w)
    return result

def tropical_mul(poly1: Dict[FrozenSet[int], float],
                poly2: Dict[FrozenSet[int], float]) -> Dict[FrozenSet[int], float]:
    """Tropical multiplication (addition of weights) with blade union."""
    result: Dict[FrozenSet[int], float] = {}
    for b1, w1 in poly1.items():
        for b2, w2 in poly2.items():
            combined = frozenset(set(b1) ^ set(b2))  # symmetric difference = union without duplicates
            weight = w1 + w2
            # If the same blade appears via different paths, keep the larger weight
            result[combined] = max(result.get(combined, -math.inf), weight)
    return result

def tropical_free_energy(poly: Dict[FrozenSet[int], float]) -> float:
    """Variational free energy approximation: F = -max_B w_B."""
    if not poly:
        return math.inf
    max_weight = max(poly.values())
    return -max_weight

# ----------------------------------------------------------------------
# Bridge utilities
# ----------------------------------------------------------------------
def multivector_to_tropical(mv: Multivector) -> Dict[FrozenSet[int], float]:
    """Log‑transform of a multivector into a tropical polynomial."""
    tropical: Dict[FrozenSet[int], float] = {}
    for blade, coeff in mv.components.items():
        if coeff != 0.0:
            tropical[blade] = math.log(abs(coeff))
        else:
            tropical[blade] = -math.inf
    return tropical

def tropical_to_multivector(tpoly: Dict[FrozenSet[int], float],
                            n: int) -> Multivector:
    """Inverse log‑transform: exponentiate tropical weights back to a multivector."""
    comps: Dict[FrozenSet[int], float] = {}
    for blade, w in tpoly.items():
        if w > -math.inf:
            comps[blade] = math.exp(w)
    return Multivector(comps, n)

# ----------------------------------------------------------------------
# StoreState (from Parent B) – a simple variational‑energy reservoir
# ----------------------------------------------------------------------
class StoreState:
    """Dynamic level that integrates inflow/outflow driven by free‑energy."""
    def __init__(self,
                 level: float = 0.0,
                 alpha: float = 1.0,
                 beta: float = 1.0,
                 dt: float = 1.0,
                 limit: float = 10.0):
        self.level = float(level)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.dt = float(dt)
        self.limit = float(limit)

    def update(self,
               inflow: List[float],
               outflow: List[float]) -> Tuple[float, float]:
        """Euler update of the reservoir level."""
        delta = self.alpha * sum(inflow) - self.beta * sum(outflow)
        self.level = max(0.0, min(self.limit, self.level + self.dt * delta))
        return self.level, delta

# ----------------------------------------------------------------------
# Hybrid operations (demonstration functions)
# ----------------------------------------------------------------------
def demo_geometric_vs_tropical():
    """Show that geometric product maps to tropical multiplication after log‑transform."""
    # Random 3‑dimensional multivectors
    n = 3
    mv1 = Multivector({
        frozenset(): random.uniform(0.5, 2.0),
        frozenset({0}): random.uniform(-2, 2),
        frozenset({1, 2}): random.uniform(-1, 1)
    }, n)
    mv2 = Multivector({
        frozenset(): random.uniform(0.5, 2.0),
        frozenset({1}): random.uniform(-2, 2),
        frozenset({0, 2}): random.uniform(-1, 1)
    }, n)

    gp = mv1 * mv2
    tp1 = multivector_to_tropical(mv1)
    tp2 = multivector_to_tropical(mv2)
    tp_product = tropical_mul(tp1, tp2)

    # Convert back to multivector for comparison
    mv_from_tropical = tropical_to_multivector(tp_product, n)

    print("Geometric product:", gp)
    print("Tropical‑derived product:", mv_from_tropical)

def demo_variational_free_energy(state: StoreState, mv: Multivector):
    """Compute free energy from a multivector, feed it to the store, and return new level."""
    tropical = multivector_to_tropical(mv)
    free_energy = tropical_free_energy(tropical)          # F = -max w_B
    # Use the magnitude of free energy as inflow; a small decay as outflow
    inflow = [abs(free_energy)]
    outflow = [0.1 * state.level]
    level, delta = state.update(inflow, outflow)
    print(f"Free energy: {free_energy:.4f}, Δlevel: {delta:.4f}, new level: {level:.4f}")
    return level

def demo_hybrid_cycle(steps: int = 5):
    """Run a short hybrid simulation where multivectors evolve and drive the store."""
    n = 4
    state = StoreState(level=1.0, alpha=0.8, beta=0.5, dt=0.5, limit=5.0)

    # Initialise a random multivector that will be updated each step
    current_mv = Multivector({
        frozenset(): 1.0,
        frozenset({0}): 0.5,
        frozenset({1, 3}): -0.8
    }, n)

    for step in range(steps):
        print(f"\n--- Step {step+1} ---")
        # Randomly perturb the multivector (simulating a dynamical system)
        perturb = {
            frozenset({random.randint(0, n-1)}): random.uniform(-0.3, 0.3)
        }
        current_mv = current_mv + Multivector(perturb, n)

        # Compute hybrid free‑energy update
        demo_variational_free_energy(state, current_mv)

        # Occasionally compute geometric vs tropical consistency check
        if step == steps - 1:
            demo_geometric_vs_tropical()

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("Running Hybrid Geometric‑Tropical demonstration...\n")
    demo_hybrid_cycle(steps=4)
    print("\nAll operations completed without error.")