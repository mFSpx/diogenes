# DARWIN HAMMER — match 1415, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_physar_hybrid_hybrid_hybrid_m1182_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1038_s0.py (gen5)
# born: 2026-05-29T23:36:16Z

"""Hybrid Algorithm: Physarum‑Bandit ↔ Geometric‑Koopman‑Fisher Fusion

Parents:
- hybrid_hybrid_hybrid_physar_hybrid_hybrid_m1182_s0.py (Physarum conductance + bandit router)
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1038_s0.py (Geometric algebra, Koopman operator, Fisher weighting)

Mathematical Bridge:
Both parents manipulate quantities that can be expressed as vectors in a common
linear space:

* In the Physarum‑Bandit side the scalar conductance `g` of an edge is updated
  by a flux‐derived quantity `q = propensity·reward·|flux|`.  This scalar can be
  embedded as the grade‑0 (scalar) component of a multivector.

* In the Geometric‑Koopman‑Fisher side a multivector `M` lives in the Clifford
  algebra `Cl(n,0)`.  Linear evolution of `M` is performed by a Koopman matrix
  `K` acting on the coefficient vector of `M`.  Fisher information provides a
  scalar weight `w` that modulates the influence of each coefficient.

The hybrid therefore:
1. Packs the Physarum conductance (and optionally a vector of geometric
   information such as edge direction) into a multivector `M`.
2. Evolves `M` with a Koopman operator `K`.
3. Computes a Fisher‑information‑derived scalar `w` from a set of sampled
   contexts.
4. Uses `w` together with the bandit propensity to produce a hybrid flux `q`
   that finally updates the original conductance via the Physarum rule.

The three core functions below illustrate this pipeline."""
import math
import random
import sys
import pathlib
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# 1.  Physarum‑Bandit primitives (parent A)
# ----------------------------------------------------------------------
def flux(conductance: float, edge_length: float, pressure_a: float, pressure_b: float, eps: float = 1e-12) -> float:
    """Ohmic flux through an edge."""
    if edge_length <= 0:
        raise ValueError('edge_length must be positive')
    return conductance / max(edge_length, eps) * (pressure_a - pressure_b)


def update_conductance(conductance: float, q: float, dt: float = 1.0,
                       gain: float = 1.0, decay: float = 0.05) -> float:
    """Physarum conductance dynamics."""
    if dt < 0 or decay < 0:
        raise ValueError('dt and decay must be non‑negative')
    return max(0.0, conductance + dt * (gain * abs(q) - decay * conductance))


def hybrid_bandit_update(conductance: float, propensity: float,
                         reward: float, dt: float = 1.0,
                         gain: float = 1.0, decay: float = 0.05) -> float:
    """Bandit‑augmented conductance update."""
    q = propensity * reward
    return update_conductance(conductance, q, dt, gain, decay)


def euclidean_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Standard Euclidean distance in ℝ²."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


# ----------------------------------------------------------------------
# 2.  Geometric‑Koopman‑Fisher primitives (parent B)
# ----------------------------------------------------------------------
def _blade_sign(indices: List[int]) -> Tuple[List[int], int]:
    """Return (sorted_blade, sign) after bubble‑sorting index list."""
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
                del lst[j:j + 2]
                n -= 2
                continue
            j += 1
        i += 1
    return lst, sign


def _multiply_blades(blade_a: frozenset, blade_b: frozenset) -> Tuple[frozenset, int]:
    """Multiply two basis blades, returning (result_blade, sign)."""
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


class Multivector:
    """Element of Cl(n,0) represented as a sparse sum of basis blades."""

    def __init__(self, components: Dict[frozenset, float], n: int):
        self.n = int(n)
        # prune near‑zero entries
        self.components = {k: v for k, v in components.items() if abs(v) > 1e-15}

    # ------------------------------------------------------------------
    # Algebraic operations
    # ------------------------------------------------------------------
    def __add__(self, other: 'Multivector') -> 'Multivector':
        if self.n != other.n:
            raise ValueError('grade mismatch')
        comp = self.components.copy()
        for b, v in other.components.items():
            comp[b] = comp.get(b, 0.0) + v
            if abs(comp[b]) < 1e-15:
                del comp[b]
        return Multivector(comp, self.n)

    def __rmul__(self, scalar: float) -> 'Multivector':
        return Multivector({b: scalar * v for b, v in self.components.items()}, self.n)

    def geometric_product(self, other: 'Multivector') -> 'Multivector':
        if self.n != other.n:
            raise ValueError('grade mismatch')
        result: Dict[frozenset, float] = {}
        for ba, va in self.components.items():
            for bb, vb in other.components.items():
                bc, sign = _multiply_blades(ba, bb)
                result[bc] = result.get(bc, 0.0) + sign * va * vb
        return Multivector(result, self.n)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def as_vector(self, blade_order: List[frozenset]) -> np.ndarray:
        """Return coefficient vector ordered by ``blade_order``."""
        return np.array([self.components.get(b, 0.0) for b in blade_order], dtype=float)

    @staticmethod
    def from_vector(vec: np.ndarray, blade_order: List[frozenset], n: int) -> 'Multivector':
        comp = {b: float(v) for b, v in zip(blade_order, vec) if abs(v) > 1e-15}
        return Multivector(comp, n)


def generate_blade_order(n: int) -> List[frozenset]:
    """All basis blades of Cl(n,0) ordered lexicographically."""
    blades: List[frozenset] = [frozenset()]  # scalar
    for grade in range(1, n + 1):
        # generate combinations of basis indices
        indices = list(range(n))
        from itertools import combinations
        for combo in combinations(indices, grade):
            blades.append(frozenset(combo))
    return blades


def koopman_step(mv: Multivector, K: np.ndarray, blade_order: List[frozenset]) -> Multivector:
    """Linear Koopman evolution: v_{t+1} = K @ v_t."""
    v = mv.as_vector(blade_order)
    if K.shape != (len(v), len(v)):
        raise ValueError('Koopman matrix dimension mismatch')
    v_next = K @ v
    return Multivector.from_vector(v_next, blade_order, mv.n)


def fisher_weight(mv: Multivector, samples: np.ndarray, eps: float = 1e-12) -> float:
    """
    Approximate Fisher information weight for a multivector.
    Treat each coefficient as a parameter θ_i and estimate
        I = Σ (∂log p/∂θ_i)^2  ≈  Σ (Δθ_i / θ_i)^2
    where Δθ_i is the sample variance.
    """
    coeffs = np.array(list(mv.components.values()))
    if coeffs.size == 0:
        return 0.0
    var = np.var(samples, axis=0)[:coeffs.size]
    # avoid division by zero
    denom = coeffs ** 2 + eps
    return float(np.sum(var / denom))


# ----------------------------------------------------------------------
# 3.  Hybrid operations
# ----------------------------------------------------------------------
def pack_edge_into_multivector(conductance: float,
                               direction: Tuple[float, float],
                               n: int = 3) -> Multivector:
    """
    Encode an edge as a multivector:
    - scalar part  = conductance
    - vector part  = direction (padded/truncated to n components)
    """
    # scalar blade
    components: Dict[frozenset, float] = {frozenset(): conductance}
    # vector blades e0, e1, e2 ...
    for i, coord in enumerate(direction[:n]):
        components[frozenset({i})] = float(coord)
    return Multivector(components, n)


def hybrid_edge_update(edge_id: str,
                       conductance: float,
                       edge_length: float,
                       pressure_a: float,
                       pressure_b: float,
                       pos_a: Tuple[float, float],
                       pos_b: Tuple[float, float],
                       bandit_action: 'BanditAction',
                       K: np.ndarray,
                       samples: np.ndarray) -> float:
    """
    Perform a single hybrid update for one edge.

    Steps:
    1. Compute physarum flux.
    2. Build a multivector from conductance and geometric direction.
    3. Evolve it with the Koopman operator.
    4. Derive a Fisher weight from ``samples``.
    5. Combine weight, bandit propensity and reward into a hybrid q.
    6. Update conductance using the physarum rule.
    """
    # 1. flux
    f = flux(conductance, edge_length, pressure_a, pressure_b)

    # 2. geometric direction (Voronoi‑like)
    direction = (pos_b[0] - pos_a[0], pos_b[1] - pos_a[1])

    # 3. multivector pack & Koopman evolution
    mv = pack_edge_into_multivector(conductance, direction, n=3)
    blade_order = generate_blade_order(mv.n)
    mv_evolved = koopman_step(mv, K, blade_order)

    # 4. Fisher weighting
    w = fisher_weight(mv_evolved, samples)

    # 5. hybrid q (propensity·reward·|f|·w)
    hybrid_q = bandit_action.propensity * bandit_action.expected_reward * abs(f) * w

    # 6. conductance update
    new_g = update_conductance(conductance, hybrid_q, dt=1.0, gain=1.0, decay=0.05)
    return new_g


# ----------------------------------------------------------------------
# 4.  Simple data structures for the bandit side
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str


# ----------------------------------------------------------------------
# 5.  Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a tiny network with a single edge
    edge_id = "e0"
    conductance = 0.8
    edge_length = 2.0
    pressure_a = 1.0
    pressure_b = 0.2
    pos_a = (0.0, 0.0)
    pos_b = (1.0, 1.0)

    # Dummy bandit action
    action = BanditAction(
        action_id="a0",
        propensity=0.6,
        expected_reward=1.2,
        confidence_bound=0.1,
        algorithm="UCB"
    )

    # Koopman matrix (identity + small random perturbation)
    n = 3  # geometric dimension used in pack_edge_into_multivector
    blade_order = generate_blade_order(n)
    dim = len(blade_order)
    np.random.seed(42)
    K = np.eye(dim) + 0.01 * np.random.randn(dim, dim)

    # Sample matrix for Fisher weight (10 samples of random coeffs)
    samples = np.random.randn(10, dim)

    new_conductance = hybrid_edge_update(
        edge_id,
        conductance,
        edge_length,
        pressure_a,
        pressure_b,
        pos_a,
        pos_b,
        action,
        K,
        samples
    )

    print(f"Old conductance: {conductance:.4f}")
    print(f"New conductance: {new_conductance:.4f}")