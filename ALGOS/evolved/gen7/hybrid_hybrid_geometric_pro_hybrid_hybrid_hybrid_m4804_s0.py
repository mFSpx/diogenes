# DARWIN HAMMER — match 4804, survivor 0
# gen: 7
# parent_a: hybrid_geometric_product_hybrid_hybrid_hybrid_m2370_s1.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1994_s0.py (gen5)
# born: 2026-05-29T23:59:43Z

"""
Hybrid Geometric‑Product Curvature‑Bandit Algorithm
===================================================

This module fuses the two parents:

* **Parent A** – *geometric_product.py* : provides a Clifford‑algebra
  ``Multivector`` and the ``geometric_product`` that can represent endpoint
  health scores as multivectors.

* **Parent B** – *hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1994_s0.py* :
  supplies the TTT‑Linear update rule and the Ollivier‑Ricci curvature that
  modulates a bandit‑style confidence term.

**Mathematical bridge**

For every endpoint we build a multivector ``h`` whose scalar part is the
raw health score ``s = w·x`` (dot‑product of a weight vector ``w`` and the
feature vector ``x``) and whose higher‑grade components encode auxiliary
statistics (here we store the curvature ``κ`` as a bivector coefficient).
The confidence term of the bandit is multiplied by a function of the
curvature, producing a *curvature‑aware* confidence ``c``.  A Hoeffding
bound on the empirical mean of the multivector health scores decides whether
the currently selected endpoint should be switched.

The three public functions below implement the full hybrid loop:

1. ``compute_multivector_health_scores`` – builds multivectors for all
   endpoints.
2. ``update_endpoint`` – applies the TTT‑Linear weight update using the
   Ollivier‑Ricci curvature.
3. ``maybe_switch_endpoint`` – uses Hoeffding’s inequality on the scalar
   parts of the health multivectors (weighted by the curvature‑aware
   confidence) to decide on a switch.

All operations stay within NumPy and the Python standard library.
"""

import math
import random
import sys
import pathlib
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Clifford algebra utilities (shared by both parents)
# ----------------------------------------------------------------------


def _blade_sign(indices: Tuple[int, ...]) -> Tuple[List[int], int]:
    """Return a sorted list of blade indices and the sign produced by the
    anticommutative swaps needed to sort them.  Identical indices cancel
    (e.g. e_i∧e_i = 0)."""
    lst = list(indices)
    sign = 1
    n = len(lst)
    for i in range(n):
        j = 0
        while j < n - 1 - i:
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
                j += 1
            elif lst[j] == lst[j + 1]:
                # cancel duplicate basis vector
                lst.pop(j)
                lst.pop(j)  # the next element shifts left
                n -= 2
                sign *= 1
                # stay at same j because we removed two items
            else:
                j += 1
    return lst, sign


def _multiply_blades(blade_a: frozenset, blade_b: frozenset) -> Tuple[frozenset, int]:
    """Geometric product of two basis blades (as frozensets of indices)."""
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(tuple(combined))
    return frozenset(result), sign


class Multivector:
    """Sparse representation of a multivector.

    ``components`` maps a blade (frozenset of basis indices) to its scalar
    coefficient.  The scalar (grade‑0) blade is represented by ``frozenset()``.
    """

    def __init__(self, components: Dict[frozenset, float] = None):
        self.components: Dict[frozenset, float] = components or {}

    def copy(self) -> "Multivector":
        return Multivector(self.components.copy())

    def __add__(self, other: "Multivector") -> "Multivector":
        result = self.copy()
        for blade, coeff in other.components.items():
            result.components[blade] = result.components.get(blade, 0.0) + coeff
        return result

    def __rmul__(self, scalar: float) -> "Multivector":
        return Multivector({b: scalar * c for b, c in self.components.items()})

    def scalar(self) -> float:
        """Return the grade‑0 part (the ordinary health score)."""
        return self.components.get(frozenset(), 0.0)


def geometric_product(a: Multivector, b: Multivector) -> Multivector:
    """Full geometric product between two multivectors."""
    result = Multivector()
    for blade_a, coeff_a in a.components.items():
        for blade_b, coeff_b in b.components.items():
            blade_res, sign = _multiply_blades(blade_a, blade_b)
            result.components[blade_res] = result.components.get(blade_res, 0.0) + sign * coeff_a * coeff_b
    return result


# ----------------------------------------------------------------------
# Parent B – curvature and bandit utilities
# ----------------------------------------------------------------------


def ollivier_ricci_curvature(W: np.ndarray, x: np.ndarray, target: np.ndarray = None) -> float:
    """Curvature = squared residual of the TTT‑Linear model."""
    pred = W @ x
    t = x if target is None else target
    residual = pred - t
    return float(residual @ residual) + 1e-12  # avoid division by zero


def ttt_linear_update(W: np.ndarray, x: np.ndarray, target: np.ndarray = None, lr: float = 0.01) -> np.ndarray:
    """Weight update using a random gradient scaled by curvature."""
    grad = np.random.rand(*W.shape)  # placeholder for a true gradient
    curvature = ollivier_ricci_curvature(W, x, target)
    W_new = W + lr * grad / curvature
    return W_new


def curvature_modulated_confidence(curvature: float, N: int) -> float:
    """Bandit confidence term where curvature boosts exploration."""
    # The original confidence term: 1 / sqrt(N)
    base = 1.0 / math.sqrt(1 + N)
    # Curvature factor: grows with curvature but is bounded.
    factor = 1.0 + math.tanh(curvature)  # in (1,2)
    return factor * base


# ----------------------------------------------------------------------
# Hybrid core – health scores as multivectors + curvature‑aware bandit
# ----------------------------------------------------------------------


def compute_multivector_health_scores(
    weights: np.ndarray,
    features: List[np.ndarray],
) -> List[Multivector]:
    """
    For each endpoint compute a multivector:

    * scalar part = w·x  (raw health score)
    * bivector part (grade‑2) = curvature κ stored on blade (0,1)
      (any fixed bivector index pair works; we use {0,1}).

    Parameters
    ----------
    weights : (d,) array
        Global weight vector used for all endpoints.
    features : list of (d,) arrays
        Feature vector for each endpoint.

    Returns
    -------
    List[Multivector]
        Health multivectors, one per endpoint.
    """
    healths = []
    for x in features:
        score = float(weights @ x)                     # scalar health
        curvature = ollivier_ricci_curvature(np.diag(weights), x)  # scalar curvature
        # Encode curvature as a bivector coefficient on blade {0,1}
        bivector_blade = frozenset({0, 1})
        mv = Multivector(
            {
                frozenset(): score,
                bivector_blade: curvature,
            }
        )
        healths.append(mv)
    return healths


def update_endpoint(
    weight_matrix: np.ndarray,
    feature: np.ndarray,
    target: np.ndarray = None,
) -> np.ndarray:
    """
    Apply the TTT‑Linear update to the endpoint's weight matrix,
    using the curvature computed from the current state.

    Returns the updated weight matrix.
    """
    return ttt_linear_update(weight_matrix, feature, target)


def maybe_switch_endpoint(
    health_multivectors: List[Multivector],
    pulls: List[int],
    delta: float = 0.05,
) -> Tuple[int, bool]:
    """
    Decide whether to switch to a different endpoint using Hoeffding's bound
    on the scalar parts of the health multivectors, weighted by a
    curvature‑modulated confidence term.

    Parameters
    ----------
    health_multivectors : list of Multivector
        Current health estimates for each endpoint.
    pulls : list of int
        Number of times each endpoint has been selected so far.
    delta : float
        Desired failure probability for the Hoeffding bound.

    Returns
    -------
    (best_index, switched)
        ``best_index`` – index of the chosen endpoint after the test.
        ``switched``   – True if a switch occurred, False otherwise.
    """
    # Compute confidence‑adjusted scores
    adjusted_scores = []
    for mv, n in zip(health_multivectors, pulls):
        scalar = mv.scalar()
        # extract curvature (grade‑2 coefficient) if present
        curvature = mv.components.get(frozenset({0, 1}), 0.0)
        conf = curvature_modulated_confidence(curvature, n)
        adjusted_scores.append(scalar + conf)  # higher is better

    # Hoeffding bound on the scalar part (range assumed [0,1] after normalization)
    # For safety we normalize scores to [0,1] using min‑max of current batch.
    mn, mx = min(adjusted_scores), max(adjusted_scores)
    if mx - mn < 1e-12:
        normalized = [0.5] * len(adjusted_scores)
    else:
        normalized = [(s - mn) / (mx - mn) for s in adjusted_scores]

    n_total = sum(pulls) + len(pulls)  # add 1 per arm for smoothing
    epsilon = math.sqrt((1.0 / (2 * n_total)) * math.log(2.0 / delta))

    # Choose the arm with maximal normalized score, but only switch if it
    # exceeds the current best by more than epsilon.
    current_idx = max(range(len(normalized)), key=lambda i: pulls[i])  # naive current
    best_idx = max(range(len(normalized)), key=lambda i: normalized[i])

    switched = False
    if normalized[best_idx] - normalized[current_idx] > epsilon:
        switched = True
        chosen = best_idx
    else:
        chosen = current_idx

    return chosen, switched


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Simple reproducible test with three endpoints
    dim = 4
    rng = np.random.default_rng(42)

    # Global weight vector (used for scalar health)
    w_global = rng.normal(size=dim)

    # Random feature vectors for three endpoints
    features = [rng.normal(size=dim) for _ in range(3)]

    # Initialise pull counts (how many times each endpoint was used)
    pulls = [5, 3, 7]

    # Compute health multivectors
    healths = compute_multivector_health_scores(w_global, features)

    # Perform a potential switch decision
    chosen, switched = maybe_switch_endpoint(healths, pulls)

    print(f"Chosen endpoint: {chosen} (switched={switched})")

    # Update the weight matrix of the chosen endpoint
    W = np.diag(w_global)  # simple diagonal weight matrix for demonstration
    W_updated = update_endpoint(W, features[chosen])

    # Show that the update ran without error
    print("Weight matrix updated. Norm change:",
          np.linalg.norm(W_updated - W))