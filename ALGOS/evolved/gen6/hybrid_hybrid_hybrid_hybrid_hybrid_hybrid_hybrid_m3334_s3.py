# DARWIN HAMMER — match 3334, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2010_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1892_s3.py (gen5)
# born: 2026-05-29T23:49:20Z

"""Hybrid Bandit‑NLMS‑Physarum Multivector Model
================================================

This module fuses the two parent algorithms:

* **Parent A** – a bandit router whose propensity scores are modified by an
  NLMS (Normalized Least‑Mean‑Squares) weight update that is weighted by
  epistemic certainty flags stored in a `StoreState` (honey‑bee style store).

* **Parent B** – a physarum‑inspired conductance dynamics expressed with a
  Clifford‑algebra `Multivector`.  Conductances are updated by a geometric‑product
  interaction with a flux multivector generated from a geometric path.

**Mathematical bridge**

Each discrete action of the bandit is associated with a basis blade of a
Clifford algebra.  The vector of propensities therefore becomes a *grade‑1*
multivector `g`.  A geometric path through a point cloud is transformed into a
flux multivector `f` (grade‑1) using a simple B‑spline‑like weighting.  The
physarum update


g_new = g + α·(g * f) – β·g


is applied directly to the propensity multivector, thus letting the geometry
of the path influence the bandit’s action scores.

After this geometric update the NLMS rule is applied to the same multivector
components, using the store’s current level as an epistemic certainty factor
`c ∈ [0,1]`:


w_i ← w_i + μ·c·e·x_i / (ε + ‖x‖²)


where `e = target – reward` is the error, `x` the context vector, and `μ` the
NLMS step size.  The result is a single hybrid system where bandit propensities,
physarum conductance dynamics and adaptive NLMS learning mutually reinforce each
other.

The public API consists of three demonstration functions:
`geometric_product`, `nlms_update`, and `hybrid_step`.  A small smoke test is
provided under ``if __name__ == "__main__":``.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# 1. Clifford algebra – Sparse Multivector
# ----------------------------------------------------------------------


class Multivector:
    """Sparse multivector in an n‑dimensional Euclidean Clifford algebra."""

    def __init__(self, components: Dict[FrozenSet[int], float], n: int):
        self.n = int(n)
        # drop near‑zero components for sparsity
        self.components = {
            k: float(v) for k, v in components.items() if abs(v) > 1e-15
        }

    # ------------------------------------------------------------------
    # basic selectors
    # ------------------------------------------------------------------
    def grade(self, k: int) -> "Multivector":
        """Return the grade‑k part of the multivector."""
        return Multivector(
            {b: c for b, c in self.components.items() if len(b) == k}, self.n
        )

    # ------------------------------------------------------------------
    # geometric product
    # ------------------------------------------------------------------
    def __mul__(self, other: "Multivector") -> "Multivector":
        """Geometric product `self * other` for Euclidean signature."""
        if not isinstance(other, Multivector):
            raise TypeError("Geometric product is defined between two Multivectors.")
        result: Dict[FrozenSet[int], float] = {}
        for blade_a, coeff_a in self.components.items():
            for blade_b, coeff_b in other.components.items():
                blade_res, sign = _blade_product(blade_a, blade_b)
                coeff = coeff_a * coeff_b * sign
                result[blade_res] = result.get(blade_res, 0.0) + coeff
        return Multivector(result, self.n)

    __rmul__ = __mul__

    # ------------------------------------------------------------------
    # utilities
    # ------------------------------------------------------------------
    def __add__(self, other: "Multivector") -> "Multivector":
        if not isinstance(other, Multivector):
            raise TypeError("Can only add two Multivectors.")
        result = self.components.copy()
        for blade, coeff in other.components.items():
            result[blade] = result.get(blade, 0.0) + coeff
        return Multivector(result, self.n)

    __radd__ = __add__

    def __sub__(self, other: "Multivector") -> "Multivector":
        return self + (-other)

    def __neg__(self) -> "Multivector":
        return Multivector({b: -c for b, c in self.components.items()}, self.n)

    def __repr__(self) -> str:
        terms = [f"{c:.3g}*e{sorted(list(b))}" if b else f"{c:.3g}" for b, c in self.components.items()]
        return " + ".join(terms) if terms else "0"

    def to_vector(self) -> np.ndarray:
        """Extract the grade‑1 (vector) part as a dense numpy array."""
        vec = np.zeros(self.n)
        for blade, coeff in self.components.items():
            if len(blade) == 1:
                idx = next(iter(blade))
                vec[idx] = coeff
        return vec

    @classmethod
    def from_vector(cls, vec: np.ndarray) -> "Multivector":
        comps = {frozenset([i]): float(v) for i, v in enumerate(vec) if abs(v) > 1e-15}
        return cls(comps, len(vec))


def _blade_product(b1: FrozenSet[int], b2: FrozenSet[int]) -> Tuple[FrozenSet[int], int]:
    """
    Compute the geometric product of two basis blades in Euclidean space.

    Returns
    -------
    blade : frozenset[int]
        Resulting blade (duplicates cancelled because e_i*e_i = 1).
    sign : int
        +1 or -1 depending on the number of swaps needed to reorder the
        concatenated basis list.
    """
    # Convert to mutable lists for ordering
    list1 = list(b1)
    list2 = list(b2)
    combined = list1 + list2
    sign = 1

    # Perform a bubble‑sort style swap count to bring `combined` into
    # ascending order while tracking sign flips.
    for i in range(len(combined)):
        for j in range(len(combined) - 1, i, -1):
            if combined[j - 1] > combined[j]:
                combined[j - 1], combined[j] = combined[j], combined[j - 1]
                sign = -sign

    # Cancel pairs of equal indices (e_i * e_i = 1)
    i = 0
    while i < len(combined) - 1:
        if combined[i] == combined[i + 1]:
            # Remove the pair
            del combined[i : i + 2]
            # No sign change because e_i*e_i = +1
            i = max(i - 1, 0)
        else:
            i += 1

    return frozenset(combined), sign


# ----------------------------------------------------------------------
# 2. Bandit / Store structures (from Parent A)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class BanditAction:
    """Result of an action selection."""

    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "HybridBandit"


@dataclass(frozen=True)
class BanditUpdate:
    """Single observation used to update the policy."""

    context_id: str
    action_id: str
    reward: float
    propensity: float


@dataclass
class StoreState:
    """Honey‑bee‑style store whose level modulates epistemic certainty."""

    level: float = 0.0
    alpha: float = 1.0
    beta: float = 1.0
    dt: float = 1.0
    base: float = 1.0
    gain: float = 1.0
    limit: float = 10.0

    def update(self, inflow: List[float], outflow: List[float]) -> Tuple[float, float]:
        """
        Apply the store equation and recompute the control signal.

        Returns
        -------
        new_level, delta
        """
        delta = self.alpha * sum(inflow) - self.beta * sum(outflow)
        self.level = max(0.0, self.level + self.dt * delta)
        return self.level, delta

    @property
    def certainty(self) -> float:
        """Normalized certainty in [0,1] derived from the current level."""
        return min(1.0, self.level / self.limit)

    @property
    def dance(self) -> float:
        """Bounded control signal derived from the last Δ (computed lazily)."""
        delta = getattr(self, "_last_delta", 0.0)
        return max(0.0, min(self.limit, self.base + self.gain * delta))

    def _store_last_delta(self, delta: float) -> None:
        """Internal helper to keep the most recent Δ for `dance`."""
        self._last_delta = delta


# ----------------------------------------------------------------------
# 3. Core hybrid operations
# ----------------------------------------------------------------------


def geometric_product(g: Multivector, f: Multivector) -> Multivector:
    """
    Wrapper around the multivector geometric product.

    Parameters
    ----------
    g, f : Multivector
        Input multivectors.

    Returns
    -------
    Multivector
        The geometric product `g * f`.
    """
    return g * f


def nlms_update(
    w: np.ndarray,
    x: np.ndarray,
    error: float,
    certainty: float,
    mu: float = 0.1,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Normalized LMS update weighted by epistemic certainty.

    w_{new} = w + μ·certainty·error·x / (ε + ‖x‖²)

    Parameters
    ----------
    w : np.ndarray
        Current weight (propensity) vector.
    x : np.ndarray
        Context (input) vector.
    error : float
        Target – actual reward.
    certainty : float
        Epistemic certainty factor in [0,1] (from StoreState).
    mu : float, optional
        Base step size.
    eps : float, optional
        Regularisation term to avoid division by zero.

    Returns
    -------
    np.ndarray
        Updated weight vector.
    """
    norm_sq = np.dot(x, x) + eps
    correction = (mu * certainty * error / norm_sq) * x
    return w + correction


def path_to_flux_multivector(path: np.ndarray, n: int) -> Multivector:
    """
    Convert a geometric path (NxD) into a grade‑1 flux multivector.

    A very lightweight B‑spline‑like weighting is used: each segment contributes
    a vector proportional to its length and direction, summed over the whole
    path.

    Parameters
    ----------
    path : np.ndarray, shape (M, D)
        Ordered points of the path.
    n : int
        Dimensionality of the Clifford algebra (must equal D).

    Returns
    -------
    Multivector
        Grade‑1 flux multivector.
    """
    if path.shape[1] != n:
        raise ValueError("Path dimensionality must match multivector dimension.")
    flux = np.zeros(n)
    for i in range(len(path) - 1):
        segment = path[i + 1] - path[i]
        length = np.linalg.norm(segment)
        if length > 0:
            flux += segment / length  # unit direction contribution
    return Multivector.from_vector(flux)


def hybrid_step(
    actions: List[BanditAction],
    context: np.ndarray,
    reward: float,
    target: float,
    path: np.ndarray,
    store: StoreState,
    alpha: float = 0.05,
    beta: float = 0.01,
    mu: float = 0.1,
) -> List[BanditAction]:
    """
    Perform one hybrid update cycle:

    1. Build a propensity multivector `g` from the current actions.
    2. Generate a flux multivector `f` from the supplied geometric path.
    3. Apply the physarum conductance update
           g ← g + α·(g * f) – β·g
    4. Extract the updated vector part and feed it to the NLMS rule,
       weighted by the store's epistemic certainty.
    5. Update the store based on inflow/outflow derived from reward dynamics.

    Parameters
    ----------
    actions : list[BanditAction]
        Current action set.
    context : np.ndarray
        Context (input) vector for NLMS.
    reward : float
        Observed reward for the selected action.
    target : float
        Desired reward (used to compute error).
    path : np.ndarray, shape (M, D)
        Geometric path that influences the conductance dynamics.
    store : StoreState
        Epistemic store whose level modulates certainty.
    alpha, beta : float
        Learning‑rate and decay for the physarum update.
    mu : float
        Base NLMS step size.

    Returns
    -------
    list[BanditAction]
        Updated actions with new propensities.
    """
    # 1. Propensity multivector (grade‑1)
    prop_vec = np.array([a.propensity for a in actions])
    g = Multivector.from_vector(prop_vec)

    # 2. Flux multivector from path
    f = path_to_flux_multivector(path, n=prop_vec.size)

    # 3. Physarum‑style conductance update
    g = g + alpha * geometric_product(g, f) - beta * g

    # 4. NLMS update on the vector part
    updated_vec = g.to_vector()
    error = target - reward
    updated_vec = nlms_update(
        updated_vec, context, error, store.certainty, mu=mu
    )

    # 5. Store dynamics: treat reward as inflow, (target‑reward) as outflow
    inflow = [max(0.0, reward)]
    outflow = [max(0.0, target - reward)]
    store.update(inflow, outflow)

    # Build new action list
    new_actions = []
    for idx, act in enumerate(actions):
        new_prop = float(updated_vec[idx])
        new_actions.append(
            BanditAction(
                action_id=act.action_id,
                propensity=new_prop,
                expected_reward=act.expected_reward,  # unchanged for demo
                confidence_bound=act.confidence_bound,
                algorithm=act.algorithm,
            )
        )
    return new_actions


# ----------------------------------------------------------------------
# 4. Demonstration / smoke test