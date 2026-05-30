# DARWIN HAMMER — match 1611, survivor 5
# gen: 5
# parent_a: hybrid_hybrid_hybrid_distri_hybrid_hybrid_regret_m25_s5.py (gen4)
# parent_b: hybrid_hybrid_hybrid_geomet_hybrid_hybrid_fisher_m31_s3.py (gen4)
# born: 2026-05-29T23:37:54Z

"""Hybrid Leader‑Election, Regret‑Weighted Tree & Geometric Fisher Fusion.

Parents:
- hybrid_hybrid_hybrid_distri_hybrid_hybrid_regret_m25_s5.py (Hoeffding‑tree,
  tropical max‑plus gain, regret‑weighted distribution, simulated‑annealing)
- hybrid_hybrid_hybrid_geomet_hybrid_hybrid_fisher_m31_s3.py (Geometric algebra
  representation of Fisher information and hygiene scores)

Mathematical bridge:
Both parents manipulate scalar “energies” (ΔE, tropical gain) and probability
vectors (regret distribution π).  The geometric‑algebra parent provides a
linear‑algebraic embedding of scalar quantities into a multivector space.
We embed the Fisher information vector **F** and the leader‑signature vector
**σ** as multivectors, compute their inner product ⟨F|σ⟩ to obtain a similarity
measure that modulates the simulated‑annealing temperature.  The resulting
hybrid acceptance probability is

    ΔE      = ε – G                         (Hoeffding bound ε, tropical gain G)
    σ_sim   = ⟨F|σ⟩ / (‖F‖·‖σ‖)               (cosine‑like similarity in multivector space)
    T_eff   = T / (1 + λ·σ_sim)              (temperature cooled by similarity)
    p_acc   = exp( –ΔE / T_eff )            (Metropolis‑Hastings rule)

The module implements:
* a minimal Clifford (geometric‑algebra) engine (Multivector)
* encoding of Fisher information and signatures as multivectors
* regret‑weighted probability handling
* the hybrid acceptance rule

Three public functions illustrate the workflow:
    build_hybrid_state(...)
    hybrid_acceptance_probability(...)
    update_fisher_information(...)
"""

from __future__ import annotations

import math
import random
import sys
import pathlib
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple, FrozenSet

import numpy as np

# ----------------------------------------------------------------------
# Parent‑A data structures (regret engine)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathAction:
    """Action with expected value, cost and risk."""
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True)
class MathCounterfactual:
    """Counterfactual outcome for an action."""
    action_id: str
    outcome_value: float
    probability: float = 1.0


# ----------------------------------------------------------------------
# Parent‑B geometric algebra core
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
                # duplicate index → blade vanishes (eₖ∧eₖ = 0)
                del lst[j : j + 2]
                n -= 2
                sign *= 1
                continue
            j += 1
        i += 1
    return lst, sign


def _multiply_blades(
    blade_a: FrozenSet[int], blade_b: FrozenSet[int]
) -> Tuple[FrozenSet[int], int]:
    """Geometric product of two basis blades (set representation)."""
    combined = list(blade_a) + list(blade_b)
    sorted_blade, sign = _blade_sign(combined)
    return frozenset(sorted_blade), sign


class Multivector:
    """Simple Clifford algebra element Cl(n,0) using dictionary of blades."""

    def __init__(self, components: Dict[FrozenSet[int], float], n: int):
        # discard near‑zero components for stability
        self.components = {
            k: float(v) for k, v in components.items() if abs(v) > 1e-15
        }
        self.n = int(n)

    # ------------------------------------------------------------------
    # Basic algebraic operations
    # ------------------------------------------------------------------
    def __add__(self, other: "Multivector") -> "Multivector":
        result = dict(self.components)
        for blade, coeff in other.components.items():
            result[blade] = result.get(blade, 0.0) + coeff
        return Multivector(result, self.n)

    def __sub__(self, other: "Multivector") -> "Multivector":
        result = dict(self.components)
        for blade, coeff in other.components.items():
            result[blade] = result.get(blade, 0.0) - coeff
        return Multivector(result, self.n)

    def __neg__(self) -> "Multivector":
        return Multivector({b: -c for b, c in self.components.items()}, self.n)

    def __mul__(self, other: "Multivector") -> "Multivector":
        """Geometric product (associative, distributes over addition)."""
        result: Dict[FrozenSet[int], float] = {}
        for blade_a, coeff_a in self.components.items():
            for blade_b, coeff_b in other.components.items():
                blade_res, sign = _multiply_blades(blade_a, blade_b)
                result[blade_res] = result.get(blade_res, 0.0) + sign * coeff_a * coeff_b
        return Multivector(result, self.n)

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------
    def grade(self, k: int) -> "Multivector":
        """Extract grade‑k part (blades of size k)."""
        return Multivector(
            {blade: coeff for blade, coeff in self.components.items() if len(blade) == k},
            self.n,
        )

    def scalar_part(self) -> float:
        """Return the scalar (grade‑0) component, 0 if absent."""
        return self.components.get(frozenset(), 0.0)

    def norm(self) -> float:
        """Euclidean norm of the multivector (treating components as a vector)."""
        return math.sqrt(sum(c * c for c in self.components.values()))

    def inner(self, other: "Multivector") -> float:
        """Scalar inner product ⟨A|B⟩ = sum_{blade} a·b (grade‑0 of reverse(A)*B)."""
        # For Euclidean metric the inner product reduces to componentwise product.
        return sum(
            self.components.get(blade, 0.0) * other.components.get(blade, 0.0)
            for blade in set(self.components) | set(other.components)
        )

    def __repr__(self) -> str:
        terms = [f"{c:.3g}*e{sorted(list(b))}" if b else f"{c:.3g}" for b, c in self.components.items()]
        return " + ".join(terms) or "0"


# ----------------------------------------------------------------------
# Helper functions to embed vectors into multivectors
# ----------------------------------------------------------------------
def vector_to_multivector(vec: np.ndarray) -> Multivector:
    """Map a 1‑D numpy array to a grade‑1 multivector (each entry → basis blade)."""
    n = len(vec)
    components = {
        frozenset({i}): float(v) for i, v in enumerate(vec) if abs(v) > 1e-15
    }
    return Multivector(components, n)


def normalize_multivector(mv: Multivector) -> Multivector:
    """Return a unit‑norm multivector (preserves direction)."""
    nrm = mv.norm()
    if nrm == 0.0:
        return mv
    return Multivector({b: c / nrm for b, c in mv.components.items()}, mv.n)


# ----------------------------------------------------------------------
# Hybrid core: combine regret distribution with Fisher‑geometric information
# ----------------------------------------------------------------------
def build_hybrid_state(
    actions: List[MathAction],
    fisher_info: np.ndarray,
    signatures: np.ndarray,
) -> Dict[str, Any]:
    """
    Construct a dictionary containing:
        - regret distribution π (softmax over expected values)
        - multivector representation of Fisher information (F)
        - multivector representation of signatures (S)
        - normalized similarity σ = ⟨F̂|Ŝ⟩  (in [0,1])
    """
    # ---- regret distribution (softmax) ---------------------------------
    ev = np.array([a.expected_value for a in actions], dtype=float)
    # numerical stability
    ev_shift = ev - np.max(ev)
    exp_ev = np.exp(ev_shift)
    pi = exp_ev / exp_ev.sum() if exp_ev.size else np.array([])

    # ---- geometric embeddings -------------------------------------------
    F_mv = normalize_multivector(vector_to_multivector(fisher_info))
    S_mv = normalize_multivector(vector_to_multivector(signatures))

    sigma = F_mv.inner(S_mv)  # cosine‑like similarity because both are unit‑norm
    sigma = max(0.0, min(1.0, sigma))  # clamp for safety

    return {
        "actions": actions,
        "pi": pi,
        "F_mv": F_mv,
        "S_mv": S_mv,
        "sigma": sigma,
    }


def hybrid_acceptance_probability(
    state: Dict[str, Any],
    hoeffding_eps: float,
    tropical_gain: float,
    temperature: float,
    lambda_sim: float = 1.0,
) -> float:
    """
    Compute Metropolis‑Hastings acceptance probability for a candidate structural
    change using the hybrid formulation.

    Parameters
    ----------
    state : dict
        Output of ``build_hybrid_state`` containing ``sigma``.
    hoeffding_eps : float
        Hoeffding bound ε (uncertainty of the split estimate).
    tropical_gain : float
        Tropical max‑plus gain G (energy reduction if the split is kept).
    temperature : float
        Base temperature T for simulated annealing.
    lambda_sim : float
        Weight of similarity in temperature cooling.

    Returns
    -------
    float
        Acceptance probability p ∈ [0,1].
    """
    sigma = float(state.get("sigma", 0.0))

    delta_E = hoeffding_eps - tropical_gain  # ΔE = ε – G
    T_eff = temperature / (1.0 + lambda_sim * sigma)  # cooled temperature

    # Guard against division by zero or negative temperature
    if T_eff <= 0.0:
        return 0.0 if delta_E > 0 else 1.0

    exponent = -delta_E / T_eff
    # Clip exponent for numerical stability
    exponent = max(min(exponent, 700), -700)  # np.exp overflows near 709
    p_accept = math.exp(exponent)
    return max(0.0, min(1.0, p_accept))


def update_fisher_information(
    state: Dict[str, Any],
    new_fisher: np.ndarray,
) -> Dict[str, Any]:
    """
    Replace the Fisher information component of the hybrid state with a new
    vector, recompute the multivector and similarity σ, and return the updated
    state (leaving other entries untouched).

    Parameters
    ----------
    state : dict
        Existing hybrid state.
    new_fisher : np.ndarray
        Updated Fisher information vector.

    Returns
    -------
    dict
        Updated state with refreshed ``F_mv`` and ``sigma``.
    """
    F_mv = normalize_multivector(vector_to_multivector(new_fisher))
    S_mv = state["S_mv"]
    sigma = F_mv.inner(S_mv)
    sigma = max(0.0, min(1.0, sigma))

    new_state = dict(state)  # shallow copy
    new_state.update({"F_mv": F_mv, "sigma": sigma})
    return new_state


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a tiny action set
    actions = [
        MathAction(id="a1", expected_value=1.2, cost=0.1, risk=0.05),
        MathAction(id="a2", expected_value=0.8, cost=0.2, risk=0.02),
        MathAction(id="a3", expected_value=1.5, cost=0.15, risk=0.07),
    ]

    # Random Fisher information (positive) and signature vectors
    rng = np.random.default_rng(42)
    fisher_vec = rng.random(5) + 0.1  # ensure non‑zero
    signature_vec = rng.random(5)

    # Build hybrid state
    state = build_hybrid_state(actions, fisher_vec, signature_vec)

    # Parameters for acceptance test
    epsilon = 0.04               # Hoeffding bound
    tropical_gain = 0.03         # Tropical max‑plus gain
    base_T = 1.0                 # Temperature
    lambda_sim = 2.0             # Similarity weight

    p = hybrid_acceptance_probability(
        state,
        hoeffding_eps=epsilon,
        tropical_gain=tropical_gain,
        temperature=base_T,
        lambda_sim=lambda_sim,
    )
    print(f"Acceptance probability: {p:.4f}")

    # Update Fisher info and recompute probability
    new_fisher = rng.random(5) + 0.1
    state = update_fisher_information(state, new_fisher)
    p2 = hybrid_acceptance_probability(
        state,
        hoeffding_eps=epsilon,
        tropical_gain=tropical_gain,
        temperature=base_T,
        lambda_sim=lambda_sim,
    )
    print(f"Acceptance probability after Fisher update: {p2:.4f}")

    # Simple sanity checks
    assert 0.0 <= p <= 1.0
    assert 0.0 <= p2 <= 1.0
    print("Smoke test completed successfully.")