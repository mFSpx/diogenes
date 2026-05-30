# DARWIN HAMMER — match 4174, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2358_s0.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1216_s5.py (gen6)
# born: 2026-05-29T23:53:56Z

"""Hybrid Algorithm Fusion of Geometric Algebra Decision Hygiene (Parent A) and Bandit‑Router Store Dynamics with Stylometric Workshare (Parent B).

Mathematical Bridge
-------------------
Parent A encodes decision‑hygiene feature counts as a *feature‑count vector* `c ∈ ℝⁿ` and lifts it into a geometric‑algebra multivector  
`C = Σ_i c_i e_i`.  The scalar part of the geometric product `C·w` (dot product with a weight vector `w`) yields the same inner product used in Parent B’s store update (Equation 1).

Parent B updates a store vector `ℓ ∈ ℝ^m` by

ℓ_{t+1} = ℓ_t + dt·(α·ℓ_t + β·⟨w, σ⟩)

where `σ` is a signature vector derived from stylometry.  In the hybrid we replace `σ` by the scalar part of `C·w`, i.e. the dot product of the multivector‑encoded hygiene features with the same weight vector `w`.  The updated store is then normalised to a probability simplex `π` and drives a work‑share allocation (Equation 2).

Thus the fusion consists of:
1. Building a stylometric‑style signature from raw text and lifting it to a multivector.
2. Using the multivector‑dot with `w` inside the store dynamics.
3. Propagating the resulting simplex probabilities to bandit‑action propensities.

The module implements this closed‑loop hybrid system with three core functions.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, replace
from typing import List, Tuple, Dict

import numpy as np


# ----------------------------------------------------------------------
# Minimal Geometric Algebra utilities (Parent A core)
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
                # duplicate index → blade vanishes
                del lst[j : j + 2]
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
    """Simple multivector limited to scalar and vector grades (Cl(n,0))."""

    def __init__(self, components: Dict[frozenset, float], n: int):
        self.components = {k: v for k, v in components.items() if abs(v) > 1e-12}
        self.n = n

    @staticmethod
    def from_vector(vec: np.ndarray) -> "Multivector":
        """Lift a real vector to a grade‑1 multivector."""
        comps = {frozenset({i}): float(v) for i, v in enumerate(vec)}
        return Multivector(comps, n=len(vec))

    def scalar_part(self) -> float:
        return self.components.get(frozenset(), 0.0)

    def vector_part(self) -> np.ndarray:
        v = np.zeros(self.n)
        for blade, coeff in self.components.items():
            if len(blade) == 1:
                idx = next(iter(blade))
                v[idx] = coeff
        return v

    def dot(self, other: "Multivector") -> float:
        """Scalar (inner) product of two grade‑1 multivectors."""
        if self.n != other.n:
            raise ValueError("Dimension mismatch in dot product")
        a = self.vector_part()
        b = other.vector_part()
        return float(np.dot(a, b))

    def __add__(self, other: "Multivector") -> "Multivector":
        if self.n != other.n:
            raise ValueError("Dimension mismatch in addition")
        new_comps = self.components.copy()
        for blade, coeff in other.components.items():
            new_comps[blade] = new_comps.get(blade, 0.0) + coeff
        return Multivector(new_comps, self.n)

    def __rmul__(self, scalar: float) -> "Multivector":
        new_comps = {blade: scalar * coeff for blade, coeff in self.components.items()}
        return Multivector(new_comps, self.n)

    def __repr__(self) -> str:
        return f"Multivector({self.components}, n={self.n})"


# ----------------------------------------------------------------------
# Stylometric signature generation (Parent B core)
# ----------------------------------------------------------------------


def compute_signature(text: str) -> np.ndarray:
    """
    Build a low‑dimensional stylometric signature from raw text.

    The signature consists of five counts:
        0 – vowels,
        1 – consonants,
        2 – digits,
        3 – whitespace characters,
        4 – punctuation marks.
    The vector is normalised to unit length (L2) to keep scale comparable.
    """
    vowels = set("aeiouAEIOU")
    punctuation = set('.,;:!?-()"\'')
    v = c = d = s = p = 0
    for ch in text:
        if ch.isalpha():
            if ch in vowels:
                v += 1
            else:
                c += 1
        elif ch.isdigit():
            d += 1
        elif ch.isspace():
            s += 1
        elif ch in punctuation:
            p += 1
    vec = np.array([v, c, d, s, p], dtype=float)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------


def store_update_and_allocate(
    sig_mv: Multivector,
    store: np.ndarray,
    dt: float,
    alpha: float,
    beta: float,
    w: np.ndarray,
    total_units: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform the fused store dynamics and work‑share allocation.

    1. Compute the scalar dot product ⟨w, σ⟩ where σ is the vector part of `sig_mv`.
    2. Update the store according to the hybrid version of Equation 1.
    3. Normalise the updated store to a probability simplex π.
    4. Allocate `total_units` proportionally to π (Equation 2).

    Returns
    -------
    store_new : np.ndarray
        Updated store vector.
    allocation : np.ndarray
        Amount of work‑share assigned to each store component.
    pi : np.ndarray
        Normalised probability simplex.
    """
    # Step 1: inner product using geometric algebra dot
    sigma_dot_w = sig_mv.dot(Multivector.from_vector(w))

    # Step 2: store update (vectorised)
    store_new = store + dt * (alpha * store + beta * sigma_dot_w)

    # Guard against negative or zero sum
    store_new = np.maximum(store_new, 0.0)
    sum_store = store_new.sum()
    if sum_store == 0:
        pi = np.full_like(store_new, 1.0 / store_new.size)
    else:
        pi = store_new / sum_store

    # Step 3: allocation
    allocation = pi * total_units
    return store_new, allocation, pi


@dataclass(frozen=True)
class BanditAction:
    """Result of an action selection (Parent B data structure)."""
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str


def adjust_bandit_propensities(
    actions: List[BanditAction],
    allocation: np.ndarray,
    total_units: float,
) -> List[BanditAction]:
    """
    Rescale bandit propensities using the allocation outcome.

    The allocation vector is interpreted as a multiplicative boost:
        new_propensity = old_propensity * (1 + allocation_i / total_units)

    The function returns a new list of immutable `BanditAction` objects.
    """
    if len(actions) != len(allocation):
        raise ValueError("Length mismatch between actions and allocation")

    new_actions = []
    for act, share in zip(actions, allocation):
        boost = 1.0 + (share / total_units)
        new_propensity = act.propensity * boost
        new_act = replace(act, propensity=new_propensity)
        new_actions.append(new_act)
    return new_actions


def hybrid_step(
    text: str,
    store: np.ndarray,
    actions: List[BanditAction],
    w: np.ndarray,
    dt: float = 0.1,
    alpha: float = 0.5,
    beta: float = 0.3,
    total_units: float = 100.0,
) -> Tuple[np.ndarray, List[BanditAction], np.ndarray]:
    """
    One complete hybrid iteration:
        1. Compute stylometric signature and lift to a multivector.
        2. Update the store and obtain allocation.
        3. Adjust bandit propensities.

    Returns
    -------
    store_new : np.ndarray
        Updated store after the iteration.
    actions_new : List[BanditAction]
        Bandit actions with refreshed propensities.
    pi : np.ndarray
        Probability simplex derived from the updated store.
    """
    # 1. Signature → multivector
    sig_vec = compute_signature(text)
    sig_mv = Multivector.from_vector(sig_vec)

    # 2. Store dynamics + allocation
    store_new, allocation, pi = store_update_and_allocate(
        sig_mv, store, dt, alpha, beta, w, total_units
    )

    # 3. Propensity adjustment
    actions_new = adjust_bandit_propensities(actions, allocation, total_units)

    return store_new, actions_new, pi


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Sample data
    sample_text = (
        "In a distant future, the algorithms converge. "
        "Numbers 12345, punctuation! And spaces."
    )

    # Random weight vector for dot product (dimension must match signature length)
    rng = np.random.default_rng(42)
    w_vec = rng.normal(size=5)

    # Initial store (3‑dimensional for illustration)
    store_vec = np.array([10.0, 5.0, 15.0])

    # Dummy bandit actions (same length as store)
    actions_list = [
        BanditAction(
            action_id=f"act_{i}",
            propensity=0.2 + 0.1 * i,
            expected_reward=1.0,
            confidence_bound=0.5,
            algorithm="HybridBandit",
        )
        for i in range(3)
    ]

    # Run a single hybrid step
    new_store, new_actions, simplex = hybrid_step(
        text=sample_text,
        store=store_vec,
        actions=actions_list,
        w=w_vec,
        dt=0.05,
        alpha=0.6,
        beta=0.4,
        total_units=120.0,
    )

    print("Updated store:", new_store)
    print("Probability simplex π:", simplex)
    print("Adjusted actions:")
    for act in new_actions:
        print(f"  {act.action_id}: propensity={act.propensity:.4f}")