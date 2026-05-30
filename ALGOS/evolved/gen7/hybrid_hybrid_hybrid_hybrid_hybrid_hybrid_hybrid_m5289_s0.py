# DARWIN HAMMER — match 5289, survivor 0
# gen: 7
# parent_a: hybrid_hybrid_hybrid_geomet_hybrid_hybrid_hybrid_m1376_s3.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1203_s2.py (gen6)
# born: 2026-05-30T00:01:04Z

"""Hybrid Algorithm Fusion of Geometric Algebra (Parent A) and Gini‑Weighted Bandit Bilinear Form (Parent B)

Parents:
- hybrid_hybrid_hybrid_geomet_hybrid_hybrid_hybrid_m1376_s3.py (Geometric Algebra multivector core)
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1203_s2.py (Gini‑modulated regret bandit with bilinear projection)

Mathematical Bridge:
The bridge is a *bilinear form* ⟨w | x⟩ where both the feature vector `x` and the
weight vector `w` are represented as **Multivectors**.  The geometric product of the
two multivectors yields a new multivector; its scalar part is the bilinear
evaluation.  This scalar is then modulated by a Gini‑coefficient computed over
the expected rewards of the candidate actions, providing a regret‑weighted
adjustment to each action’s propensity in the bandit policy.
"""

import numpy as np
import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Set, Tuple, FrozenSet

# ----------------------------------------------------------------------
# Parent A – Geometric Algebra core
# ----------------------------------------------------------------------
class Multivector:
    """Sparse representation of a multivector in an n‑dimensional GA."""
    def __init__(self, components: Dict[FrozenSet[int], float], n: int):
        # filter near‑zero components for stability
        self.components = {k: float(v) for k, v in components.items() if abs(v) > 1e-15}
        self.n = int(n)

    def grade(self, k: int) -> 'Multivector':
        """Return the grade‑k part of the multivector."""
        return Multivector(
            {blade: coef for blade, coef in self.components.items() if len(blade) == k},
            self.n,
        )

    def scalar_part(self) -> float:
        """Extract the scalar (grade‑0) component."""
        return self.components.get(frozenset(), 0.0)

    # ------------------------------------------------------------------
    # Linear operations
    # ------------------------------------------------------------------
    def __add__(self, other: 'Multivector') -> 'Multivector':
        result = dict(self.components)
        for blade, coef in other.components.items():
            result[blade] = result.get(blade, 0.0) + coef
        return Multivector({k: v for k, v in result.items() if abs(v) > 1e-15}, self.n)

    def __sub__(self, other: 'Multivector') -> 'Multivector':
        return self + (-other)

    def __neg__(self) -> 'Multivector':
        return Multivector({k: -v for k, v in self.components.items()}, self.n)

    # ------------------------------------------------------------------
    # Multiplication
    # ------------------------------------------------------------------
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Multivector({k: v * other for k, v in self.components.items()}, self.n)
        if isinstance(other, Multivector):
            return geometric_product(self, other)
        raise TypeError("Unsupported multiplication with type {}".format(type(other)))

    __rmul__ = __mul__

def blade_mul(b1: FrozenSet[int], b2: FrozenSet[int]) -> Tuple[FrozenSet[int], int]:
    """
    Geometric product of two basis blades represented as frozensets of basis indices.
    Returns (resulting blade, sign) where sign = ±1 accounts for anti‑commutation.
    """
    result = list(b1) + list(b2)
    sign = 1
    i = 0
    while i < len(result):
        j = i + 1
        while j < len(result):
            if result[i] == result[j]:
                # duplicate basis vector squares to scalar → remove both
                del result[j]
                del result[i]
                i -= 1
                break
            elif result[i] > result[j]:
                # swap to enforce canonical order, flipping sign
                result[i], result[j] = result[j], result[i]
                sign = -sign
                j += 1
            else:
                j += 1
        i += 1
    return frozenset(result), sign

def geometric_product(a: Multivector, b: Multivector) -> Multivector:
    """Full geometric product a ∘ b."""
    result: Dict[FrozenSet[int], float] = {}
    for blade_a, coeff_a in a.components.items():
        for blade_b, coeff_b in b.components.items():
            blade_res, sign = blade_mul(blade_a, blade_b)
            result[blade_res] = result.get(blade_res, 0.0) + sign * coeff_a * coeff_b
    return Multivector({k: v for k, v in result.items() if abs(v) > 1e-15}, a.n)

# ----------------------------------------------------------------------
# Parent B – Bandit & Gini utilities
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathAction:
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0

@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float               # probability of selection
    expected_reward: float
    confidence_bound: float
    algorithm: str = "hybrid"

# ----------------------------------------------------------------------
# Hybrid Functions
# ----------------------------------------------------------------------
def gini_coefficient(values: List[float]) -> float:
    """
    Compute the Gini coefficient of a list of non‑negative numbers.
    G = (∑_i ∑_j |x_i - x_j|) / (2 * n² * mean)
    """
    if not values:
        return 0.0
    arr = np.array(values, dtype=float)
    if np.any(arr < 0):
        raise ValueError("Gini coefficient is defined for non‑negative values.")
    n = len(arr)
    mean = np.mean(arr)
    if mean == 0:
        return 0.0
    diff_sum = np.abs(arr[:, None] - arr[None, :]).sum()
    return diff_sum / (2 * n * n * mean)

def bilinear_form_mv(features: Multivector, weights: Multivector) -> float:
    """
    Evaluate the bilinear form ⟨weights | features⟩ using the geometric product.
    The scalar part of the product is the desired scalar output.
    """
    prod = geometric_product(weights, features)
    return prod.scalar_part()

def hybrid_regret_weighted_update(
    actions: List[BanditAction],
    features: Multivector,
    weight_mv: Multivector,
    temperature: float = 0.1,
) -> List[BanditAction]:
    """
    Perform a single policy update:
    1. Compute a bilinear score s_i = ⟨w | x_i⟩ for each action (here x_i is the same
       feature multivector; in practice each action could have its own embedding).
    2. Modulate each score by the Gini coefficient of the expected rewards across
       the action set, producing a regret‑aware scaling factor.
    3. Apply a soft‑max with temperature to obtain new propensities.
    4. Return a new list of BanditAction objects with updated propensities.
    """
    # Step 1 – bilinear scores (identical features for simplicity)
    base_score = bilinear_form_mv(features, weight_mv)

    # Step 2 – Gini‑based scaling
    rewards = [a.expected_reward for a in actions]
    gini = gini_coefficient(rewards)
    scaling = 1.0 + gini  # higher inequality → stronger exploitation

    # Step 3 – soft‑max over scaled scores
    exp_vals = np.array([math.exp((base_score * scaling) / temperature) for _ in actions])
    probs = exp_vals / exp_vals.sum()

    # Step 4 – construct updated actions
    updated = []
    for a, p in zip(actions, probs):
        updated.append(
            BanditAction(
                action_id=a.action_id,
                propensity=p,
                expected_reward=a.expected_reward,
                confidence_bound=a.confidence_bound,
                algorithm=a.algorithm,
            )
        )
    return updated

def sample_random_multivector(dim: int, max_grade: int = 2) -> Multivector:
    """
    Utility to generate a random sparse multivector up to `max_grade`.
    """
    components: Dict[FrozenSet[int], float] = {}
    for grade in range(max_grade + 1):
        # choose a random number of blades of this grade
        count = random.randint(0, max(1, dim // (grade + 1)))
        for _ in range(count):
            blade = frozenset(random.sample(range(dim), grade))
            components[blade] = random.uniform(-1.0, 1.0)
    return Multivector(components, dim)

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a 4‑dimensional feature multivector and a weight multivector
    dim = 4
    features_mv = sample_random_multivector(dim)
    weights_mv = sample_random_multivector(dim)

    # Define a small bandit action set
    actions = [
        BanditAction(action_id="A", propensity=0.33, expected_reward=1.2, confidence_bound=0.1),
        BanditAction(action_id="B", propensity=0.33, expected_reward=0.8, confidence_bound=0.2),
        BanditAction(action_id="C", propensity=0.34, expected_reward=1.0, confidence_bound=0.15),
    ]

    # Perform hybrid update
    updated_actions = hybrid_regret_weighted_update(actions, features_mv, weights_mv)

    # Print results for verification
    print("Original propensities:", [a.propensity for a in actions])
    print("Updated propensities :", [a.propensity for a in updated_actions])
    print("Gini of rewards      :", gini_coefficient([a.expected_reward for a in actions]))
    print("Bilinear score       :", bilinear_form_mv(features_mv, weights_mv))