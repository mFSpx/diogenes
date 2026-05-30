# DARWIN HAMMER — match 5138, survivor 3
# gen: 4
# parent_a: hybrid_hybrid_label_foundry_hybrid_hybrid_minimu_m542_s0.py (gen3)
# parent_b: hybrid_geometric_product_hybrid_model_vram_sc_m22_s1.py (gen2)
# born: 2026-05-30T00:00:15Z

"""Hybrid algorithm merging:
- Parent A: morphological indices, Bayesian label update utilities.
- Parent B: Clifford geometric product via Multivector representation.

Mathematical bridge:
Morphological scalar features (sphericity, flatness, righting‑time) are embedded as
grade‑1 blades of a multivector.  A weight multivector (grade‑1) encodes learned
coefficients.  The geometric product of feature and weight multivectors yields a
multivector whose scalar part aggregates the dot‑like interaction of the two
vectors.  This scalar is interpreted as an *effective likelihood* that feeds
into the Bayesian update from Parent A.  Consequently the hybrid system performs
a Bayesian posterior computation while the learning rule for the weight
multivector is expressed through geometric‑product‑based gradient updates.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from typing import Dict, FrozenSet, Tuple

# ---------- Parent A utilities ----------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length

def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)

def righting_time_index(m: Morphology, b: float = 1.0 / 3.0,
                        k: float = 0.35, neck_lever: float = 1.0) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    return b * (m.length / neck_lever) + k * m.mass

def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    if not all(0 <= x <= 1 for x in (prior, likelihood, false_positive)):
        raise ValueError("probabilities must be in [0,1]")
    return likelihood * prior + false_positive * (1.0 - prior)

def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    if marginal <= 0:
        raise ValueError("P(E) must be > 0")
    return prior * likelihood / marginal

# ---------- Parent B geometric product ----------
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
                # duplicate index cancels (e_i * e_i = 1 -> removed)
                lst.pop(j)
                lst.pop(j)  # second element shifts left
                return tuple(lst), sign
    return tuple(lst), sign

def _multiply_blades(blade_a: FrozenSet[int], blade_b: FrozenSet[int]) -> Tuple[FrozenSet[int], int]:
    """Geometric product of two basis blades."""
    combined = list(blade_a) + list(blade_b)
    sorted_blade, sign = _blade_sign(tuple(combined))
    return frozenset(sorted_blade), sign

class Multivector:
    """Clifford algebra element Cl(n,0) stored as a dict {blade: coefficient}."""

    def __init__(self, components: Dict[FrozenSet[int], float], n: int):
        self.n = int(n)
        # prune zero entries
        self.components = {k: float(v) for k, v in components.items() if abs(v) > 1e-15}

    def grade(self, k: int) -> "Multivector":
        return Multivector({blade: coef for blade, coef in self.components.items()
                            if len(blade) == k}, self.n)

    def scalar_part(self) -> float:
        return self.components.get(frozenset(), 0.0)

    def __add__(self, other: "Multivector") -> "Multivector":
        result = dict(self.components)
        for blade, coef in other.components.items():
            result[blade] = result.get(blade, 0.0) + coef
        return Multivector(result, self.n)

    def __mul__(self, other: "Multivector") -> "Multivector":
        """Geometric product."""
        result: Dict[FrozenSet[int], float] = {}
        for blade_a, coef_a in self.components.items():
            for blade_b, coef_b in other.components.items():
                blade_res, sign = _multiply_blades(blade_a, blade_b)
                result[blade_res] = result.get(blade_res, 0.0) + sign * coef_a * coef_b
        return Multivector(result, self.n)

    def __rmul__(self, scalar: float) -> "Multivector":
        return Multivector({blade: scalar * coef for blade, coef in self.components.items()}, self.n)

    def __repr__(self) -> str:
        terms = [f"{coef:.3g}*e{sorted(blade) if blade else ''}" for blade, coef in self.components.items()]
        return " + ".join(terms) if terms else "0"

# ---------- Hybrid core ----------
def morphology_to_multivector(m: Morphology) -> Multivector:
    """
    Encode a Morphology instance into a grade‑1 multivector.
    Basis mapping:
        e0 → length
        e1 → width
        e2 → height
        e3 → mass
    """
    comps = {
        frozenset({0}): m.length,
        frozenset({1}): m.width,
        frozenset({2}): m.height,
        frozenset({3}): m.mass,
    }
    return Multivector(comps, n=4)

def features_to_multivector(m: Morphology) -> Multivector:
    """
    Compute the three scalar morphological features and embed them as a
    grade‑1 multivector (different basis than morphology_to_multivector).
    Mapping:
        e0 → sphericity
        e1 → flatness
        e2 → righting‑time
    """
    sph = sphericity_index(m.length, m.width, m.height)
    flt = flatness_index(m.length, m.width, m.height)
    rgt = righting_time_index(m)
    comps = {
        frozenset({0}): sph,
        frozenset({1}): flt,
        frozenset({2}): rgt,
    }
    return Multivector(comps, n=4)

def hybrid_bayesian_posterior(prior: float,
                              morph: Morphology,
                              weight: Multivector,
                              false_positive: float = 0.01) -> float:
    """
    1. Convert morphological features to a multivector `f`.
    2. Geometric product `f * weight` yields a multivector whose scalar part
       behaves like a dot product between feature and weight vectors.
    3. Treat the scalar part as an *effective likelihood* (clipped to [0,1]).
    4. Perform a Bayesian update using `bayes_marginal` and `bayes_update`.
    """
    f = features_to_multivector(morph)
    product = f * weight
    effective_likelihood = max(0.0, min(1.0, product.scalar_part()))
    marginal = bayes_marginal(prior, effective_likelihood, false_positive)
    posterior = bayes_update(prior, effective_likelihood, marginal)
    return posterior

def gradient_from_error(prior: float,
                        posterior: float,
                        morph: Morphology,
                        weight: Multivector,
                        learning_rate: float = 0.05) -> Multivector:
    """
    Simple gradient approximation for weight update:
        grad ≈ (posterior - prior) * feature_multivector
    The geometric product is used to keep the update within the Clifford algebra.
    """
    error = posterior - prior
    f = features_to_multivector(morph)
    # Scale feature multivector by the scalar error term
    scaled = error * f  # scalar multiplication via __rmul__
    # Apply learning rate
    return learning_rate * scaled

def update_weight(weight: Multivector,
                  grad: Multivector) -> Multivector:
    """
    Perform a weight update using geometric addition:
        w_new = w + grad
    """
    return weight + grad

# ---------- Demonstration functions ----------
def hybrid_endpoint_circuit_breaker(morph: Morphology,
                                    weight: Multivector,
                                    prior: float = 0.5) -> float:
    """
    High‑level wrapper that mimics the “circuit breaker” concept:
    Computes a posterior probability that the given morphology belongs to the
    target class, using the hybrid geometric‑Bayesian pipeline.
    """
    return hybrid_bayesian_posterior(prior, morph, weight)

def train_hybrid_model(samples: Tuple[Morphology, int],
                       epochs: int = 5,
                       lr: float = 0.1) -> Multivector:
    """
    Very lightweight trainer.
    `samples` is an iterable of (Morphology, label) where label ∈ {0,1}.
    The weight multivector is initialised randomly.
    """
    # Initialise weight with small random values on the same grades as features (e0‑e2)
    init_comps = {
        frozenset({0}): random.uniform(-0.1, 0.1),
        frozenset({1}): random.uniform(-0.1, 0.1),
        frozenset({2}): random.uniform(-0.1, 0.1),
    }
    weight = Multivector(init_comps, n=4)

    for epoch in range(epochs):
        for morph, label in samples:
            prior = 0.5  # uninformed prior for each sample
            posterior = hybrid_bayesian_posterior(prior, morph, weight)
            # Interpret label as desired posterior (1 → high confidence, 0 → low)
            target = float(label)
            # Simple error‑driven gradient
            grad = gradient_from_error(prior, target, morph, weight, learning_rate=lr)
            weight = update_weight(weight, grad)
    return weight

def evaluate_hybrid_model(weight: Multivector,
                          test_set: Tuple[Morphology, int]) -> float:
    """
    Returns accuracy over the provided test set.
    """
    correct = 0
    total = len(test_set)
    for morph, label in test_set:
        prob = hybrid_endpoint_circuit_breaker(morph, weight, prior=0.5)
        pred = 1 if prob >= 0.5 else 0
        if pred == label:
            correct += 1
    return correct / total if total else 0.0

# ---------- Smoke test ----------
if __name__ == "__main__":
    # Create synthetic dataset
    random.seed(42)
    def random_morph():
        return Morphology(
            length=random.uniform(0.5, 2.0),
            width=random.uniform(0.5, 2.0),
            height=random.uniform(0.5, 2.0),
            mass=random.uniform(0.1, 5.0)
        )
    # Generate 20 samples with arbitrary binary labels
    dataset = [(random_morph(), random.randint(0, 1)) for _ in range(20)]
    train_set = dataset[:15]
    test_set = dataset[15:]

    # Train
    learned_weight = train_hybrid_model(train_set, epochs=10, lr=0.05)
    print("Learned weight:", learned_weight)

    # Evaluate
    acc = evaluate_hybrid_model(learned_weight, test_set)
    print(f"Test accuracy on {len(test_set)} samples: {acc:.2%}")

    # Single example demonstration
    example = random_morph()
    prob = hybrid_endpoint_circuit_breaker(example, learned_weight, prior=0.5)
    print(f"Posterior probability for a random morphology: {prob:.4f}")