# DARWIN HAMMER — match 1973, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hoeffding_tre_hybrid_hybrid_hybrid_m979_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_percyphon_hyb_m1057_s1.py (gen4)
# born: 2026-05-29T23:40:12Z

import math
import random
import sys
import pathlib
import numpy as np
from dataclasses import dataclass
from typing import Iterable, Dict

# ----------------------------------------------------------------------
# Hoeffding bound utilities
# ----------------------------------------------------------------------

def hoeffding_bound(r: float, delta: float, n: int) -> float:
    """Return the Hoeffding bound ε for range ``r``, confidence ``1‑δ`` and
    sample size ``n``.
    
    Raises:
        ValueError: if arguments are out of the admissible domain.
    """
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0 required")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))


# ----------------------------------------------------------------------
# Gini utilities
# ----------------------------------------------------------------------

def gini_coefficient(values: Iterable[float]) -> float:
    """Compute the Gini inequality coefficient for a non-negative iterable."""
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0.0:
        return 0.0
    if xs[0] < 0:
        raise ValueError("values must be non-negative")
    n = len(xs)
    return sum((2 * i - n - 1) * x for i, x in enumerate(xs, 1)) / (n * sum(xs))


# ----------------------------------------------------------------------
# Morphology utilities
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class Morphology:
    """Geometric description of an entity."""
    length: float
    width: float
    height: float
    mass: float


def morphological_indices(morphology: Morphology) -> tuple[float, float]:
    """Compute the sphericity and flatness indices for a morphology."""
    volume = morphology.length * morphology.width * morphology.height
    surface_area = 2 * (morphology.length * morphology.width + morphology.width * morphology.height + morphology.height * morphology.length)
    sphericity = volume / (surface_area / 6) ** (1/3)
    flatness = (morphology.length + morphology.width + morphology.height) / (2 * math.sqrt((morphology.length * morphology.width) + (morphology.width * morphology.height) + (morphology.height * morphology.length)))
    return sphericity, flatness


# ----------------------------------------------------------------------
# Hybrid Regret-Gini Morphology Engine
# ----------------------------------------------------------------------

def hybrid_engine(regrets: Dict[str, Dict[str, float]], work_shares: Dict[str, float], morphologies: Dict[str, Morphology], base_budget: int) -> Dict[str, int]:
    """Run the hybrid engine to compute the allocation of procedural slots to each model."""
    gini_values_per_model = {}
    for model, regrets_per_model in regrets.items():
        gini_values = list(regrets_per_model.values())
        gini_values_per_model[model] = gini_values

    gini_coefficients = {model: gini_coefficient(values) for model, values in gini_values_per_model.items()}
    gini_avg = sum(gini_coefficients.values()) / len(gini_coefficients)

    allocations = {}
    for model, work_share in work_shares.items():
        morphology = morphologies[model]
        sphericity, flatness = morphological_indices(morphology)
        mu = (sphericity + flatness) / 2
        sigma = (1 + gini_coefficients[model]) / (1 + gini_avg)
        allocation = max(0, round(sigma * work_share * mu * base_budget))
        allocations[model] = allocation

    # Normalize allocations to sum up to base_budget
    total_allocation = sum(allocations.values())
    if total_allocation > 0:
        scaling_factor = base_budget / total_allocation
        allocations = {model: max(0, round(allocation * scaling_factor)) for model, allocation in allocations.items()}
        # Adjust allocations to ensure they sum up to base_budget
        diff = base_budget - sum(allocations.values())
        model_with_min_allocation = min(allocations, key=allocations.get)
        allocations[model_with_min_allocation] += diff

    return allocations


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------

if __name__ == "__main__":
    regrets = {
        "model1": {"regret1": 0.5, "regret2": 0.6},
        "model2": {"regret1": 0.3, "regret2": 0.4},
        "model3": {"regret1": 0.2, "regret2": 0.1}
    }
    work_shares = {"model1": 0.7, "model2": 0.2, "model3": 0.1}
    morphologies = {
        "model1": Morphology(length=10.0, width=5.0, height=2.0, mass=10.0),
        "model2": Morphology(length=5.0, width=3.0, height=1.0, mass=5.0),
        "model3": Morphology(length=2.0, width=1.0, height=0.5, mass=2.0)
    }
    base_budget = 100
    allocations = hybrid_engine(regrets, work_shares, morphologies, base_budget)
    print(allocations)