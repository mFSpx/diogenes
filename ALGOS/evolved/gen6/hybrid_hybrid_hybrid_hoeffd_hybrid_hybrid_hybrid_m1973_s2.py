# DARWIN HAMMER — match 1973, survivor 2
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

@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

def hoeffding_bound(r: float, delta: float, n: int) -> float:
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0 required")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))

def gini_coefficient(values: Iterable[float]) -> float:
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0.0:
        return 0.0
    if xs[0] < 0:
        raise ValueError("values must be non-negative")
    n = len(xs)
    return sum((2 * i - n - 1) * x for i, x in enumerate(xs, 1)) / (n * sum(xs))

def morphological_indices(morphology: Morphology) -> tuple[float, float]:
    volume = morphology.length * morphology.width * morphology.height
    surface_area = 2 * (morphology.length * morphology.width + morphology.width * morphology.height + morphology.height * morphology.length)
    sphericity = volume / (surface_area / 6) ** (1/3)
    flatness = (morphology.length + morphology.width + morphology.height) / (2 * math.sqrt((morphology.length * morphology.width) + (morphology.width * morphology.height) + (morphology.height * morphology.length)))
    return sphericity, flatness

def hybrid_engine(regrets: Dict[str, float], work_shares: Dict[str, float], morphologies: Dict[str, Morphology], base_budget: int) -> Dict[str, int]:
    gini_avgs = {model: gini_coefficient([regrets[model]]) for model in regrets}
    gini_avg = sum(gini_avgs.values()) / len(gini_avgs)
    
    allocations = {}
    for model, work_share in work_shares.items():
        morphology = morphologies[model]
        sphericity, flatness = morphological_indices(morphology)
        mu = (sphericity + flatness) / 2
        sigma = (1 + gini_avgs[model]) / (1 + gini_avg)
        allocation = round(sigma * work_share * mu * base_budget)
        allocations[model] = allocation
    
    # Normalize allocations to ensure they sum to base_budget
    total_allocation = sum(allocations.values())
    if total_allocation > 0:
        allocations = {model: round(allocation / total_allocation * base_budget) for model, allocation in allocations.items()}
    else:
        allocations = {model: base_budget // len(regrets) for model in regrets}
    
    return allocations

if __name__ == "__main__":
    regrets = {"model1": 0.5, "model2": 0.3, "model3": 0.2}
    work_shares = {"model1": 0.7, "model2": 0.2, "model3": 0.1}
    morphologies = {
        "model1": Morphology(length=10.0, width=5.0, height=2.0, mass=10.0),
        "model2": Morphology(length=5.0, width=3.0, height=1.0, mass=5.0),
        "model3": Morphology(length=2.0, width=1.0, height=0.5, mass=2.0)
    }
    base_budget = 100
    allocations = hybrid_engine(regrets, work_shares, morphologies, base_budget)
    print(allocations)