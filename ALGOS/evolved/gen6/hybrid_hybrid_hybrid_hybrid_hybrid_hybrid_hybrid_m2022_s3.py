# DARWIN HAMMER — match 2022, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_worksh_hybrid_caputo_fracti_m46_s4.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_decisi_m855_s1.py (gen5)
# born: 2026-05-29T23:40:31Z

import numpy as np
import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List

# Shared data structures
@dataclass(frozen=True)
class MathAction:
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0

@dataclass(frozen=True)
class MathCounterfactual:
    action_id: str
    outcome_value: float
    probability: float = 1.0

# Constants & Helpers
GROUPS = ("codex", "groq", "cohere", "local_models")

def _pct(value: float) -> float:
    """Round a float to six decimal places."""
    return round(float(value), 6)

def _gamma(z: float) -> float:
    """Lanczos approximation of the Gamma function."""
    if z < 0.5:
        # Reflection formula
        return math.pi / (math.sin(math.pi * z) * _gamma(1 - z))
    z -= 1
    x = _LANCZOS_C[0]
    for i in range(1, _LANCZOS_G + 2):
        x += _LANCZOS_C[i] / (z + i)
    t = z + _LANCZOS_G + 0.5
    return math.sqrt(2 * math.pi) * t ** (z + 0.5) * math.exp(-t) * x

_LANCZOS_G = 7
_LANCZOS_C = np.array([
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857
])

def caputo_kernel(alpha: float, t: np.ndarray) -> np.ndarray:
    """
    Compute the raw (unnormalized) Caputo kernel values for a vector of time indices.
    The kernel is t^{alpha-1} / Gamma(alpha).
    """
    if alpha <= 0:
        raise ValueError("Fractional order alpha must be positive.")
    # avoid t=0 when alpha<1 (singularity) by starting at 1
    t = np.where(t == 0, 1e-12, t)
    return t ** (alpha - 1) / _gamma(alpha)

def normalized_caputo_weights(alpha: float, length: int, scale: float = 1.0) -> np.ndarray:
    """
    Produce a normalized weight vector of given length.
    ``scale`` allows a dynamic modulation of the fractional order (e.g. via LTC).
    """
    effective_alpha = max(alpha * scale, 1e-6)  # keep >0
    times = np.arange(1, length + 1)
    return caputo_kernel(effective_alpha, times)

def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Compute the Kullback-Leibler divergence."""
    return np.sum(p * np.log(p / q))

def shannon_entropy(p: np.ndarray) -> float:
    """Compute the Shannon entropy."""
    return -np.sum(p * np.log(p))

def exponential_pruning_probability(lambda_val: float, alpha: float, t: float) -> float:
    """Compute the exponential pruning probability."""
    return min(1, lambda_val * math.exp(-alpha * t))

def hybrid_pruning_probability(lambda_val: float, alpha: float, t: float, v: np.ndarray) -> float:
    """Compute the hybrid pruning probability."""
    max_entropy = np.log(len(v))
    return exponential_pruning_probability(lambda_val, alpha, t) / (1 + shannon_entropy(v) / max_entropy)

def regret(math_counterfactuals: List[MathCounterfactual]) -> float:
    """Compute the regret."""
    return sum(cf.outcome_value * cf.probability for cf in math_counterfactuals)

def hybrid_score(regret: float, math_counterfactuals: List[MathCounterfactual]) -> float:
    """Compute the hybrid score."""
    probabilities = np.array([cf.probability for cf in math_counterfactuals])
    return regret * (1 - hybrid_pruning_probability(1.0, 1.0, 1.0, probabilities))

def improved_hybrid_score(regret: float, math_counterfactuals: List[MathCounterfactual]) -> float:
    """Compute the improved hybrid score."""
    probabilities = np.array([cf.probability for cf in math_counterfactuals])
    max_entropy = np.log(len(probabilities))
    entropy = shannon_entropy(probabilities)
    return regret * (1 - exponential_pruning_probability(1.0, 1.0, 1.0) * (1 + entropy / max_entropy))

# Smoke test
if __name__ == "__main__":
    math_actions = [
        MathAction("action1", 10.0, 1.0),
        MathAction("action2", 20.0, 2.0),
        MathAction("action3", 30.0, 3.0)
    ]
    math_counterfactuals = [
        MathCounterfactual("action1", 10.0, 0.5),
        MathCounterfactual("action2", 20.0, 0.3),
        MathCounterfactual("action3", 30.0, 0.2)
    ]
    print(improved_hybrid_score(regret(math_counterfactuals), math_counterfactuals))