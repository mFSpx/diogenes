# DARWIN HAMMER — match 2022, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_worksh_hybrid_caputo_fracti_m46_s4.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_decisi_m855_s1.py (gen5)
# born: 2026-05-29T23:40:31Z

import numpy as np
import math
from dataclasses import dataclass
from typing import List

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

def _gamma(z: float) -> float:
    if z < 0.5:
        return math.pi / (math.sin(math.pi * z) * _gamma(1 - z))
    z -= 1
    x = 0.99999999999980993
    for i in range(1, 8):
        x += [676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857][i-1] / (z + i)
    t = z + 8 + 0.5
    return math.sqrt(2 * math.pi) * t ** (z + 0.5) * math.exp(-t) * x

def caputo_kernel(alpha: float, t: np.ndarray) -> np.ndarray:
    if alpha <= 0:
        raise ValueError("Fractional order alpha must be positive.")
    t = np.where(t == 0, 1e-12, t)
    return t ** (alpha - 1) / _gamma(alpha)

def normalized_caputo_weights(alpha: float, length: int, scale: float = 1.0) -> np.ndarray:
    effective_alpha = max(alpha * scale, 1e-6)  
    times = np.arange(1, length + 1)
    return caputo_kernel(effective_alpha, times) / np.sum(caputo_kernel(effective_alpha, times))

def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    epsilon = 1e-15
    p = np.clip(p, epsilon, 1 - epsilon)
    q = np.clip(q, epsilon, 1 - epsilon)
    return np.sum(p * np.log(p / q))

def shannon_entropy(p: np.ndarray) -> float:
    epsilon = 1e-15
    p = np.clip(p, epsilon, 1 - epsilon)
    return -np.sum(p * np.log(p))

def exponential_pruning_probability(lambda_val: float, alpha: float, t: float) -> float:
    return min(1, lambda_val * math.exp(-alpha * t))

def hybrid_pruning_probability(lambda_val: float, alpha: float, t: float, v: np.ndarray) -> float:
    h_max = shannon_entropy(np.array([1.0/len(v) for _ in range(len(v))]))
    return exponential_pruning_probability(lambda_val, alpha, t) / (1 + shannon_entropy(v) / h_max)

def regret(math_counterfactuals: List[MathCounterfactual]) -> float:
    return sum(cf.outcome_value * cf.probability for cf in math_counterfactuals)

def hybrid_score(math_counterfactuals: List[MathCounterfactual], 
                 lambda_val: float = 1.0, 
                 alpha: float = 1.0, 
                 t: float = 1.0) -> float:
    regret_val = regret(math_counterfactuals)
    probabilities = np.array([cf.probability for cf in math_counterfactuals])
    return regret_val * (1 - hybrid_pruning_probability(lambda_val, alpha, t, probabilities))

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
    print(hybrid_score(math_counterfactuals))