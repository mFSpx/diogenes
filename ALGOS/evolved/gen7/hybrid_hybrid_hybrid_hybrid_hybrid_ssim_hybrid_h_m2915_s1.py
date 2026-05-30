# DARWIN HAMMER — match 2915, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_shap_attribution_m2113_s0.py (gen6)
# parent_b: hybrid_ssim_hybrid_hybrid_fracti_m934_s1.py (gen3)
# born: 2026-05-29T23:46:34Z

"""
Hybrid algorithm fusing 
- Hybrid Resource Allocation via Shapley-Based Model Selection (hybrid_hybrid_hybrid_hybrid_shap_attribution_m2113_s0.py)
- Hybrid algorithm that fuses the Structural Similarity Index (SSIM) from ssim.py and the Hybrid Fractional-Hoeffding algorithm (hybrid_ssim_hybrid_hybrid_fracti_m934_s1.py).

The mathematical bridge lies in applying the Shapley values as weights to compute a 
weighted SSIM (wSSIM) between model outputs. The wSSIM is then used as input to 
the Hybrid Fractional-Hoeffding's fractional power operation, thus quantifying 
uncertainty in both model performance and data distributions.
"""

import numpy as np
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Iterable, Optional
import random
import sys
import pathlib

class ModelTier:
    """Lightweight descriptor of a model."""
    def __init__(self, name: str, ram_mb: int, tier: str, reward: float):
        self.name = name
        self.ram_mb = ram_mb
        self.tier = tier
        self.reward = reward  

class ModelPool:
    """Manages loaded models under a RAM ceiling."""
    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded: Dict[str, ModelTier] = {}

    def is_loaded(self, name: str) -> bool:
        return name in self.loaded

    def _used(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def can_load(self, model: ModelTier) -> bool:
        return (self._used() + model.ram_mb) <= self.ram_ceiling_mb

    def load(self, model: ModelTier) -> None:
        if self.is_loaded(model.name):
            return
        if not self.can_load(model):
            raise RuntimeError(
                f"Insufficient RAM to load {model.name}: "
                f"required {model.ram_mb} MB, available {self.ram_ceiling_mb - self._used()}")

def shapley_value(models: List[ModelTier], coalition: List[ModelTier]) -> Dict[str, float]:
    n = len(models)
    values = {}
    for model in models:
        marginal_contributions = []
        for i in range(n):
            subset = coalition[:i] + coalition[i+1:]
            if model in subset:
                subset.remove(model)
            subset_value = sum(m.reward for m in subset)
            marginal_contributions.append(coalition_value(coalition) - subset_value)
        values[model.name] = sum(marginal_contributions) / n
    return values

def coalition_value(coalition: List[ModelTier]) -> float:
    return sum(model.reward for model in coalition)

def ssim(x: List[float], y: List[float], dynamic_range: float = 255.0, k1: float = 0.01, k2: float = 0.03) -> float:
    if len(x) != len(y):
        raise ValueError('samples must have equal length')
    if not x:
        raise ValueError('samples must not be empty')
    n = len(x)
    mx = sum(x) / n
    my = sum(y) / n
    vx = sum((v - mx) ** 2 for v in x) / n
    vy = sum((v - my) ** 2 for v in y) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y)) / n
    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2
    return ((2 * mx * my + c1) * (2 * cov + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))

def wssim(models: List[ModelTier], outputs: List[List[float]]) -> float:
    shapley_values = shapley_value(models, models)
    weights = [shapley_values[model.name] for model in models]
    weighted_outputs = [output * weight for output, weight in zip(outputs, weights)]
    return ssim([o[0] for o in weighted_outputs], [o[1] for o in weighted_outputs])

def fractional_power(X: np.ndarray, alpha: float) -> np.ndarray:
    return np.abs(X)**alpha * np.sign(X)

def hybrid_operation(models: List[ModelTier], outputs: List[List[float]], alpha: float) -> np.ndarray:
    wssim_value = wssim(models, outputs)
    return fractional_power(np.array([wssim_value]), alpha)

def main():
    model_tier1 = ModelTier("model1", 1000, "tier1", 10.0)
    model_tier2 = ModelTier("model2", 2000, "tier2", 20.0)
    model_pool = ModelPool()
    model_pool.load(model_tier1)
    model_pool.load(model_tier2)

    outputs = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    alpha = 0.5
    result = hybrid_operation([model_tier1, model_tier2], outputs, alpha)
    print(result)

if __name__ == "__main__":
    main()