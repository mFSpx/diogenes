# DARWIN HAMMER — match 2113, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m558_s1.py (gen5)
# parent_b: shap_attribution.py (gen0)
# born: 2026-05-29T23:40:47Z

"""Hybrid Resource Allocation via Shapley‑Based Model Selection.

Parents:
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m558_s1.py (ModelPool with RAM ceiling)
- shap_attribution.py (Exact Shapley value computation)

Mathematical bridge:
Each model is treated as a player in a cooperative game.  
The coalition value V(S) is the sum of the intrinsic rewards of the models in
subset S.  The Shapley value φ_i gives a fair marginal contribution of model i
to the overall reward.  By feeding these φ_i into the ModelPool decision rule,
the algorithm loads the most valuable models while respecting the RAM limit,
thus fusing the resource‑allocation topology of Parent A with the Shapley
valuation topology of Parent B.
"""

import math
import random
import sys
import pathlib
from typing import Dict, List, Callable, Any, Iterable, Tuple
import numpy as np

# ----------------------------------------------------------------------
# Parent A – model pool management
# ----------------------------------------------------------------------
class ModelTier:
    """Lightweight descriptor of a model."""
    def __init__(self, name: str, ram_mb: int, tier: str, reward: float):
        self.name = name
        self.ram_mb = ram_mb
        self.tier = tier
        self.reward = reward  # intrinsic performance metric used by Shapley

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
        """True if the model fits within the remaining RAM."""
        return (self._used() + model.ram_mb) <= self.ram_ceiling_mb

    def load(self, model: ModelTier) -> None:
        """Load a model if RAM permits; silently ignore if already loaded."""
        if self.is_loaded(model.name):
            return
        if not self.can_load(model):
            raise RuntimeError(
                f"Insufficient RAM to load {model.name}: "
                f"required {model.ram_mb} MB, available {self.ram_ceiling_mb - self._used()} MB"
            )
        self.loaded[model.name] = model

    def unload(self, name: str) -> None:
        """Unload a model if present."""
        self.loaded.pop(name, None)

    def status(self) -> Tuple[int, int]:
        """Return (used_ram, remaining_ram)."""
        used = self._used()
        return used, self.ram_ceiling_mb - used

# ----------------------------------------------------------------------
# Parent B – exact Shapley value utilities
# ----------------------------------------------------------------------
def shapley_kernel_weight(subset_size: int, feature_count: int) -> float:
    """Weight for a coalition of size `subset_size` in a game with `feature_count` players."""
    return (
        math.factorial(subset_size)
        * math.factorial(feature_count - subset_size - 1)
        / math.factorial(feature_count)
    )

def exact_shapley_value(
    value_fn: Callable[[frozenset[int]], float],
    feature_index: int,
    feature_count: int,
) -> float:
    """Exact Shapley value for a single player by enumerating all coalitions."""
    others = [i for i in range(feature_count) if i != feature_index]
    total = 0.0
    for k in range(len(others) + 1):
        for subset in combinations(others, k):
            s = frozenset(subset)
            marginal = value_fn(s | {feature_index}) - value_fn(s)
            total += shapley_kernel_weight(k, feature_count) * marginal
    return total

# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------
def compute_shapley_for_models(models: List[ModelTier]) -> Dict[str, float]:
    """
    Treat each model as a player.  The coalition value is the sum of rewards.
    Returns a mapping from model name to its Shapley value.
    """
    rewards = [m.reward for m in models]
    n = len(models)

    # Pre‑compute value of every possible coalition using bit masks for speed.
    # Since n is expected to be small (resource‑allocation context), O(2^n) is fine.
    coalition_value = {}
    for mask in range(1 << n):
        subset = frozenset(i for i in range(n) if mask & (1 << i))
        coalition_value[subset] = sum(rewards[i] for i in subset)

    def value_fn(subset: frozenset[int]) -> float:
        return coalition_value[subset]

    shapley: Dict[str, float] = {}
    for idx, model in enumerate(models):
        phi = exact_shapley_value(value_fn, idx, n)
        shapley[model.name] = phi
    return shapley

def allocate_models_by_shapley(pool: ModelPool, models: List[ModelTier]) -> List[str]:
    """
    Load models into the pool in descending order of Shapley value
    until the RAM ceiling is reached.  Returns the list of loaded model names.
    """
    shapley = compute_shapley_for_models(models)
    # Sort models by descending Shapley contribution
    sorted_models = sorted(models, key=lambda m: shapley[m.name], reverse=True)

    loaded_names = []
    for model in sorted_models:
        if pool.is_loaded(model.name):
            loaded_names.append(model.name)
            continue
        try:
            pool.load(model)
            loaded_names.append(model.name)
        except RuntimeError:
            # Not enough RAM – skip this model
            continue
    return loaded_names

def simulate_hybrid_step(pool: ModelPool, models: List[ModelTier]) -> None:
    """
    One simulation step:
      1. Compute Shapley values.
      2. Unload any loaded model whose Shapley value is negative or
         below the median of the current set (optional heuristic).
      3. Re‑run allocation to fill freed RAM with higher‑value models.
    """
    shapley = compute_shapley_for_models(models)

    # Unload low‑value models (heuristic: below median)
    if pool.loaded:
        median_phi = np.median(list(shapley[name] for name in pool.loaded))
        for name in list(pool.loaded.keys()):
            if shapley.get(name, 0.0) < median_phi:
                pool.unload(name)

    # Re‑allocate
    allocate_models_by_shapley(pool, models)

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a handful of dummy models with random RAM and reward
    random.seed(42)
    models = [
        ModelTier(name=f"model_{i}", ram_mb=random.randint(500, 2000),
                  tier="A", reward=random.uniform(0.0, 10.0))
        for i in range(8)
    ]

    # Instantiate a pool with a modest RAM ceiling
    pool = ModelPool(ram_ceiling_mb=6000)

    # Initial allocation
    loaded = allocate_models_by_shapley(pool, models)
    used, remaining = pool.status()
    print("Initial allocation:")
    print(f"  Loaded models: {loaded}")
    print(f"  RAM used / remaining: {used} MB / {remaining} MB")

    # Run a few hybrid steps to demonstrate dynamic re‑allocation
    for step in range(3):
        simulate_hybrid_step(pool, models)
        used, remaining = pool.status()
        print(f"\nAfter step {step + 1}:")
        print(f"  Loaded models: {list(pool.loaded.keys())}")
        print(f"  RAM used / remaining: {used} MB / {remaining} MB")