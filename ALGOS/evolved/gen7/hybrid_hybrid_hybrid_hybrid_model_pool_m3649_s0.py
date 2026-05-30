# DARWIN HAMMER — match 3649, survivor 0
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1394_s0.py (gen6)
# parent_b: model_pool.py (gen0)
# born: 2026-05-29T23:51:03Z

"""Hybrid algorithm merging multivector geometry (Parent A) with model‑resource management (Parent B).

Mathematical bridge:
- Each `ModelTier` is encoded as a multivector `M` whose scalar part reflects its RAM
  requirement and whose bivector part encodes its tier level.
- Regret for a model is defined as the shortfall between available RAM and the
  model’s RAM demand.
- The geometric product `G = R ⊗ M` (where `R` is a multivector representing the
  current resource state) couples the resource vector with the model vector.
- The hybrid weight `w = scalar(G) * regret` modulates the regret‑based priority
  with the geometric interaction, yielding a nuanced ordering for loading
  models under the decreasing‑rate pruning policy of Parent B.
"""

import math
import random
import sys
import pathlib
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, FrozenSet

# ----------------------------------------------------------------------
# Multivector utilities (from Parent A)
# ----------------------------------------------------------------------
def _blade_sign(indices: Tuple[int, ...]) -> Tuple[List[int], int]:
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
                # duplicate index → cancel
                lst.pop(j)
                lst.pop(j)
                n -= 2
                continue
            j += 1
        i += 1
    return lst, sign

def _multiply_blades(blade_a: FrozenSet[int], blade_b: FrozenSet[int]) -> Tuple[FrozenSet[int], int]:
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(tuple(combined))
    return frozenset(result), sign

class Multivector:
    """Sparse representation of a multivector in an n‑dimensional geometric algebra."""
    def __init__(self, components: Dict[FrozenSet[int], float], n: int):
        # keep only non‑zero components
        self.components = {k: float(v) for k, v in components.items() if abs(v) > 1e-12}
        self.n = int(n)

    def __repr__(self) -> str:
        terms = [f"{c}*{sorted(b)}" if b else f"{c}" for b, c in self.components.items()]
        return f"Multivector({', '.join(terms)})"

    def scalar_part(self) -> float:
        """Return the coefficient of the empty blade (grade‑0)."""
        return self.components.get(frozenset(), 0.0)

    def geometric_product(self, other: "Multivector") -> "Multivector":
        """Geometric product ⊗ between two multivectors."""
        result: Dict[FrozenSet[int], float] = {}
        for blade_a, coef_a in self.components.items():
            for blade_b, coef_b in other.components.items():
                blade_res, sign = _multiply_blades(blade_a, blade_b)
                result[blade_res] = result.get(blade_res, 0.0) + sign * coef_a * coef_b
        return Multivector(result, self.n)

    def __add__(self, other: "Multivector") -> "Multivector":
        new_comp = self.components.copy()
        for b, c in other.components.items():
            new_comp[b] = new_comp.get(b, 0.0) + c
        return Multivector(new_comp, self.n)

    def __sub__(self, other: "Multivector") -> "Multivector":
        new_comp = self.components.copy()
        for b, c in other.components.items():
            new_comp[b] = new_comp.get(b, 0.0) - c
        return Multivector(new_comp, self.n)

    def __mul__(self, scalar: float) -> "Multivector":
        return Multivector({b: c * scalar for b, c in self.components.items()}, self.n)

# ----------------------------------------------------------------------
# Model tier definitions (from Parent B)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str   # "T1", "T2", "T3"

class ModelLoadError(RuntimeError):
    pass

class ModelPool:
    """Simple RAM‑capped pool with mutual‑exclusion rules."""
    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded: Dict[str, ModelTier] = {}

    def is_loaded(self, name: str) -> bool:
        return name in self.loaded

    def _used(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def load(self, model: ModelTier) -> None:
        if model.tier == "T3" and any(m.tier == "T2" for m in self.loaded.values()):
            raise ModelLoadError("T3 mutually exclusive with T2 resident")
        if model.ram_mb + self._used() > self.ram_ceiling_mb:
            raise ModelLoadError("RAM ceiling exceeded")
        self.loaded[model.name] = model

    def load_with_eviction(self, model: ModelTier) -> None:
        """Evict oldest entries until the new model fits."""
        while self.loaded and model.ram_mb + self._used() > self.ram_ceiling_mb:
            evicted_name = next(iter(self.loaded))
            self.loaded.pop(evicted_name)
        self.load(model)

# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def encode_resource_state(ram_available_mb: int, tier_bias: str = "T1") -> Multivector:
    """
    Encode the current resource state as a multivector.
    - Scalar (grade‑0) = available RAM.
    - Bivector (grade‑2) encodes a bias toward a tier:
        T1 → basis e1∧e2, T2 → e1∧e3, T3 → e2∧e3
    """
    basis_map = {"T1": frozenset({1, 2}),
                 "T2": frozenset({1, 3}),
                 "T3": frozenset({2, 3})}
    components = {
        frozenset(): float(ram_available_mb),          # scalar part
        basis_map.get(tier_bias, frozenset()): 1.0     # unit bivector bias
    }
    return Multivector(components, n=3)

def encode_model_as_multivector(model: ModelTier) -> Multivector:
    """
    Encode a model tier as a multivector.
    - Scalar = model RAM demand.
    - Bivector = tier identifier (same mapping as in encode_resource_state).
    """
    tier_basis = {"T1": frozenset({1, 2}),
                  "T2": frozenset({1, 3}),
                  "T3": frozenset({2, 3})}
    components = {
        frozenset(): float(model.ram_mb),
        tier_basis[model.tier]: 1.0
    }
    return Multivector(components, n=3)

def hybrid_weight(resource_mv: Multivector, model_mv: Multivector) -> float:
    """
    Compute a hybrid weight:
        w = scalar( resource_mv ⊗ model_mv ) * regret
    Regret = max(0, model_ram - available_ram)
    """
    product = resource_mv.geometric_product(model_mv)
    scalar = product.scalar_part()
    model_ram = model_mv.scalar_part()
    available_ram = resource_mv.scalar_part()
    regret = max(0.0, model_ram - available_ram)
    return scalar * regret

def prioritize_models(models: List[ModelTier], resource_mv: Multivector) -> List[Tuple[ModelTier, float]]:
    """
    Return models sorted descending by hybrid weight.
    """
    weighted = []
    for m in models:
        mv = encode_model_as_multivector(m)
        w = hybrid_weight(resource_mv, mv)
        weighted.append((m, w))
    # Higher weight → higher priority (larger regret * geometric interaction)
    weighted.sort(key=lambda x: x[1], reverse=True)
    return weighted

def load_models_hybrid(pool: ModelPool, prioritized: List[Tuple[ModelTier, float]]) -> List[str]:
    """
    Attempt to load models in priority order using `load_with_eviction`.
    Returns the list of successfully loaded model names.
    """
    loaded_names = []
    for model, _weight in prioritized:
        try:
            pool.load_with_eviction(model)
            loaded_names.append(model.name)
        except ModelLoadError as e:
            # If eviction cannot satisfy constraints, skip this model.
            print(f"Skipping {model.name}: {e}", file=sys.stderr)
    return loaded_names

# ----------------------------------------------------------------------
# Demonstration / smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a pool with a 6000 MB ceiling
    pool = ModelPool(ram_ceiling_mb=6000)

    # Catalogue of models (mirrors Parent B's tiers)
    catalog = [
        ModelTier("qwen-0.5b", 512, "T1"),
        ModelTier("reasoning-t2", 3000, "T2"),
        ModelTier("tool-t2", 2600, "T2"),
        ModelTier("qwen-7b", 7000, "T3")  # exceeds ceiling, will trigger eviction or skip
    ]

    # Current resource state: 4000 MB free, bias toward T2 (more reasoning tools)
    resource_state = encode_resource_state(ram_available_mb=4000, tier_bias="T2")

    # Compute priorities
    prioritized = prioritize_models(catalog, resource_state)

    print("Priority list (model, weight):")
    for m, w in prioritized:
        print(f"  {m.name:15s} weight={w:.2f}")

    # Load according to hybrid policy
    loaded = load_models_hybrid(pool, prioritized)

    print("\nSuccessfully loaded models:")
    for name in loaded:
        print(f"  {name}")

    print("\nFinal pool usage:", pool._used(), "MB /", pool.ram_ceiling_mb, "MB")