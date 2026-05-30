# DARWIN HAMMER — match 1073, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_hybrid_regret_hybrid_sparse_wta_hy_m173_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_distri_hybrid_hybrid_regret_m25_s6.py (gen4)
# born: 2026-05-29T23:32:43Z

import numpy as np
import math
import random
import sys
import pathlib
import hashlib
from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

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

@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str

class ModelPool:
    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded = {}
        self.tier_hierarchy = {"T1": 0, "T2": 1, "T3": 2}

    def is_loaded(self, name: str) -> bool: 
        return name in self.loaded

    def _used(self) -> int: 
        return sum(m.ram_mb for m in self.loaded.values())

    def load(self, model: ModelTier) -> None:
        if model.tier=="T3" and any(m.tier=="T2" for m in self.loaded.values()): 
            raise Exception("T3 mutually exclusive with T2 resident")
        if model.ram_mb + self._used() > self.ram_ceiling_mb: 
            raise Exception("RAM ceiling exceeded")
        if model.tier not in self.tier_hierarchy:
            raise Exception("Invalid model tier")
        self.loaded[model.name]=model

    def load_with_eviction(self, model: ModelTier) -> None:
        while self.loaded and model.ram_mb + self._used() > self.ram_ceiling_mb:
            eviction_candidate = min(self.loaded, key=lambda m: self.tier_hierarchy[self.loaded[m].tier])
            del self.loaded[eviction_candidate]
        self.load(model)

def _hash(seed: int, token: str) -> int:
    """Deterministic 64‑bit integer hash."""
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")

def signature(tokens: Iterable[str], k: int = 128) -> List[int]:
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [(1 << 64) - 1] * k
    return [_hash(hash(t), str(i)) for i, t in enumerate(toks)]

def broadcast_probability(total_phases: int, current_phase: int) -> float:
    if total_phases < 1 or current_phase < 1:
        raise ValueError("phases and phase must be positive")
    decay = 2 ** max(0, total_phases - current_phase)
    return min(1.0, 1.0 / decay)

def hoeffding_bound(R: float, delta: float, n: int) -> float:
    if n <= 0:
        raise ValueError("sample size n must be positive")
    if not (0 < delta < 1):
        raise ValueError("delta must be in (0,1)")
    return math.sqrt((R ** 2 * math.log(1.0 / delta)) / (2.0 * n))

def tropical_max_plus_gain(gains: np.ndarray) -> float:
    return float(np.max(gains))

def hybrid_model_loading(model_pool: ModelPool, model_tier: ModelTier, 
                         total_phases: int, current_phase: int, 
                         R: float, delta: float, n: int) -> None:
    probability = broadcast_probability(total_phases, current_phase)
    bound = hoeffding_bound(R, delta, n)
    gain = tropical_max_plus_gain(np.array([model_tier.ram_mb, bound]))
    if gain > 0 and probability > 0:
        model_pool.load_with_eviction(model_tier)

def hybrid_model_selection(model_pool: ModelPool, tokens: Iterable[str], 
                           k: int = 128) -> List[int]:
    model_signatures = signature(tokens, k)
    loaded_models = [m.name for m in model_pool.loaded.values()]
    loaded_signatures = signature(loaded_models, k)
    return [model_signatures[i] ^ loaded_signatures[i] for i in range(k)]

def model_eviction_policy(model_pool: ModelPool) -> None:
    if len(model_pool.loaded) > 1:
        eviction_candidate = min(model_pool.loaded, key=lambda m: model_pool.loaded[m].ram_mb)
        del model_pool.loaded[eviction_candidate]

if __name__ == "__main__":
    model_pool = ModelPool()
    model_tier = ModelTier("test_model", 1000, "T1")
    hybrid_model_loading(model_pool, model_tier, 10, 5, 1.0, 0.1, 100)
    tokens = ["token1", "token2"]
    print(hybrid_model_selection(model_pool, tokens))
    model_eviction_policy(model_pool)