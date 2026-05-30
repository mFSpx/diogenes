# DARWIN HAMMER — match 319, survivor 1
# gen: 3
# parent_a: model_pool.py (gen0)
# parent_b: hybrid_hybrid_workshare_all_hybrid_krampus_brain_m23_s3.py (gen2)
# born: 2026-05-29T23:28:13Z

"""
Hybrid Module: model_pool + hybrid_hybrid_workshare_all_hybrid_krampus_brain_m23_s3
This fusion links the two parent algorithms through a *dynamic RAM allocation* 
that modulates the model load/unload based on the feature-curvature matrix 
from the Krampus extractor.

The model pool's load/unload logic is scaled by the curvature-based 
allocation weights, ensuring that the RAM ceiling is respected while 
prioritizing model loads based on their feature-curvature scores.

- The deterministic portion of the allocation is scaled by the classic 
  **doomsday weekday value** (0-6) as in `hybrid_workshare_allocator_doomsday_calendar`.
- The stochastic model-load portion is no longer split evenly; instead a 
  **24-dimensional feature vector** extracted deterministically from an input text 
  (Krampus extractor) is transformed into a **curvature matrix** `C = v·vᵀ` 
  (outer product of the normalized feature vector).
- The per-model share is obtained by projecting the curvature matrix onto a 
  one-hot encoding of the model name, yielding a weight proportional to the 
  corresponding entry of the vector `w = C·g`, where `g` is a one-hot vector for 
  the model.

The three public functions demonstrate the hybrid behaviour:
`load_model_with_curvature`, `compute_feature_curvature`, and
`hybrid_summary`.
"""

import math
import random
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

# ---------------------------------------------------------------------------
# Helper utilities (shared by both parents)
# ---------------------------------------------------------------------------

GROUPS = ("codex", "groq", "cohere", "local_models")

def _pct(value: float) -> float:
    """Round a float to six decimal places."""
    return round(float(value), 6)

def doomsday(year: int, month: int, day: int) -> int:
    """
    Return the weekday index used by the original doomsday calendar:
    Monday → 1, …, Sunday → 0 (mod 7).
    """
    return (date(year, month, day).weekday() + 1) % 7

def _rng_from_text(text: str) -> random.Random:
    """Deterministic RNG seeded from a stable SHA-256 hash of *text*."""
    import hashlib

    h = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(h[:8], "big")
    return random.Random(seed)

# ModelPool class from parent A
@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str

TIER_T1_QWEN_0_5B=ModelTier("qwen-0.5b", 512, "T1")
TIER_T2_REASONING=ModelTier("reasoning-t2", 3000, "T2")
TIER_T2_TOOL=ModelTier("tool-t2", 2600, "T2")
TIER_T3_QWEN_7B=ModelTier("qwen-7b", 7000, "T3")

class ModelLoadError(RuntimeError): pass

class ModelPool:
    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb=ram_ceiling_mb; self.loaded={}
    def is_loaded(self, name: str) -> bool: return name in self.loaded
    def _used(self) -> int: return sum(m.ram_mb for m in self.loaded.values())
    def load(self, model: ModelTier) -> None:
        if model.tier=="T3" and any(m.tier=="T2" for m in self.loaded.values()): raise ModelLoadError("T3 mutually exclusive with T2 resident")
        if model.ram_mb + self._used() > self.ram_ceiling_mb: raise ModelLoadError("RAM ceiling exceeded")
        self.loaded[model.name]=model
    def load_with_eviction(self, model: ModelTier) -> None:
        while self.loaded and model.ram_mb + self._used() > self.ram_ceiling_mb:
            self.loaded.pop(next(iter(self.loaded)))
        self.load(model)

# Hybrid functions
def compute_feature_curvature(text: str) -> np.ndarray:
    rng = _rng_from_text(text)
    feature_vector = np.array([rng.random() for _ in range(24)])
    feature_vector /= np.linalg.norm(feature_vector)
    curvature_matrix = np.outer(feature_vector, feature_vector)
    return curvature_matrix

def allocate_workshare_with_features(curvature_matrix: np.ndarray, model_names: List[str]) -> Dict[str, float]:
    one_hot_encodings = np.eye(len(model_names))
    allocation_weights = {}
    for i, model_name in enumerate(model_names):
        one_hot_encoding = one_hot_encodings[i]
        weight = np.dot(curvature_matrix, one_hot_encoding).sum()
        allocation_weights[model_name] = _pct(weight)
    return allocation_weights

def load_model_with_curvature(model_pool: ModelPool, model_tiers: List[ModelTier], text: str) -> None:
    curvature_matrix = compute_feature_curvature(text)
    model_names = [model_tier.name for model_tier in model_tiers]
    allocation_weights = allocate_workshare_with_features(curvature_matrix, model_names)
    sorted_model_tiers = sorted(zip(model_names, model_tiers), key=lambda x: allocation_weights[x[0]], reverse=True)
    for model_name, model_tier in sorted_model_tiers:
        try:
            model_pool.load_with_eviction(model_tier)
        except ModelLoadError:
            print(f"Failed to load {model_tier.name}")

def hybrid_summary(model_pool: ModelPool, model_tiers: List[ModelTier], text: str) -> None:
    load_model_with_curvature(model_pool, model_tiers, text)
    print("Loaded models:")
    for model_name, model_tier in model_pool.loaded.items():
        print(f"{model_name}: {model_tier.ram_mb} MB")

if __name__ == "__main__":
    model_pool = ModelPool()
    model_tiers = [TIER_T1_QWEN_0_5B, TIER_T2_REASONING, TIER_T2_TOOL, TIER_T3_QWEN_7B]
    text = "This is a sample text for feature extraction."
    hybrid_summary(model_pool, model_tiers, text)