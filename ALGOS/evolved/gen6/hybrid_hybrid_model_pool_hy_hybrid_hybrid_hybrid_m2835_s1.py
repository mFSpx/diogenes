# DARWIN HAMMER — match 2835, survivor 1
# gen: 6
# parent_a: hybrid_model_pool_hybrid_hybrid_worksh_m319_s0.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2490_s2.py (gen5)
# born: 2026-05-29T23:46:09Z

"""Hybrid Model‑Pool / Physarum‑Risk Algorithm
Parents:
- hybrid_model_pool_hybrid_hybrid_worksh_m319_s0.py (ModelPool, feature curvature)
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2490_s2.py (risk scoring, multivector conductance, DP aggregation)

Mathematical bridge:
The scalar *feature curvature* derived from the text and current model‑pool state is mapped to a
*reconstruction risk score* (0‑1). This risk score rescales a conductance multivector whose scalar
part reflects total RAM utilisation and whose vector part encodes the per‑model RAM distribution.
The rescaled vector is fed through a resource‑allocation matrix derived from tier‑wise RAM sums,
combined with a geometric sphericity index, and finally privately aggregated via a deterministic
DP mean. The result fuses memory‑aware model management with privacy‑aware geometric resource
allocation in a single unified workflow.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, Tuple, List

import numpy as np
from datetime import date

# ----------------------------------------------------------------------
# Parent A – ModelPool and feature curvature utilities
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str

TIER_T1_QWEN_0_5B = ModelTier("qwen-0.5b", 512, "T1")
TIER_T2_REASONING = ModelTier("reasoning-t2", 3000, "T2")
TIER_T2_TOOL = ModelTier("tool-t2", 2600, "T2")
TIER_T3_QWEN_7B = ModelTier("qwen-7b", 7000, "T3")

GROUPS = ("codex", "groq", "cohere", "local_models")

class ModelPool:
    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded: Dict[str, ModelTier] = {}

    def is_loaded(self, name: str) -> bool:
        return name in self.loaded

    def _used(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def load(self, model: ModelTier) -> None:
        if model.tier == "T3" and any(m.tier == "T2" for m in self.loaded.values()):
            raise RuntimeError("T3 mutually exclusive with T2 resident")
        if model.ram_mb + self._used() > self.ram_ceiling_mb:
            raise RuntimeError("RAM ceiling exceeded")
        self.loaded[model.name] = model

    def load_with_eviction(self, model: ModelTier) -> None:
        while self.loaded and model.ram_mb + self._used() > self.ram_ceiling_mb:
            self.loaded.pop(next(iter(self.loaded)))
        self.load(model)

def _pct(value: float) -> float:
    return round(float(value), 6)

def doomsday(year: int, month: int, day: int) -> int:
    return (date(year, month, day).weekday() + 1) % 7

def _rng_from_text(text: str) -> random.Random:
    import hashlib
    h = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(h[:8], "big")
    return random.Random(seed)

def compute_feature_curvature(text: str, model_pool: ModelPool) -> float:
    """
    A lightweight curvature proxy:
    - Generate 10 pseudo‑random numbers seeded by the input text.
    - Compute their standard deviation (a measure of spread → curvature).
    - Modulate by the proportion of RAM currently used.
    """
    rng = _rng_from_text(text)
    samples = [rng.random() for _ in range(10)]
    stddev = np.std(samples)
    usage_ratio = model_pool._used() / max(1, model_pool.ram_ceiling_mb)
    return stddev * usage_ratio

# ----------------------------------------------------------------------
# Parent B – Privacy risk, multivector conductance, DP aggregation
# ----------------------------------------------------------------------
def reconstruction_risk_score(unique_quasi_identifiers: int, total_records: int) -> float:
    """Probability that a record can be re‑identified (0‑1)."""
    if total_records <= 0:
        return 0.0
    return max(0.0, min(1.0, unique_quasi_identifiers / total_records))

def dp_aggregate(values: Iterable[float], epsilon: float = 1.0, sensitivity: float = 1.0) -> float:
    """
    Deterministic differential‑privacy mean aggregator.
    (Real DP would add Laplace noise; here we keep it deterministic for reproducibility.)
    """
    vals = list(values)
    if not vals:
        return 0.0
    return sum(vals) / len(vals)

def sphericity_index(length: float, width: float, height: float) -> float:
    """Geometric sphericity: (geometric mean) / (max dimension)."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / max(length, width, height)

# ----------------------------------------------------------------------
# Hybrid constructs
# ----------------------------------------------------------------------
def build_conductance_multivector(pool: ModelPool) -> Tuple[float, np.ndarray]:
    """
    Construct a simple multivector:
    - scalar part `c` = used RAM / ceiling (global conductance factor).
    - vector part `g` = normalized RAM distribution across loaded models.
    Returns (c, g) where g is a unit‑length numpy array (or zero vector if empty).
    """
    used = pool._used()
    c = used / max(1, pool.ram_ceiling_mb)

    if not pool.loaded:
        return c, np.zeros(0)

    ram_values = np.array([m.ram_mb for m in pool.loaded.values()], dtype=float)
    norm = np.linalg.norm(ram_values)
    g = ram_values / norm if norm > 0 else ram_values
    return c, g

def hybrid_resource_matrix(pool: ModelPool) -> np.ndarray:
    """
    Produce a resource‑allocation matrix R (size N×N where N = number of loaded models).
    Elements are proportional to tier‑wise RAM sums, encouraging intra‑tier sharing.
    """
    models = list(pool.loaded.values())
    n = len(models)
    if n == 0:
        return np.zeros((0, 0))

    # Tier‑wise aggregates
    tier_sums = {}
    for m in models:
        tier_sums.setdefault(m.tier, 0)
        tier_sums[m.tier] += m.ram_mb

    R = np.zeros((n, n), dtype=float)
    for i, mi in enumerate(models):
        for j, mj in enumerate(models):
            if mi.tier == mj.tier:
                R[i, j] = tier_sums[mi.tier]
            else:
                R[i, j] = min(mi.ram_mb, mj.ram_mb) * 0.1  # cross‑tier leakage factor
    return R

def hybrid_allocation(text: str,
                      pool: ModelPool,
                      epsilon: float = 1.0) -> Dict[str, float]:
    """
    End‑to‑end hybrid operation:
    1. Compute feature curvature → map to reconstruction risk.
    2. Build conductance multivector (c, g) from the current pool.
    3. Scale multivector by risk_score.
    4. Allocate resources via R·ĝ (R from tier aggregates, ĝ = scaled g).
    5. Modulate allocation by a sphericity index derived from tier RAM sums.
    6. Return a DP‑aggregated mean of the final allocation vector.
    """
    # 1. Curvature & risk
    curvature = compute_feature_curvature(text, pool)
    # Map curvature (≈0‑0.5) to a quasi‑identifier count for risk scoring
    quasi_ids = int(_pct(curvature) * 1000)
    risk = reconstruction_risk_score(quasi_ids, total_records=1000)

    # 2. Multivector
    c, g = build_conductance_multivector(pool)

    # 3. Scale
    c_scaled = c * risk
    g_scaled = g * risk

    # 4. Allocation matrix
    R = hybrid_resource_matrix(pool)
    if R.size == 0:
        alloc_vec = np.zeros(0)
    else:
        alloc_vec = R @ g_scaled

    # 5. Sphericity modulation
    # Derive dimensions from tier RAM sums
    tier_ram = {"T1": 0.0, "T2": 0.0, "T3": 0.0}
    for m in pool.loaded.values():
        tier_ram[m.tier] += m.ram_mb
    sph = sphericity_index(tier_ram["T1"] + 1e-6,
                           tier_ram["T2"] + 1e-6,
                           tier_ram["T3"] + 1e-6)
    alloc_vec = alloc_vec * (1.0 + sph)

    # 6. DP aggregation
    dp_mean = dp_aggregate(alloc_vec.tolist(), epsilon=epsilon)

    return {
        "curvature": curvature,
        "risk_score": risk,
        "scalar_conductance": c_scaled,
        "allocation_mean_dp": dp_mean,
        "sphericity": sph,
    }

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Initialise a pool with a modest ceiling
    pool = ModelPool(ram_ceiling_mb=8000)

    # Load a mixture of tiers respecting the T2/T3 exclusivity rule
    pool.load(TIER_T1_QWEN_0_5B)
    pool.load(TIER_T2_REASONING)
    pool.load(TIER_T2_TOOL)

    # Example text input
    sample_text = "The quick brown fox jumps over the lazy dog."

    result = hybrid_allocation(sample_text, pool, epsilon=0.5)
    for k, v in result.items():
        print(f"{k}: {v}")

    # Demonstrate eviction + re‑load of a T3 model (forces eviction of T2 models)
    try:
        pool.load(TIER_T3_QWEN_7B)
    except RuntimeError as e:
        print(f"Expected load error (T3 vs T2): {e}")

    # Evict with policy and load T3
    pool.loaded.clear()  # simple eviction for demo
    pool.load(TIER_T3_QWEN_7B)

    # Run allocation again with the new pool state
    result2 = hybrid_allocation(sample_text, pool, epsilon=0.5)
    print("\nAfter loading T3 model:")
    for k, v in result2.items():
        print(f"{k}: {v}")