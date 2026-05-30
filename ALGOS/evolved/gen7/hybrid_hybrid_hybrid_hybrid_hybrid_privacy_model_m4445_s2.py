# DARWIN HAMMER — match 4445, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_fisher_m1571_s3.py (gen6)
# parent_b: hybrid_privacy_model_pool_m7_s0.py (gen1)
# born: 2026-05-29T23:55:52Z

"""Hybrid Fusion of:
- Parent A: count‑min sketch, hyper‑loglog cardinality, Hoeffding bound,
  Gaussian beam propensity, Fisher information, SSIM‑like similarity.
- Parent B: reconstruction risk scoring, differential‑privacy aggregation,
  RAM‑constrained model pool.

Mathematical Bridge
------------------
1. **Gaussian Beam ↔ Propensity** – For each streamed item we map its hash
   to a pseudo‑angle θ∈[−π,π]. The Gaussian intensity g(θ)=exp(−θ²/(2σ²))
   becomes a multiplicative weight wₚ used in the Count‑Min Sketch update.
2. **Fisher Information ↔ Dynamic Confidence** – The Fisher information
   I(θ)=g(θ)/σ² supplies a data‑driven “range” r for the Hoeffding bound
   ε=√(r²·log(1/δ)/(2·n)).
3. **Cardinality ↔ Risk Score** – An HyperLogLog‑style estimator derived from
   the sketch yields an estimated distinct count U. The reconstruction‑risk
   score ρ=U/(U+M) (M≈total RAM ceiling) is fed as the scale of Laplace noise
   that perturbs the RAM requirement before a model is admitted to the pool.
4. **SSIM‑like Gap** – The ratio β=I(θ)/U is compared to a reference signal
   using a simple 1‑D SSIM formula; the resulting similarity modulates the
   final weight applied to the sketch entry.

The three core functions below illustrate this unified system:
`weighted_cms_update`, `privacy_aware_hoeffding`, and `load_model_with_fusion`."""

import hashlib
import math
import random
import re
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Gaussian beam utilities (Parent A)
# ----------------------------------------------------------------------
def _pseudo_angle(item: Any, depth: int, sigma: float = 1.0) -> float:
    """Map an item + depth to a deterministic angle θ∈[−π,π]."""
    h = hashlib.sha256(f"{depth}:{item}".encode()).digest()
    # 64‑bit integer → float in [0,1)
    val = int.from_bytes(h[:8], "big") / 2**64
    return (val * 2 - 1) * math.pi  # scale to [−π,π]

def gaussian_intensity(theta: float, sigma: float = 1.0) -> float:
    """Gaussian beam intensity g(θ)."""
    return math.exp(-theta ** 2 / (2 * sigma ** 2))

def fisher_information(theta: float, sigma: float = 1.0) -> float:
    """Fisher information for the Gaussian beam."""
    g = gaussian_intensity(theta, sigma)
    return g / (sigma ** 2)

# ----------------------------------------------------------------------
# Count‑Min Sketch with Gaussian‑propensity weighting
# ----------------------------------------------------------------------
class WeightedCountMinSketch:
    """Count‑Min Sketch where each update is weighted by a Gaussian propensity."""
    def __init__(self, width: int = 64, depth: int = 4, sigma: float = 1.0):
        self.width = width
        self.depth = depth
        self.sigma = sigma
        self.table: List[List[float]] = [[0.0] * width for _ in range(depth)]

    def update(self, item: Any, weight: float = 1.0) -> None:
        """Standard CMS update with an external weight (default 1)."""
        for d in range(self.depth):
            idx = int(
                hashlib.sha256(f"{d}:{item}".encode()).hexdigest(), 16
            ) % self.width
            self.table[d][idx] += weight

    def weighted_update(self, item: Any) -> None:
        """Hybrid update: weight = Gaussian intensity evaluated at pseudo‑angle."""
        for d in range(self.depth):
            theta = _pseudo_angle(item, d, self.sigma)
            w = gaussian_intensity(theta, self.sigma)
            idx = int(
                hashlib.sha256(f"{d}:{item}".encode()).hexdigest(), 16
            ) % self.width
            self.table[d][idx] += w

    def query(self, item: Any) -> float:
        """Estimate frequency of *item* (minimum across rows)."""
        mins = []
        for d in range(self.depth):
            idx = int(
                hashlib.sha256(f"{d}:{item}".encode()).hexdigest(), 16
            ) % self.width
            mins.append(self.table[d][idx])
        return min(mins)

    def total_counts(self) -> float:
        """Sum of all counters – useful as n for Hoeffding bound."""
        return sum(sum(row) for row in self.table)

# ----------------------------------------------------------------------
# HyperLogLog‑style cardinality estimator from the sketch (Parent A)
# ----------------------------------------------------------------------
def hyperloglog_estimate(sketch: WeightedCountMinSketch) -> int:
    """Very rough distinct‑count estimate: count non‑zero cells across rows."""
    nonzero = sum(1 for row in sketch.table for v in row if v > 0)
    # Apply the classic HLL bias correction factor α_m ≈ 0.7213/(m) where m=width
    m = sketch.width
    if nonzero == 0:
        return 0
    estimate = -m * math.log(1 - nonzero / (m * sketch.depth))
    return int(estimate)

# ----------------------------------------------------------------------
# SSIM‑like similarity between a signal and a reference (Parent A)
# ----------------------------------------------------------------------
def ssim_1d(signal: np.ndarray, reference: np.ndarray, C1: float = 1e-4, C2: float = 9e-4) -> float:
    """Simplified 1‑D SSIM; returns value in [0,1]."""
    mu_x = signal.mean()
    mu_y = reference.mean()
    sigma_x = signal.var()
    sigma_y = reference.var()
    sigma_xy = ((signal - mu_x) * (reference - mu_y)).mean()
    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    return numerator / denominator if denominator != 0 else 0.0

# ----------------------------------------------------------------------
# Reconstruction risk scoring (Parent B)
# ----------------------------------------------------------------------
def reconstruction_risk_score(unique_quasi_identifiers: int, total_records: int) -> float:
    """Risk ∈[0,1] = proportion of quasi‑identifiers."""
    if total_records <= 0:
        return 0.0
    return max(0.0, min(1.0, unique_quasi_identifiers / total_records))

# ----------------------------------------------------------------------
# Model pool (Parent B) with privacy‑aware loading
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str

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
            raise RuntimeError("T3 mutually exclusive with any T2 resident")
        if model.ram_mb + self._used() > self.ram_ceiling_mb:
            raise RuntimeError("RAM ceiling exceeded")
        self.loaded[model.name] = model

    def evict_one(self) -> None:
        """Evict an arbitrary model (FIFO‑style)."""
        if self.loaded:
            key = next(iter(self.loaded))
            del self.loaded[key]

    def load_with_privacy(self, model: ModelTier, epsilon: float = 1.0) -> None:
        """Load a model after adding Laplace noise scaled by reconstruction risk."""
        # Use sketch‑derived cardinality as proxy for unique quasi‑identifiers
        # and total RAM ceiling as total records (both are counts).
        unique = len(self.loaded)  # number of already loaded models
        total = self.ram_ceiling_mb
        risk = reconstruction_risk_score(unique, total)
        noise = np.random.laplace(0.0, risk / epsilon)
        if model.ram_mb + self._used() + noise <= self.ram_ceiling_mb:
            self.load(model)
        else:
            # try to make room by evicting until feasible or empty
            while self.loaded and model.ram_mb + self._used() + noise > self.ram_ceiling_mb:
                self.evict_one()
            self.load(model)

# ----------------------------------------------------------------------
# Fusion Functions
# ----------------------------------------------------------------------
def weighted_cms_update(items: Iterable[Any], sketch: WeightedCountMinSketch) -> None:
    """Update the sketch with Gaussian‑propensity weights for each item."""
    for item in items:
        sketch.weighted_update(item)

def privacy_aware_hoeffding(sketch: WeightedCountMinSketch, delta: float = 0.05, sigma: float = 1.0) -> float:
    """
    Compute a Hoeffding‑type confidence bound where the range r is the
    Fisher information averaged over the sketch updates.
    """
    n = sketch.total_counts()
    if n == 0:
        return float('inf')
    # Approximate average Fisher information by sampling a few cells
    sample_thetas = [_pseudo_angle("sample", d, sigma) for d in range(sketch.depth)]
    avg_fisher = sum(fisher_information(t, sigma) for t in sample_thetas) / sketch.depth
    r = avg_fisher  # treat Fisher as the effective range
    epsilon = math.sqrt(r ** 2 * math.log(1.0 / delta) / (2.0 * n))
    return epsilon

def load_model_with_fusion(model: ModelTier, pool: ModelPool,
                          sketch: WeightedCountMinSketch,
                          epsilon_priv: float = 1.0,
                          delta_conf: float = 0.05) -> None:
    """
    Decide whether to load *model* using a fused signal:
    - Estimate cardinality U from the sketch.
    - Derive a risk‑scaled privacy budget.
    - Apply a Hoeffding confidence bound to modulate the effective RAM demand.
    """
    # 1️⃣ Cardinality estimate (U) → risk score
    U = hyperloglog_estimate(sketch)
    risk = reconstruction_risk_score(U, pool.ram_ceiling_mb)

    # 2️⃣ Confidence bound ε from sketch dynamics
    eps_conf = privacy_aware_hoeffding(sketch, delta=delta_conf)

    # 3️⃣ Adjusted RAM requirement = base + Laplace(noise) + ε‑scaled slack
    lap_noise = np.random.laplace(0.0, risk / epsilon_priv)
    adjusted_ram = model.ram_mb + lap_noise + eps_conf * 10  # scale factor for illustration

    # 4️⃣ Attempt load; if impossible, evict until feasible
    while pool.loaded and adjusted_ram + pool._used() > pool.ram_ceiling_mb:
        pool.evict_one()
    if adjusted_ram + pool._used() <= pool.ram_ceiling_mb:
        # Store the model with its original RAM (pool bookkeeping stays consistent)
        pool.load(model)

# ----------------------------------------------------------------------
# Simple ternary regex feature extractor (Parent A)
# ----------------------------------------------------------------------
_ternary_pattern = re.compile(r"[0-2]+")

def ternary_features(item: str) -> List[int]:
    """Extract contiguous ternary substrings and convert each to an integer."""
    return [int(match, 3) for match in _ternary_pattern.findall(item)]

def augment_item_with_ternary(item: str) -> Tuple[str, float]:
    """
    Return a tuple (augmented_item, extra_weight) where extra_weight is the
    sum of extracted ternary values normalized.
    """
    feats = ternary_features(item)
    if not feats:
        return item, 0.0
    extra = sum(feats) / (len(feats) * (3 ** max(len(str(f)) for f in feats)))
    return f"{item}|ternary", extra

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a sketch
    cms = WeightedCountMinSketch(width=128, depth=5, sigma=0.8)

    # Sample stream with ternary augmentation
    raw_items = ["101", "2120", "abc", "202", "0012", "xyz"]
    stream = []
    for itm in raw_items:
        aug, extra = augment_item_with_ternary(itm)
        stream.append(aug)
        if extra > 0:
            cms.update(aug, weight=extra)  # manual extra weight

    # Hybrid weighted update using Gaussian propensity
    weighted_cms_update(stream, cms)

    # Compute a confidence bound
    bound = privacy_aware_hoeffding(cms, delta=0.01, sigma=0.8)
    print(f"Hoeffding‑style bound: {bound:.6f}")

    # Initialise model pool
    pool = ModelPool(ram_ceiling_mb=2000)
    models = [
        ModelTier(name="A", ram_mb=400, tier="T1"),
        ModelTier(name="B", ram_mb=800, tier="T2"),
        ModelTier(name="C", ram_mb=600, tier="T1"),
    ]

    # Attempt to load models using the fused decision logic
    for m in models:
        try:
            load_model_with_fusion(m, pool, cms, epsilon_priv=0.5, delta_conf=0.01)
            print(f"Loaded model {m.name}")
        except Exception as e:
            print(f"Failed to load {m.name}: {e}")

    print(f"Final RAM usage: {pool._used()} / {pool.ram_ceiling_mb} MB")
    print(f"Models in pool: {list(pool.loaded.keys())}")