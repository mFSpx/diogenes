# DARWIN HAMMER — match 4591, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_decisi_m1260_s1.py (gen4)
# parent_b: hybrid_hybrid_geometric_pro_hybrid_percyphon_hyb_m1442_s2.py (gen3)
# born: 2026-05-29T23:56:57Z

import numpy as np
import math
import random
import sys
import pathlib
from dataclasses import dataclass, field
from typing import Dict, Any

# ----------------------------------------------------------------------
# Parent A – Model pool & entropy utilities
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str  # e.g., "T1", "T2", "T3"

class ModelPool:
    """Manages loaded models under a RAM ceiling with optional eviction."""
    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded: Dict[str, ModelTier] = {}
        self.morphology_lookup: Dict[str, 'Morphology'] = {}

    def is_loaded(self, name: str) -> bool:
        return name in self.loaded

    def _used(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def load(self, model: ModelTier, morphology: 'Morphology') -> None:
        if model.tier == "T3" and any(m.tier == "T2" for m in self.loaded.values()):
            raise RuntimeError("T3 mutually exclusive with T2 resident")
        if model.ram_mb + self._used() > self.ram_ceiling_mb:
            raise RuntimeError("RAM ceiling exceeded")
        self.loaded[model.name] = model
        self.morphology_lookup[model.name] = morphology

    def load_with_eviction(self, model: ModelTier, morphology: 'Morphology') -> None:
        """Evicts oldest entries until the new model fits."""
        while self.loaded and model.ram_mb + self._used() > self.ram_ceiling_mb:
            # FIFO eviction: pop the first inserted key
            self.loaded.pop(next(iter(self.loaded)))
            self.morphology_lookup.pop(next(iter(self.morphology_lookup)))
        self.load(model, morphology)

def calculate_entropy(feature_counts: Dict[str, int]) -> float:
    """Shannon entropy H = - Σ p_i log2 p_i."""
    total = sum(feature_counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in feature_counts.values():
        prob = count / total
        if prob > 0:
            entropy -= prob * math.log2(prob)
    return entropy

# Alias kept for compatibility with parent code
shannon_entropy = calculate_entropy

def reconstruction_risk_score(unique_quasi_identifiers: int,
                              total_records: int) -> float:
    """Simple risk metric: proportion of unique quasi‑identifiers."""
    if total_records == 0:
        return 0.0
    return unique_quasi_identifiers / total_records

# ----------------------------------------------------------------------
# Parent B – Geometry & Clifford‑style operations
# ----------------------------------------------------------------------
@dataclass
class Morphology:
    length: float
    width: float
    height: float
    mass: float = 1.0  # default mass if not supplied

def sphericity_index(length: float, width: float, height: float) -> float:
    """Ratio of the geometric mean of dimensions to the maximum dimension."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    gm = (length * width * height) ** (1.0 / 3.0)
    return gm / max(length, width, height)

def flatness_index(length: float, width: float, height: float) -> float:
    """Flatness = (length+width) / (2*height)."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)

def geometric_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Clifford‑style product: dot plus element‑wise multiplication."""
    return np.dot(a, b) + np.multiply(a, b)

def ttt_ga_forward(W: np.ndarray, R: np.ndarray, x: np.ndarray,
                   eta_w: float, eta_r: float) -> np.ndarray:
    """Three‑tier genetic‑algorithm forward pass with additive Gaussian noise."""
    noise_w = eta_w * np.random.rand(*W.shape)
    noise_r = eta_r * np.random.rand(*R.shape)
    return np.dot(W + noise_w, np.dot(R + noise_r, x))

# ----------------------------------------------------------------------
# Fusion – Entropy‑weighted geometric transformation
# ----------------------------------------------------------------------
def hybrid_ttt_ga_entropy(x_seq: np.ndarray,
                          W: np.ndarray,
                          R: np.ndarray,
                          eta_w: float,
                          eta_r: float,
                          pool: ModelPool,
                          feature_counts: Dict[str, int]) -> np.ndarray:
    """
    Core hybrid operation:
    1. Compute Shannon entropy H from `feature_counts`.
    2. Derive geometry scalars S (sphericity) and F (flatness) from a randomly selected morphology.
    3. Form entropy‑weighted scalars S' = S * H, F' = F * H.
    4. Scale matrices: W' = W * S', R' = R * F'.
    5. Run the three‑tier GA forward pass.
    """
    H = calculate_entropy(feature_counts)
    morphology_name = random.choice(list(pool.loaded.keys()))
    morphology = pool.morphology_lookup[morphology_name]
    S = sphericity_index(morphology.length, morphology.width, morphology.height)
    F = flatness_index(morphology.length, morphology.width, morphology.height)

    # Guard against zero entropy (no information) – fall back to unit scaling.
    if H == 0.0:
        scale_s = S
        scale_f = F
    else:
        scale_s = S * H
        scale_f = F * H

    adjusted_W = W * scale_s
    adjusted_R = R * scale_f

    return ttt_ga_forward(adjusted_W, adjusted_R, x_seq, eta_w, eta_r)

def load_morphology_into_pool(pool: ModelPool,
                              morphology: Morphology,
                              name: str,
                              tier: str = "T1") -> None:
    """
    Convert a `Morphology` into a `ModelTier` whose RAM cost is proportional
    to its physical volume (rounded to the nearest MB) and load it, evicting
    if necessary.
    """
    volume = morphology.length * morphology.width * morphology.height  # cubic units
    ram_mb = max(1, int(volume * 10))  # arbitrary density factor: 10 MB per unit³
    model = ModelTier(name=name, ram_mb=ram_mb, tier=tier)
    pool.load_with_eviction(model, morphology)

def evaluate_hybrid_system(pool: ModelPool,
                           x_seq: np.ndarray,
                           W: np.ndarray,
                           R: np.ndarray,
                           eta_w: float,
                           eta_r: float,
                           feature_counts: Dict[str, int],
                           unique_quasi_identifiers: int,
                           total_records: int) -> Dict[str, Any]:
    """
    Demonstrates the full pipeline:
    * Randomly pick a loaded morphology.
    * Compute the hybrid transformation.
    * Compute reconstruction risk.
    Returns a dictionary with results for inspection.
    """
    if not pool.loaded:
        raise RuntimeError("ModelPool is empty – load at least one Morphology.")
    result = hybrid_ttt_ga_entropy(x_seq, W, R, eta_w, eta_r, pool, feature_counts)
    risk = reconstruction_risk_score(unique_quasi_identifiers, total_records)
    return {
        'result': result,
        'risk': risk,
    }

# Example usage:
if __name__ == '__main__':
    pool = ModelPool()
    morphology = Morphology(length=1.0, width=2.0, height=3.0)
    load_morphology_into_pool(pool, morphology, 'example_morphology')
    x_seq = np.array([1.0, 2.0, 3.0])
    W = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    R = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    eta_w = 0.1
    eta_r = 0.1
    feature_counts = {'feature1': 10, 'feature2': 20}
    unique_quasi_identifiers = 5
    total_records = 100
    result = evaluate_hybrid_system(pool, x_seq, W, R, eta_w, eta_r, feature_counts, unique_quasi_identifiers, total_records)
    print(result)