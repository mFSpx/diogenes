# DARWIN HAMMER — match 4591, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_decisi_m1260_s1.py (gen4)
# parent_b: hybrid_hybrid_geometric_pro_hybrid_percyphon_hyb_m1442_s2.py (gen3)
# born: 2026-05-29T23:56:57Z

"""Hybrid algorithm combining entropy‑driven model pool management (Parent A) with
geometry‑aware tensor transformation (Parent B).

Mathematical bridge:
- Parent A provides a Shannon entropy `H` computed from feature frequencies.
- Parent B scales weight matrices `W` and rotor `R` by morphology‑derived
  sphericity `S` and flatness `F`.
- The fusion multiplies these geometric scalars by the entropy `H`,
  yielding entropy‑weighted scaling factors `S·H` and `F·H`.  These factors
  modulate the matrices before the three‑tier genetic‑algorithm forward
  operation `ttt_ga_forward`.
- Model loading/eviction is governed by a `ModelPool` whose RAM budget is
  consumed by `Morphology` instances; the RAM cost is proportional to the
  physical volume of the morphology, linking the geometric domain to the
  resource‑management domain.

The resulting system can:
1. Compute entropy from arbitrary feature counts.
2. Load and evict morphology objects respecting a RAM ceiling.
3. Produce a hybrid transformation output that reflects both information‑
   theoretic uncertainty and geometric shape characteristics.
"""

import numpy as np
import math
import random
import sys
import pathlib
from dataclasses import dataclass
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
        """Evicts oldest entries until the new model fits."""
        while self.loaded and model.ram_mb + self._used() > self.ram_ceiling_mb:
            # FIFO eviction: pop the first inserted key
            self.loaded.pop(next(iter(self.loaded)))
        self.load(model)

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
                          morphology: Morphology,
                          feature_counts: Dict[str, int]) -> np.ndarray:
    """
    Core hybrid operation:
    1. Compute Shannon entropy H from `feature_counts`.
    2. Derive geometry scalars S (sphericity) and F (flatness).
    3. Form entropy‑weighted scalars S' = S * H, F' = F * H.
    4. Scale matrices: W' = W * S', R' = R * F'.
    5. Run the three‑tier GA forward pass.
    """
    H = calculate_entropy(feature_counts)
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
    pool.load_with_eviction(model)

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
    morphology_name = random.choice(list(pool.loaded.keys()))
    # Re‑reconstruct the Morphology from its name (encoding length,width,height)
    # For the demo we store the actual object in a side‑dict.
    # In a real system the Morphology would be persisted; here we keep a lookup.
    # We'll attach a hidden attribute to ModelTier for simplicity.
    morphology = getattr(pool.loaded[morphology_name], "morphology_obj", None)
    if morphology is None:
        raise RuntimeError(f"Morphology object missing for model '{morphology_name}'")
    output = hybrid_ttt_ga_entropy(x_seq, W, R, eta_w, eta_r,
                                   morphology, feature_counts)
    risk = reconstruction_risk_score(unique_quasi_identifiers, total_records)
    return {
        "model_name": morphology_name,
        "entropy": calculate_entropy(feature_counts),
        "sphericity": sphericity_index(morphology.length, morphology.width, morphology.height),
        "flatness": flatness_index(morphology.length, morphology.width, morphology.height),
        "hybrid_output": output,
        "reconstruction_risk": risk
    }

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Initialise a model pool with a modest RAM ceiling.
    pool = ModelPool(ram_ceiling_mb=500)

    # Create a few Morphology instances and load them.
    morphologies = [
        Morphology(length=2.0, width=1.5, height=1.0),
        Morphology(length=3.5, width=2.0, height=0.8),
        Morphology(length=1.2, width=1.2, height=1.2)
    ]

    for i, morph in enumerate(morphologies, start=1):
        name = f"morph_{i}"
        load_morphology_into_pool(pool, morph, name, tier="T1")
        # Attach the actual Morphology object to the ModelTier for later retrieval.
        setattr(pool.loaded[name], "morphology_obj", morph)

    # Dummy feature counts for entropy calculation.
    feature_counts = {"age": 120, "zip": 80, "gender": 50, "income": 30}

    # Random matrices/vectors respecting dimensions.
    dim = 4
    np.random.seed(42)
    W = np.random.rand(dim, dim)
    R = np.random.rand(dim, dim)
    x_seq = np.random.rand(dim)

    # Hyper‑parameters for the GA forward pass.
    eta_w = 0.01
    eta_r = 0.01

    # Risk parameters.
    unique_qi = 150
    total_recs = 1000

    # Run the evaluation.
    result = evaluate_hybrid_system(pool, x_seq, W, R,
                                    eta_w, eta_r,
                                    feature_counts,
                                    unique_qi,
                                    total_recs)

    # Print a concise summary.
    print("Hybrid evaluation result:")
    for key, val in result.items():
        if isinstance(val, np.ndarray):
            print(f"{key}: {val.tolist()}")
        else:
            print(f"{key}: {val}")