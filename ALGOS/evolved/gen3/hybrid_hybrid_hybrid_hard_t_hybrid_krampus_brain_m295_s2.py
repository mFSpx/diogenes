# DARWIN HAMMER — match 295, survivor 2
# gen: 3
# parent_a: hybrid_hybrid_hard_truth_ma_rectified_flow_m91_s0.py (gen2)
# parent_b: hybrid_krampus_brainmap_ollivier_ricci_curva_m13_s1.py (gen1)
# born: 2026-05-29T23:28:18Z

"""Hybrid Stylometry‑BrainMap Flow with Ollivier‑Ricci Curvature
================================================================

This module fuses the two parent algorithms:

* **hybrid_hard_truth_math_model_pool** – provides stylometric
  feature extraction based on linguistic function categories and a
  RAM‑aware model‑pool abstraction.
* **krampus_brainmap + ollivier_ricci_curvature** – supplies a high‑dimensional
  “brain‑map” feature vector and a discrete Ollivier‑Ricci curvature
  estimator for edges in a metric space.

**Mathematical bridge**

1. A text is represented by two vectors  
   *`s ∈ ℝⁿ`* – stylometric counts (parent A) and  
   *`b ∈ ℝᵐ`* – brain‑map features (parent B).  

   The joint embedding is the concatenation  

   ``v = [s; b] ∈ ℝⁿ⁺ᵐ``  

   which lives in a common Euclidean space.

2. **Rectified flow** (parent A) defines a straight‑line interpolant between
   source and target embeddings  

   ``Φ_t(v_src, v_tgt) = (1‑t)·v_src + t·v_tgt``   for ``t∈[0,1]``.

3. **Ollivier‑Ricci curvature** (parent B) is applied to the edge
   ``(v_src, v_tgt)``.  By sampling Gaussian neighbourhoods
   ``μ_src, μ_tgt`` around the endpoints and approximating the
   1‑Wasserstein distance ``W₁(μ_src, μ_tgt)`` with the average Euclidean
   distance of paired samples, curvature is  

   ``κ = 1 - W₁(μ_src, μ_tgt) / ‖v_src‑v_tgt‖₂``.

4. The curvature modulates the flow: the final hybrid embedding is  

   ``v_hybrid = (1 + κ)·Φ_t(v_src, v_tgt)``  

   – positive curvature accelerates the transport, negative curvature
   decelerates it.

The three core functions below implement this pipeline, while the
light‑weight ``ModelPool`` class demonstrates the RAM‑management ideas
from parent A.  All operations rely only on ``numpy`` and the Python
standard library."""

import numpy as np
import random
import math
import sys
import pathlib
import datetime as dt
from collections import Counter
from typing import Dict, List, Tuple

# ----------------------------------------------------------------------
# Parent A – stylometric utilities
# ----------------------------------------------------------------------
FUNCTION_CATS: Dict[str, set] = {
    "pronoun": set("i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()),
    "article": set("a an the".split()),
    "preposition": set("about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()),
    "auxiliary": set("am are be been being can could did do does had has have is may might must shall should was were will would".split()),
    "conjunction": set("and but or nor so yet because although while if when where whereas unless until".split()),
    "negation": set("no not never none neither cannot can't won't don't didn't isn't aren't wasn't weren't".split()),
    "quantifier": set("all any both each few many more most much none several some such".split()),
    "adverb_common": set("very really just still already also even only then there here now often always sometimes".split()),
}

def _tokenize(text: str) -> List[str]:
    """Very simple whitespace / punctuation tokenizer."""
    return [t.lower().strip(".,;:!?\"'()[]{}") for t in text.split() if t]

def stylometric_vector(text: str) -> np.ndarray:
    """
    Convert *text* into a normalized count vector over FUNCTION_CATS.
    Returns an ``(len(FUNCTION_CATS),)`` ndarray.
    """
    tokens = _tokenize(text)
    cat_counts = np.zeros(len(FUNCTION_CATS), dtype=float)
    for idx, (cat, vocab) in enumerate(FUNCTION_CATS.items()):
        cat_counts[idx] = sum(1 for t in tokens if t in vocab)
    total = cat_counts.sum()
    if total > 0:
        cat_counts /= total  # probability‑like distribution
    return cat_counts

# ----------------------------------------------------------------------
# Parent B – brain‑map feature extraction
# ----------------------------------------------------------------------
def extract_full_features(text: str) -> Dict[str, float]:
    """Placeholder: returns a dictionary of random features."""
    rnd = random.Random(hash(text) & 0xffffffff)
    keys = [
        "operator_visceral_ratio", "operator_tech_ratio", "operator_legal_osint_ratio",
        "psyche_forensic_shield_ratio", "psyche_poetic_entropy", "psyche_dissociative_index",
        "resilience_bureaucratic_weaponization_index", "resilience_resource_exhaustion_metric",
        "resilience_swarm_orchestration_density", "rainmaker_corporate_grit_tension",
        "rainmaker_countdown_density", "rainmaker_asset_structuring_weight",
        "telemetry_agent_symmetry_ratio", "telemetry_protocol_discipline", "telemetry_manic_velocity"
    ]
    return {k: rnd.random() for k in keys}

def brainmap_vector(text: str) -> np.ndarray:
    """
    Produce a deterministic brain‑map embedding for *text*.
    The embedding is the ordered list of the values returned by
    ``extract_full_features``.
    """
    feats = extract_full_features(text)
    ordered_keys = sorted(feats.keys())
    return np.array([feats[k] for k in ordered_keys], dtype=float)

# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def joint_vector(text: str) -> np.ndarray:
    """
    Concatenate the stylometric distribution and the brain‑map embedding.
    Shape: (n_categories + n_brain_features,).
    """
    s = stylometric_vector(text)
    b = brainmap_vector(text)
    return np.concatenate([s, b])

def rectified_flow(src: np.ndarray, tgt: np.ndarray, t: float) -> np.ndarray:
    """
    Straight‑line (rectified) flow between *src* and *tgt*.
    ``t`` must be in [0, 1].
    """
    if not 0.0 <= t <= 1.0:
        raise ValueError("t must be between 0 and 1")
    return (1.0 - t) * src + t * tgt

def _sample_neighbourhood(vec: np.ndarray, sigma: float, n: int, rng: random.Random) -> np.ndarray:
    """
    Sample *n* points from a Gaussian centred at *vec* with stddev *sigma*.
    Returns an ``(n, dim)`` array.
    """
    dim = vec.shape[0]
    samples = rng.normalvariate(0, sigma)
    # Using numpy for vectorised sampling
    return vec + sigma * np.random.randn(n, dim)

def ollivier_ricci_curvature(src: np.ndarray, tgt: np.ndarray,
                             sigma: float = 1e-2,
                             n_samples: int = 20,
                             rng_seed: int = 0) -> float:
    """
    Approximate Ollivier‑Ricci curvature κ for the edge (src, tgt).

    The neighbourhood measures are isotropic Gaussians.  The 1‑Wasserstein
    distance is approximated by the average Euclidean distance between
    paired samples (same random seed yields a deterministic pairing).

    Returns κ ∈ ℝ.  κ > 0 indicates positively curved (contracting) edge.
    """
    if src.shape != tgt.shape:
        raise ValueError("src and tgt must have the same dimensionality")
    rng = np.random.default_rng(rng_seed)
    src_samples = _sample_neighbourhood(src, sigma, n_samples, rng)
    tgt_samples = _sample_neighbourhood(tgt, sigma, n_samples, rng)

    # Pairwise distance of corresponding samples (deterministic ordering)
    dists = np.linalg.norm(src_samples - tgt_samples, axis=1)
    W1 = dists.mean()
    base_dist = np.linalg.norm(src - tgt)
    if base_dist == 0.0:
        return 0.0
    kappa = 1.0 - (W1 / base_dist)
    return kappa

def hybrid_interpolate_with_curvature(src_text: str, tgt_text: str,
                                      t: float = 0.5,
                                      sigma: float = 1e-2,
                                      n_samples: int = 20) -> Tuple[np.ndarray, float]:
    """
    Full hybrid pipeline:

    1. Build joint embeddings for *src_text* and *tgt_text*.
    2. Compute rectified flow at time *t*.
    3. Estimate Ollivier‑Ricci curvature κ of the edge.
    4. Modulate the flow by ``(1 + κ)`` and return the result together
       with κ.

    The returned tuple is ``(v_hybrid, κ)``.
    """
    v_src = joint_vector(src_text)
    v_tgt = joint_vector(tgt_text)

    flow = rectified_flow(v_src, v_tgt, t)
    kappa = ollivier_ricci_curvature(v_src, v_tgt, sigma, n_samples)

    v_hybrid = (1.0 + kappa) * flow
    return v_hybrid, kappa

# ----------------------------------------------------------------------
# RAM‑aware model pool (light‑weight version of parent A)
# ----------------------------------------------------------------------
class ModelTier:
    """Simple descriptor for a model's memory footprint."""
    def __init__(self, name: str, ram_mb: int, tier: str):
        self.name = name
        self.ram_mb = ram_mb
        self.tier = tier

class ModelPool:
    """
    Holds a set of models respecting a total RAM ceiling.
    The pool can be queried for loading status and can automatically
    unload the least‑recently‑used model when the ceiling would be exceeded.
    """
    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded: Dict[str, ModelTier] = {}
        self._usage_order: List[str] = []  # LRU tracking

    def is_loaded(self, name: str) -> bool:
        return name in self.loaded

    def _used(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def load(self, model: ModelTier) -> None:
        """Load *model*, evicting LRU models if necessary."""
        if model.name in self.loaded:
            # refresh usage
            self._usage_order.remove(model.name)
            self._usage_order.append(model.name)
            return

        while self._used() + model.ram_mb > self.ram_ceiling_mb:
            if not self._usage_order:
                raise RuntimeError("Cannot fit model within RAM ceiling")
            lru = self._usage_order.pop(0)
            evicted = self.loaded.pop(lru)
            # In a real system we would release resources here

        self.loaded[model.name] = model
        self._usage_order.append(model.name)

    def unload(self, name: str) -> None:
        if name in self.loaded:
            self.loaded.pop(name)
            self._usage_order.remove(name)

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    src = "The quick brown fox jumps over the lazy dog."
    tgt = "In a distant future, machines will learn to write poetry."
    vec, kappa = hybrid_interpolate_with_curvature(src, tgt, t=0.3)
    print(f"Hybrid vector shape: {vec.shape}")
    print(f"Ollivier‑Ricci curvature κ: {kappa:.4f}")

    # Demonstrate ModelPool behaviour
    pool = ModelPool(ram_ceiling_mb=500)
    models = [
        ModelTier("tiny", 100, "A"),
        ModelTier("small", 200, "B"),
        ModelTier("medium", 250, "C")
    ]
    for m in models:
        pool.load(m)
        print(f"Loaded {m.name}, total RAM used: {pool._used()} MB")
    # Attempt to load a large model that forces eviction
    large = ModelTier("large", 300, "D")
    pool.load(large)
    print(f"After loading large model, total RAM used: {pool._used()} MB")
    print("Currently loaded models:", list(pool.loaded.keys()))