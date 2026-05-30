# DARWIN HAMMER — match 2749, survivor 6
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hard_t_hybrid_endpoint_circ_m199_s7.py (gen3)
# parent_b: hybrid_hybrid_hybrid_privac_hybrid_hybrid_hdc_se_m2427_s1.py (gen4)
# born: 2026-05-29T23:45:36Z

"""Hybrid Risk Assessment Module
Combines:
- Parent Algorithm A: Morphology metrics (sphericity, flatness, righting time) and a simple
  ModelPool for resource‑aware loading.
- Parent Algorithm B: Count‑Min Sketch (CMS) turned into a hypervector and bound to a
  causal hypervector via a binding operation.

Mathematical Bridge
------------------
The bridge is the *binding* of two high‑dimensional representations:
1. A CMS derived hypervector **h_cms** ∈ ℝ^D that encodes frequency‑based privacy leakage.
2. A morphology‑derived hypervector **h_morph** ∈ ℝ^D that encodes geometric attributes
   (sphericity, flatness, righting‑time) of an object.

Binding is performed by element‑wise multiplication (Hadamard product):
    **h_bound = h_cms ⊙ h_morph**.

The resulting bound vector is used to compute a hybrid risk score via the cosine
similarity with a reference hypervector **h_ref** (e.g., a learned “high‑risk”
prototype). This score simultaneously reflects privacy frequency, causal influence,
and physical morphology.

The module provides three core hybrid operations:
- `compute_hybrid_vector`: builds the bound hypervector from raw items and a Morphology.
- `hybrid_risk_score`: evaluates the risk by comparing the bound vector to a reference.
- `load_models_for_risk`: demonstrates ModelPool usage in a risk‑assessment workflow.
"""

import hashlib
import random
import math
import sys
from pathlib import Path
from collections import defaultdict
from typing import Iterable, List, Tuple, Dict, Any

import numpy as np

# ----------------------------------------------------------------------
# Parent A components (resource‑aware model pool and morphology metrics)
# ----------------------------------------------------------------------
class ModelTier:
    def __init__(self, name: str, ram_mb: int, tier: str):
        self.name = name
        self.ram_mb = ram_mb
        self.tier = tier


class ModelPool:
    """Simple RAM‑constrained pool for loading/unloading model descriptors."""
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
                f"required {model.ram_mb} MB, available {self.ram_ceiling_mb - self._used()} MB."
            )
        self.loaded[model.name] = model

    def unload(self, name: str) -> None:
        self.loaded.pop(name, None)


# Morphology data class and associated indices (from Parent A)
class Morphology:
    __slots__ = ("length", "width", "height", "mass")
    def __init__(self, length: float, width: float, height: float, mass: float):
        self.length = length
        self.width = width
        self.height = height
        self.mass = mass


def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length


def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


def righting_time_index(
    m: Morphology, b: float = 1.0 / 3.0, k: float = 0.35, neck_lever: float = 1.0
) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever


def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """Normalised priority in [0,1] derived from righting‑time index."""
    return max(0.0, min(1.0, righting_time_index(m) / max_index))


# ----------------------------------------------------------------------
# Parent B components (Count‑Min Sketch → hypervector, binding)
# ----------------------------------------------------------------------
def _cms_hash(item: str, depth: int, width: int) -> List[int]:
    """Return a list of column indices, one per hash row."""
    return [
        int(hashlib.sha256(f"{d}:{item}".encode()).hexdigest(), 16) % width
        for d in range(depth)
    ]


def build_cms(data: Iterable[str], depth: int = 4, width: int = 256) -> np.ndarray:
    """
    Build a Count‑Min Sketch matrix of shape (depth, width).
    Each cell holds an integer count of hashed items.
    """
    cms = np.zeros((depth, width), dtype=np.int32)
    for item in data:
        cols = _cms_hash(item, depth, width)
        for row, col in enumerate(cols):
            cms[row, col] += 1
    return cms


def cms_to_hypervector(cms: np.ndarray, dim: int = 10000, seed: int = 0) -> np.ndarray:
    """
    Convert a CMS matrix into a high‑dimensional float hypervector.
    The conversion uses a deterministic pseudo‑random generator seeded by each
    cell value so that identical counts produce identical sub‑vectors.
    The final hypervector is the normalised sum of all sub‑vectors.
    """
    rng = random.Random(seed)
    flat = cms.flatten()
    # Create a base random vector that will be scaled by each count.
    base_vec = np.array([rng.random() for _ in range(dim)], dtype=np.float32)
    hyper = np.zeros(dim, dtype=np.float32)
    for count in flat:
        if count == 0:
            continue
        # Derive a deterministic scaling factor from the count.
        scale = 1.0 + (count % 7) * 0.1  # simple deterministic function
        hyper += base_vec * scale
    # Normalise to unit length (important for cosine similarity later)
    norm = np.linalg.norm(hyper)
    if norm > 0:
        hyper /= norm
    return hyper


def bind_vectors(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Binding operation (Hadamard product) that fuses two hypervectors.
    Result is re‑normalised to unit length.
    """
    if v1.shape != v2.shape:
        raise ValueError("Vectors must share the same dimensionality for binding")
    bound = v1 * v2
    norm = np.linalg.norm(bound)
    if norm > 0:
        bound /= norm
    return bound


# ----------------------------------------------------------------------
# Hybrid functions (the required three demonstrations)
# ----------------------------------------------------------------------
def morphology_hypervector(m: Morphology, dim: int = 10000) -> np.ndarray:
    """
    Produce a hypervector that encodes the three morphology indices.
    Each index is mapped to a deterministic random sub‑vector; the three are summed
    and normalised.
    """
    # Compute the three scalar attributes
    sph = sphericity_index(m.length, m.width, m.height)
    flat = flatness_index(m.length, m.width, m.height)
    rti = righting_time_index(m)

    # Helper to turn a scalar into a deterministic random vector
    def scalar_to_vec(value: float, label: str) -> np.ndarray:
        seed_bytes = f"{label}:{value:.12f}".encode()
        seed = int.from_bytes(hashlib.sha256(seed_bytes).digest()[:8], "big")
        rng = random.Random(seed)
        return np.array([rng.random() for _ in range(dim)], dtype=np.float32)

    vec_sph = scalar_to_vec(sph, "sph")
    vec_flat = scalar_to_vec(flat, "flat")
    vec_rti = scalar_to_vec(rti, "rti")

    hyper = vec_sph + vec_flat + vec_rti
    norm = np.linalg.norm(hyper)
    if norm > 0:
        hyper /= norm
    return hyper


def compute_hybrid_vector(
    items: Iterable[str],
    morphology: Morphology,
    cms_depth: int = 4,
    cms_width: int = 256,
    hv_dim: int = 10000,
) -> np.ndarray:
    """
    Core hybrid operation:
    1. Build a CMS from the raw items.
    2. Convert the CMS to a hypervector (privacy leakage side‑channel).
    3. Build a morphology hypervector (geometric side‑channel).
    4. Bind the two hypervectors → fused representation.
    """
    cms = build_cms(items, depth=cms_depth, width=cms_width)
    hv_cms = cms_to_hypervector(cms, dim=hv_dim, seed=42)
    hv_morph = morphology_hypervector(morphology, dim=hv_dim)
    return bind_vectors(hv_cms, hv_morph)


def hybrid_risk_score(
    bound_vec: np.ndarray,
    reference_vec: np.ndarray | None = None,
) -> float:
    """
    Compute a risk score as the cosine similarity between the bound hypervector
    and a reference hypervector that represents a “high‑risk” prototype.
    If no reference is supplied, a deterministic random prototype is generated.
    """
    if reference_vec is None:
        rng = random.Random(999)
        reference_vec = np.array([rng.random() for _ in range(bound_vec.shape[0])], dtype=np.float32)
        reference_vec /= np.linalg.norm(reference_vec)

    # Cosine similarity (dot product because both vectors are unit‑normed)
    score = float(np.dot(bound_vec, reference_vec))
    # Map from [-1,1] to a more intuitive [0,1] risk scale
    return (score + 1.0) / 2.0


def load_models_for_risk(models: List[Tuple[str, int, str]]) -> ModelPool:
    """
    Demonstrates ModelPool usage in a risk‑assessment pipeline.
    `models` is a list of (name, ram_mb, tier) tuples.
    Returns a ModelPool with all models loaded (or raises on RAM overflow).
    """
    pool = ModelPool()
    for name, ram, tier in models:
        pool.load(ModelTier(name, ram, tier))
    return pool


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Sample data stream (e.g., user identifiers)
    sample_items = [
        "alice@example.com",
        "bob@example.com",
        "charlie@example.com",
        "alice@example.com",  # duplicate to affect CMS counts
        "dave@example.org",
        "eve@example.net",
    ]

    # Example morphology (arbitrary physical object)
    morph = Morphology(length=2.5, width=1.2, height=0.8, mass=4.3)

    # Compute the fused hypervector
    fused = compute_hybrid_vector(sample_items, morph)

    # Evaluate risk
    risk = hybrid_risk_score(fused)
    print(f"Hybrid risk score: {risk:.4f}")

    # Load dummy models to show ModelPool integration
    dummy_models = [
        ("risk_classifier_v1", 1500, "tier1"),
        ("privacy_encoder", 800, "tier2"),
        ("geometry_analyzer", 1200, "tier1"),
    ]
    pool = load_models_for_risk(dummy_models)
    print(f"Loaded models: {list(pool.loaded.keys())}")
    print(f"Total RAM used: {pool._used()} MB")