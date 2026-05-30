# DARWIN HAMMER — match 2749, survivor 5
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hard_t_hybrid_endpoint_circ_m199_s7.py (gen3)
# parent_b: hybrid_hybrid_hybrid_privac_hybrid_hybrid_hdc_se_m2427_s1.py (gen4)
# born: 2026-05-29T23:45:36Z

"""
Hybrid Algorithm: Fusion of Model Management & Morphology (Parent A) with
Count‑Min Sketch Hypervector Binding (Parent B).

Mathematical Bridge
-------------------
- Parent A provides geometric descriptors of an object via :func:`sphericity_index`,
  :func:`flatness_index` and :func:`righting_time_index`.  These descriptors are
  collected in a *morphology vector* (a dense real‑valued hypervector) that
  encodes the shape and mass of the object.
- Parent B builds a Count‑Min Sketch (CMS) for a multiset of items and converts
  the sketch into a hypervector by scaling a deterministic random base vector
  with the per‑row counts.  This yields a *frequency hypervector* that captures
  privacy‑leakage risk.
- The fusion binds the two hypervectors with element‑wise multiplication
  (the classic HDC binding operator).  The bound vector is then projected onto
  the morphology space by a dot‑product, producing a *hybrid risk score* that
  simultaneously accounts for (i) the frequency‑based privacy exposure of the
  items and (ii) the geometric susceptibility of the object (recovery priority).

The module therefore implements:
1. CMS construction and conversion to a hypervector.
2. Morphology‑based hypervector generation.
3. Binding + scoring functions that combine both worlds.
"""

import math
import random
import hashlib
from pathlib import Path
from typing import List, Dict, Iterable, Tuple
import numpy as np

# ----------------------------------------------------------------------
# Parent A – Model pool and morphology utilities (kept unchanged)
# ----------------------------------------------------------------------


class ModelTier:
    def __init__(self, name: str, ram_mb: int, tier: str):
        self.name = name
        self.ram_mb = ram_mb
        self.tier = tier


class ModelPool:
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
    """Normalized susceptibility in [0,1] derived from the righting‑time index."""
    return max(0.0, min(1.0, righting_time_index(m) / max_index))


# ----------------------------------------------------------------------
# Parent B – CMS → hypervector utilities (completed & adapted)
# ----------------------------------------------------------------------


def _cms_hash(item: str, depth: int, width: int) -> List[int]:
    """Return a list of column indices, one per hash row."""
    return [
        int(hashlib.sha256(f"{d}:{item}".encode()).hexdigest(), 16) % width
        for d in range(depth)
    ]


def random_vector(dim: int = 10000, seed: int | str | None = None) -> np.ndarray:
    """Deterministic pseudo‑random real vector in [0,1)."""
    rng = random.Random(seed)
    return np.fromiter((rng.random() for _ in range(dim)), dtype=np.float64)


def cms_hypervector(
    items: Iterable[str],
    depth: int = 4,
    width: int = 256,
    dim: int = 10000,
) -> np.ndarray:
    """
    Build a Count‑Min Sketch for *items* and convert it to a hypervector.
    Each row of the sketch is turned into a random base vector; the row count
    scales the base vector, and the scaled rows are summed.
    """
    # 1) Build the sketch (counts per row/column)
    sketch = np.zeros((depth, width), dtype=np.int32)
    for it in items:
        cols = _cms_hash(it, depth, width)
        for d, c in enumerate(cols):
            sketch[d, c] += 1

    # 2) Convert each row to a random vector and weight by the row sum
    hv = np.zeros(dim, dtype=np.float64)
    for d in range(depth):
        base = random_vector(dim, seed=f"cms_row_{d}")
        row_sum = sketch[d].sum()
        if row_sum == 0:
            continue
        hv += base * row_sum
    # Normalise to unit length to keep binding stable
    norm = np.linalg.norm(hv)
    return hv / norm if norm > 0 else hv


def morphology_hypervector(m: Morphology, dim: int = 10000) -> np.ndarray:
    """
    Encode the geometric descriptors of *m* into a dense hypervector.
    The vector is deterministic: a seed derived from the morphology values
    drives a random base vector which is then element‑wise scaled by a
    normalised feature vector.
    """
    # Feature vector: [sphericity, flatness, mass] – all positive
    sph = sphericity_index(m.length, m.width, m.height)
    flt = flatness_index(m.length, m.width, m.height)
    features = np.array([sph, flt, m.mass], dtype=np.float64)

    # Normalise features to unit sum to avoid exploding scales
    if features.sum() == 0:
        raise ValueError("Morphology features sum to zero")
    features = features / features.sum()

    # Deterministic seed from morphology values
    seed_bytes = hashlib.sha256(
        f"{m.length}:{m.width}:{m.height}:{m.mass}".encode()
    ).digest()[:8]
    seed = int.from_bytes(seed_bytes, "big")
    base = random_vector(dim, seed=seed)

    # Broadcast features across the vector (repeat pattern)
    repeat = dim // len(features) + 1
    scaling = np.tile(features, repeat)[:dim]

    hv = base * scaling
    norm = np.linalg.norm(hv)
    return hv / norm if norm > 0 else hv


def bind_vectors(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Hyperdimensional binding via element‑wise multiplication followed by
    L2 normalisation.  Both vectors are assumed to be unit‑norm.
    """
    bound = v1 * v2
    norm = np.linalg.norm(bound)
    return bound / norm if norm > 0 else bound


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------


def hybrid_risk_score(
    morphology: Morphology,
    items: Iterable[str],
    depth: int = 4,
    width: int = 256,
    dim: int = 10000,
    max_index: float = 10.0,
) -> float:
    """
    Compute a risk score that blends privacy leakage (CMS) with geometric
    susceptibility (recovery priority).

    Steps
    -----
    1. Build a morphology hypervector `hv_m`.
    2. Build a CMS hypervector `hv_c`.
    3. Bind them → `hv_b`.
    4. Project `hv_b` onto `hv_m` (dot product) to obtain a similarity term.
    5. Multiply by the normalized recovery priority.
    """
    hv_m = morphology_hypervector(morphology, dim=dim)
    hv_c = cms_hypervector(items, depth=depth, width=width, dim=dim)
    hv_b = bind_vectors(hv_m, hv_c)

    similarity = float(np.dot(hv_b, hv_m))  # ∈ [0,1] for unit vectors
    priority = recovery_priority(morphology, max_index=max_index)

    return similarity * priority


def load_models_for_morphology(pool: ModelPool, morphology: Morphology) -> List[ModelTier]:
    """
    Demonstrates interaction between the ModelPool (Parent A) and the hybrid
    risk computation (Parent B).  Models whose *ram_mb* is proportional to the
    object's mass are attempted to be loaded; only those that fit within the
    RAM ceiling are returned.
    """
    # Simple heuristic: one model per kilogram of mass, each 10 MB
    required_models = [
        ModelTier(name=f"model_{i}", ram_mb=10, tier="standard")
        for i in range(int(math.ceil(morphology.mass)))
    ]

    loaded = []
    for mdl in required_models:
        try:
            pool.load(mdl)
            loaded.append(mdl)
        except RuntimeError:
            # Stop loading once RAM is exhausted
            break
    return loaded


def hybrid_decision(
    morphology: Morphology,
    items: Iterable[str],
    pool: ModelPool,
    risk_threshold: float = 0.3,
) -> Tuple[bool, float, List[ModelTier]]:
    """
    Returns a decision tuple:
        (accept, risk_score, loaded_models)

    *accept* is True if the hybrid risk score is below *risk_threshold* and at
    least one model could be loaded for the given morphology.
    """
    score = hybrid_risk_score(morphology, items)
    loaded_models = load_models_for_morphology(pool, morphology)
    accept = (score < risk_threshold) and (len(loaded_models) > 0)
    return accept, score, loaded_models


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a sample morphology
    demo_morph = Morphology(length=2.0, width=1.0, height=0.5, mass=3.7)

    # Simulated item stream (e.g., privacy‑sensitive tokens)
    demo_items = ["alpha", "beta", "gamma", "alpha", "delta", "beta", "epsilon"]

    # Initialise model pool
    pool = ModelPool(ram_ceiling_mb=200)

    # Compute hybrid risk
    risk = hybrid_risk_score(demo_morph, demo_items)
    print(f"Hybrid risk score: {risk:.4f}")

    # Perform decision
    accept, score, models = hybrid_decision(demo_morph, demo_items, pool)
    print(f"Decision: {'ACCEPT' if accept else 'REJECT'} (score={score:.4f})")
    print(f"Loaded {len(models)} model(s): {[m.name for m in models]}")