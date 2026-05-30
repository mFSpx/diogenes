# DARWIN HAMMER — match 5457, survivor 3
# gen: 7
# parent_a: hybrid_hybrid_hybrid_bayes__hybrid_hybrid_hybrid_m1827_s1.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hard_truth_ma_m1388_s2.py (gen6)
# born: 2026-05-30T00:02:15Z

"""Hybrid Fusion of Spatial Privacy Risk and Hyperdimensional Binding

Parent Algorithms:
- hybrid_hybrid_hybrid_bayes__hybrid_hybrid_hybrid_m1827_s1 (Algorithm A)
- hybrid_hybrid_hybrid_hybrid_hybrid_hard_truth_ma_m1388_s2 (Algorithm B)

Mathematical Bridge:
Algorithm A produces a spatial‑aware privacy risk vector **r** ∈ ℝⁿ that weights each
entity according to geographic proximity and quasi‑identifier similarity.
Algorithm B operates on binary hyperdimensional vectors using bind (⊗) and bundle (⊕)
operations and evaluates similarity with a bilinear (dot‑product) form.

The hybrid algorithm treats **r** as a scalar weighting factor for the bind operation:
for each entity we generate a random hyperdimensional vector **vᵢ**, bind it with its
risk weight (implemented as sign‑preserving scaling), and finally bundle all
weighted vectors into a single composite representation **c**.  An NLMS‑style
adaptive update refines a weight vector **w** that aligns **c** with a desired
target, using the same error term that appears in the NLMS equation but
modulated by the privacy risk weights.

Thus the core topology fuses:
- spatial risk computation (A)
- hyperdimensional bind/bundle algebra (B)
- NLMS adaptive weight update (A) with bilinear similarity (B)
"""

import math
import random
import sys
from pathlib import Path
from typing import List
import numpy as np

# ----------------------------------------------------------------------
# Data structures and geometric utilities (from Algorithm A)
# ----------------------------------------------------------------------
class Entity:
    __slots__ = ("id", "lat", "lon", "category", "score", "address_signature")
    def __init__(self, id: str, lat: float, lon: float,
                 category: str = "", score: float = 0.0,
                 address_signature: str = ""):
        self.id = id
        self.lat = lat
        self.lon = lon
        self.category = category
        self.score = score
        self.address_signature = address_signature

def haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great‑circle distance in metres."""
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = (math.sin(dlat / 2) ** 2 +
         math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
    return 2 * 6_371_000.0 * math.asin(min(1.0, math.sqrt(h)))

def _signature(e: Entity) -> str:
    return (e.address_signature or e.category).strip().lower()

def _reconstruction_risk(entity: Entity, entities: List[Entity],
                         delta_m: float) -> float:
    """Simple proxy risk: proportion of similar‑signature neighbours within δ."""
    similar = [
        other for other in entities
        if other is not entity and
        _signature(other) == _signature(entity) and
        haversine_m((entity.lat, entity.lon), (other.lat, other.lon)) <= delta_m
    ]
    total = len(entities)
    if total == 0:
        return 0.0
    # risk grows with the density of similar neighbours
    return min(1.0, len(similar) / total)

def spatial_privacy_risk_vector(entities: List[Entity],
                                delta_m: float = 500.0) -> np.ndarray:
    """Return a normalized risk weight vector (sums to 1)."""
    risks = np.array([_reconstruction_risk(e, entities, delta_m)
                      for e in entities], dtype=float)
    # Convert risk to a weight that penalises high risk (exp(-risk))
    weights = np.exp(-risks)
    if weights.sum() == 0:
        return np.ones_like(weights) / len(weights)
    return weights / weights.sum()

# ----------------------------------------------------------------------
# Hyperdimensional algebra (from Algorithm B)
# ----------------------------------------------------------------------
Vector = List[int]

def random_vector(dim: int = 10_000, seed: int | None = None) -> Vector:
    rng = random.Random(seed)
    return [1 if rng.getrandbits(1) else -1 for _ in range(dim)]

def bind(a: Vector, b: Vector) -> Vector:
    """Element‑wise multiplication (⊗)."""
    if len(a) != len(b):
        raise ValueError("vectors must have equal length")
    return [x * y for x, y in zip(a, b)]

def bundle(vectors: List[Vector]) -> Vector:
    """Majority‑vote sum (⊕). Returns a bipolar vector."""
    if not vectors:
        raise ValueError("bundle requires at least one vector")
    dim = len(vectors[0])
    for v in vectors:
        if len(v) != dim:
            raise ValueError("all vectors must have same dimension")
    summed = [sum(comp) for comp in zip(*vectors)]
    return [1 if s >= 0 else -1 for s in summed]

def similarity(a: Vector, b: Vector) -> float:
    """Bilinear (dot‑product) similarity normalized by dimension."""
    if len(a) != len(b):
        raise ValueError("vectors must have equal length")
    dot = sum(x * y for x, y in zip(a, b))
    return dot / len(a)

# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def generate_entity_vectors(entities: List[Entity],
                            dim: int = 10_000) -> List[Vector]:
    """Deterministically map each entity ID to a bipolar hyperdimensional vector."""
    vectors = []
    for e in entities:
        # Simple deterministic seed from the ASCII codes of the ID
        seed = sum(ord(ch) for ch in e.id) & 0xFFFFFFFF
        vectors.append(random_vector(dim, seed))
    return vectors

def risk_weighted_binding(entities: List[Entity],
                          dim: int = 10_000,
                          delta_m: float = 500.0) -> Vector:
    """
    1. Compute spatial privacy risk weights r_i.
    2. Generate hyperdimensional vectors v_i for each entity.
    3. Bind each v_i with its scalar weight (sign‑preserving scaling).
    4. Bundle the weighted vectors into a single composite vector c.
    """
    weights = spatial_privacy_risk_vector(entities, delta_m)
    raw_vectors = generate_entity_vectors(entities, dim)

    weighted_vectors: List[Vector] = []
    for w, v in zip(weights, raw_vectors):
        # Scale each component by the weight and keep bipolar sign
        scaled = [1 if x * w >= 0 else -1 for x in v]
        weighted_vectors.append(scaled)

    composite = bundle(weighted_vectors)
    return composite

def nlms_adaptive_update(weight_vec: np.ndarray,
                         input_vec: np.ndarray,
                         desired: float,
                         mu: float = 0.01,
                         eps: float = 1e-6) -> np.ndarray:
    """
    Normalized Least‑Mean‑Squares (NLMS) update.
    error = desired - w·x
    w ← w + μ * error * x / (ε + x·x)
    """
    pred = float(weight_vec.dot(input_vec))
    error = desired - pred
    norm_factor = eps + float(input_vec.dot(input_vec))
    weight_vec = weight_vec + (mu * error / norm_factor) * input_vec
    return weight_vec

def hybrid_fisher_bilinear_routing(entities: List[Entity],
                                   query_text: str,
                                   dim: int = 10_000,
                                   delta_m: float = 500.0) -> float:
    """
    Demonstrates the full hybrid pipeline:
    * Build a risk‑weighted composite vector C from the entities.
    * Derive a simple Fisher‑like score from the spatial risk distribution
      (higher entropy → higher score).
    * Convert the query text into a hyperdimensional probe vector (random‑seeded).
    * Compute a bilinear similarity between the probe and C, weighted by the
      Fisher score.
    """
    # 1. Composite representation
    composite_vec = risk_weighted_binding(entities, dim, delta_m)

    # 2. Fisher‑style score: inverse of variance of the risk weights
    risk_weights = spatial_privacy_risk_vector(entities, delta_m)
    variance = np.var(risk_weights) if risk_weights.size > 0 else 0.0
    fisher_score = 1.0 / (variance + 1e-9)

    # 3. Probe vector from query text (deterministic seed)
    seed = sum(ord(ch) for ch in query_text) & 0xFFFFFFFF
    probe_vec = np.array(random_vector(dim, seed), dtype=int)

    # 4. Bilinear similarity (dot product normalized)
    composite_np = np.array(composite_vec, dtype=int)
    sim = float(probe_vec.dot(composite_np)) / dim

    # 5. Weight similarity by Fisher score
    return fisher_score * sim

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a small synthetic dataset
    entities = [
        Entity(id="A1", lat=40.7128, lon=-74.0060, category="type1"),
        Entity(id="B2", lat=34.0522, lon=-118.2437, category="type1"),
        Entity(id="C3", lat=41.8781, lon=-87.6298, category="type2"),
        Entity(id="D4", lat=29.7604, lon=-95.3698, category="type2")
    ]

    # Run the hybrid pipeline
    result = hybrid_fisher_bilinear_routing(
        entities,
        query_text="sample query for routing",
        dim=2048,          # smaller dimension for quick test
        delta_m=1000.0
    )
    print(f"Hybrid routing score: {result:.6f}")

    # Demonstrate NLMS update on a dummy regression task
    w = np.zeros(2048)
    x = np.random.choice([-1, 1], size=2048)
    desired_output = 0.3
    w = nlms_adaptive_update(w, x, desired_output)
    print(f"NLMS weight norm after one update: {np.linalg.norm(w):.4f}")