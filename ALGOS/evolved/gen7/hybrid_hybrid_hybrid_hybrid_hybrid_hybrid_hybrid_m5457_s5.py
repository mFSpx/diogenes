# DARWIN HAMMER — match 5457, survivor 5
# gen: 7
# parent_a: hybrid_hybrid_hybrid_bayes__hybrid_hybrid_hybrid_m1827_s1.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hard_truth_ma_m1388_s2.py (gen6)
# born: 2026-05-30T00:02:15Z

import math
import random
import hashlib
from dataclasses import dataclass
from typing import List, Tuple, Sequence

import numpy as np

# ----------------------------------------------------------------------
# Parent A – spatial privacy risk utilities
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Entity:
    id: str
    lat: float
    lon: float
    category: str
    score: float = 0.0
    address_signature: str = ""


def haversine_m(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Great‑circle distance in metres between two (lat,lon) points."""
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    return 2 * 6_371_000.0 * math.asin(min(1.0, math.sqrt(h)))


def _signature(e: Entity) -> str:
    return (e.address_signature or e.category).strip().lower()


def reconstruction_risk_score(entities: List[Entity], delta_m: float) -> np.ndarray:
    """Proportion of quasi‑identifiers that are unique within a spatial radius."""
    n = len(entities)
    risks = np.zeros(n, dtype=float)
    for i, ent in enumerate(entities):
        similar = [
            other
            for j, other in enumerate(entities)
            if i != j
            and _signature(ent) == _signature(other)
            and haversine_m((ent.lat, ent.lon), (other.lat, other.lon)) <= delta_m
        ]
        denom = max(1, len(similar) + 1)
        risks[i] = 1.0 / denom
    if risks.max() > 0:
        risks = risks / risks.max()
    return risks


def spatial_aware_privacy_risk_vector(
    entities: List[Entity], delta_m: float
) -> np.ndarray:
    """Probability‑like weight vector derived from reconstruction risks."""
    risks = reconstruction_risk_score(entities, delta_m)
    # higher risk → lower weight, but keep a smooth mapping
    weights = np.exp(-risks)
    weights /= weights.sum() + 1e-12
    return weights


# ----------------------------------------------------------------------
# Parent B – hyperdimensional (bind / bundle) utilities
# ----------------------------------------------------------------------
Vector = np.ndarray  # real‑valued hypervector, values in [-1, 1]


def random_vector(dim: int = 10000, seed: int | str | None = None) -> Vector:
    rng = np.random.default_rng(seed)
    return rng.choice([-1, 1], size=dim).astype(np.int8)


def symbol_vector(symbol: str, dim: int = 10000) -> Vector:
    seed_bytes = hashlib.sha256(symbol.encode("utf-8")).digest()[:8]
    seed = int.from_bytes(seed_bytes, "big")
    return random_vector(dim, seed)


def bind(a: Vector, b: Vector) -> Vector:
    """Element‑wise multiplication (XOR for bipolar vectors)."""
    if a.shape != b.shape:
        raise ValueError("vectors must have equal shape")
    return a * b


def bundle(vectors: List[Vector]) -> Vector:
    """Normalized sum (real‑valued bundling)."""
    if not vectors:
        raise ValueError("bundle requires at least one vector")
    stacked = np.stack(vectors, axis=0).astype(np.float32)
    summed = stacked.sum(axis=0)
    # keep magnitude information but normalise to [-1,1]
    norm = np.linalg.norm(summed) + 1e-12
    return summed / norm


def similarity(a: Vector, b: Vector) -> float:
    """Cosine similarity in [-1,1]."""
    if a.shape != b.shape:
        raise ValueError("vectors must have equal shape")
    dot = float(np.dot(a, b))
    norm = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return dot / norm


# ----------------------------------------------------------------------
# Enhanced fusion utilities
# ----------------------------------------------------------------------
def _risk_to_real_hyper(risk_vec: np.ndarray, dim: int, seed: int | None = None) -> Vector:
    """
    Map a probability‑like risk distribution to a real‑valued hypervector.
    Steps:
        1. Upsample / repeat risk values to reach `dim`.
        2. Apply a symmetric non‑linear transform to spread values in [-1,1].
        3. Randomly permute to break tiling artefacts.
    """
    # 1. repeat / truncate
    tiled = np.resize(risk_vec, dim).astype(np.float32)

    # 2. centre around 0 and squash with tanh for smoothness
    centred = tiled - tiled.mean()
    scaled = np.tanh(5 * centred)  # factor 5 sharpens extremes

    # 3. optional permutation for decorrelation
    rng = np.random.default_rng(seed)
    perm = rng.permutation(dim)
    return scaled[perm]


def nlms_weight_update(
    w: Vector,
    x: Vector,
    d: float,
    mu: float,
    epsilon: float,
    risk_factor: Vector,
) -> Vector:
    """
    NLMS weight update where `risk_factor` is a real‑valued scaling vector
    (same shape as `w`). The adaptation magnitude is modulated by the
    element‑wise risk factor, allowing nuanced attenuation/amplification.
    """
    if w.shape != x.shape:
        raise ValueError("weight and input dimensions must match")
    if risk_factor.shape != w.shape:
        raise ValueError("risk factor must match weight dimension")
    y = float(np.dot(w, x))
    e = d - y
    norm = epsilon + float(np.dot(x, x))
    adaptation = (mu / norm) * e * x * risk_factor
    return w + adaptation


def hybrid_route_vector(
    entities: List[Entity],
    delta_m: float,
    text: str,
    dim: int = 1024,
    pool_size: int = 7,
    seed: int = 42,
) -> Tuple[Vector, float]:
    """
    Produce a routing hypervector that tightly couples privacy risk with
    hyperdimensional binding/bundling.

    Returns:
        routing_vector – bundled real‑valued hypervector.
        fisher_weight   – enriched Fisher‑style weight derived from the risk distribution.
    """
    # 1. Risk distribution → real‑valued hypervector
    risk_weights = spatial_aware_privacy_risk_vector(entities, delta_m)
    risk_hyper = _risk_to_real_hyper(risk_weights, dim, seed=seed)

    # 2. Text → symbolic hypervector (bipolar)
    text_hyper = symbol_vector(text, dim).astype(np.float32)

    # 3. Bind text with risk hypervector (preserves magnitude)
    bound = bind(text_hyper, risk_hyper)

    # 4. Create a pool of random hypervectors, modulate each by the same risk hypervector
    rng = np.random.default_rng(seed)
    pool = [
        bind(random_vector(dim, seed=int(rng.integers(0, 2**31 - 1))), risk_hyper)
        for _ in range(pool_size)
    ]

    # 5. Bundle bound vector with the risk‑modulated pool
    routing_vec = bundle([bound] + pool)

    # 6. Fisher‑style weight: combine variance and distance of mean from 0.5
    var = np.var(risk_weights)
    mean_offset = (risk_weights.mean() - 0.5) ** 2
    fisher_weight = var + mean_offset

    return routing_vec, float(fisher_weight)


def evaluate_model_compatibility(
    routing_vec: Vector,
    model_vec: Vector,
    fisher_weight: float,
) -> float:
    """
    Compatibility score = cosine similarity * Fisher weight.
    The score is bounded in [-fisher_weight, +fisher_weight].
    """
    sim = similarity(routing_vec, model_vec)
    return sim * fisher_weight


# ----------------------------------------------------------------------
# Simple sanity test (executed only when run as script)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # synthetic entities
    ents = [
        Entity(id="e1", lat=40.0, lon=-73.0, category="A"),
        Entity(id="e2", lat=40.0005, lon=-73.0005, category="A"),
        Entity(id="e3", lat=41.0, lon=-74.0, category="B"),
        Entity(id="e4", lat=41.001, lon=-74.001, category="B"),
        Entity(id="e5", lat=42.0, lon=-75.0, category="C"),
    ]

    # generate routing vector
    route_vec, fisher = hybrid_route_vector(
        entities=ents,
        delta_m=100.0,
        text="privacy‑aware model",
        dim=2048,
        pool_size=9,
        seed=12345,
    )

    # dummy model vector (random but fixed)
    model_vec = random_vector(2048, seed=999).astype(np.float32)

    score = evaluate_model_compatibility(route_vec, model_vec, fisher)

    print(f"Fisher weight   : {fisher:.6f}")
    print(f"Compatibility   : {score:.6f}")