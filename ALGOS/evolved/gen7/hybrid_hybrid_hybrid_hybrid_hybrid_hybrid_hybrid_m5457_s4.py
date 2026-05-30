# DARWIN HAMMER — match 5457, survivor 4
# gen: 7
# parent_a: hybrid_hybrid_hybrid_bayes__hybrid_hybrid_hybrid_m1827_s1.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hard_truth_ma_m1388_s2.py (gen6)
# born: 2026-05-30T00:02:15Z

"""
Hybrid Algorithm integrating:
- Parent A: spatial-aware privacy risk weighting combined with NLMS adaptive filtering.
- Parent B: hyperdimensional computing (bind, bundle) with Fisher‑style similarity weighting.

Mathematical Bridge:
The privacy risk vector `r ∈ ℝ^d` produced by Parent A is transformed into a binary
hypervector `r̂ ∈ {−1,+1}^d` (sign of the risk).  This hypervector is then used as a
multiplicative factor in the NLMS weight‑update:
    w←w+μ·e·x·r̂ / (ε+‖x‖²)
where `e` is the error, `x` the input, and `μ,ε` NLMS parameters.
Simultaneously, `r̂` participates in hyperdimensional binding with a text‑derived
feature hypervector `f` (Parent B).  The bound vector `b = bind(f, r̂)` is bundled
with a pool of random hypervectors to obtain a routing hypervector `v`.  The
similarity between `v` and a model‑resource hypervector is finally weighted by a
Fisher‑style score derived from the risk distribution, thus fusing both parent
topologies into a single unified system.
"""

import math
import random
import hashlib
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple
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
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6_371_000.0 * math.asin(min(1.0, math.sqrt(h)))

def _signature(e: Entity) -> str:
    return (e.address_signature or e.category).strip().lower()

def reconstruction_risk_score(entities: List[Entity], delta_m: float) -> np.ndarray:
    """Simple risk: proportion of quasi‑identifiers that are unique within a spatial radius."""
    n = len(entities)
    risks = np.zeros(n)
    for i, ent in enumerate(entities):
        similar = [
            other for j, other in enumerate(entities)
            if i != j and _signature(ent) == _signature(other)
            and haversine_m((ent.lat, ent.lon), (other.lat, other.lon)) <= delta_m
        ]
        # risk proportional to the inverse of the size of the similarity set
        denom = max(1, len(similar) + 1)
        risks[i] = 1.0 / denom
    # normalise to [0,1]
    if risks.max() > 0:
        risks = risks / risks.max()
    return risks

def spatial_aware_privacy_risk_vector(entities: List[Entity], delta_m: float) -> np.ndarray:
    """Return a probability‑like weight vector derived from reconstruction risks."""
    risks = reconstruction_risk_score(entities, delta_m)
    weights = np.exp(-risks)            # higher risk → lower weight
    weights /= weights.sum() + 1e-12    # normalise to sum 1
    return weights

# ----------------------------------------------------------------------
# Parent B – hyperdimensional (bind / bundle) utilities
# ----------------------------------------------------------------------
Vector = List[int]   # hypervector of ±1

def random_vector(dim: int = 10000, seed: int | str | None = None) -> Vector:
    rng = random.Random(seed)
    return [1 if rng.getrandbits(1) else -1 for _ in range(dim)]

def symbol_vector(symbol: str, dim: int = 10000) -> Vector:
    seed_bytes = hashlib.sha256(symbol.encode("utf-8")).digest()[:8]
    seed = int.from_bytes(seed_bytes, "big")
    return random_vector(dim, seed)

def bind(a: Vector, b: Vector) -> Vector:
    """Element‑wise multiplication (XOR for binary ±1 vectors)."""
    if len(a) != len(b):
        raise ValueError("vectors must have equal length")
    return [x * y for x, y in zip(a, b)]

def bundle(vectors: List[Vector]) -> Vector:
    """Majority vote (sum then sign) across a list of hypervectors."""
    if not vectors:
        raise ValueError("bundle requires at least one vector")
    dim = len(vectors[0])
    for v in vectors:
        if len(v) != dim:
            raise ValueError("all vectors must have same dimension")
    summed = [sum(comp) for comp in zip(*vectors)]
    return [1 if s >= 0 else -1 for s in summed]

def similarity(a: Vector, b: Vector) -> float:
    """Normalized dot product, yields cosine‑like similarity in [-1,1]."""
    if len(a) != len(b):
        raise ValueError("vectors must have equal length")
    dot = sum(x * y for x, y in zip(a, b))
    return dot / len(a)

# ----------------------------------------------------------------------
# Hybrid core functions (fusion of both parents)
# ----------------------------------------------------------------------
def expand_risk_to_hyper(risk_vec: np.ndarray, dim: int) -> Vector:
    """
    Convert a real‑valued risk distribution into a ±1 hypervector.
    Steps:
        1. Tile / repeat the risk values to reach `dim`.
        2. Binarise by sign (positive → +1, non‑positive → -1).
    """
    tiled = np.resize(risk_vec, dim)
    signs = np.sign(tiled)
    signs[signs == 0] = -1
    return signs.astype(int).tolist()

def nlms_weight_update(
    w: np.ndarray,
    x: np.ndarray,
    d: float,
    mu: float,
    epsilon: float,
    risk_hyper: Vector
) -> np.ndarray:
    """
    Normalised Least‑Mean‑Squares (NLMS) weight update with risk‑aware scaling.
    `risk_hyper` is a ±1 vector; we cast it to float and multiply element‑wise.
    """
    if w.shape != x.shape:
        raise ValueError("weight and input dimensions must match")
    if len(risk_hyper) != len(w):
        raise ValueError("risk hypervector length must match weight dimension")
    risk_factor = np.array(risk_hyper, dtype=float)
    y = np.dot(w, x)                # current output
    e = d - y                       # error
    norm = epsilon + np.dot(x, x)   # normalising term
    adaptation = (mu / norm) * e * x * risk_factor
    w_new = w + adaptation
    return w_new

def hybrid_route_vector(
    entities: List[Entity],
    delta_m: float,
    text: str,
    dim: int = 1024,
    pool_size: int = 5
) -> Tuple[Vector, float]:
    """
    Produce a routing hypervector that fuses privacy risk (Parent A) and
    hyperdimensional binding/bundling (Parent B).

    Returns:
        routing_vector – the bundled hypervector.
        fisher_weight   – Fisher‑style weight derived from the risk distribution.
    """
    # 1. Risk vector → binary hypervector
    risk_weights = spatial_aware_privacy_risk_vector(entities, delta_m)
    risk_hyper = expand_risk_to_hyper(risk_weights, dim)

    # 2. Text → feature hypervector (symbolic encoding)
    text_hyper = symbol_vector(text, dim)

    # 3. Bind text with risk hypervector
    bound = bind(text_hyper, risk_hyper)

    # 4. Create a pool of random hypervectors and bundle together with the bound vector
    pool = [random_vector(dim, seed=i) for i in range(pool_size)]
    bundle_input = [bound] + pool
    routing_vec = bundle(bundle_input)

    # 5. Fisher‑style weight: variance of risk distribution (higher variance → higher discriminative power)
    fisher_weight = np.var(risk_weights)

    return routing_vec, fisher_weight

def evaluate_model_compatibility(
    routing_vec: Vector,
    model_vec: Vector,
    fisher_weight: float
) -> float:
    """
    Compatibility score = similarity * Fisher weight.
    This mirrors the bridge where Fisher scores weight the bilinear (inner‑product) compatibility.
    """
    sim = similarity(routing_vec, model_vec)
    return sim * fisher_weight

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a tiny synthetic population
    entities = [
        Entity(id="A", lat=40.7128, lon=-74.0060, category="residential"),
        Entity(id="B", lat=40.7130, lon=-74.0055, category="residential"),
        Entity(id="C", lat=34.0522, lon=-118.2437, category="commercial"),
        Entity(id="D", lat=34.0520, lon=-118.2440, category="commercial"),
    ]

    delta_m = 50.0                     # 50 m spatial radius
    dim = 1024                         # hypervector dimensionality
    mu = 0.5                           # NLMS step size
    epsilon = 1e-6                     # NLMS regulariser

    # 1. Compute risk‑aware hypervector
    risk_weights = spatial_aware_privacy_risk_vector(entities, delta_m)
    risk_hyper = expand_risk_to_hyper(risk_weights, dim)

    # 2. NLMS demo
    w = np.zeros(dim)
    x = np.array(random_vector(dim, seed=42), dtype=float)
    desired_output = 1.0
    w_updated = nlms_weight_update(w, x, desired_output, mu, epsilon, risk_hyper)

    # 3. Hybrid routing vector
    text = "sample query for routing"
    routing_vec, fisher_w = hybrid_route_vector(entities, delta_m, text, dim=dim)

    # 4. Model compatibility test
    model_vec = symbol_vector("model_42", dim)
    score = evaluate_model_compatibility(routing_vec, model_vec, fisher_w)

    # Output results (just to verify no exception)
    print("Risk weights:", risk_weights.round(3))
    print("First 5 updated NLMS weights:", w_updated[:5].round(3))
    print("Fisher weight (variance of risk):", fisher_w.round(5))
    print("Compatibility score:", round(score, 5))