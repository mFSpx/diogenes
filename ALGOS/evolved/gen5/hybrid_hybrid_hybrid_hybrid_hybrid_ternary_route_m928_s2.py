# DARWIN HAMMER — match 928, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_fisher_hybrid_hybrid_hdc_se_m172_s4.py (gen4)
# parent_b: hybrid_ternary_router_hybrid_hybrid_hybrid_m133_s0.py (gen4)
# born: 2026-05-29T23:31:39Z

"""Hybrid Hyperdimensional Shapley Router
=======================================

This module fuses the two parent algorithms:

* **Parent A** – Hyperdimensional computing with Fisher‑information scoring
  (binding, bundling, similarity, timestamp encoding, etc.).
* **Parent B** – Ternary routing combined with Shapley kernel weighting.

**Mathematical bridge**

The bridge is the *weighting* of hypervectors generated from Fisher scores by the
Shapley kernel weight that originates from the ternary‑router/Shapley side.
Each feature (e.g. a morphological measurement) yields a Fisher score, which is
converted to a hypervector.  The Shapley kernel weight for the feature’s index
acts as a scalar multiplier in a *weighted bundle* operation.  The resulting
feature hypervector is then bound to a ternary‑encoded command vector and
blended with a previous state vector, giving a single unified hypervector that
encodes both the statistical (Fisher) and combinatorial (Shapley) information.

The three core functions below demonstrate this hybrid pipeline:
`ternary_route`, `shapley_weighted_hypervector`, and `hybrid_predictor`. """

import math
import random
import hashlib
from datetime import datetime, timezone
import numpy as np
import sys
import pathlib
from typing import List, Dict, Tuple

Vector = List[int]

# ----------------------------------------------------------------------
# Parent A building blocks (hyperdimensional vectors, Fisher score, etc.)
# ----------------------------------------------------------------------
def random_vector(dim: int = 10000, seed: int | str | None = None) -> Vector:
    rng = random.Random(seed)
    return [1 if rng.getrandbits(1) else -1 for _ in range(dim)]

def bind(a: Vector, b: Vector) -> Vector:
    if len(a) != len(b):
        raise ValueError("vectors must have equal length")
    return [x * y for x, y in zip(a, b)]

def weighted_bundle(vectors: List[Vector], weights: List[float]) -> Vector:
    """Weighted majority vote.  Each component is summed with its weight,
    then the sign determines the bundled bit."""
    if not vectors:
        raise ValueError("bundle requires at least one vector")
    dim = len(vectors[0])
    for v in vectors:
        if len(v) != dim:
            raise ValueError("all vectors must have same dimension")
    if len(vectors) != len(weights):
        raise ValueError("weights length must match vectors length")
    summed = [0.0] * dim
    for vec, w in zip(vectors, weights):
        for i, val in enumerate(vec):
            summed[i] += w * val
    return [1 if s >= 0 else -1 for s in summed]

def similarity(a: Vector, b: Vector) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have equal length")
    dot = sum(x * y for x, y in zip(a, b))
    return dot / len(a)

def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float = 0.0, width: float = 1.0, eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

def fisher_to_hypervector(score: float, dim: int = 10000) -> Vector:
    """Deterministic hypervector from a floating‑point score."""
    score_str = f"{score:.12g}"
    seed = int.from_bytes(hashlib.sha256(score_str.encode()).digest()[:8], "big")
    hv = random_vector(dim, seed)
    if score < 0:
        hv = [-x for x in hv]
    return hv

# ----------------------------------------------------------------------
# Parent B building blocks (ternary routing, Shapley kernel weight)
# ----------------------------------------------------------------------
def ternary_route(command: str, dim: int = 10000) -> Vector:
    """Encode a command string as a ternary hypervector.
    Each character contributes -1, 0, or +1 based on its hash modulo 3."""
    rng = random.Random(hash(command) & ((1 << 64) - 1))
    # Produce a base random vector then map its values to ternary {-1,0,1}
    base = random_vector(dim, rng.randint(0, 2**63 - 1))
    ternary = []
    for val in base:
        mod = rng.getrandbits(2) % 3  # 0,1,2
        ternary.append({0: -1, 1: 0, 2: 1}[mod])
    return ternary

def shapley_kernel_weight(subset_size: int, feature_count: int) -> float:
    """Exact Shapley kernel weight for a given subset size."""
    if feature_count <= 0:
        raise ValueError("feature_count must be positive")
    return (math.factorial(subset_size) *
            math.factorial(feature_count - subset_size - 1) /
            math.factorial(feature_count))

# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def shapley_weighted_hypervector(features: Dict[str, float],
                                 dim: int = 10000) -> Vector:
    """Convert a dict of feature values into a single hypervector.
    Each feature:
      1. gets a Fisher score (treating the raw value as theta),
      2. is turned into a hypervector,
      3. is weighted by its Shapley kernel weight (based on index order).
    The weighted vectors are combined with a weighted bundle."""
    if not features:
        raise ValueError("features dictionary cannot be empty")
    # Deterministic ordering
    keys = sorted(features.keys())
    n = len(keys)
    vectors: List[Vector] = []
    weights: List[float] = []
    for idx, key in enumerate(keys):
        theta = float(features[key])
        score = fisher_score(theta)
        hv = fisher_to_hypervector(score, dim)
        # Shapley weight uses subset size = idx (0‑based) and total count n
        w = shapley_kernel_weight(idx, n)
        vectors.append(hv)
        weights.append(w)
    return weighted_bundle(vectors, weights)

def hybrid_predictor(prev_vec: Vector,
                     command: str,
                     features: Dict[str, float],
                     alpha: float = 0.5,
                     dim: int = 10000) -> Vector:
    """Hybrid predictor that merges three ingredients:
        * previous state vector (prev_vec)
        * ternary‑encoded command vector
        * Shapley‑weighted feature hypervector
    The command vector is bound to the previous vector, then blended with the
    feature vector using a linear interpolation controlled by alpha."""
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in [0, 1]")
    if len(prev_vec) != dim:
        raise ValueError("prev_vec dimension mismatch")
    cmd_vec = ternary_route(command, dim)
    bound = bind(prev_vec, cmd_vec)
    feat_vec = shapley_weighted_hypervector(features, dim)
    # Linear blend in the integer domain; keep sign after blending
    blended = [(1 - alpha) * b + alpha * f for b, f in zip(bound, feat_vec)]
    return [1 if v >= 0 else -1 for v in blended]

def encode_timestamp_vector(ts: float, dim: int = 10000) -> Vector:
    """Timestamp encoding from Parent A, exposed for external use."""
    iso = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    return random_vector(dim, hashlib.sha256(iso.encode()).digest()[:8])

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    DIM = 4096  # smaller size for quick testing
    # Previous hypervector (random seed for reproducibility)
    prev = random_vector(DIM, seed=42)

    # Example command
    cmd = "MOVE_FORWARD"

    # Example feature set (morphology‑like values)
    feats = {
        "length": 12.3,
        "width": 4.7,
        "height": 5.1,
        "mass": 8.9,
        "density": 0.85,
    }

    # Run hybrid predictor
    result = hybrid_predictor(prev, cmd, feats, alpha=0.3, dim=DIM)

    # Simple sanity checks
    sim_prev = similarity(prev, result)
    print(f"Similarity with previous state: {sim_prev:.4f}")

    # Verify that the result is a proper hypervector
    assert len(result) == DIM
    assert all(v in (-1, 1) for v in result)

    # Demonstrate timestamp encoding does not raise
    ts_vec = encode_timestamp_vector(datetime.now(tz=timezone.utc).timestamp(), dim=DIM)
    print(f"Timestamp vector generated, first 5 bits: {ts_vec[:5]}")

    sys.exit(0)