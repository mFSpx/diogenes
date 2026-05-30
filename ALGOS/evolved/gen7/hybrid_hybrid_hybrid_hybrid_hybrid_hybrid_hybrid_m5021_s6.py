# DARWIN HAMMER — match 5021, survivor 6
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_endpoint_circ_m1225_s1.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hard_truth_ma_m1388_s3.py (gen6)
# born: 2026-05-29T23:59:29Z

import math
import random
import hashlib
import sys
import pathlib
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import List, Tuple, Iterable

import numpy as np

# ----------------------------------------------------------------------
# Parent A building blocks
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / max(length, width, height)

def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)

def gaussian_beam(theta: float, center: float, width: float, sphericity: float) -> float:
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return sphericity * math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float,
                 sphericity: float, eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width, sphericity), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

def _minhash_token(token: str) -> int:
    return int(hashlib.md5(token.encode()).hexdigest(), 16)

def minhash_signature(tokens: List[str]) -> int:
    if not tokens:
        raise ValueError("token list cannot be empty")
    return min(_minhash_token(t) for t in tokens)

def minhash_similarity(tokens1: List[str], tokens2: List[str]) -> float:
    sig1 = minhash_signature(tokens1)
    sig2 = minhash_signature(tokens2)
    return 1.0 if sig1 == sig2 else 0.0  

def geometric_similarity(m1: Morphology, m2: Morphology) -> float:
    sph1 = sphericity_index(m1.length, m1.width, m1.height)
    sph2 = sphericity_index(m2.length, m2.width, m2.height)
    flat1 = flatness_index(m1.length, m1.width, m1.height)
    flat2 = flatness_index(m2.length, m2.width, m2.height)

    sph_sim = 1.0 - abs(sph1 - sph2) / max(sph1, sph2, 1e-12)
    flat_sim = 1.0 - abs(flat1 - flat2) / max(flat1, flat2, 1e-12)
    return (sph_sim + flat_sim) / 2.0

# ----------------------------------------------------------------------
# Parent B building blocks
# ----------------------------------------------------------------------
Vector = List[int]  

def random_vector(dim: int = 10000, seed: int | str | None = None) -> Vector:
    rng = random.Random(seed)
    return [1 if rng.getrandbits(1) else -1 for _ in range(dim)]

def symbol_vector(symbol: str, dim: int = 10000) -> Vector:
    seed_bytes = hashlib.sha256(symbol.encode("utf-8")).digest()[:8]
    seed = int.from_bytes(seed_bytes, "big")
    return random_vector(dim, seed)

def bind(a: Vector, b: Vector) -> Vector:
    if len(a) != len(b):
        raise ValueError("vectors must have equal length")
    return [x * y for x, y in zip(a, b)]

def bundle(vectors: Iterable[Vector]) -> Vector:
    vecs = list(vectors)
    if not vecs:
        raise ValueError("bundle requires at least one vector")
    dim = len(vecs[0])
    for v in vecs:
        if len(v) != dim:
            raise ValueError("all vectors must have same dimension")
    summed = [sum(comp) for comp in zip(*vecs)]
    return [1 if s >= 0 else -1 for s in summed]

# ----------------------------------------------------------------------
# Hybrid Functions (mathematical fusion)
# ----------------------------------------------------------------------
def hybrid_similarity(tokens1: List[str], tokens2: List[str],
                     morph1: Morphology, morph2: Morphology,
                     alpha: float = 0.5) -> float:
    geom_sim = geometric_similarity(morph1, morph2)
    token_sim = minhash_similarity(tokens1, tokens2)
    return alpha * geom_sim + (1.0 - alpha) * token_sim

def weighted_bind(theta: float, center: float, width: float,
                  morph: Morphology, vec_a: Vector, vec_b: Vector) -> Vector:
    spher = sphericity_index(morph.length, morph.width, morph.height)
    fisher = fisher_score(theta, center, width, spher)

    if fisher < 0:
        raise ValueError("Fisher score must be non-negative")

    sign = np.sign(fisher)  # Use numpy.sign for more robust handling
    bound = bind(vec_a, vec_b)
    return [int(sign * x) for x in bound]  # Explicitly convert to int

def bundle_morphologies(morphs: List[Morphology],
                        theta: float, center: float, width: float,
                        dim: int) -> Vector:
    if not morphs:
        raise ValueError("bundle_morphologies requires at least one morphology")

    vectors = []
    for morph in morphs:
        vec = symbol_vector(str(morph), dim)
        weighted = weighted_bind(theta, center, width, morph, vec, vec)
        vectors.append(weighted)

    return bundle(vectors)

def improved_hybrid_similarity(tokens1: List[str], tokens2: List[str],
                              morphs1: List[Morphology], morphs2: List[Morphology],
                              alpha: float = 0.5) -> float:
    if len(morphs1) != len(morphs2):
        raise ValueError("Both lists of morphologies must have the same length")

    sims = []
    for morph1, morph2 in zip(morphs1, morphs2):
        tokens1_filtered = [t for t in tokens1 if t != '']
        tokens2_filtered = [t for t in tokens2 if t != '']
        sim = hybrid_similarity(tokens1_filtered, tokens2_filtered, morph1, morph2, alpha)
        sims.append(sim)

    return np.mean(sims)  # Use numpy.mean for more robust averaging

def improved_bundle_morphologies(morphs: List[Morphology],
                                 theta: float, center: float, width: float,
                                 dim: int) -> Vector:
    return bundle_morphologies(morphs, theta, center, width, dim)