# DARWIN HAMMER — match 4016, survivor 6
# gen: 5
# parent_a: hybrid_hybrid_hybrid_endpoi_epistemic_certainty_m59_s0.py (gen3)
# parent_b: hybrid_hybrid_hybrid_pherom_hybrid_hybrid_hybrid_m397_s3.py (gen4)
# born: 2026-05-29T23:53:04Z

"""Hybrid Algorithm integrating Morphology‑based Epistemic Certainty (Parent A) 
and MinHash‑RBF Pheromone Similarity with Entropy (Parent B).

Mathematical Bridge
-------------------
Parent A provides a *recovery_priority* 𝑝∈[0,1] derived from morphology that
quantifies epistemic certainty about a physical entity.  
Parent B supplies a similarity measure 𝑠∈[0,1] based on MinHash signatures
and a Gaussian‑weighted Euclidean distance 𝑔∈[0,1], together with an
entropy‑based uncertainty 𝐻.

The fusion treats 𝑝 as a confidence weight that modulates the influence of
the similarity 𝑠·𝑔 and the entropy term.  The core hybrid metric is

    χ = p · (s·g) · C(flag) / (1 + H)

where C(flag) maps the epistemic flag to a numeric certainty factor and H is
the expected entropy of a binary hit/miss model.  This creates a single
scalar that simultaneously reflects physical morphology, epistemic confidence,
set‑theoretic similarity, and information‑theoretic uncertainty.

The module implements the full pipeline: morphology feature extraction,
MinHash‑RBF similarity, entropy calculation, and the combined hybrid score.
"""

import math
import random
import sys
import pathlib
import hashlib
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Morphology & Epistemic Certainty utilities
# ----------------------------------------------------------------------
EPISTEMIC_FLAGS: Tuple[str, ...] = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")
FLAG_CONFIDENCE: Dict[str, float] = {
    "FACT": 1.0,
    "PROBABLE": 0.8,
    "POSSIBLE": 0.5,
    "BULLSHIT": 0.0,
    "SURE_MAYBE": 0.3,
}


@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length


def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


def righting_time_index(m: Morphology, b: float = 1.0 / 3.0,
                        k: float = 0.35, neck_lever: float = 1.0) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever


def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """Normalized confidence derived from morphology (0 ≤ p ≤ 1)."""
    return max(0.0, min(1.0, righting_time_index(m) / max_index))


# ----------------------------------------------------------------------
# Parent B – MinHash, RBF, Entropy utilities
# ----------------------------------------------------------------------
def deterministic_hash(token: str, seed: int) -> int:
    h = hashlib.sha256(f"{token}:{seed}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], byteorder="big", signed=False)


def minhash_signature(tokens: List[str], num_hash_functions: int) -> List[int]:
    signature = []
    for i in range(num_hash_functions):
        min_hash = (1 << 64) - 1
        for token in tokens:
            h = deterministic_hash(token, i)
            if h < min_hash:
                min_hash = h
        signature.append(min_hash)
    return signature


def minhash_similarity(sig1: List[int], sig2: List[int]) -> float:
    if len(sig1) != len(sig2) or not sig1:
        return 0.0
    matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
    return matches / len(sig1)


def calculate_entropy(probs: List[float], eps: float = 1e-12) -> float:
    total = sum(probs)
    if total <= 0:
        raise ValueError("positive probability mass required")
    probs = np.array(probs) / total
    probs = np.clip(probs, eps, 1.0)
    return -float(np.sum(probs * np.log(probs)))


def expected_entropy(p_hit: float,
                     hit_state: List[float],
                     miss_state: List[float]) -> float:
    if not 0.0 <= p_hit <= 1.0:
        raise ValueError("p_hit must be in [0, 1]")
    return p_hit * calculate_entropy(hit_state) + (1.0 - p_hit) * calculate_entropy(miss_state)


def gaussian(r: float, epsilon: float = 1.0) -> float:
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: List[float], b: List[float]) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def hybrid_rbf_pheromone_similarity(x: List[float],
                                    y: List[float],
                                    num_hash_functions: int) -> float:
    """Parent B core similarity: Gaussian‑weighted Euclidean distance × MinHash similarity."""
    sig1 = minhash_signature([str(i) for i in x], num_hash_functions)
    sig2 = minhash_signature([str(i) for i in y], num_hash_functions)
    similarity = minhash_similarity(sig1, sig2)
    distance = euclidean(x, y)
    return gaussian(distance) * similarity


# ----------------------------------------------------------------------
# Hybrid Layer – Fusion of the two parents
# ----------------------------------------------------------------------
def morphology_features(m: Morphology) -> Dict[str, float]:
    """Extract a dictionary of morphology‑derived indices."""
    return {
        "sphericity": sphericity_index(m.length, m.width, m.height),
        "flatness": flatness_index(m.length, m.width, m.height),
        "righting_time": righting_time_index(m),
        "recovery_priority": recovery_priority(m),
    }


def epistemic_confidence(flag: str) -> float:
    """Map an epistemic flag to a numeric confidence factor."""
    return FLAG_CONFIDENCE.get(flag.upper(), 0.0)


def hybrid_confidence_score(m: Morphology,
                            x: List[float],
                            y: List[float],
                            flag: str,
                            num_hash_functions: int = 64,
                            epsilon: float = 1.0) -> float:
    """
    Core hybrid metric χ = p · (s·g) · C(flag) / (1 + H)

    - p : recovery_priority from morphology (0‑1)
    - s·g : similarity from Parent B (0‑1)
    - C(flag) : epistemic confidence factor (0‑1)
    - H : expected entropy of a binary hit/miss model (≥0)
    """
    # Morphology‑derived confidence
    p = recovery_priority(m)

    # Parent B similarity
    s_g = hybrid_rbf_pheromone_similarity(x, y, num_hash_functions)

    # Epistemic flag confidence
    c_flag = epistemic_confidence(flag)

    # Entropy term – we treat hit/miss probabilities as the similarity vs. (1‑similarity)
    p_hit = s_g
    hit_state = [p_hit, 1 - p_hit]   # simplistic binary distribution
    miss_state = [1 - p_hit, p_hit]
    H = expected_entropy(p_hit, hit_state, miss_state)

    # Final hybrid score
    chi = p * s_g * c_flag / (1.0 + H)
    return chi


def weighted_semiseparable_matrix(m: Morphology,
                                  vectors: List[List[float]],
                                  flag: str,
                                  num_hash_functions: int = 32) -> np.ndarray:
    """
    Build a symmetric matrix M where M[i,j] = hybrid_confidence_score
    between vectors[i] and vectors[j] weighted by morphology‑derived priority.
    The matrix is semiseparable because each entry can be expressed as
    u_i * v_j + v_i * u_j with u_i = recovery_priority and v_j = similarity term.
    """
    n = len(vectors)
    if n == 0:
        return np.empty((0, 0))

    priority = recovery_priority(m)
    flag_factor = epistemic_confidence(flag)

    # Pre‑compute pairwise similarities
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            s = hybrid_rbf_pheromone_similarity(vectors[i], vectors[j], num_hash_functions)
            sim[i, j] = sim[j, i] = s

    # Construct semiseparable representation
    u = np.full(n, priority * flag_factor)          # same for all rows (could be vectorized per row)
    v = sim                                        # similarity matrix acts as second factor

    M = np.outer(u, np.ones(n)) * v + np.outer(np.ones(n), u) * v.T
    # Ensure symmetry
    M = (M + M.T) / 2.0
    return M


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a sample morphology
    sample_morph = Morphology(length=2.0, width=1.5, height=0.8, mass=3.0)

    # Random vectors for similarity testing
    random.seed(42)
    vec_a = [random.uniform(0, 10) for _ in range(5)]
    vec_b = [random.uniform(0, 10) for _ in range(5)]
    vec_c = [random.uniform(0, 10) for _ in range(5)]

    # Demonstrate individual components
    feats = morphology_features(sample_morph)
    print("Morphology features:", feats)

    sim_ab = hybrid_rbf_pheromone_similarity(vec_a, vec_b, num_hash_functions=64)
    print("Hybrid RBF‑Pheromone similarity (A,B):", sim_ab)

    score = hybrid_confidence_score(sample_morph, vec_a, vec_b,
                                   flag="PROBABLE", num_hash_functions=64)
    print("Hybrid confidence score (A,B):", score)

    # Build a semiseparable matrix for three vectors
    matrix = weighted_semiseparable_matrix(sample_morph,
                                           [vec_a, vec_b, vec_c],
                                           flag="FACT",
                                           num_hash_functions=32)
    print("Weighted semiseparable matrix:\n", matrix)