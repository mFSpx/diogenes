# DARWIN HAMMER — match 4832, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1857_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_geomet_m2702_s3.py (gen5)
# born: 2026-05-29T23:58:23Z

"""
Hybrid Algorithm: hybrid_hybrid_hybrid_hybrid_unified_system.py

This module fuses the core topologies of two parent algorithms:

* Parent A – hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1857_s0.py  
  (provides morphology-driven righting-time index, recovery priority, and a
  structural-similarity (SSIM) based evaluation).

* Parent B – hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_geomet_m2702_s3.py  
  (provides Gaussian similarity utilities and a perceptual hash framework).

The mathematical bridge between the two parents lies in the use of geometric 
and morphological features to inform both physical stability and service 
reliability assessments. Specifically, we use the *sphericity_index* and 
*flatness_index* from Parent A to modulate the Gaussian similarity matrix 
from Parent B, producing a unified assessment that respects both physical 
stability and service reliability.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Shared data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    """Physical description of an endpoint."""
    length: float
    width: float
    height: float
    mass: float

def sphericity_index(length: float, width: float, height: float) -> float:
    """Geometric sphericity used by Parent A."""
    if min(length, width, height) <= 0:
        raise ValueError("All dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length

def flatness_index(length: float, width: float, height: float) -> float:
    """Flatness metric from Parent A."""
    if min(length, width, height) <= 0:
        raise ValueError("All dimensions must be positive")
    return (length + width) / (2.0 * height)

def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian RBF kernel from Parent B."""
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
    """Euclidean distance between two feature vectors from Parent B."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def compute_phash(values: List[float]) -> int:
    """Simple perceptual hash from Parent B."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits

def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two 64-bit integers from Parent B."""
    return (a ^ b).bit_count()

# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------
def hybrid_similarity_matrix(morphologies: Dict[int, Morphology], 
                             features: Dict[int, Tuple[float, ...]], 
                             epsilon: float = 1.0) -> np.ndarray:
    """
    Build a hybrid similarity matrix that combines morphological and 
    feature-based similarities.

    Args:
    morphologies: A dictionary of Morphology objects.
    features: A dictionary of feature vectors.
    epsilon: The epsilon value for the Gaussian kernel.

    Returns:
    A 2D numpy array representing the hybrid similarity matrix.
    """
    nodes = list(morphologies.keys())
    n = len(nodes)
    S = np.empty((n, n), dtype=np.float64)

    for i, ni in enumerate(nodes):
        for j, nj in enumerate(nodes):
            if j < i:
                S[i, j] = S[j, i]
                continue

            # Compute morphological similarity
            mi = morphologies[ni]
            mj = morphologies[nj]
            si = sphericity_index(mi.length, mi.width, mi.height)
            sj = sphericity_index(mj.length, mj.width, mj.height)
            fi = flatness_index(mi.length, mi.width, mi.height)
            fj = flatness_index(mj.length, mj.width, mj.height)
            morphological_sim = gaussian(abs(si - sj), epsilon) * gaussian(abs(fi - fj), epsilon)

            # Compute feature-based similarity
            ai = features[ni]
            aj = features[nj]
            euclidean_dist = euclidean(ai, aj)
            feature_sim = gaussian(euclidean_dist, epsilon)

            # Compute hybrid similarity
            S[i, j] = morphological_sim * feature_sim

    return S

def hybrid_priority(morphology: Morphology, 
                    health_score: float, 
                    epsilon: float = 1.0) -> float:
    """
    Compute a hybrid priority score that combines morphological and 
    health-based priorities.

    Args:
    morphology: A Morphology object.
    health_score: A health score value.
    epsilon: The epsilon value for the Gaussian kernel.

    Returns:
    A hybrid priority score.
    """
    si = sphericity_index(morphology.length, morphology.width, morphology.height)
    fi = flatness_index(morphology.length, morphology.width, morphology.height)
    morphological_priority = gaussian(abs(si - 1.0), epsilon) * gaussian(abs(fi - 1.0), epsilon)
    hybrid_priority = morphological_priority * health_score
    return hybrid_priority

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    morphologies = {
        0: Morphology(1.0, 2.0, 3.0, 10.0),
        1: Morphology(4.0, 5.0, 6.0, 20.0),
    }
    features = {
        0: (1.0, 2.0, 3.0),
        1: (4.0, 5.0, 6.0),
    }
    S = hybrid_similarity_matrix(morphologies, features)
    print(S)

    morphology = Morphology(1.0, 2.0, 3.0, 10.0)
    health_score = 0.8
    priority = hybrid_priority(morphology, health_score)
    print(priority)