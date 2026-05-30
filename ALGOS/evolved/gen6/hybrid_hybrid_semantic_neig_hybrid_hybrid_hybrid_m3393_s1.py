# DARWIN HAMMER — match 3393, survivor 1
# gen: 6
# parent_a: hybrid_semantic_neighbors_hybrid_endpoint_circ_m17_s5.py (gen2)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_infota_m1404_s5.py (gen5)
# born: 2026-05-29T23:49:57Z

"""
Hybrid Semantic-Morphology Neighbor System with Caputo Fractional-Derivative Kernel.
Parents:
- hybrid_semantic_neighbors_hybrid_endpoint_circ_m17_s5.py
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_infota_m1404_s5.py

Mathematical Bridge:
The recovery priority `p ∈ [0,1]` derived from a document's morphology is used as a
multiplicative scaling factor for the cosine similarity `c ∈ [-1,1]` between vectors.
The hybrid affinity is defined as `h = c * p_other` where `p_other` is the recovery priority of the candidate neighbor.
The Caputo fractional-derivative kernel provides a set of long-range memory weights
`w_i = Δt_i^{‑α}`. By interpreting the normalized weight vector as a phase
`φ = 2π·∑w_i` we obtain a rotation angle that can be used as a GA-rotor
(in a 2-D sub-space) to rotate any vector.
The hybrid similarity combines both measures, thereby fusing fractional-memory-driven geometry with set-based similarity and morphology-driven recovery priority.
"""

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np

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

def righting_time_index(m: Morphology, b: float = 1.0 / 3.0, k: float = 0.35, neck_lever: float = 1.0) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever

def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """Maps righting time index to a normalized priority in [0,1]."""
    return max(0.0, min(1.0, righting_time_index(m) / max_index))

def lanczos_gamma(z: float) -> float:
    """Lanczos approximation of Gamma(z) for z > 0."""
    if z < 0.5:
        return math.pi / (math.sin(math.pi * z) * lanczos_gamma(1 - z))
    g = 7
    z += g + 0.5
    p = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ]
    a = 0.99999999999980993
    for i, c in enumerate(p[1:], 1):
        a += c / (z - i)
    t = z + g + 0.5
    return math.sqrt(2 * math.pi) * t ** (z - 0.5) * math.exp(-t) * a

def caputo_weights(times: List[float], alpha: float) -> np.ndarray:
    """
    Compute normalized Caputo kernel weights for a monotonic time vector.

    w_i ∝ (Δt_i)^{‑α}  where Δt_i = t_i – t_{i‑1}.
    The returned array has length len(times)‑1 and sums to 1.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1) for a proper Caputo kernel")
    weights = np.array([(times[i] - times[i-1]) ** (-alpha) for i in range(1, len(times))])
    return weights / np.sum(weights)

def hybrid_affinity(v1: np.ndarray, v2: np.ndarray, m1: Morphology, m2: Morphology) -> float:
    """
    Compute the hybrid affinity between two vectors, taking into account their morphologies.

    The hybrid affinity is defined as `h = c * p_other` where `c` is the cosine similarity
    between the two vectors and `p_other` is the recovery priority of the second morphology.
    """
    c = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    p_other = recovery_priority(m2)
    return c * p_other

def rotated_affinity(v1: np.ndarray, v2: np.ndarray, times: List[float], alpha: float) -> float:
    """
    Compute the rotated affinity between two vectors, taking into account their Caputo kernel.

    The rotated affinity is defined as the cosine similarity between the two vectors
    after rotating the first vector by the phase angle obtained from the Caputo kernel.
    """
    weights = caputo_weights(times, alpha)
    phase_angle = 2 * math.pi * np.sum(weights)
    rotation_matrix = np.array([[math.cos(phase_angle), -math.sin(phase_angle)],
                                 [math.sin(phase_angle), math.cos(phase_angle)]])
    rotated_v1 = np.dot(rotation_matrix, v1[:2])
    return np.dot(rotated_v1, v2[:2]) / (np.linalg.norm(rotated_v1) * np.linalg.norm(v2[:2]))

def hybrid_similarity(v1: np.ndarray, v2: np.ndarray, m1: Morphology, m2: Morphology, times: List[float], alpha: float) -> float:
    """
    Compute the hybrid similarity between two vectors, taking into account their morphologies and Caputo kernel.

    The hybrid similarity is defined as the average of the hybrid affinity and the rotated affinity.
    """
    return (hybrid_affinity(v1, v2, m1, m2) + rotated_affinity(v1, v2, times, alpha)) / 2

if __name__ == "__main__":
    m1 = Morphology(1.0, 2.0, 3.0, 4.0)
    m2 = Morphology(5.0, 6.0, 7.0, 8.0)
    v1 = np.array([1.0, 2.0, 3.0])
    v2 = np.array([4.0, 5.0, 6.0])
    times = [1.0, 2.0, 3.0]
    alpha = 0.5
    print(hybrid_affinity(v1, v2, m1, m2))
    print(rotated_affinity(v1, v2, times, alpha))
    print(hybrid_similarity(v1, v2, m1, m2, times, alpha))