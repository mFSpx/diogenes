# DARWIN HAMMER — match 3331, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_ternar_m1335_s3.py (gen5)
# parent_b: serpentina_self_righting.py (gen0)
# born: 2026-05-29T23:49:30Z

"""Hybrid module combining:
- Parent A: Gaussian RBF similarity, Gaussian beam pruning, Fisher‑score weighting.
- Parent B: Morphology‑based flatness/sphericity indices and righting‑time recovery priority.

Mathematical bridge:
Morphology attributes (length, width, height, mass) are treated as a 4‑D feature vector.
The Euclidean distance between two such vectors feeds the Gaussian RBF (Parent A) to obtain a
similarity score `s`. This similarity is then interpreted as a “θ” parameter for the Gaussian‑beam
pruning function of Parent A. The resulting beam value is passed through a Fisher‑information‑like
score, producing a weighting `w`. Finally the recovery priority derived from Parent B
(`p = recovery_priority(m)`) is modulated by `w` → `hybrid_priority = p * w`. This fuses the
geometric‑similarity core of Parent A with the biomechanics‑driven priority of Parent B into a
single unified metric.
"""

import math
import random
import sys
from pathlib import Path
import numpy as np

# ----------------------------------------------------------------------
# Parent A building blocks
# ----------------------------------------------------------------------
def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian radial basis function."""
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two equal‑length vectors."""
    if a.shape != b.shape:
        raise ValueError("vectors must have same shape")
    return float(np.linalg.norm(a - b))


def rbf_similarity(a: np.ndarray, b: np.ndarray, epsilon: float = 1.0) -> float:
    """Gaussian RBF kernel based on Euclidean distance."""
    return gaussian(euclidean(a, b), epsilon)


def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian beam used for pruning probabilities."""
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """
    Fisher‑information‑like score derived from a Gaussian beam.
    The derivative of the log‑likelihood of a Gaussian w.r.t. its mean gives
    (theta‑center)/(width²). Multiplying by the beam value yields a score that
    grows with distance from the centre but is tempered by the Gaussian envelope.
    """
    beam = gaussian_beam(theta, center, width)
    return ((theta - center) / (width ** 2 + eps)) * beam


# ----------------------------------------------------------------------
# Parent B building blocks
# ----------------------------------------------------------------------
class Morphology:
    """Simple immutable container for shape and mass."""
    __slots__ = ('length', 'width', 'height', 'mass')

    def __init__(self, length: float, width: float, height: float, mass: float):
        if min(length, width, height, mass) <= 0:
            raise ValueError("All morphology parameters must be positive")
        self.length = length
        self.width = width
        self.height = height
        self.mass = mass


def sphericity_index(length: float, width: float, height: float) -> float:
    """Geometric sphericity: cube‑root of volume divided by longest dimension."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length


def flatness_index(length: float, width: float, height: float) -> float:
    """Flatness: average of length & width relative to height."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


def righting_time_index(m: Morphology, b: float = 1.0 / 3.0,
                        k: float = 0.35, neck_lever: float = 1.0) -> float:
    """Estimated time to self‑right, derived from biology."""
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever


def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """Normalized priority (0‥1) for external assistance."""
    return max(0.0, min(1.0, righting_time_index(m) / max_index))


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def morphology_to_vector(m: Morphology) -> np.ndarray:
    """Map a Morphology instance to a 4‑D feature vector."""
    return np.array([m.length, m.width, m.height, m.mass], dtype=float)


def hybrid_similarity(m: Morphology, reference: np.ndarray,
                      epsilon: float = 1.0) -> float:
    """
    Compute Gaussian RBF similarity between a morphology and a reference vector.
    The reference is expected to be a 4‑D NumPy array.
    """
    vec = morphology_to_vector(m)
    return rbf_similarity(vec, reference, epsilon)


def hybrid_priority(m: Morphology, reference: np.ndarray,
                    center: float = 0.5, width: float = 0.1,
                    epsilon: float = 1.0) -> float:
    """
    Fuse recovery priority (Parent B) with similarity‑based Fisher weighting (Parent A).

    Steps:
    1. Base priority `p` from morphology biomechanics.
    2. Similarity `s` via Gaussian RBF.
    3. Fisher weight `w = fisher_score(s, center, width)`.
    4. Return `p * w`, clipped to [0,1].
    """
    p = recovery_priority(m)
    s = hybrid_similarity(m, reference, epsilon)
    w = fisher_score(s, center, width)
    return max(0.0, min(1.0, p * w))


def bandit_update(weights: np.ndarray, reward: float, alpha: float = 0.1) -> np.ndarray:
    """
    Very simple stochastic bandit update: increase probability of the
    arm (here the single hybrid metric) proportionally to the received reward.
    The function keeps the weight vector normalized.
    """
    if weights.ndim != 1:
        raise ValueError("weights must be a 1‑D array")
    if not (0.0 <= reward <= 1.0):
        raise ValueError("reward must be in [0,1]")
    new_weights = weights + alpha * reward * (1.0 - weights)
    return new_weights / new_weights.sum()


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a small population of morphologies
    pop = [
        Morphology(30.0, 20.0, 10.0, 5.0),
        Morphology(25.0, 22.0, 12.0, 4.5),
        Morphology(28.0, 18.0, 9.0, 5.5),
    ]

    # Build a reference vector as the centroid of the population
    vectors = np.stack([morphology_to_vector(m) for m in pop])
    reference_vec = vectors.mean(axis=0)

    # Initialise a trivial bandit weight (single arm)
    weight = np.array([1.0])

    for i, m in enumerate(pop, 1):
        sim = hybrid_similarity(m, reference_vec)
        prio = hybrid_priority(m, reference_vec)
        weight = bandit_update(weight, prio)
        print(f"Sample {i}: similarity={sim:.4f}, hybrid_priority={prio:.4f}, bandit_weight={weight[0]:.4f}")

    sys.exit(0)