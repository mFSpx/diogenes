# DARWIN HAMMER — match 3346, survivor 0
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_path_s_m1618_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_fisher_hybrid_hybrid_hybrid_m1893_s1.py (gen6)
# born: 2026-05-29T23:49:22Z

"""Hybrid Fusion Module

Parents:
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_path_s_m1618_s0.py
- hybrid_hybrid_hybrid_fisher_hybrid_hybrid_hybrid_m1893_s1.py

Mathematical Bridge:
The bridge connects the *path‑signature* machinery of Parent A with the *Fisher‑information* machinery of Parent B.
A path signature S(p) (vector of iterated integrals) quantifies geometric structure of a trajectory.
Parent B supplies a scalar Fisher information F(θ) derived from a Gaussian beam model.
The hybrid algorithm weights each component of the path signature by the corresponding Fisher score and by a pheromone field
π(i) defined over discrete nodes visited by the path.  The final hybrid score is

    H(p) = Σ_i  π(i) · ||S_i(p)|| · F(θ_i)

where θ_i is a geometric angle extracted from the local segment of the path at node i.
Thus the core topologies of both parents are fused into a single unified system."""
import math
import random
import sys
from pathlib import Path
import numpy as np
from collections import Counter, defaultdict
from dataclasses import dataclass, frozen
from datetime import datetime, timezone
from typing import Callable, Dict, Any, Tuple, List

# ----------------------------------------------------------------------
# Parent B – Gaussian beam, Fisher information, morphological mapping
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity I(θ) of a beam centred at *center* with *width*."""
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a single angle θ.

    F(θ) = (∂I/∂θ)² / I  where I = Gaussian beam intensity.
    """
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def probabilistic_morphological_feature_mapping(length: int,
                                                theta: float,
                                                center: float,
                                                width: float) -> float:
    """Probabilistic morphological feature mapping using Fisher information."""
    return fisher_score(theta, center, width) * math.exp(-0.5 * (math.log(length)) ** 2)


# ----------------------------------------------------------------------
# Parent A – Path signature & pheromone handling (simplified)
# ----------------------------------------------------------------------
def compute_path_signature(path: np.ndarray, order: int = 2) -> np.ndarray:
    """
    Compute a simple truncated path signature up to *order*.
    For order=1 we return the displacement vector.
    For order=2 we return the element‑wise product of displacements (approx. second‑order iterated integrals).
    """
    if path.ndim != 2 or path.shape[0] < 2:
        raise ValueError("path must be a 2‑D array with at least two points")
    displacements = np.diff(path, axis=0)                     # shape (N‑1, d)
    sig1 = np.sum(displacements, axis=0)                     # first order
    if order == 1:
        return sig1
    # second order approximation: cumulative outer products summed over segments
    d = displacements.shape[1]
    sig2 = np.zeros(d * d)
    for delta in displacements:
        outer = np.outer(delta, delta).reshape(-1)
        sig2 += outer
    return np.concatenate([sig1, sig2])


def update_pheromones(pheromones: Dict[int, float],
                     path: np.ndarray,
                     decay: float = 0.01,
                     deposit: float = 1.0) -> Dict[int, float]:
    """
    Simple pheromone update:
    - decay all existing pheromones,
    - deposit *deposit* on each integer‑rounded node visited by the path.
    Nodes are identified by rounding the first coordinate to the nearest int.
    """
    # decay
    for node in list(pheromones.keys()):
        pheromones[node] *= (1.0 - decay)
        if pheromones[node] < 1e-6:
            del pheromones[node]
    # deposit
    visited = np.rint(path[:, 0]).astype(int)
    for node in visited:
        pheromones[node] = pheromones.get(node, 0.0) + deposit
    return pheromones


# ----------------------------------------------------------------------
# Hybrid core – integrating both topologies
# ----------------------------------------------------------------------
def hybrid_score(path: np.ndarray,
                 pheromones: Dict[int, float],
                 morphology_length: int,
                 center: float,
                 width: float) -> float:
    """
    Compute the hybrid score H(p) = Σ_i π(i) · ||S_i(p)|| · F(θ_i)

    - path: NxD array of points.
    - pheromones: dict mapping node identifiers to scalar pheromone levels.
    - morphology_length: integer length used in the morphological mapping.
    - center, width: parameters of the Gaussian beam (shared across all segments).
    """
    # 1. Path signature (first order only for simplicity)
    signature = compute_path_signature(path, order=1)          # shape (D,)
    sig_norm = np.linalg.norm(signature)                       # scalar ||S(p)||

    # 2. Extract local angles θ_i from successive segments
    displacements = np.diff(path, axis=0)                     # (N‑1, D)
    angles = np.arctan2(displacements[:, 1], displacements[:, 0]) if displacements.shape[1] >= 2 else np.zeros(displacements.shape[0])

    # 3. Fisher scores per segment
    fisher_vals = np.array([fisher_score(theta, center, width) for theta in angles])

    # 4. Pheromone contribution per visited node (using first coordinate rounding)
    nodes = np.rint(path[:-1, 0]).astype(int)                 # associate each segment with its start node
    pheromone_vals = np.array([pheromones.get(node, 0.0) for node in nodes])

    # 5. Morphological weighting (same for all segments)
    morph_weight = probabilistic_morphological_feature_mapping(morphology_length,
                                                               np.mean(angles) if angles.size else 0.0,
                                                               center,
                                                               width)

    # 6. Combine
    combined = pheromone_vals * fisher_vals * morph_weight
    return sig_norm * np.sum(combined)


# ----------------------------------------------------------------------
# Helper data structures for a higher‑level API
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: int


class HybridSystem:
    """
    High‑level interface that maintains pheromone state and provides
    methods to evaluate hybrid scores for incoming trajectories.
    """
    def __init__(self, center: float = 0.0, width: float = 1.0):
        self.pheromones: Dict[int, float] = {}
        self.center = center
        self.width = width

    def ingest_path(self, path: np.ndarray, morphology: Morphology) -> float:
        """
        Update pheromones based on *path* and return the hybrid score.
        """
        # update internal pheromone map
        self.pheromones = update_pheromones(self.pheromones, path)
        # compute score
        return hybrid_score(path,
                            self.pheromones,
                            morphology.length,
                            self.center,
                            self.width)

    def get_pheromone_map(self) -> Dict[int, float]:
        """Return a copy of the current pheromone map."""
        return dict(self.pheromones)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # deterministic seed for reproducibility
    random.seed(0)
    np.random.seed(0)

    # generate a simple 2‑D random walk
    steps = np.cumsum(np.random.randn(10, 2), axis=0)

    # instantiate system
    system = HybridSystem(center=0.0, width=2.0)

    # dummy morphology
    morph = Morphology(length=5)

    # evaluate
    score = system.ingest_path(steps, morph)

    print("Hybrid score:", score)
    print("Pheromone map:", system.get_pheromone_map())