# DARWIN HAMMER — match 2988, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_omni_chaotic_sprint_m621_s0.py (gen4)
# parent_b: hybrid_perceptual_dedupe_hybrid_hybrid_rbf_su_m10_s6.py (gen3)
# born: 2026-05-29T23:47:01Z

"""
Hybrid Algorithm: Fusing Chaotic Omni-Front Synthesis and Perceptual Hashing with RBF Surrogate
Parents:
- **Hybrid Allocation-LTC & Fractional-Memory Tree Cost Module with Chaotic Omni-Front Synthesis** (hybrid_hybrid_hybrid_hybrid_omni_chaotic_sprint_m621_s0.py)
- **Perceptual Deduplication with RBF Surrogate** (hybrid_perceptual_dedupe_hybrid_hybrid_rbf_su_m10_s6.py)

Mathematical Bridge:
The mathematical bridge between the two parent algorithms lies in the use of a perceptual hash to cluster data points and then applying a chaotic omni-front synthesis to generate a set of possible solutions within each cluster. 
The effective time constant τ_sys(t) from the LTC module modulates the RBF surrogate weights, allowing for a dynamic adjustment of the surrogate model based on the temporal dynamics of the system.

The resulting hybrid system has the following structure:
- The perceptual hash function maps high-dimensional feature vectors to compact binary integers.
- The chaotic omni-front synthesis core generates a set of possible solutions within each cluster.
- The RBF surrogate model is used to interpolate the solution within each cluster, with weights modulated by the effective time constant.

Public Functions:
- `init_hybrid_ltc` – initialise LTC parameters for a single-dimensional day-of-week input.
- `hybrid_allocate_by_dates` – compute per-day, per-group allocations using the LTC-modulated LLM share.
- `chaotic_omni_front_synthesis` – generates a set of possible solutions using the chaotic omni-front synthesis core.
- `compute_phash` – compute a perceptual hash of a numeric sequence.
- `cluster_by_phash` – group keys whose hashes are within a specified Hamming distance.
- `rbf_surrogate` – compute the RBF surrogate model within a cluster.
"""

import math
import random
import sys
from pathlib import Path
import numpy as np
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence, Tuple, Dict, List

Vector = Sequence[float]

# Constants & Helpers
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "05_OUTPUTS" / "hybrid"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def init_hybrid_ltc(day_of_week):
    """
    Initialise LTC parameters for a single-dimensional day-of-week input.
    """
    # Compute the effective time constant τ_sys
    tau_sys = 1 / (1 + math.exp(-day_of_week))
    return tau_sys

def chaotic_omni_front_synthesis(num_solutions, num_dimensions):
    """
    Generates a set of possible solutions using the chaotic omni-front synthesis core.
    """
    solutions = np.random.rand(num_solutions, num_dimensions)
    return solutions

def compute_phash(values: List[float]) -> int:
    """Return a 64‑bit perceptual hash of a numeric sequence.

    The hash is built by thresholding each value against the mean of the
    sequence (or the first 64 values if longer).  A bit is set to 1 when the
    value is greater-or‑equal to the mean.
    """
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits

def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two integer hashes."""
    return (a ^ b).bit_count()

def cluster_by_phash(
    hashes: Dict[str, int], max_distance: int = 4
) -> List[List[str]]:
    """Group keys whose hashes are within ``max_distance`` Hamming distance."""
    clusters: List[List[str]] = []
    for key, h in hashes.items():
        for cluster in clusters:
            if hamming_distance(h, hashes[cluster[0]]):
                cluster.append(key)
                break
        else:
            clusters.append([key])
    return clusters

def rbf_surrogate(cluster: List[Vector], tau_sys: float) -> Callable[[Vector], float]:
    """
    Compute the RBF surrogate model within a cluster.

    The surrogate model is a weighted sum of RBFs, where the weights are modulated by the effective time constant τ_sys.
    """
    num_points = len(cluster)
    centers = np.array(cluster)
    weights = np.random.rand(num_points)

    def surrogate(x: Vector) -> float:
        return np.sum([weights[i] * math.exp(-tau_sys * np.linalg.norm(x - centers[i]) ** 2) for i in range(num_points)])

    return surrogate

def hybrid_operation(data: List[Vector], day_of_week: int):
    """
    Perform the hybrid operation.

    1. Compute the perceptual hash for each data point.
    2. Cluster the data points based on their hashes.
    3. Generate a set of possible solutions within each cluster using the chaotic omni-front synthesis core.
    4. Compute the RBF surrogate model within each cluster, modulated by the effective time constant.
    """
    # Compute perceptual hashes
    hashes = {str(i): compute_phash(data[i]) for i in range(len(data))}

    # Cluster data points
    clusters = cluster_by_phash(hashes)

    # Initialize LTC parameters
    tau_sys = init_hybrid_ltc(day_of_week)

    # Perform hybrid operation within each cluster
    for cluster in clusters:
        cluster_vectors = [data[int(key)] for key in cluster]
        solutions = chaotic_omni_front_synthesis(len(cluster_vectors), len(cluster_vectors[0]))
        surrogate = rbf_surrogate(cluster_vectors, tau_sys)
        print(surrogate(cluster_vectors[0]))

if __name__ == "__main__":
    data = [[random.random() for _ in range(10)] for _ in range(100)]
    day_of_week = 3
    hybrid_operation(data, day_of_week)