# DARWIN HAMMER — match 2988, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_omni_chaotic_sprint_m621_s0.py (gen4)
# parent_b: hybrid_perceptual_dedupe_hybrid_hybrid_rbf_su_m10_s6.py (gen3)
# born: 2026-05-29T23:47:01Z

"""
Hybrid Algorithm: Combining Temporal Dynamics, Chaotic Omni-Front Synthesis, and Perceptual Hash-Based Clustering

Parents:
- **Hybrid Allocation-LTC & Fractional-Memory Tree Cost Module** (hybrid_hybrid_hybrid_worksh_hybrid_caputo_fracti_m46_s0.py)
- **LUCIDOTA Chaotic Omni-Front Synthesis Core** (omni_chaotic_sprint.py)
- **Perceptual Hash-Based Clustering and RBF Surrogate** (perceptual_dedupe.py)

Mathematical Bridge:
The temporal dynamics of the Liquid Time-Constant (LTC) module are integrated with the chaotic omni-front synthesis core and the perceptual hash-based clustering. 
The effective time constant τ_sys(t) from the LTC module is used as a multiplicative factor on the RBF weights, 
and the hash clustering provides the matrix-partitioning that feeds the linear system solved for the RBF weights.
The chaotic omni-front synthesis core is used to generate a set of possible solutions, 
which are then filtered and refined using the effective time constant and the perceptual hash-based clustering.

"""

import math
import random
import sys
from pathlib import Path
import numpy as np

# Constants & Helpers
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "05_OUTPUTS" / "hybrid"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# Perceptual hash utilities
# ----------------------------------------------------------------------
def compute_phash(values: List[float]) -> int:
    """Return a 64-bit perceptual hash of a numeric sequence."""
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
            if hamming_distance(h, hashes[cluster[0]]) <= max_distance:
                cluster.append(key)
                break
        else:
            clusters.append([key])
    return clusters


# ----------------------------------------------------------------------
# Chaotic omni-front synthesis utilities
# ----------------------------------------------------------------------
def chaotic_omni_front_synthesis(values: List[float]) -> List[float]:
    """Generate a set of possible solutions using the chaotic omni-front synthesis core."""
    return [x + random.uniform(-0.1, 0.1) for x in values]


def increment_hybrid_fractional_tree_cost(values: List[float]) -> float:
    """Builds the tree edge-by-edge, updates distances, and evaluates the hybrid cost using the fractional memory term."""
    # Compute the effective time constant τ_sys
    tau_sys = init_hybrid_ltc(values[0])
    
    # Compute the RBF weights using the effective time constant
    rbf_weights = [w * tau_sys for w in chaotic_omni_front_synthesis(values[1:])]
    
    # Compute the tree cost using the RBF weights
    tree_cost = 0
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            tree_cost += rbf_weights[i] * rbf_weights[j] * (values[i] - values[j]) ** 2
    
    return tree_cost


# ----------------------------------------------------------------------
# Hybrid LTC and perceptual hash-based clustering
# ----------------------------------------------------------------------
def init_hybrid_ltc(day_of_week: float) -> float:
    """
    Initialise LTC parameters for a single-dimensional day-of-week input.
    """
    # Compute the effective time constant τ_sys
    tau_sys = 1 / (1 + np.exp(-day_of_week))
    return tau_sys


def hybrid_allocate_by_dates(day_of_week: float, values: List[float]) -> List[float]:
    """
    Compute per-day, per-group allocations using the LTC-modulated LLM share.
    """
    # Compute the effective time constant τ_sys
    tau_sys = init_hybrid_ltc(day_of_week)
    
    # Compute the RBF weights using the effective time constant
    rbf_weights = [w * tau_sys for w in chaotic_omni_front_synthesis(values)]
    
    # Compute the allocations using the RBF weights
    allocations = [rfb_weights[i] * values[i] for i in range(len(values))]
    
    return allocations


def hybrid_perceptual_rbf(values: List[float]) -> List[float]:
    """
    Compute the hybrid perceptual RBF using the LTC-modulated RBF weights and the perceptual hash-based clustering.
    """
    # Compute the perceptual hash
    phash = compute_phash(values)
    
    # Compute the RBF weights using the perceptual hash
    rbf_weights = chaotic_omni_front_synthesis(values)
    
    # Compute the allocations using the RBF weights and the perceptual hash-based clustering
    allocations = []
    for cluster in cluster_by_phash({f"cluster_{i}": phash for i in range(len(values))}):
        cluster_values = [values[i] for i in cluster]
        cluster_rbf_weights = [rfb_weights[i] for i in cluster]
        cluster_allocations = [cluster_rbf_weights[i] * cluster_values[i] for i in range(len(cluster_values))]
        allocations.extend(cluster_allocations)
    
    return allocations


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    day_of_week = 0.5
    values = [1.0, 2.0, 3.0, 4.0]
    print(hybrid_allocate_by_dates(day_of_week, values))
    print(hybrid_perceptual_rbf(values))
    print(increment_hybrid_fractional_tree_cost(values))