# DARWIN HAMMER — match 4586, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_sketches_rlct_hybrid_hybrid_bayes__m98_s2.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_krampu_m1391_s0.py (gen6)
# born: 2026-05-29T23:56:50Z

import hashlib
import math
import random
import sys
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Utility hashing (deterministic)
# ----------------------------------------------------------------------
def _hash(item: str, seed: int) -> int:
    """Deterministic integer hash for a given seed."""
    h = hashlib.blake2b(digest_size=8)
    h.update(item.encode("utf-8"))
    h.update(seed.to_bytes(2, "little"))
    return int.from_bytes(h.digest(), "little")


# ----------------------------------------------------------------------
# Count‑Min Sketch with numerical stability
# ----------------------------------------------------------------------
def count_min_sketch(
    items: Iterable[str], width: int = 128, depth: int = 5
) -> np.ndarray:
    """
    Return a depth×width Count‑Min sketch as a NumPy array.
    Zero counts are left as‑is; later stages add a small epsilon
    before taking logarithms to avoid ``log(0)``.
    """
    sketch = np.zeros((depth, width), dtype=np.int64)
    for item in items:
        for i in range(depth):
            idx = _hash(item, i) % width
            sketch[i, idx] += 1
    return sketch


# ----------------------------------------------------------------------
# Bayesian update using a Gaussian prior on the (log) sketch counts
# ----------------------------------------------------------------------
def bayesian_sketch_update(
    sketch: np.ndarray,
    prior_mean: float,
    prior_var: float,
    epsilon: float = 1e-9,
) -> Tuple[float, float]:
    """
    Perform a conjugate Gaussian update where the likelihood is approximated
    by the (log) counts of the first hash row.  The log‑likelihood is
    ``log(count + epsilon)`` to keep the computation stable.
    """
    # Use only the first hash row as the representative statistic
    counts = sketch[0, :].astype(np.float64)
    log_likelihood = np.log(counts + epsilon).sum()

    # Gaussian conjugate update (scalar case)
    posterior_var = 1.0 / (1.0 / prior_var + 1.0)  # observation variance = 1
    posterior_mean = posterior_var * (prior_mean / prior_var + log_likelihood)

    return float(posterior_mean), float(posterior_var)


# ----------------------------------------------------------------------
# RLCT (Real Log‑Canonical Threshold) estimator
# ----------------------------------------------------------------------
def hybrid_rlct_estimate(
    posterior_mean: float,
    posterior_var: float,
    sketch: np.ndarray,
) -> float:
    """
    A more faithful RLCT asymptotic:
        RLCT ≈ (d/2) * log(N) - λ * log(log(N))

    where:
        d = effective dimension (taken as the number of sketch cells)
        N = total count observed (sum of first row)
        λ = posterior variance (acts as a proxy for the RLCT constant)
    """
    d = sketch.size  # total number of cells in the sketch
    N = sketch[0, :].sum() + 1e-9  # avoid log(0)
    lambda_const = max(posterior_var, 1e-9)

    rlct = (d / 2.0) * math.log(N) - lambda_const * math.log(math.log(N + 1.0))
    # Incorporate the posterior mean as a scaling factor to keep the estimate
    # sensitive to the observed data.
    return rlct * (1.0 + 0.01 * posterior_mean)


# ----------------------------------------------------------------------
# Graph construction and Ollivier‑Ricci curvature approximation
# ----------------------------------------------------------------------
def _euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    return float(np.linalg.norm(vec1 - vec2))


def _transport_cost(
    src: np.ndarray, dst: np.ndarray, weight: float
) -> float:
    """
    Simple 1‑Wasserstein (Earth Mover) distance between two Dirac masses
    placed at src and dst with equal mass.  The cost is weight * distance.
    """
    return weight * _euclidean_distance(src, dst)


def krampus_brain_map_curvature(
    morphology: Dict[str, float],
    workshare_lane: Dict[str, float],
) -> np.ndarray:
    """
    Build a tiny weighted graph from the morphology parameters and the
    workshare lane, then compute a pairwise Ollivier‑Ricci curvature matrix.
    The graph has four nodes (L, W, H, M) representing length, width,
    height, mass.  Edges are weighted by the product of the two node values.
    The curvature between two nodes i, j is approximated by:

        κ(i,j) = 1 - (W_ij / d_ij)

    where W_ij is the optimal transport cost between the unit masses at i
    and j (here simply the edge weight) and d_ij is the Euclidean distance
    in the feature space.
    """
    # Feature vector for each morphological attribute
    attrs = ["length", "width", "height", "mass"]
    vecs = {a: np.array([morphology[a]]) for a in attrs}

    # Edge weights based on interaction with workshare lane
    llm_units = workshare_lane.get("llm_units", 1.0)
    edge_weight = lambda a, b: (morphology[a] * morphology[b]) / llm_units

    n = len(attrs)
    curvature_mat = np.zeros((n, n), dtype=np.float64)

    for i, a in enumerate(attrs):
        for j, b in enumerate(attrs):
            if i == j:
                curvature_mat[i, j] = 1.0
                continue
            d_ij = _euclidean_distance(vecs[a], vecs[b]) + 1e-9
            w_ij = edge_weight(a, b) + 1e-9
            curvature_mat[i, j] = 1.0 - (w_ij / d_ij)

    # Normalise to keep values in a reasonable range
    curvature_mat = np.clip(curvature_mat, -1.0, 1.0)
    return curvature_mat


# ----------------------------------------------------------------------
# Workshare allocator with curvature‑aware soft allocation
# ----------------------------------------------------------------------
def workshare_allocator(
    output_projections: List[float],
    health_score: float,
    curvature_mat: np.ndarray,
) -> List[float]:
    """
    Allocate workshare resources by:
      1. Scaling projections by the health score.
      2. Modulating each scaled projection with a curvature‑derived weight.
         The weight for projection *i* is the average curvature of node *i*
         with all other nodes (row‑mean of the curvature matrix).
      3. Normalising the final allocation to sum to the health‑adjusted total.
    """
    if not output_projections:
        return []

    # Step 1: health weighting
    weighted = np.array(output_projections, dtype=np.float64) * health_score

    # Step 2: curvature weighting (row‑mean)
    curvature_weights = curvature_mat.mean(axis=1)
    # If the number of projections differs from curvature dimensions,
    # broadcast or truncate safely.
    if curvature_weights.size != weighted.size:
        # Simple broadcast by repeating or truncating
        repeats = int(np.ceil(weighted.size / curvature_weights.size))
        curvature_weights = np.tile(curvature_weights, repeats)[: weighted.size]

    allocated = weighted * curvature_weights

    # Step 3: normalisation
    total = allocated.sum()
    if total > 0:
        allocated = allocated / total * (health_score * sum(output_projections))
    else:
        allocated = np.zeros_like(allocated)

    return allocated.tolist()


# ----------------------------------------------------------------------
# Example usage (smoke test)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic items
    items = ["item1", "item2", "item3", "item1", "item2"]
    sketch = count_min_sketch(items, width=64, depth=4)

    # Bayesian update
    prior_mean = 0.0
    prior_var = 1.0
    post_mean, post_var = bayesian_sketch_update(sketch, prior_mean, prior_var)

    # RLCT estimate
    rlct = hybrid_rlct_estimate(post_mean, post_var, sketch)

    # Morphology & workshare lane
    morphology = {"length": 12.0, "width": 7.0, "height": 4.5, "mass": 25.0}
    workshare_lane = {"llm_units": 150.0}
    curvature_matrix = krampus_brain_map_curvature(morphology, workshare_lane)

    # Output projections (e.g., from an endpoint circuit)
    output_projections = [0.15, 0.35, 0.25, 0.10]
    health_score = 0.87

    allocation = workshare_allocator(output_projections, health_score, curvature_matrix)

    print("RLCT estimate :", rlct)
    print("Curvature matrix:\n", curvature_matrix)
    print("Allocation :", allocation)