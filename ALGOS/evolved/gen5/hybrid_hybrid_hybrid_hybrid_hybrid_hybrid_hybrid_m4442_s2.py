# DARWIN HAMMER — match 4442, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_model__hybrid_ssim_hybrid_d_m478_s0.py (gen3)
# parent_b: hybrid_hybrid_hybrid_fisher_hybrid_hybrid_vorono_m1361_s4.py (gen4)
# born: 2026-05-29T23:55:51Z

"""Hybrid Algorithm integrating:
- Parent A: Ollivier‑Ricci curvature weighted by Structural Similarity Index (SSIM)
- Parent B: Fisher‑score derived curvature, Voronoi partitioning of (SSIM, Fisher) feature space,
  and a sparse Winner‑Take‑All (WTA) selector.

Mathematical Bridge
-------------------
For each node *i* we first compute an SSIM‑based similarity *s_i* with its neighbours.
The SSIM value feeds a Fisher‑score *f_i = FisherScore(s_i; μ_f, σ_f)* producing a
curvature‑like weight.  The pair *(s_i, f_i)* is treated as a 2‑D point in a feature
space that is partitioned by a Voronoi diagram defined by a set of seed points
*{c_r}*.  Each Voronoi region *r* owns a model in a shared pool.  The Ollivier‑Ricci
curvature for node *i* is then modulated by the SSIM weight *s_i* (as in Parent A) and
by the Voronoi region membership (as in Parent B).  Finally a sparse WTA selector
chooses the region with the largest aggregate risk score
    R_r = Σ_{i∈region r} s_i·f_i,
driving model‑selection or resource‑management decisions.

The implementation below provides three core functions that demonstrate this
fusion:
1. `hybrid_curvature` – computes curvature μ_i(v) using SSIM‑weighted Ollivier‑Ricci.
2. `voronoi_partition` – assigns each node to a Voronoi region in (s, f) space.
3. `wta_selector` – selects the region(s) with highest reconstruction‑risk score.

All other helper utilities are kept minimal and self‑contained.
"""

import os
import sys
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple, Iterable

import numpy as np

# ----------------------------------------------------------------------
# Helper: Structural Similarity Index (SSIM) – NumPy version (Parent B)
# ----------------------------------------------------------------------
def ssim(x: np.ndarray, y: np.ndarray,
         dynamic_range: float = 255.0,
         k1: float = 0.01,
         k2: float = 0.03) -> float:
    """Return SSIM between two equally‑shaped 1‑D arrays."""
    if x.shape != y.shape:
        raise ValueError('samples must have equal shape')
    if x.size == 0:
        raise ValueError('samples must not be empty')
    mx = np.mean(x)
    my = np.mean(y)
    vx = np.var(x)
    vy = np.var(y)
    cov = np.mean((x - mx) * (y - my))
    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2
    numerator = (2 * mx * my + c1) * (2 * cov + c2)
    denominator = (mx * mx + my * my + c1) * (vx + vy + c2)
    return numerator / denominator

# ----------------------------------------------------------------------
# Gaussian beam and Fisher score (Parent B)
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian kernel value."""
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information score derived from a Gaussian beam."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

# ----------------------------------------------------------------------
# Ollivier‑Ricci curvature weighted by SSIM (Parent A)
# ----------------------------------------------------------------------
def hybrid_curvature(
    graph: Dict[int, List[int]],
    node_features: Dict[int, np.ndarray],
    alpha: float,
    vram_weights: Dict[int, float],
    ssims: Dict[Tuple[int, int], float] = None,
) -> Dict[int, float]:
    """
    Compute μ_i(v) for every node i using the formula:

        μ_i(v) = α·w_i·δ_{i=v}
               + (1-α)·w_i·(1/deg(i))· Σ_{u∈N(i)} δ_{u=v}·SSIM(i,u)

    If `ssims` is not supplied it is computed on‑the‑fly.
    Returns a dictionary mapping node i → curvature value μ_i(i) (self‑mass).
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError('alpha must be in [0,1]')

    # Pre‑compute SSIM for all edges if not given
    if ssims is None:
        ssims = {}
        for i, nbrs in graph.items():
            for u in nbrs:
                if (i, u) not in ssims and (u, i) not in ssims:
                    ssims[(i, u)] = ssim(node_features[i], node_features[u])

    curvature: Dict[int, float] = {}
    for i, nbrs in graph.items():
        w_i = vram_weights.get(i, 1.0)
        deg = len(nbrs) if nbrs else 1  # avoid division by zero
        self_term = alpha * w_i
        neighbor_term = 0.0
        for u in nbrs:
            s = ssims.get((i, u), ssims.get((u, i), 0.0))
            neighbor_term += s
        neighbor_term = (1 - alpha) * w_i * (neighbor_term / deg)
        curvature[i] = self_term + neighbor_term
    return curvature

# ----------------------------------------------------------------------
# Voronoi partition in (SSIM, Fisher) feature space (Parent B)
# ----------------------------------------------------------------------
def voronoi_partition(
    points: List[Tuple[float, float]],
    seeds: List[Tuple[float, float]],
) -> List[int]:
    """
    Assign each point to the index of the nearest seed (Euclidean distance).
    Returns a list of region indices parallel to `points`.
    """
    if not seeds:
        raise ValueError('seed list must not be empty')
    assignments = []
    for p in points:
        dists = [(p[0] - s[0]) ** 2 + (p[1] - s[1]) ** 2 for s in seeds]
        assignments.append(int(np.argmin(dists)))
    return assignments

# ----------------------------------------------------------------------
# Sparse Winner‑Take‑All selector (Parent B)
# ----------------------------------------------------------------------
def wta_selector(
    region_assignments: List[int],
    scores: List[float],
    top_k: int = 1,
) -> List[int]:
    """
    Compute aggregate score per region as Σ scores_i for nodes i belonging to
    that region. Return the `top_k` region indices with highest aggregate score.
    """
    agg: Dict[int, float] = {}
    for reg, sc in zip(region_assignments, scores):
        agg[reg] = agg.get(reg, 0.0) + sc
    # Sort regions by descending aggregate score
    sorted_regs = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)
    return [reg for reg, _ in sorted_regs[:top_k]]

# ----------------------------------------------------------------------
# Full hybrid operation combining all pieces
# ----------------------------------------------------------------------
def hybrid_step(
    graph: Dict[int, List[int]],
    node_features: Dict[int, np.ndarray],
    vram_weights: Dict[int, float],
    alpha: float,
    fisher_center: float,
    fisher_width: float,
    voronoi_seeds: List[Tuple[float, float]],
    top_k_regions: int = 1,
) -> Tuple[Dict[int, float], List[int], List[int]]:
    """
    Execute one hybrid iteration:
    1. Compute SSIM‑weighted Ollivier‑Ricci curvature (μ_i).
    2. Derive per‑node SSIM statistic s_i as the average SSIM with neighbours.
    3. Compute Fisher score f_i = FisherScore(s_i; fisher_center, fisher_width).
    4. Form 2‑D points (s_i, f_i) and partition them via Voronoi.
    5. Apply WTA on region risk scores r_r = Σ s_i·f_i.
    Returns:
        curvature dict,
        region assignment list (per node),
        winning region indices list.
    """
    # 1. Curvature (also gives us SSIM values for neighbours)
    curvature = hybrid_curvature(graph, node_features, alpha, vram_weights)

    # 2. Compute per‑node average SSIM with neighbours
    avg_ssim: Dict[int, float] = {}
    for i, nbrs in graph.items():
        if not nbrs:
            avg_ssim[i] = 0.0
            continue
        sims = [ssim(node_features[i], node_features[u]) for u in nbrs]
        avg_ssim[i] = sum(sims) / len(sims)

    # 3. Fisher scores
    fisher_vals = {
        i: fisher_score(avg_ssim[i], fisher_center, fisher_width)
        for i in avg_ssim
    }

    # 4. Voronoi partition
    points = [(avg_ssim[i], fisher_vals[i]) for i in sorted(avg_ssim.keys())]
    region_assignments = voronoi_partition(points, voronoi_seeds)

    # 5. WTA selector based on risk score s_i * f_i
    risk_scores = [avg_ssim[i] * fisher_vals[i] for i in sorted(avg_ssim.keys())]
    winning_regions = wta_selector(region_assignments, risk_scores, top_k=top_k_regions)

    return curvature, region_assignments, winning_regions

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple synthetic graph: 5 nodes in a line
    graph = {
        0: [1],
        1: [0, 2],
        2: [1, 3],
        3: [2, 4],
        4: [3],
    }

    # Random 8‑pixel patches per node (values 0‑255)
    rng = np.random.default_rng(42)
    node_features = {i: rng.integers(0, 256, size=8).astype(float) for i in graph}

    # Mock VRAM weights (normalized)
    vram_weights = {i: 1.0 for i in graph}

    # Hyper‑parameters
    alpha = 0.6
    fisher_center = 0.5   # typical SSIM range centre
    fisher_width = 0.2
    # Voronoi seeds placed in (s, f) space
    voronoi_seeds = [
        (0.2, 0.1),
        (0.5, 0.3),
        (0.8, 0.6),
    ]

    curvature, regions, winners = hybrid_step(
        graph,
        node_features,
        vram_weights,
        alpha,
        fisher_center,
        fisher_width,
        voronoi_seeds,
        top_k_regions=2,
    )

    print("Curvature per node:", curvature)
    print("Voronoi region assignment:", regions)
    print("Winning region(s):", winners)