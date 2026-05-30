# DARWIN HAMMER — match 2527, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1088_s0.py (gen4)
# parent_b: hybrid_krampus_brainmap_ollivier_ricci_curva_m13_s1.py (gen1)
# born: 2026-05-29T23:42:42Z

"""
Hybrid Voronoi‑Curvature‑Feature Algorithm
========================================

This module fuses the two parent algorithms:

* **Parent A** – sinusoidal weekday weighting combined with a Voronoi
  (geometric product) partition of a point cloud.
* **Parent B** – randomised feature extraction from free‑form text (the
  “master vector”) and a notion of curvature between abstract dimensions.

**Mathematical bridge**

1. The Voronoi regions produced by *Parent A* give a probability
   distribution over the seed indices (region size ÷ total points).
2. The sinusoidal weekday vector (also from *Parent A*) provides a
   deterministic weighting for each seed.
3. The master feature vector (from *Parent B*) is projected onto the
   seed space by matching the first *k* features to *k* seeds and
   normalising – this yields a second probability distribution.
4. The two distributions are fused element‑wise (Hadamard product) to
   obtain a **combined weight vector**.
5. From the combined weights a symmetric curvature‑like matrix  

   \(C_{ij}= \dfrac{w_i w_j}{1+d_{ij}}\)  

   where \(d_{ij}\) is the Euclidean distance between seeds *i* and *j*,
   is constructed.  This matrix plays the role of the Ollivier‑Ricci
   curvature operator of *Parent B*.
6. Finally each point in a Voronoi region is linearly transformed by the
   2 × 2 principal sub‑matrix of *C* (the first two seeds) – a concrete
   realisation of “weighting the matrix operations” from *Parent A*.

The three public functions below expose the core of this hybrid pipeline.
"""

import numpy as np
import math
import random
import sys
import pathlib
from datetime import datetime
from typing import List, Tuple, Dict

# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
Point = Tuple[float, float]

# ----------------------------------------------------------------------
# Parent‑A building blocks
# ----------------------------------------------------------------------
def distance(a: Point, b: Point) -> float:
    """Euclidean distance between two 2‑D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def nearest(point: Point, seeds: List[Point]) -> int:
    """Index of the closest seed to *point* (ties broken by index)."""
    if not seeds:
        raise ValueError('seeds required')
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))

def assign(points: List[Point], seeds: List[Point]) -> Dict[int, List[Point]]:
    """Assign each point to the nearest seed, returning a region dict."""
    regions: Dict[int, List[Point]] = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions

def weekday_weight_vector(groups: List[str], dow: int) -> np.ndarray:
    """
    Row‑stochastic sinusoidal weight vector for *groups* on weekday *dow*.
    *dow* follows Python's Monday=0 … Sunday=6 convention.
    """
    n = len(groups)
    if n == 0:
        raise ValueError("groups must contain at least one element")
    base_angles = np.arange(n) * (2.0 * math.pi) / n
    phase = (2.0 * math.pi) * (dow / 7.0)
    weight_vec = 1.0 + 0.5 * np.sin(base_angles + phase)
    return weight_vec / weight_vec.sum()

# ----------------------------------------------------------------------
# Parent‑B building blocks
# ----------------------------------------------------------------------
def extract_full_features(text: str) -> Dict[str, float]:
    """Generate a dense random feature dictionary from *text*."""
    # In a real system the text would be parsed; here we use random numbers.
    features: Dict[str, float] = {}
    features.update({
        "operator_visceral_ratio": random.random(),
        "operator_tech_ratio": random.random(),
        "operator_legal_osint_ratio": random.random(),
        "psyche_forensic_shield_ratio": random.random(),
        "psyche_poetic_entropy": random.random(),
        "psyche_dissociative_index": random.random(),
        "resilience_bureaucratic_weaponization_index": random.random(),
        "resilience_resource_exhaustion_metric": random.random(),
        "resilience_swarm_orchestration_density": random.random(),
        "rainmaker_corporate_grit_tension": random.random(),
        "rainmaker_countdown_density": random.random(),
        "rainmaker_asset_structuring_weight": random.random(),
        "telemetry_agent_symmetry_ratio": random.random(),
        "telemetry_protocol_discipline": random.random(),
        "telemetry_manic_velocity": random.random(),
    })
    return features

def extract_master_vector(text: str) -> Dict[str, float]:
    """Condense the full feature set to a smaller master vector."""
    f = extract_full_features(text)
    return {
        "visceral_ratio": f["operator_visceral_ratio"],
        "tech_ratio": f["operator_tech_ratio"],
        "legal_osint_ratio": f["operator_legal_osint_ratio"],
        "forensic_shield_ratio": f["psyche_forensic_shield_ratio"],
        "poetic_entropy": f["psyche_poetic_entropy"],
        "dissociative_index": f["psyche_dissociative_index"],
        "bureaucratic_weaponization_index": f["resilience_bureaucratic_weaponization_index"],
        "resource_exhaustion_metric": f["resilience_resource_exhaustion_metric"],
        "swarm_orchestration_density": f["resilience_swarm_orchestration_density"],
        "corporate_grit_tension": f["rainmaker_corporate_grit_tension"],
        "countdown_density": f["rainmaker_countdown_density"],
        "asset_structuring_weight": f["rainmaker_asset_structuring_weight"],
    }

# ----------------------------------------------------------------------
# Hybrid core
# ----------------------------------------------------------------------
def combined_seed_weights(seeds: List[Point], dow: int, text: str) -> np.ndarray:
    """
    Produce a unified weight vector for the seeds.

    Steps:
    1. Sinusoidal weekday vector (size = |seeds|).
    2. Feature‑derived probability (first |seeds| entries of master vector,
       normalised).
    3. Hadamard product of the two vectors, renormalised to sum to 1.
    """
    n = len(seeds)
    groups = [f"seed_{i}" for i in range(n)]
    weekday_vec = weekday_weight_vector(groups, dow)                     # (n,)

    # Project master vector onto seed space
    master = extract_master_vector(text)
    master_vals = np.array(list(master.values()))[:n]                     # truncate / pad
    if master_vals.size < n:
        # Pad with a tiny epsilon to keep stochasticity
        pad = np.full(n - master_vals.size, 1e-6)
        master_vals = np.concatenate([master_vals, pad])
    feature_vec = master_vals / master_vals.sum()                         # (n,)

    combined = weekday_vec * feature_vec
    if combined.sum() == 0:
        # fallback to uniform distribution
        combined = np.full(n, 1.0 / n)
    else:
        combined = combined / combined.sum()
    return combined

def curvature_matrix(seeds: List[Point], weights: np.ndarray) -> np.ndarray:
    """
    Construct a symmetric curvature‑like matrix C where

        C_ij = w_i * w_j / (1 + d_ij)

    with d_ij the Euclidean distance between seeds i and j.
    The matrix is positive‑definite for positive weights.
    """
    n = len(seeds)
    C = np.empty((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            dij = distance(seeds[i], seeds[j])
            C[i, j] = weights[i] * weights[j] / (1.0 + dij)
    return C

def hybrid_transform(
    seeds: List[Point],
    points: List[Point],
    dow: int,
    text: str
) -> Dict[int, List[Point]]:
    """
    Full hybrid pipeline:

    1. Voronoi partition of *points* by *seeds*.
    2. Compute combined seed weights (weekday + feature).
    3. Build curvature matrix C.
    4. Use the 2×2 principal sub‑matrix of C as a linear transform
       applied to every point inside its region.
    5. Return a dict mapping seed index → list of transformed points.
    """
    # 1. Voronoi regions
    regions = assign(points, seeds)

    # 2. Combined weights
    w = combined_seed_weights(seeds, dow, text)          # (n,)

    # 3. Curvature matrix
    C = curvature_matrix(seeds, w)                       # (n, n)

    # 4. Extract a 2×2 transformation matrix.
    #    If fewer than 2 seeds exist we fall back to the identity.
    if C.shape[0] >= 2:
        M = C[:2, :2]
    else:
        M = np.eye(2)

    # Ensure M is a proper 2×2 matrix for multiplication.
    if M.shape != (2, 2):
        M = np.eye(2)

    # 5. Apply transformation region‑wise.
    transformed: Dict[int, List[Point]] = {}
    for idx, pts in regions.items():
        transformed_pts: List[Point] = []
        for (x, y) in pts:
            vec = np.array([x, y])
            new_vec = M @ vec
            transformed_pts.append((float(new_vec[0]), float(new_vec[1])))
        transformed[idx] = transformed_pts
    return transformed

# ----------------------------------------------------------------------
# Additional helper demonstrating the hybrid operation in isolation
# ----------------------------------------------------------------------
def region_statistics(regions: Dict[int, List[Point]]) -> Dict[int, Tuple[int, float]]:
    """
    Return simple statistics for each Voronoi region:
    (point count, average distance from region centroid).
    """
    stats: Dict[int, Tuple[int, float]] = {}
    for idx, pts in regions.items():
        count = len(pts)
        if count == 0:
            stats[idx] = (0, 0.0)
            continue
        xs, ys = zip(*pts)
        centroid = (sum(xs) / count, sum(ys) / count)
        avg_dist = sum(distance(p, centroid) for p in pts) / count
        stats[idx] = (count, avg_dist)
    return stats

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # deterministic seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Example seeds and points
    seeds = [(0.0, 0.0), (10.0, 0.0), (5.0, 8.0)]
    points = [(random.uniform(-5, 15), random.uniform(-5, 15)) for _ in range(200)]

    # Current weekday (Monday=0)
    dow = datetime.utcnow().weekday()

    # Sample text – content is irrelevant for the random feature extractor
    sample_text = "The quick brown fox jumps over the lazy dog."

    transformed_regions = hybrid_transform(seeds, points, dow, sample_text)

    # Print a concise summary
    stats = region_statistics(transformed_regions)
    for idx, (cnt, avg_dist) in stats.items():
        print(f"Region {idx}: {cnt} points, avg distance from centroid = {avg_dist:.3f}")

    # Verify that the transformation produced finite numbers
    flat = [coord for pts in transformed_regions.values() for pt in pts for coord in pt]
    assert all(math.isfinite(v) for v in flat), "Non‑finite values detected"
    print("Hybrid transformation completed successfully.")