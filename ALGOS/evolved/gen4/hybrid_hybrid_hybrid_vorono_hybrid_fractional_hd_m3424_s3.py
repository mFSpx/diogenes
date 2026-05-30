# DARWIN HAMMER — match 3424, survivor 3
# gen: 4
# parent_a: hybrid_hybrid_voronoi_parti_hybrid_sparse_wta_hy_m228_s1.py (gen3)
# parent_b: hybrid_fractional_hdc_counterfactual_effec_m38_s1.py (gen1)
# born: 2026-05-29T23:50:01Z

"""
Hybrid Voronoi‑HDC Causal Model
================================

This module fuses the two parent algorithms:

* **Parent A** – Voronoi partitioning combined with sparse winner‑take‑all (WTA) tags
  used for privacy‑model pool management.
* **Parent B** – Fractional‑power hyperdimensional computing (HDC) together with
  lightweight causal / counterfactual effect estimation.

**Mathematical bridge**

1. The point cloud is first split into Voronoi cells `R_i` by a set of seed points
   `S = {s_i}` (Parent A).
2. Each cell is encoded as a high‑dimensional hypervector `h_i` by binding a
   *seed hypervector* `v(s_i)` with a *centroid hypervector* `c(R_i)`.  
   The binding operator is element‑wise multiplication (complex phase multiplication).
3. The strength of a cell (its cardinality `|R_i|`) is injected by a fractional‑power
   operation `h_i ← h_i ** (|R_i| / N)` where `N` is the total number of points.
   This yields a *weighted* hypervector that respects both geometry (region size)
   and semantics (binding of seed ↔ region).
4. A sparse WTA layer selects the `k` most salient hypervectors (largest L2 norm),
   mirroring the sparse tag mechanism of Parent A.
5. The selected hypervectors can be fed to the causal‑effect estimator of Parent B,
   allowing causal reasoning directly on spatially‑aware representations.

The resulting pipeline is a single, mathematically coherent system that maps
geometric partitions to hyperdimensional symbols, weights them by region importance,
and performs sparse selection before causal inference.
"""

import math
import random
import sys
import pathlib
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Voronoi utilities
# ----------------------------------------------------------------------
Point = Tuple[float, float]

def distance(a: Point, b: Point) -> float:
    """Euclidean distance between two 2‑D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def nearest(point: Point, seeds: List[Point]) -> int:
    """Index of the nearest seed to *point* (ties broken by index)."""
    if not seeds:
        raise ValueError("seed list is empty")
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))

def voronoi_assign(points: List[Point], seeds: List[Point]) -> Dict[int, List[Point]]:
    """
    Assign every point to the index of its nearest seed, returning a mapping
    ``seed_index -> list[points]``.
    """
    regions: Dict[int, List[Point]] = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions

# ----------------------------------------------------------------------
# Parent B – Hyperdimensional primitives
# ----------------------------------------------------------------------
def random_hv(d: int = 10000, kind: str = "complex", seed: int | None = None) -> np.ndarray:
    """Generate a random hypervector."""
    rng = np.random.default_rng(seed)
    if kind == "complex":
        theta = rng.uniform(0.0, 2.0 * np.pi, size=d)
        return np.exp(1j * theta)  # unit‑magnitude complex vector
    elif kind == "bipolar":
        return rng.choice([-1, 1], size=d).astype(np.float64)
    elif kind == "real":
        vec = rng.normal(size=d)
        return vec / np.linalg.norm(vec)
    else:
        raise ValueError(f"unknown kind {kind!r}")

def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Phase‑preserving binding (element‑wise multiplication)."""
    return a * b

def fractional_power(v: np.ndarray, alpha: float) -> np.ndarray:
    """
    Raise a complex hypervector to a fractional power ``alpha``.
    For real‑valued vectors we fall back to element‑wise exponentiation.
    """
    if np.iscomplexobj(v):
        # v = exp(i*phi) => v**alpha = exp(i*alpha*phi)
        phi = np.angle(v)
        return np.exp(1j * alpha * phi)
    else:
        # Preserve sign for bipolar vectors
        return np.sign(v) * (np.abs(v) ** alpha)

def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity (real part for complex vectors)."""
    dot = np.vdot(a, b)  # conjugate dot product
    return float(np.real(dot) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def cleanup(v: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    """Zero out components with magnitude below *threshold*."""
    mask = np.abs(v) >= threshold
    return v * mask

# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------
def centroid(points: List[Point]) -> Point:
    """Arithmetic mean of a list of 2‑D points (returns (0,0) for empty list)."""
    if not points:
        return (0.0, 0.0)
    xs, ys = zip(*points)
    return (sum(xs) / len(points), sum(ys) / len(points))

def point_to_hv(p: Point, dim: int = 10000, seed_offset: int = 0) -> np.ndarray:
    """
    Deterministically map a 2‑D point to a complex hypervector by seeding the RNG
    with a hash of the coordinates.
    """
    # Simple deterministic hash → integer seed
    h = hash((round(p[0], 6), round(p[1], 6), seed_offset)) & 0xFFFFFFFF
    return random_hv(d=dim, kind="complex", seed=int(h))

def encode_region_hv(region_points: List[Point],
                     seed_point: Point,
                     dim: int = 10000,
                     total_points: int = 1) -> np.ndarray:
    """
    Encode a Voronoi region as a hypervector.

    1. Generate a seed hypervector from the seed point.
    2. Generate a centroid hypervector from the region centroid.
    3. Bind them (phase multiplication).
    4. Apply fractional power proportional to region cardinality / total_points.
    """
    seed_hv = point_to_hv(seed_point, dim=dim, seed_offset=0)
    cent = centroid(region_points)
    cent_hv = point_to_hv(cent, dim=dim, seed_offset=1)

    bound = bind(seed_hv, cent_hv)
    weight = len(region_points) / max(total_points, 1)
    weighted = fractional_power(bound, weight)
    return weighted

def sparse_wta(hv_dict: Dict[int, np.ndarray], k: int = 3) -> Dict[int, np.ndarray]:
    """
    Sparse winner‑take‑all: keep the *k* hypervectors with largest L2 norm,
    zero‑out the rest.
    """
    if k <= 0:
        return {}
    norms = {i: np.linalg.norm(v) for i, v in hv_dict.items()}
    top_indices = sorted(norms, key=norms.get, reverse=True)[:k]
    return {i: hv_dict[i] for i in top_indices}

def estimate_causal_effect(query_hv: np.ndarray,
                           pool: List[np.ndarray]) -> float:
    """
    Very lightweight causal‑effect estimator: similarity of *query_hv* to the
    most similar hypervector in *pool*.  In a full system this would be replaced
    by the sophisticated counterfactual machinery of Parent B.
    """
    if not pool:
        return 0.0
    sims = [similarity(query_hv, hv) for hv in pool]
    return max(sims)

# ----------------------------------------------------------------------
# End‑to‑end hybrid pipeline
# ----------------------------------------------------------------------
def hybrid_voronoi_hdc(points: List[Point],
                       seed_count: int = 5,
                       dim: int = 10000,
                       wta_k: int = 3) -> Tuple[Dict[int, np.ndarray], float]:
    """
    Full hybrid workflow:

    1. Randomly generate ``seed_count`` seed points inside the bounding box of *points*.
    2. Partition *points* into Voronoi cells.
    3. Encode each cell as a hypervector (binding + fractional power).
    4. Apply sparse WTA to obtain the most salient region hypervectors.
    5. Pick a random query point, encode it, and return a causal‑effect estimate
       with respect to the selected hypervectors.

    Returns
    -------
    selected_hvs : dict[int, np.ndarray]
        The hypervectors that survived the WTA stage.
    effect_estimate : float
        Similarity‑based causal‑effect estimate for a random query.
    """
    # 1. Seed generation (uniform in the data bounding box)
    xs, ys = zip(*points)
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    seeds = [(random.uniform(min_x, max_x), random.uniform(min_y, max_y))
             for _ in range(seed_count)]

    # 2. Voronoi partition
    regions = voronoi_assign(points, seeds)

    # 3. Encode each region
    total_pts = len(points)
    region_hvs: Dict[int, np.ndarray] = {}
    for idx, pts in regions.items():
        region_hvs[idx] = encode_region_hv(pts, seeds[idx],
                                           dim=dim,
                                           total_points=total_pts)

    # 4. Sparse WTA
    selected_hvs = sparse_wta(region_hvs, k=wta_k)

    # 5. Query & causal estimate
    query_pt = (random.uniform(min_x, max_x), random.uniform(min_y, max_y))
    query_hv = point_to_hv(query_pt, dim=dim, seed_offset=2)
    effect = estimate_causal_effect(query_hv, list(selected_hvs.values()))
    return selected_hvs, effect

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Generate a synthetic point cloud (e.g., 500 points in a unit square)
    N = 500
    pts = [(random.random(), random.random()) for _ in range(N)]

    selected, est = hybrid_voronoi_hdc(pts,
                                       seed_count=7,
                                       dim=4096,
                                       wta_k=4)

    print(f"Selected {len(selected)} region hypervectors after WTA.")
    for idx, hv in selected.items():
        print(f"  Region {idx}: norm={np.linalg.norm(hv):.3f}")

    print(f"Causal‑effect estimate for a random query: {est:.4f}")