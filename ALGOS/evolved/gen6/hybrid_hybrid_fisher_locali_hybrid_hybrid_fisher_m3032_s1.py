# DARWIN HAMMER — match 3032, survivor 1
# gen: 6
# parent_a: hybrid_fisher_localization_krampus_chrono_m17_s0.py (gen1)
# parent_b: hybrid_hybrid_fisher_locali_hybrid_hybrid_vorono_m1224_s0.py (gen5)
# born: 2026-05-29T23:47:21Z

"""
Hybrid Algorithm: Fisher‑Chrono‑Voronoi (FCV)

Parents:
- hybrid_fisher_localization_krampus_chrono_m17_s0.py (temporal Fisher scoring & Gaussian smoothing)
- hybrid_hybrid_fisher_locali_hybrid_hybrid_vorono_m1224_s0.py (Fisher‑optimized Gaussian beams + Voronoi partitioning + multivector algebra)

Mathematical Bridge:
Both parents rely on the Gaussian beam g(θ)=exp[-½((θ−c)/w)²] and its Fisher‑information score
I(θ)= (∂g/∂θ)² / g.
We treat each timestamp (seconds since epoch) as a scalar θ and embed it into ℝ² by adding an auxiliary
dimension (e.g. a random “feature”). Voronoi regions are built over these 2‑D points.
Inside each region we aggregate the Fisher scores, producing a scalar weight per region.
Finally the region‑weights are lifted into a geometric algebra multivector where each region
corresponds to a basis blade; the multivector encodes the full hybrid state.
"""

import math
import random
import sys
import pathlib
from datetime import datetime
from typing import List, Tuple, Dict, Set, FrozenSet

import numpy as np

# ----------------------------------------------------------------------
# Core Types
# ----------------------------------------------------------------------
Point = Tuple[float, float]                 # (time, auxiliary)
Blade = FrozenSet[int]                      # basis blade represented by a set of indices
Multivector = Dict[Blade, float]            # mapping blade → coefficient

# ----------------------------------------------------------------------
# Shared Gaussian / Fisher utilities (from both parents)
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian beam g(θ) = exp(-½ ((θ‑c)/w)²)."""
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher‑information score I(θ) = (∂g/∂θ)² / g."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

# ----------------------------------------------------------------------
# Chronological candidate generation (Parent A)
# ----------------------------------------------------------------------
def parse_loose_datetime(raw: str) -> datetime | None:
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None

def chrono_candidates_for_path(path: pathlib.Path, text_sample: str = "") -> List[Dict[str, str]]:
    """Generate a dense list of ISO‑date strings (1900‑2099) as candidate timestamps."""
    candidates: List[Dict[str, str]] = []
    for year in range(1900, 2100):
        for month in range(1, 13):
            for day in range(1, 32):
                raw = f"{year}-{month:02d}-{day:02d}"
                parsed = parse_loose_datetime(raw)
                if parsed:
                    candidates.append({
                        "timestamp": raw,
                        "source": "path",
                        "raw": raw,
                    })
    return candidates

# ----------------------------------------------------------------------
# Voronoi utilities (Parent B)
# ----------------------------------------------------------------------
def euclidean_distance(a: Point, b: Point) -> float:
    """Standard Euclidean distance in ℝ²."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def compute_voronoi_regions(points: List[Point], sites: List[Point]) -> Dict[int, List[Point]]:
    """
    Assign each point to the index of the nearest site.
    Returns a dict ``site_index → list[points]``.
    """
    regions: Dict[int, List[Point]] = {i: [] for i in range(len(sites))}
    for pt in points:
        distances = [euclidean_distance(pt, s) for s in sites]
        nearest = int(np.argmin(distances))
        regions[nearest].append(pt)
    return regions

# ----------------------------------------------------------------------
# Hybrid Functions
# ----------------------------------------------------------------------
def generate_temporal_points(
    candidates: List[Dict[str, str]],
    center: float,
    width: float,
    aux_range: Tuple[float, float] = (0.0, 1.0),
    max_points: int = 500
) -> List[Point]:
    """
    Convert timestamp candidates into 2‑D points.
    - x coordinate: epoch seconds (float)
    - y coordinate: random auxiliary value in ``aux_range``
    The function samples up to ``max_points`` candidates, weighted by their Fisher scores,
    to favour timestamps that sit near the Gaussian centre.
    """
    # Compute Fisher scores for all candidates
    scored = []
    for cand in candidates:
        ts = datetime.fromisoformat(cand["timestamp"]).timestamp()
        score = fisher_score(ts, center, width)
        scored.append((ts, score))

    # Normalise scores to a probability distribution
    scores = np.array([s for _, s in scored], dtype=float)
    if scores.sum() == 0:
        probs = np.full_like(scores, 1.0 / len(scores))
    else:
        probs = scores / scores.sum()

    # Sample indices according to probabilities
    chosen_idx = np.random.choice(len(scored), size=min(max_points, len(scored)), replace=False, p=probs)
    points: List[Point] = []
    for idx in chosen_idx:
        ts, _ = scored[idx]
        aux = random.uniform(*aux_range)
        points.append((ts, aux))
    return points

def voronoi_fisher_aggregate(
    points: List[Point],
    sites: List[Point],
    center: float,
    width: float
) -> Dict[int, float]:
    """
    1. Build Voronoi regions for ``points`` using ``sites``.
    2. Inside each region compute the sum of Fisher scores of the *time* coordinate.
    Returns a mapping ``site_index → aggregated Fisher score``.
    """
    regions = compute_voronoi_regions(points, sites)
    region_scores: Dict[int, float] = {}
    for idx, pts in regions.items():
        agg = 0.0
        for (t, _) in pts:
            agg += fisher_score(t, center, width)
        region_scores[idx] = agg
    return region_scores

def multivector_from_region_scores(region_scores: Dict[int, float]) -> Multivector:
    """
    Lift scalar region scores into a geometric algebra multivector.
    Each region index i becomes a basis vector e_i; the scalar weight w_i becomes the
    coefficient of the 1‑blade {i}. For demonstration we also form 2‑blades by wedging
    every pair of distinct regions (product of their weights).
    """
    mv: Multivector = {}
    # 1‑blades
    for i, w in region_scores.items():
        mv[frozenset({i})] = w

    # 2‑blades (simple outer product)
    indices = list(region_scores.keys())
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            blade = frozenset({indices[i], indices[j]})
            weight = region_scores[indices[i]] * region_scores[indices[j]]
            mv[blade] = weight
    return mv

def hybrid_fcv_pipeline(
    path: pathlib.Path,
    center: float,
    width: float,
    n_sites: int = 5,
    max_points: int = 300
) -> Multivector:
    """
    End‑to‑end pipeline:
    1. Generate chronological candidates from ``path`` (Parent A).
    2. Sample and embed them as 2‑D points.
    3. Randomly initialise Voronoi sites.
    4. Aggregate Fisher scores per Voronoi region.
    5. Return a multivector representing the hybrid state.
    """
    candidates = chrono_candidates_for_path(path)
    points = generate_temporal_points(candidates, center, width, max_points=max_points)

    # Randomly pick site positions inside the convex hull of points
    if not points:
        raise RuntimeError("No points generated for Voronoi partitioning.")
    xs, ys = zip(*points)
    site_xs = np.random.uniform(min(xs), max(xs), size=n_sites)
    site_ys = np.random.uniform(min(ys), max(ys), size=n_sites)
    sites = list(zip(site_xs, site_ys))

    region_scores = voronoi_fisher_aggregate(points, sites, center, width)
    mv = multivector_from_region_scores(region_scores)
    return mv

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Use a temporary path (the actual filesystem is irrelevant for candidate generation)
    dummy_path = pathlib.Path("dummy_folder")
    # Parameters for the Gaussian beam
    gaussian_center = datetime(2000, 1, 1).timestamp()   # centre around Y2K
    gaussian_width = 5 * 365 * 24 * 3600.0               # ~5 years in seconds

    try:
        result_mv = hybrid_fcv_pipeline(
            path=dummy_path,
            center=gaussian_center,
            width=gaussian_width,
            n_sites=4,
            max_points=200
        )
        print("Hybrid multivector (blade → coefficient):")
        for blade, coeff in sorted(result_mv.items(), key=lambda x: (len(x[0]), sorted(x[0]))):
            print(f"  {sorted(blade)} : {coeff:.6e}")
    except Exception as e:
        print(f"Error during smoke test: {e}", file=sys.stderr)
        sys.exit(1)