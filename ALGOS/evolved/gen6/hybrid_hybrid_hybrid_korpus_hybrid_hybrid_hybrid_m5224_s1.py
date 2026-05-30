# DARWIN HAMMER — match 5224, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_korpus_text_h_hybrid_geometric_pro_m523_s1.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_regret_m1227_s4.py (gen5)
# born: 2026-05-30T00:00:53Z

"""Hybrid MinHash‑Voronoi‑Fisher‑SSIM Algorithm
================================================

Parents
-------
* **Algorithm A** – `hybrid_hybrid_korpus_text_h_hybrid_geometric_pro_m523_s1.py`
  (min‑hash generation for text → 2‑D point cloud → Voronoi partition).

* **Algorithm B** – `hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_regret_m1227_s4.py`
  (Gaussian beam → Fisher information score for adaptive weighting,
  and SSIM for similarity between signature vectors).

Mathematical Bridge
-------------------
The bridge is built on the observation that the **min‑hash signature** produced by
Algorithm A is a *finite integer vector* that can be interpreted as a set of
2‑D points.  Each Voronoi region therefore yields a **regional signature**
(e.g. the centroid of its points).  Algorithm B operates on real‑valued
vectors; we reuse its **Fisher score** as an *adaptive weight* that depends on
the distance of a region’s centroid from a moving centre, and we reuse its
**SSIM** function to measure similarity between a region’s signature and a
reference signature (e.g. from a previous iteration).  The final hybrid score
for a region is:


score_region = fisher_score(distance_to_center) * ssim(region_sig, ref_sig)


The region with the maximal hybrid score is selected as the “bandit action”.

The module below implements this unified system with three core functions:
`minhash_for_text`, `voronoi_partition`, and `hybrid_region_decision`.
"""

import re
import math
import random
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

# ----------------------------------------------------------------------
# Parent A – MinHash & Voronoi utilities
# ----------------------------------------------------------------------
def minhash_for_text(text: str, k: int = 64) -> list[int]:
    """Create a k‑length min‑hash signature from the input text."""
    text = re.sub(r"\s+", " ", text or "").strip().lower()
    shingles = [text[i:i + 5] for i in range(len(text) - 4)]
    signature = np.random.randint(0, 1_000_000, size=k)
    for s in shingles:
        idx = hash(s) % k
        signature[idx] = min(signature[idx], hash(s) % 1_000_000)
    return signature.tolist()


def _point_from_hash(value: int) -> tuple[float, float]:
    """Map an integer hash to a 2‑D point inside a 100×100 square."""
    x = value % 100
    y = (value // 100) % 100
    return float(x), float(y)


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Euclidean distance."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def nearest(point: tuple[float, float], seeds: list[tuple[float, float]]) -> int:
    """Index of the closest seed to *point* (ties broken by index)."""
    if not seeds:
        raise ValueError("seeds required")
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))


def assign(points: list[tuple[float, float]], seeds: list[tuple[float, float]]) -> dict[int, list[tuple[float, float]]]:
    """Voronoi assignment of *points* to the nearest *seeds*."""
    regions = defaultdict(list)
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return dict(regions)


def voronoi_partition(signature: list[int], num_seeds: int = 5) -> dict[int, list[tuple[float, float]]]:
    """Convert a min‑hash signature to a Voronoi partition."""
    points = [_point_from_hash(v) for v in signature]
    seeds = [(random.random() * 100, random.random() * 100) for _ in range(num_seeds)]
    return assign(points, seeds)


# ----------------------------------------------------------------------
# Parent B – Fisher score & SSIM utilities
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian beam intensity."""
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a Gaussian beam (used as an adaptive weight)."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def ssim(x: np.ndarray, y: np.ndarray, dynamic_range: float = 255.0,
         k1: float = 0.01, k2: float = 0.03) -> float:
    """Structural similarity index between two 1‑D signals."""
    if x.shape != y.shape:
        raise ValueError("samples must have equal shape")
    if x.size == 0:
        raise ValueError("samples must not be empty")
    mx, my = np.mean(x), np.mean(y)
    vx, vy = np.var(x), np.var(y)
    cov = np.mean((x - mx) * (y - my))
    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2
    numerator = (2 * mx * my + c1) * (2 * cov + c2)
    denominator = (mx * mx + my * my + c1) * (vx + vy + c2)
    return numerator / denominator


# ----------------------------------------------------------------------
# Hybrid Core Functions
# ----------------------------------------------------------------------
def region_signature(points: list[tuple[float, float]]) -> np.ndarray:
    """
    Produce a deterministic signature for a Voronoi region.
    The signature is the flattened (x, y) coordinates of the region's centroid
    followed by the region's point count, all scaled to [0, 255].
    """
    if not points:
        # Empty region – return a zero vector of length 3
        return np.zeros(3, dtype=np.float64)
    xs, ys = zip(*points)
    centroid = (sum(xs) / len(xs), sum(ys) / len(ys))
    count = len(points)
    # Scale to 0‑255 for SSIM compatibility
    vec = np.array([centroid[0], centroid[1], count], dtype=np.float64)
    vec_scaled = np.clip(vec / 100.0 * 255.0, 0, 255)
    return vec_scaled


def hybrid_region_score(region_pts: list[tuple[float, float]],
                        ref_sig: np.ndarray,
                        centre: float = 50.0,
                        width: float = 20.0) -> float:
    """
    Compute the hybrid score for a region:
        score = fisher_score(dist_to_centre) * ssim(region_sig, ref_sig)
    """
    # Region signature
    reg_sig = region_signature(region_pts)
    # Distance of region centroid from the moving centre (1‑D projection)
    if region_pts:
        xs, ys = zip(*region_pts)
        centroid = (sum(xs) / len(xs), sum(ys) / len(ys))
        theta = (centroid[0] + centroid[1]) / 2.0  # simple projection
    else:
        theta = centre  # neutral distance for empty region
    weight = fisher_score(theta, centre, width)
    similarity = ssim(reg_sig, ref_sig)
    return weight * similarity


def hybrid_region_decision(text: str,
                           reference_signature: np.ndarray,
                           k: int = 64,
                           num_seeds: int = 5,
                           centre: float = 50.0,
                           width: float = 20.0) -> int:
    """
    End‑to‑end hybrid operation:

    1. Generate a min‑hash signature from *text*.
    2. Partition the signature into Voronoi regions.
    3. Score each region with Fisher‑weighted SSIM against *reference_signature*.
    4. Return the index of the region with the highest hybrid score.
    """
    signature = minhash_for_text(text, k)
    regions = voronoi_partition(signature, num_seeds)

    best_region = -1
    best_score = -math.inf
    for idx, pts in regions.items():
        score = hybrid_region_score(pts, reference_signature, centre, width)
        if score > best_score:
            best_score = score
            best_region = idx
    return best_region


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_text = (
        "In the age of quantum computing, classical algorithms must adapt "
        "to new paradigms of information processing."
    )
    # Reference signature: a random but deterministic vector for reproducibility
    random.seed(0)
    ref_vec = np.array([random.random() * 255 for _ in range(3)], dtype=np.float64)

    chosen_region = hybrid_region_decision(
        text=sample_text,
        reference_signature=ref_vec,
        k=64,
        num_seeds=5,
        centre=50.0,
        width=20.0,
    )
    print(f"Chosen Voronoi region index: {chosen_region}")
    sys.exit(0)