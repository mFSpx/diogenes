# DARWIN HAMMER — match 4614, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_geomet_hybrid_hybrid_hybrid_m784_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_fisher_hybrid_hybrid_ternar_m1215_s4.py (gen4)
# born: 2026-05-29T23:56:55Z

"""Hybrid Voronoi‑Fisher‑SSIM Algorithm
===================================

This module fuses **Parent Algorithm A** (Voronoi partitioning, multivector
geometric algebra and Gaussian‑beam dimensionality reduction) with
**Parent Algorithm B** (Fisher information for a Gaussian beam, MinHash‑like
similarity and SSIM‑weighted bandit decision).

Mathematical bridge
-------------------
* The **Voronoi regions** produced by *assign()* are treated as probabilistic
  partitions of the data space.
* For each region we evaluate a **Fisher information score** (parent B) of the
  region centre with respect to a Gaussian‑beam model.  This score quantifies
  the epistemic certainty of the region and is used as a **weight**.
* The **Gaussian‑beam intensity** (parent B) scales the components of the
  multivector that represents the geometric product of the region’s points
  (parent A).  The scaling implements a data‑driven dimensionality reduction.
* A **structural similarity index (SSIM)** between a region‑wise feature vector
  and a reference signal is computed.  The SSIM value weights the expected
  information‑gain term of a simple multi‑armed bandit, producing a decision
  that respects both geometric certainty (Fisher) and perceptual similarity
  (SSIM).

The three core functions below embody this bridge:
`region_fisher_weights`, `scale_multivector_by_beam`, and `ssim_bandit_select`."""

import math
import random
import sys
import pathlib
import hashlib
from collections import Counter
from typing import List, Tuple, Dict, Any, Sequence, FrozenSet

import numpy as np

# ----------------------------------------------------------------------
# Basic geometry (Parent A)
# ----------------------------------------------------------------------
Point = Tuple[float, float]

def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def nearest(point: Point, seeds: List[Point]) -> int:
    if not seeds:
        raise ValueError('seeds required')
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))

def assign(points: List[Point], seeds: List[Point]) -> Dict[int, List[Point]]:
    """Voronoi assignment of `points` to the nearest `seeds`."""
    regions: Dict[int, List[Point]] = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions

# ----------------------------------------------------------------------
# Geometric algebra helpers (Parent A)
# ----------------------------------------------------------------------
def _blade_sign(indices: Sequence[int]) -> Tuple[List[int], int]:
    """Return a sorted list of unique indices and the sign of the permutation."""
    lst = list(indices)
    sign = 1
    n = len(lst)
    i = 0
    while i < n:
        j = 0
        while j < n - 1 - i:
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                # cancel duplicate basis vectors (e_i ^ e_i = 0)
                lst.pop(j)
                lst.pop(j)
                n -= 2
                i -= 1
                j -= 1
                continue
            j += 1
        i += 1
    return lst, sign

def _multiply_blades(blade_a: FrozenSet[int], blade_b: FrozenSet[int]) -> Tuple[FrozenSet[int], int]:
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign

class Multivector:
    """Simple multivector storing scalar components keyed by basis blades."""
    def __init__(self, components: Dict[FrozenSet[int], float], n: int):
        self.n = n
        # ensure all keys are frozensets of ints
        self.components = {frozenset(k): float(v) for k, v in components.items()}

    def __add__(self, other: 'Multivector') -> 'Multivector':
        if self.n != other.n:
            raise ValueError('grade mismatch')
        new = self.components.copy()
        for b, v in other.components.items():
            new[b] = new.get(b, 0.0) + v
        return Multivector(new, self.n)

    def __mul__(self, other: 'Multivector') -> 'Multivector':
        if self.n != other.n:
            raise ValueError('grade mismatch')
        result: Dict[FrozenSet[int], float] = {}
        for a_blade, a_val in self.components.items():
            for b_blade, b_val in other.components.items():
                blade, sign = _multiply_blades(a_blade, b_blade)
                result[blade] = result.get(blade, 0.0) + sign * a_val * b_val
        return Multivector(result, self.n)

    def scale(self, factor: float) -> 'Multivector':
        return Multivector({b: v * factor for b, v in self.components.items()}, self.n)

    def norm(self) -> float:
        return math.sqrt(sum(v * v for v in self.components.values()))

    def __repr__(self) -> str:
        return f"Multivector({self.components}, n={self.n})"

# ----------------------------------------------------------------------
# Gaussian‑beam, Fisher and SSIM (Parent B)
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian beam intensity."""
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a Gaussian beam."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

def compute_ssim(x: List[float], y: List[float],
                 dynamic_range: float = 1.0,
                 k1: float = 0.01,
                 k2: float = 0.03) -> float:
    """Simplified SSIM for 1‑D signals."""
    if len(x) != len(y):
        raise ValueError("signals must have equal length")
    if not x:
        raise ValueError("signals must not be empty")
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    mu_x = x_arr.mean()
    mu_y = y_arr.mean()
    sigma_x = x_arr.var()
    sigma_y = y_arr.var()
    sigma_xy = ((x_arr - mu_x) * (y_arr - mu_y)).mean()

    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2

    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    return float(numerator / denominator)

# ----------------------------------------------------------------------
# Hybrid core functions (the new contribution)
# ----------------------------------------------------------------------
def region_fisher_weights(seeds: List[Point],
                         beam_center: float,
                         beam_width: float) -> List[float]:
    """
    Compute a Fisher‑information weight for each Voronoi seed.
    The seed’s polar angle (atan2) is used as the `theta` argument of the
    Gaussian‑beam model.
    """
    weights = []
    for (x, y) in seeds:
        theta = math.atan2(y, x)  # angle in radians
        w = fisher_score(theta, beam_center, beam_width)
        weights.append(w)
    # Normalise to sum to 1 for later probabilistic use
    total = sum(weights) or 1.0
    return [w / total for w in weights]

def scale_multivector_by_beam(mv: Multivector,
                              seed: Point,
                              beam_center: float,
                              beam_width: float) -> Multivector:
    """
    Scale a multivector by the Gaussian‑beam intensity evaluated at the seed’s
    angle.  This implements a data‑driven dimensionality reduction that favours
    directions with higher beam intensity.
    """
    theta = math.atan2(seed[1], seed[0])
    intensity = gaussian_beam(theta, beam_center, beam_width)
    return mv.scale(intensity)

def ssim_bandit_select(region_features: List[List[float]],
                       reference: List[float],
                       fisher_weights: List[float],
                       exploration: float = 0.1) -> int:
    """
    Multi‑armed bandit selector that combines:
      * SSIM similarity between each region’s feature vector and a reference.
      * Fisher weight (certainty) of the region.
    The arm with maximal weighted score is chosen, with ε‑greedy exploration.
    Returns the index of the selected region.
    """
    if len(region_features) != len(fisher_weights):
        raise ValueError("feature count must match weight count")
    ssim_scores = [compute_ssim(feat, reference) for feat in region_features]
    weighted_scores = [s * w for s, w in zip(ssim_scores, fisher_weights)]

    # ε‑greedy exploration
    if random.random() < exploration:
        return random.randrange(len(region_features))
    return int(np.argmax(weighted_scores))

def hybrid_process(points: List[Point],
                   seeds: List[Point],
                   beam_center: float = 0.0,
                   beam_width: float = 1.0,
                   reference_signal: List[float] | None = None) -> Tuple[int, Multivector]:
    """
    End‑to‑end hybrid pipeline:
      1. Voronoi assignment of points to seeds.
      2. Fisher‑information weighting of seeds.
      3. For each region build a multivector from its points (using a simple
         outer‑product encoding) and scale it by the Gaussian‑beam intensity.
      4. Derive a 1‑D feature (norm of the scaled multivector) per region.
      5. Use SSIM‑bandit to select the most promising region.
    Returns the index of the selected region and its scaled multivector.
    """
    # 1. Partition
    regions = assign(points, seeds)

    # 2. Fisher weights
    fisher_w = region_fisher_weights(seeds, beam_center, beam_width)

    # 3. Build & scale multivectors
    scaled_mvs: List[Multivector] = []
    region_features: List[List[float]] = []
    for idx, pts in regions.items():
        # Encode each point as a grade‑1 blade (e1 for x, e2 for y) and sum them
        components: Dict[FrozenSet[int], float] = {}
        for (x, y) in pts:
            # e1 ↔ index 0, e2 ↔ index 1
            components[frozenset({0})] = components.get(frozenset({0}), 0.0) + x
            components[frozenset({1})] = components.get(frozenset({1}), 0.0) + y
        mv = Multivector(components, n=2)
        mv_scaled = scale_multivector_by_beam(mv, seeds[idx], beam_center, beam_width)
        scaled_mvs.append(mv_scaled)
        # Feature vector: [norm, scalar component if present]
        scalar = mv_scaled.components.get(frozenset(), 0.0)
        region_features.append([mv_scaled.norm(), scalar])

    # 4. Reference signal (default: uniform vector)
    if reference_signal is None:
        reference_signal = [1.0] * len(region_features[0])

    # 5. Bandit selection
    chosen_idx = ssim_bandit_select(region_features, reference_signal, fisher_w)

    return chosen_idx, scaled_mvs[chosen_idx]

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # generate synthetic data
    random.seed(42)
    np.random.seed(42)

    # 20 random points in the unit square
    pts = [(random.random(), random.random()) for _ in range(20)]

    # 4 seed points (centroids)
    seeds = [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)]

    # reference signal for SSIM (simple constant)
    ref = [0.5, 0.5]

    idx, mv = hybrid_process(
        points=pts,
        seeds=seeds,
        beam_center=0.0,
        beam_width=0.8,
        reference_signal=ref,
    )

    print(f"Selected region index: {idx}")
    print(f"Scaled multivector of selected region: {mv}")