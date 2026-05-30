# DARWIN HAMMER — match 1528, survivor 5
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hard_t_hybrid_hybrid_hybrid_m546_s3.py (gen5)
# parent_b: hybrid_hybrid_hybrid_vorono_hybrid_hybrid_hybrid_m892_s0.py (gen5)
# born: 2026-05-29T23:37:06Z

"""Hybrid algorithm merging stylometric feature extraction (Parent A) with Voronoi‑based geometric
partitioning and temperature‑dependent developmental rates (Parent B).

Mathematical bridge:
- Parent A yields a normalized vector **f** ∈ ℝⁿ (n = number of function‑word categories).
- Parent B operates on a set of 2‑D seed points **S** = {s_i} and assigns arbitrary points to
  Voronoi cells defined by **S**.
- The fusion maps each component f_i to a polar coordinate (r_i,θ_i) with radius proportional
  to f_i and angle θ_i = 2π·i/n, thereby turning the stylometric signature into a deterministic
  seed configuration **S(f)**.
- Cell‑wise geometric descriptors (sphericity) are combined with a temperature derived from
  the same feature vector (T = 273.15 + 30·∑f_i) and fed to the Schoolfield developmental‑rate
  model. The resulting rates are finally modulated by a weekday‑dependent sinusoidal linear
  operator from Parent A, producing a hybrid output matrix.

The module provides three high‑level functions that demonstrate this pipeline:
1. `generate_seeds_from_text`
2. `partition_points`
3. `compute_hybrid_matrix`
"""

import sys
import math
import random
import pathlib
import re
from datetime import datetime
from collections import Counter
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Iterable

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Stylometry
# ----------------------------------------------------------------------
FUNCTION_CATS: dict[str, set[str]] = {
    "pronoun": set(
        "i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()
    ),
    "article": set("a an the".split()),
    "preposition": set(
        "about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()
    ),
    "auxiliary": set(
        "am are be been being can could did do does had has have is may might must shall should was were will would".split()
    ),
    "conjunction": set(
        "and but or nor so yet because although while if when where whereas unless until".split()
    ),
    "negation": set("no not never none neither cannot can't won't don't".split()),
}
CATEGORY_ORDER = list(FUNCTION_CATS.keys())
NUM_CATS = len(CATEGORY_ORDER)


def _tokenize(text: str) -> List[str]:
    """Very simple word tokenizer."""
    return re.findall(r"\b\w+'\w+|\b\w+\b", text.lower())


def stylometry_features(text: str) -> np.ndarray:
    """
    Normalized frequency vector over FUNCTION_CATS.
    Returns a (NUM_CATS,) float array that sums to 1 (or zeros if no matches).
    """
    tokens = _tokenize(text)
    counts = Counter(tokens)
    vec = np.zeros(NUM_CATS, dtype=float)
    for idx, cat in enumerate(CATEGORY_ORDER):
        cat_words = FUNCTION_CATS[cat]
        cat_count = sum(counts[w] for w in cat_words if w in counts)
        vec[idx] = cat_count
    total = vec.sum()
    if total > 0.0:
        vec /= total
    return vec


def _base_sinusoid(pool_size: int) -> np.ndarray:
    """
    Produce a (pool_size, pool_size) matrix whose rows are shifted sinusoids.
    Row i: sin(2π * (j + i) / pool_size) for j = 0..pool_size-1
    """
    if pool_size <= 0:
        raise ValueError("pool_size must be positive")
    j = np.arange(pool_size)
    matrix = np.empty((pool_size, pool_size), dtype=float)
    for i in range(pool_size):
        matrix[i] = np.sin(2 * np.pi * (j + i) / pool_size)
    return matrix


# ----------------------------------------------------------------------
# Parent B – Voronoi & Schoolfield
# ----------------------------------------------------------------------
def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def nearest(point: Tuple[float, float], seeds: List[Tuple[float, float]]) -> int:
    if not seeds:
        raise ValueError('seeds required')
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))


def assign(points: List[Tuple[float, float]], seeds: List[Tuple[float, float]]) -> Dict[int, List[Tuple[float, float]]]:
    regions = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions


@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0
    delta_h_activation: float = 12_000.0
    t_low: float = 283.15
    t_high: float = 307.15
    delta_h_low: float = -45_000.0
    delta_h_high: float = 65_000.0
    r_cal: float = 1.987


def c_to_k(celsius: float) -> float:
    return celsius + 273.15


def developmental_rate(temp_k: float, params: SchoolfieldParams = SchoolfieldParams()) -> float:
    if temp_k <= 0 or params.rho_25 < 0:
        raise ValueError("temperature must be Kelvin-positive and rho_25 non-negative")
    numerator = params.rho_25 * (temp_k / 298.15) * np.exp(
        (params.delta_h_activation / params.r_cal) * ((1.0 / 298.15) - (1.0 / temp_k))
    )
    low = np.exp((params.delta_h_low / params.r_cal) * ((1.0 / params.t_low) - (1.0 / temp_k)))
    high = np.exp((params.delta_h_high / params.r_cal) * ((1.0 / params.t_high) - (1.0 / temp_k)))
    return numerator / (1.0 + low + high)


def sphericity_index(length: float, width: float, height: float) -> float:
    """
    Ratio of the volume of the shape to the volume of a sphere with the same surface area.
    For a rectangular prism, a simple proxy is ( (l*w*h) ** (1/3) ) / ( ( (l*w + w*h + h*l) / math.pi ) ** 0.5 )
    Returns a dimensionless number; 1 indicates perfect sphericity.
    """
    if min(length, width, height) <= 0:
        raise ValueError("All dimensions must be positive")
    volume = length * width * height
    surface = 2 * (length * width + width * height + height * length)
    sphere_radius = math.sqrt(surface / (4 * math.pi))
    sphere_volume = (4 / 3) * math.pi * sphere_radius ** 3
    return volume / sphere_volume


# ----------------------------------------------------------------------
# Fusion utilities
# ----------------------------------------------------------------------
def generate_seeds_from_text(text: str, scale: float = 10.0) -> List[Tuple[float, float]]:
    """
    Convert stylometric feature vector into polar coordinates.
    Each category i becomes a seed:
        r_i = scale * f_i          (radius proportional to frequency)
        θ_i = 2π * i / NUM_CATS    (even angular spacing)
    Returns a list of (x, y) tuples.
    """
    f = stylometry_features(text)
    seeds: List[Tuple[float, float]] = []
    for i, magnitude in enumerate(f):
        theta = 2 * math.pi * i / NUM_CATS
        r = scale * magnitude
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        seeds.append((x, y))
    return seeds


def partition_points(points: List[Tuple[float, float]], seeds: List[Tuple[float, float]]) -> Dict[int, List[Tuple[float, float]]]:
    """
    Wrapper around Parent B's `assign` that also validates that at least one seed exists.
    """
    if not seeds:
        raise ValueError("At least one seed required for Voronoi partition")
    return assign(points, seeds)


def compute_cell_metrics(
    regions: Dict[int, List[Tuple[float, float]]],
    temperature_c: float = 20.0,
) -> Dict[int, Dict[str, float]]:
    """
    For each Voronoi region compute:
    - sphericity of the axis‑aligned bounding box of its points,
    - developmental rate at a temperature derived from `temperature_c`,
    - a sinusoid‑modulated weight based on the region index.
    Returns a nested dict: {region_id: {"sphericity": ..., "rate": ..., "weight": ...}}.
    """
    temp_k = c_to_k(temperature_c)
    rate = developmental_rate(temp_k)

    # sinusoid matrix for weighting
    pool_size = len(regions)
    sinusoid_mat = _base_sinusoid(pool_size)

    metrics: Dict[int, Dict[str, float]] = {}
    for rid, pts in regions.items():
        if not pts:
            # Empty cell: assign neutral metrics
            sph = 1.0
        else:
            xs, ys = zip(*pts)
            length = max(xs) - min(xs) or 1e-6
            width = max(ys) - min(ys) or 1e-6
            # Use a dummy third dimension proportional to point count
            height = len(pts) * 0.1 + 1e-6
            sph = sphericity_index(length, width, height)
        weight = sinusoid_mat[rid, rid]  # diagonal element as region‑specific factor
        metrics[rid] = {"sphericity": sph, "rate": rate, "weight": weight}
    return metrics


def compute_hybrid_matrix(text: str, points: List[Tuple[float, float]]) -> np.ndarray:
    """
    Full pipeline:
    1. Generate seeds from `text`.
    2. Partition `points` into Voronoi cells.
    3. Compute per‑cell metrics.
    4. Assemble a matrix M where row i contains [sphericity, rate, weight] for cell i.
    Returns an (N,3) ndarray where N = number of cells (i.e., number of seeds).
    """
    seeds = generate_seeds_from_text(text)
    regions = partition_points(points, seeds)
    metrics = compute_cell_metrics(regions, temperature_c=20.0 + 10 * np.mean(stylometry_features(text)))
    rows = []
    for i in range(len(seeds)):
        m = metrics[i]
        rows.append([m["sphericity"], m["rate"], m["weight"]])
    return np.array(rows, dtype=float)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_text = (
        "I think that the quick brown fox jumps over the lazy dog, "
        "but it does not mean that it will always do so. "
        "Nevertheless, we can observe the behavior."
    )
    # Generate 200 random points in a square [-5,5]²
    random.seed(42)
    pts = [(random.uniform(-5, 5), random.uniform(-5, 5)) for _ in range(200)]

    hybrid_mat = compute_hybrid_matrix(sample_text, pts)
    print("Hybrid matrix shape:", hybrid_mat.shape)
    print("First three rows:\n", hybrid_mat[:3])