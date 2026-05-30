# DARWIN HAMMER — match 4917, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_rectified_flo_hybrid_hybrid_gini_c_m1244_s3.py (gen6)
# parent_b: hybrid_hybrid_hybrid_vorono_hybrid_hybrid_hybrid_m706_s2.py (gen5)
# born: 2026-05-29T23:58:52Z

"""Hybrid Rectified Flow – Voronoi‑Gini Fusion

This module merges the core mathematics of two parent algorithms:

* **Parent A** – Rectified Flow with a straight‑line interpolant
  `Z_t = t·x1 + (1‑t)·x0` and a scalar Gini impurity that drives a
  Hoeffding‑style split.
* **Parent B** – Voronoi partitioning whose distance is weighted by an
  epistemic certainty scalar `confidence` (basis points).

**Mathematical bridge** – Both parents expose a *scalar weighting* that
modulates a primary operation:

* In the flow side the scalar `t∈[0,1]` blends two vectors.
* In the Voronoi side the scalar `w_i = 1‑confidence_i/10_000` scales a
  Euclidean distance.

We fuse them by letting the *Gini impurity* of a site act as an additional
confidence term.  For a site `i` we define an **effective confidence**


conf_eff_i = confidence_i + α·gini_i·10_000          (α∈[0,1] is a mixing factor)


and a **weighted distance**


d̂_i(p) = ‖p – s_i‖ · (1 – conf_eff_i / 10_000) .


The point `p` is assigned to the site that minimises `d̂_i(p)`.  The resulting
minimal weighted distance is normalised to `[0,1]` and fed as the flow
parameter `t` for the rectified‑flow interpolant between the point and its
assigned site.  Thus a single scalar – derived from spatial proximity,
epistemic certainty and Gini impurity – drives both the geometric partition
and the flow interpolation, providing a unified hybrid system.

The module offers three high‑level functions that demonstrate this hybrid
behaviour and a small smoke‑test when executed as a script.
"""

import math
import random
import sys
import pathlib
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, FrozenSet, Any

import numpy as np

# ----------------------------------------------------------------------
# Types used by the Voronoi‑Epistemic side
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Blade = FrozenSet[int]          # placeholder for multivector blades
Multivector = Dict[Blade, float]  # placeholder for multivector coefficients

# ----------------------------------------------------------------------
# Epistemic certainty helpers (Parent B)
# ----------------------------------------------------------------------
EPISTEMIC_FLAGS = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")

@dataclass(frozen=True)
class CertaintyFlag:
    """Immutable description of epistemic certainty."""
    label: str
    confidence_bps: int               # 0 … 10 000 basis points = 0 % … 100 %
    authority_class: Any = None       # optional meta‑information

    def __post_init__(self):
        if self.label not in EPISTEMIC_FLAGS:
            raise ValueError(f"Invalid label {self.label!r}")
        if not (0 <= self.confidence_bps <= 10_000):
            raise ValueError("confidence_bps must be in [0, 10_000]")

# ----------------------------------------------------------------------
# Rectified Flow utilities (Parent A)
# ----------------------------------------------------------------------
def interpolant(x0: np.ndarray, x1: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Straight‑line interpolant Z_t = t·x1 + (1‑t)·x0.

    Parameters
    ----------
    x0, x1 : (B, d) arrays – start and end vectors.
    t      : (B,) array – interpolation coefficient per batch element.

    Returns
    -------
    (B, d) array of interpolated points.
    """
    t = np.reshape(t, (-1, 1))
    return t * x1 + (1.0 - t) * x0

def flow_target(x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
    """Target vector field vθ(Z_t, t) = (x1 - x0)."""
    return x1 - x0

# ----------------------------------------------------------------------
# Gini impurity utilities (Parent A)
# ----------------------------------------------------------------------
def gini_coefficient(values: List[float]) -> float:
    """Gini impurity of a non‑negative numeric collection."""
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0.0:
        return 0.0
    if xs[0] < 0:
        raise ValueError("values must be non‑negative")
    n = len(xs)
    # Gini = (2 * Σ i·x_i) / (n·Σ x_i) - (n + 1) / n
    cumulative = sum((2 * i - n + 1) * x for i, x in enumerate(xs, start=1))
    return cumulative / (n * sum(xs))

# ----------------------------------------------------------------------
# Hybrid core: weighted distance, assignment, and flow
# ----------------------------------------------------------------------
def weighted_distance_matrix(
    points: np.ndarray,
    sites: np.ndarray,
    flags: List[CertaintyFlag],
    gini_vals: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Compute the matrix of weighted distances d̂_i(p) for every point‑site pair.

    Parameters
    ----------
    points    : (P, d) array of query points.
    sites     : (S, d) array of Voronoi sites.
    flags     : list of length S with CertaintyFlag objects.
    gini_vals : (S,) array of Gini impurity values per site (in [0,1]).
    alpha     : mixing factor for Gini contribution (0 → pure epistemic,
                1 → pure Gini).

    Returns
    -------
    (P, S) array where entry (p,i) = weighted distance from point p to site i.
    """
    # Euclidean distance matrix (P, S)
    diff = points[:, None, :] - sites[None, :, :]          # shape (P, S, d)
    euclid = np.linalg.norm(diff, axis=2)                 # (P, S)

    # Effective confidence per site (scale to basis points)
    conf = np.array([f.confidence_bps for f in flags], dtype=float)   # (S,)
    gini_bp = gini_vals * 10_000.0                                      # (S,)

    conf_eff = (1.0 - alpha) * conf + alpha * gini_bp                     # (S,)
    conf_eff = np.clip(conf_eff, 0.0, 10_000.0)

    weight = 1.0 - conf_eff / 10_000.0                                     # (S,)
    return euclid * weight[None, :]                                        # (P, S)

def assign_points_to_sites(
    points: np.ndarray,
    sites: np.ndarray,
    flags: List[CertaintyFlag],
    gini_vals: np.ndarray,
    alpha: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assign each point to the site that minimises the weighted distance.

    Returns
    -------
    assignments : (P,) integer array of site indices.
    min_weights : (P,) array of the minimal weighted distance (used later as flow t).
    """
    dist_mat = weighted_distance_matrix(points, sites, flags, gini_vals, alpha)
    assignments = np.argmin(dist_mat, axis=1)
    min_weights = dist_mat[np.arange(len(points)), assignments]
    return assignments, min_weights

def hybrid_flow_voronoi(
    points: np.ndarray,
    sites: np.ndarray,
    flags: List[CertaintyFlag],
    site_features: List[List[float]],
    alpha: float = 0.5,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Perform the hybrid operation:

    1. Compute Gini impurity per site from `site_features`.
    2. Assign each point to a site using the weighted distance that mixes
       epistemic confidence and Gini impurity.
    3. Normalise the minimal weighted distance to the interval [0,1] and use it
       as the flow interpolation coefficient `t`.
    4. Return the rectified‑flow interpolant between each point and its assigned
       site.

    Parameters
    ----------
    points        : (P, d) array of query points.
    sites         : (S, d) array of Voronoi sites.
    flags         : list of S CertaintyFlag objects.
    site_features : list of S feature lists (numeric) used to compute Gini.
    alpha         : mixing factor for Gini contribution in the weighted distance.
    eps           : small constant to avoid division by zero.

    Returns
    -------
    (P, d) array of interpolated vectors Z_t.
    """
    # 1. Gini per site
    gini_vals = np.array([gini_coefficient(feat) for feat in site_features], dtype=float)

    # 2. Assignment
    assignments, min_weights = assign_points_to_sites(points, sites, flags, gini_vals, alpha)

    # 3. Normalise weighted distance to [0,1] → flow parameter t
    max_w = np.max(min_weights) if np.max(min_weights) > eps else 1.0
    t = 1.0 - min_weights / max_w                     # larger distance → smaller t
    t = np.clip(t, 0.0, 1.0)

    # 4. Interpolant between point and its assigned site
    assigned_sites = sites[assignments]               # (P, d)
    return interpolant(points, assigned_sites, t)

# ----------------------------------------------------------------------
# Demonstration utilities (three high‑level functions)
# ----------------------------------------------------------------------
def demo_assign(points: np.ndarray, sites: np.ndarray, flags: List[CertaintyFlag],
                site_features: List[List[float]]) -> None:
    """Print a simple assignment table for a demo."""
    assignments, _ = assign_points_to_sites(
        points, sites, flags,
        np.array([gini_coefficient(f) for f in site_features])
    )
    for i, site_idx in enumerate(assignments):
        print(f"Point {i} -> Site {site_idx}")

def demo_flow(points: np.ndarray, sites: np.ndarray, flags: List[CertaintyFlag],
              site_features: List[List[float]]) -> np.ndarray:
    """Run the full hybrid flow and return the interpolated points."""
    return hybrid_flow_voronoi(points, sites, flags, site_features)

def demo_split_score(points: np.ndarray, sites: np.ndarray, flags: List[CertaintyFlag],
                     site_features: List[List[float]], alpha: float = 0.5) -> np.ndarray:
    """
    Compute a hybrid split score for each point:
        score = (1 - normalized_weighted_distance) * (1 - gini_of_assigned_site)

    Higher score indicates a “clean” assignment (close and low impurity).
    """
    gini_vals = np.array([gini_coefficient(f) for f in site_features], dtype=float)
    dist_mat = weighted_distance_matrix(points, sites, flags, gini_vals, alpha)
    min_weights = np.min(dist_mat, axis=1)
    max_w = np.max(min_weights) if np.max(min_weights) > 0 else 1.0
    norm_w = min_weights / max_w                         # in [0,1]

    assignments = np.argmin(dist_mat, axis=1)
    assigned_gini = gini_vals[assignments]

    return (1.0 - norm_w) * (1.0 - assigned_gini)

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    # Generate synthetic 2‑D points and sites
    P, S, d = 10, 3, 2
    points = np.random.rand(P, d) * 10.0
    sites = np.random.rand(S, d) * 10.0

    # Random epistemic flags per site
    flags = []
    for _ in range(S):
        label = random.choice(EPISTEMIC_FLAGS)
        confidence = random.randint(0, 10_000)
        flags.append(CertaintyFlag(label=label, confidence_bps=confidence))

    # Random feature vectors per site for Gini calculation
    site_features = [list(np.random.rand(random.randint(5, 15))) for _ in range(S)]

    # Demo 1: assignment table
    print("=== Assignment Demo ===")
    demo_assign(points, sites, flags, site_features)

    # Demo 2: hybrid flow interpolation
    print("\n=== Hybrid Flow Output (first 3 rows) ===")
    interpolated = demo_flow(points, sites, flags, site_features)
    print(interpolated[:3])

    # Demo 3: hybrid split score
    print("\n=== Hybrid Split Scores ===")
    scores = demo_split_score(points, sites, flags, site_features)
    print(scores)

    # Verify shapes
    assert interpolated.shape == points.shape
    assert scores.shape == (P,)

    print("\nSmoke test completed successfully.")