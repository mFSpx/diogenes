# DARWIN HAMMER — match 4830, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_fisher_hybrid_hybrid_minimu_m140_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_fisher_m2309_s1.py (gen4)
# born: 2026-05-29T23:58:15Z

"""Hybrid Gaussian‑Geometric‑Fisher‑Variational (HGGF‑V) Engine.

Parents
-------
* **Parent A** – provides Gaussian‑beam intensity `gaussian_beam`,
  Fisher‑information scorer `fisher_score`, and a count‑min sketch utility.
* **Parent B** – supplies geometric utilities (`distance`, `nearest`), a tiny
  geometric‑algebra container `Multivector`, and a Voronoi‑based workflow.

Mathematical Bridge
-------------------
Both ancestors revolve around a *Gaussian* model:

* The beam intensity `I(θ)` is a Gaussian in the angular variable `θ`.
* Fisher information `𝓘(θ)` is derived analytically from the same Gaussian.
* The geometric side partitions space into Voronoi regions; each region’s
  centroid defines an angle `θ_c = atan2(y, x)`.  

The hybrid therefore **weights each Voronoi region by the beam intensity at
its centroid**, computes the **Fisher score** of that intensity, and embeds
the scalar score together with the centroid coordinates into a `Multivector`.
The scalar part of the multivector is then interpreted as the observation
variance `σ²` inside a **variational free‑energy** functional, letting the
geometric structure directly influence probabilistic inference.

The module implements this fused pipeline in three core public functions:

1. `assign_weighted_regions` – Voronoi assignment + Gaussian weighting.
2. `region_fisher_multivector` – builds a `Multivector` from centroid and its
   Fisher information.
3. `variational_free_energy` – evaluates the free energy using the multivector
   scalar as the variance term.

All code relies only on the Python standard library and NumPy.
"""

import math
import random
import sys
import pathlib
import hashlib
from typing import List, Tuple, Dict, Sequence, Any

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Gaussian beam & Fisher information
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity I(θ) of a beam centred at *center* with *width*."""
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a single angle θ derived from the Gaussian beam."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

# ----------------------------------------------------------------------
# Parent B – Geometry and a tiny geometric‑algebra container
# ----------------------------------------------------------------------
Point = Tuple[float, float]

def distance(a: Point, b: Point) -> float:
    """Euclidean distance between two 2‑D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def nearest(point: Point, seeds: List[Point]) -> int:
    """Index of the nearest seed (ties broken by lower index)."""
    if not seeds:
        raise ValueError("seed list is empty")
    return min(range(len(seeds)), key=lambda i: distance(point, seeds[i]))

class Multivector:
    """
    Minimal geometric‑algebra container for a scalar (grade‑0) and a
    2‑D vector (grade‑1).  The scalar part will carry the Fisher score,
    the vector part the region centroid.
    """
    __slots__ = ("scalar", "vector")

    def __init__(self, scalar: float = 0.0, vector: Point = (0.0, 0.0)):
        self.scalar = float(scalar)
        self.vector = (float(vector[0]), float(vector[1]))

    def __repr__(self) -> str:
        return f"Multivector(scalar={self.scalar:.4g}, vector=({self.vector[0]:.4g}, {self.vector[1]:.4g}))"

    def copy(self) -> "Multivector":
        return Multivector(self.scalar, self.vector)

# ----------------------------------------------------------------------
# Hybrid Core Functions
# ----------------------------------------------------------------------
def assign_weighted_regions(
    points: Sequence[Point],
    seeds: Sequence[Point],
    beam_center: float,
    beam_width: float,
) -> Dict[int, Dict[str, Any]]:
    """
    Partition *points* into Voronoi cells defined by *seeds*.
    For each cell compute:
        • centroid (mean of assigned points)
        • angular coordinate θ_c = atan2(y, x) of the centroid
        • Gaussian‑beam weight w = I(θ_c)
    Returns a dict keyed by seed index with the computed metadata.
    """
    # Initialise containers
    assignments: Dict[int, List[Point]] = {i: [] for i in range(len(seeds))}
    for pt in points:
        idx = nearest(pt, list(seeds))
        assignments[idx].append(pt)

    region_info: Dict[int, Dict[str, Any]] = {}
    for idx, pts in assignments.items():
        if not pts:  # empty cell → use the seed itself as centroid
            centroid = seeds[idx]
        else:
            arr = np.array(pts)
            centroid = (float(arr[:, 0].mean()), float(arr[:, 1].mean()))
        theta_c = math.atan2(centroid[1], centroid[0])  # angle in radians
        weight = gaussian_beam(theta_c, beam_center, beam_width)
        region_info[idx] = {
            "points": pts,
            "centroid": centroid,
            "theta": theta_c,
            "weight": weight,
        }
    return region_info

def region_fisher_multivector(
    centroid: Point,
    theta: float,
    beam_center: float,
    beam_width: float,
) -> Multivector:
    """
    Build a `Multivector` whose scalar part is the Fisher information of the
    Gaussian beam evaluated at *theta* and whose vector part is the *centroid*.
    """
    fisher = fisher_score(theta, beam_center, beam_width)
    return Multivector(scalar=fisher, vector=centroid)

def variational_free_energy(
    mv: Multivector,
    observation: Point,
    prior_mean: Point,
    prior_variance: float,
    eps: float = 1e-12,
) -> float:
    """
    Compute a simple variational free‑energy (negative ELBO) for a Gaussian
    observation model.

    Posterior (q):
        mean = mv.vector
        variance = max(mv.scalar, eps)   # Fisher score interpreted as σ²

    Likelihood (p(x|θ)):
        Gaussian with mean = posterior mean, variance = σ²·I₂

    Prior (p(θ)):
        Isotropic Gaussian N(prior_mean, prior_variance·I₂)

    The free energy is:
        F = ½ * ((x-μ_q)ᵀ Σ_q⁻¹ (x-μ_q)) + ½ log|2π Σ_q|
            + KL(q‖p)
    where Σ_q = σ² I₂.
    """
    sigma2 = max(mv.scalar, eps)

    # --- Expected negative log‑likelihood term (Gaussian) ---
    diff = np.subtract(observation, mv.vector)  # shape (2,)
    nll = 0.5 * np.dot(diff, diff) / sigma2
    nll += 0.5 * 2 * math.log(2 * math.pi * sigma2)  # 2 dimensions

    # --- KL divergence between two isotropic Gaussians ---
    mu_q = np.array(mv.vector)
    mu_p = np.array(prior_mean)
    var_q = sigma2
    var_p = prior_variance

    kl = 0.5 * (
        (var_q / var_p)
        + np.dot(mu_q - mu_p, mu_q - mu_p) / var_p
        - 2
        + math.log(var_p / var_q)
    )

    return nll + kl

def hybrid_pipeline(
    points: Sequence[Point],
    seeds: Sequence[Point],
    beam_center: float,
    beam_width: float,
    observation: Point,
    prior_mean: Point,
    prior_variance: float,
) -> List[Tuple[int, Multivector, float]]:
    """
    End‑to‑end hybrid workflow:

    1. Voronoi partition + Gaussian weighting (`assign_weighted_regions`).
    2. For each region build a `Multivector` with Fisher information
       (`region_fisher_multivector`).
    3. Evaluate the variational free energy for the supplied *observation*
       using the multivector (`variational_free_energy`).

    Returns a list of tuples ``(seed_index, multivector, free_energy)``.
    """
    region_info = assign_weighted_regions(points, seeds, beam_center, beam_width)
    results = []
    for idx, info in region_info.items():
        mv = region_fisher_multivector(
            centroid=info["centroid"],
            theta=info["theta"],
            beam_center=beam_center,
            beam_width=beam_width,
        )
        fe = variational_free_energy(
            mv,
            observation=observation,
            prior_mean=prior_mean,
            prior_variance=prior_variance,
        )
        results.append((idx, mv, fe))
    return results

# ----------------------------------------------------------------------
# Optional utility from Parent A (count‑min sketch) – retained for completeness
# ----------------------------------------------------------------------
def count_min_sketch(items: Sequence[Any], width: int = 64, depth: int = 4) -> List[List[int]]:
    """
    Simple count‑min sketch implementation.
    Returns a 2‑D list ``depth × width`` of integer counters.
    """
    table = [[0] * width for _ in range(depth)]
    for item in items:
        for d in range(depth):
            h = hashlib.sha256(f"{d}:{item}".encode()).hexdigest()
            idx = int(h, 16) % width
            table[d][idx] += 1
    return table

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # deterministic seed for reproducibility
    random.seed(0)
    np.random.seed(0)

    # Generate random 2‑D points
    points = [(random.uniform(-5, 5), random.uniform(-5, 5)) for _ in range(200)]

    # Choose a few seed points (Voronoi generators)
    seeds = [(-2.0, -2.0), (2.0, -2.0), (0.0, 3.0)]

    # Beam parameters (radians)
    beam_center = 0.0          # aligned with the x‑axis
    beam_width = 0.8           # moderate spread

    # Observation and prior for free‑energy calculation
    observation = (0.5, -0.2)
    prior_mean = (0.0, 0.0)
    prior_variance = 1.5

    results = hybrid_pipeline(
        points=points,
        seeds=seeds,
        beam_center=beam_center,
        beam_width=beam_width,
        observation=observation,
        prior_mean=prior_mean,
        prior_variance=prior_variance,
    )

    for idx, mv, fe in results:
        print(f"Seed {idx}: {mv}, Free Energy = {fe:.4f}")

    # Demonstrate the count‑min sketch works (optional)
    sketch = count_min_sketch([f"pt{i}" for i in range(50)])
    print("\nCount‑Min Sketch sample (first row):", sketch[0][:8])