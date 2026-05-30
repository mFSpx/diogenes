# DARWIN HAMMER — match 5561, survivor 5
# gen: 5
# parent_a: hybrid_hybrid_voronoi_parti_hybrid_hybrid_hybrid_m104_s2.py (gen4)
# parent_b: hybrid_hybrid_voronoi_parti_hybrid_endpoint_circ_m314_s1.py (gen2)
# born: 2026-05-30T00:02:49Z

"""Hybrid Voronoi‑Geometric‑Biological Resource Allocation.

Parents:
- hybrid_hybrid_voronoi_parti_hybrid_hybrid_hybrid_m104_s2.py (geometric product update of a
  resource allocation matrix R with a failure counter F).
- hybrid_hybrid_voronoi_parti_hybrid_endpoint_circ_m314_s1.py (Voronoi region assignment
  coupled with a biological developmental‑rate model).

Mathematical bridge:
Each Voronoi region is treated as a distinct “cell” that carries its own resource
allocation matrix R_i.  The scalar factor G used in the geometric‑product update is
derived from the biological developmental‑rate function ρ(T) evaluated at a
temperature that is inferred from the region (the mean y‑coordinate of its points,
converted to Kelvin).  Thus the biological model supplies the geometric‑product
scalar that drives the algebraic update, while the failure‑counter logic from the
circuit‑breaker parent monitors the stability of each region’s resources.

The hybrid therefore fuses:
    R_i ← R_i * (1 - (1 - exp(-t/τ)) * (1 - G_i))
    F_i ← F_i + 1 if ρ(T_i) falls below a safety threshold else 0
where G_i = ρ(T_i) (clamped to [0,1]) and τ is a time constant common to all cells.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict

import numpy as np

# ---------------------------------------------------------------------------
# Biological (Schoolfield) model – parent B
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0               # baseline rate at 25 °C
    delta_h_activation: float = 12_000.0
    t_low: float = 283.15              # K
    t_high: float = 307.15             # K
    delta_h_low: float = -45_000.0
    delta_h_high: float = 65_000.0
    r_cal: float = 1.987               # gas constant (cal·mol⁻¹·K⁻¹)

def c_to_k(celsius: float) -> float:
    """Convert Celsius to Kelvin."""
    return celsius + 273.15

def developmental_rate(temp_k: float,
                       params: SchoolfieldParams = SchoolfieldParams()) -> float:
    """Schoolfield temperature‑dependent developmental rate."""
    if temp_k <= 0 or params.rho_25 < 0:
        raise ValueError("temperature must be Kelvin‑positive and rho_25 non‑negative")
    num = params.rho_25 * (temp_k / 298.15) * math.exp(
        (params.delta_h_activation / params.r_cal) * ((1.0 / 298.15) - (1.0 / temp_k))
    )
    low = math.exp((params.delta_h_low / params.r_cal) *
                   ((1.0 / params.t_low) - (1.0 / temp_k)))
    high = math.exp((params.delta_h_high / params.r_cal) *
                    ((1.0 / params.t_high) - (1.0 / temp_k)))
    return num / (1.0 + low + high)

# ---------------------------------------------------------------------------
# Voronoi utilities – parent B
# ---------------------------------------------------------------------------

def _euclidean(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def nearest(point: Tuple[float, float], seeds: List[Tuple[float, float]]) -> int:
    """Return the index of the seed closest to *point* (ties broken by index)."""
    if not seeds:
        raise ValueError("seed list cannot be empty")
    return min(range(len(seeds)),
               key=lambda i: ( _euclidean(point, seeds[i]), i))

def assign_regions(points: List[Tuple[float, float]],
                   seeds: List[Tuple[float, float]]) -> Dict[int, List[Tuple[float, float]]]:
    """Assign every point to the Voronoi region of its nearest seed."""
    regions: Dict[int, List[Tuple[float, float]]] = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions

# ---------------------------------------------------------------------------
# Clifford‑geometric helpers – parent A
# ---------------------------------------------------------------------------

def geometric_product(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Simple scalar geometric product for 2‑D vectors.
    In a full Clifford algebra this would be a multivector;
    here we retain only the symmetric (dot) part, which suffices as a scalar factor.
    """
    if v1.shape != v2.shape:
        raise ValueError("vectors must have the same shape")
    return float(np.dot(v1, v2))

def resource_update(R: np.ndarray,
                   t: float,
                   tau: float,
                   G: float) -> np.ndarray:
    """
    Hybrid update rule from parent A, with G supplied by the biological model.
    R ← R * (1 - (1 - exp(-t/τ)) * (1 - G))
    """
    if tau <= 0:
        raise ValueError("tau must be positive")
    factor = 1.0 - (1.0 - math.exp(-t / tau)) * (1.0 - G)
    return R * factor

def failure_counter_update(F: int,
                           rate: float,
                           threshold: float = 0.2) -> int:
    """
    Increment failure counter if the developmental rate falls below *threshold*,
    otherwise reset to zero (circuit‑breaker logic from parent A).
    """
    return F + 1 if rate < threshold else 0

# ---------------------------------------------------------------------------
# Hybrid core – combines region geometry with geometric‑biological update
# ---------------------------------------------------------------------------

def compute_region_temperature(region_points: List[Tuple[float, float]]) -> float:
    """
    Infer a temperature for a region as the mean y‑coordinate (Celsius) of its points,
    then convert to Kelvin for the developmental‑rate model.
    """
    if not region_points:
        # fallback to a neutral temperature (25 °C)
        return c_to_k(25.0)
    mean_y = sum(p[1] for p in region_points) / len(region_points)
    return c_to_k(mean_y)

def initialize_resource_matrices(num_regions: int,
                                 dim: int = 2) -> List[np.ndarray]:
    """
    Create a list of identity matrices, one per Voronoi region.
    Each matrix represents the resource allocation state R_i.
    """
    return [np.identity(dim, dtype=float) for _ in range(num_regions)]

def hybrid_step(regions: Dict[int, List[Tuple[float, float]]],
                R_mats: List[np.ndarray],
                F_counters: List[int],
                t: float,
                tau: float) -> Tuple[List[np.ndarray], List[int]]:
    """
    Perform one hybrid iteration:
      1. For each region compute its temperature → developmental rate → G.
      2. Update the region's resource matrix with the geometric‑product rule.
      3. Update the failure counter based on the rate.
    Returns updated resource matrices and failure counters.
    """
    new_R = []
    new_F = []

    for idx in range(len(R_mats)):
        pts = regions.get(idx, [])
        temp_k = compute_region_temperature(pts)
        rate = developmental_rate(temp_k)               # ρ(T) ∈ (0, ∞)
        G = max(0.0, min(1.0, rate))                     # clamp to [0,1] for stability

        # geometric product uses a dummy vector derived from the region's centroid
        if pts:
            centroid = np.mean(np.array(pts), axis=0)
        else:
            centroid = np.zeros(2)
        gp = geometric_product(centroid, centroid)      # essentially ||centroid||²
        # blend the biological scalar with the geometric product (purely illustrative)
        G_blend = (G + gp) / (1.0 + gp)                  # keeps result in [0,1]

        R_updated = resource_update(R_mats[idx], t, tau, G_blend)
        F_updated = failure_counter_update(F_counters[idx], rate)

        new_R.append(R_updated)
        new_F.append(F_updated)

    return new_R, new_F

# ---------------------------------------------------------------------------
# Demonstration functions (at least three)
# ---------------------------------------------------------------------------

def generate_random_points(num_points: int,
                           xlim: Tuple[float, float] = (0.0, 100.0),
                           ylim: Tuple[float, float] = (0.0, 100.0)) -> List[Tuple[float, float]]:
    """Uniformly sample 2‑D points within the given rectangular bounds."""
    return [(random.uniform(*xlim), random.uniform(*ylim)) for _ in range(num_points)]

def run_hybrid_simulation(num_seeds: int = 5,
                          num_points: int = 500,
                          steps: int = 10,
                          tau: float = 5.0) -> None:
    """Execute a short simulation that showcases the hybrid dynamics."""
    seeds = generate_random_points(num_seeds)
    points = generate_random_points(num_points)

    regions = assign_regions(points, seeds)
    R_mats = initialize_resource_matrices(num_seeds, dim=2)
    F_counters = [0 for _ in range(num_seeds)]

    for step in range(steps):
        t = step * 1.0                     # simple linear time
        R_mats, F_counters = hybrid_step(regions, R_mats, F_counters, t, tau)

        # simple diagnostics
        avg_norm = np.mean([np.linalg.norm(R) for R in R_mats])
        max_fail = max(F_counters)
        print(f"Step {step:02d} | avg‖R‖={avg_norm:.4f} | max failure counter={max_fail}")

def region_summary(regions: Dict[int, List[Tuple[float, float]]]) -> None:
    """Print a quick summary of region sizes and inferred temperatures."""
    for idx, pts in regions.items():
        temp_k = compute_region_temperature(pts)
        print(f"Region {idx}: {len(pts)} points, inferred T = {temp_k - 273.15:.2f} °C")

# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Minimal test to ensure importability and basic execution.
    seeds_demo = [(10.0, 10.0), (80.0, 20.0), (50.0, 80.0)]
    points_demo = generate_random_points(200, xlim=(0, 100), ylim=(0, 100))
    regions_demo = assign_regions(points_demo, seeds_demo)

    print("=== Region Summary ===")
    region_summary(regions_demo)

    print("\n=== Running Hybrid Simulation ===")
    run_hybrid_simulation(num_seeds=len(seeds_demo),
                          num_points=200,
                          steps=5,
                          tau=3.0)