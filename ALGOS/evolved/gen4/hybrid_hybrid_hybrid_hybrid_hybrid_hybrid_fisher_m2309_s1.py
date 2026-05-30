# DARWIN HAMMER — match 2309, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_hybrid_geomet_hybrid_rbf_surrogate_m409_s7.py (gen3)
# parent_b: hybrid_hybrid_fisher_locali_hybrid_hybrid_ternar_m177_s0.py (gen3)
# born: 2026-05-29T23:41:46Z

"""Hybrid Geometric‑Gaussian‑Fisher‑Variational (HGGF‑V) module.

This module fuses the two parent algorithms:

* **Parent A** – geometric utilities (distance, Voronoi assignment) and a tiny
  geometric‑algebra (GA) container (`Multivector`).  The GA provides a
  coordinate‑free way to embed scalar and vector quantities together.

* **Parent B** – Gaussian beam intensity, Fisher information for a single
  angle, and a variational free‑energy formulation that treats a belief
  mean μ₍q₎ and an observation x with a Gaussian likelihood.

**Mathematical bridge** – Both parents rely on a Gaussian distribution.
In HGGF‑V we weight each Voronoi region by the Gaussian‑beam intensity of the
region’s centroid angle, compute the Fisher information of that intensity,
and embed the resulting scalar together with the centroid coordinates into a
`Multivector`.  The scalar part of the multivector (the Fisher score) is then
used as the variance term σ² in the variational free‑energy expression,
thereby letting the geometric structure directly inform the probabilistic
inference step.

The public API demonstrates the hybrid workflow through three core
functions:

1. `assign_weighted_regions` – deterministic Voronoi assignment + Gaussian
   weighting.
2. `region_fisher_multivector` – builds a `Multivector` from region centroid
   and its Fisher information.
3. `variational_free_energy` – evaluates the free energy using the multivector
   scalar as the observation variance.

A tiny ternary router is also provided for completeness.
"""

import math
import random
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Sequence, FrozenSet, Any

import numpy as np

# ----------------------------------------------------------------------
# Basic geometric utilities (Parent A)
# ----------------------------------------------------------------------
Point = Tuple[float, float]

def distance(a: Point, b: Point) -> float:
    """Euclidean distance between two 2‑D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def nearest(point: Point, seeds: List[Point]) -> int:
    """Index of the nearest seed (ties broken by lower index)."""
    if not seeds:
        raise ValueError('seeds required')
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))

def assign(points: List[Point], seeds: List[Point]) -> Dict[int, List[Point]]:
    """Deterministic Voronoi assignment of points to the nearest seed."""
    regions: Dict[int, List[Point]] = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions

# ----------------------------------------------------------------------
# Minimal geometric‑algebra implementation (Parent A)
# ----------------------------------------------------------------------
def _blade_sign(indices: Sequence[int]) -> Tuple[List[int], int]:
    """Sort a list of basis indices and return the sign of the permutation.
    Duplicate indices cancel (grade‑2+ becomes 0)."""
    lst = list(indices)
    sign = 1
    n = len(lst)
    i = 0
    while i < n - 1:
        j = 0
        while j < n - 1 - i:
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                # cancel duplicate basis vector
                del lst[j:j + 2]
                n -= 2
                i = -1  # restart outer loop
                break
            j += 1
        i += 1
    return lst, sign

def _multiply_blades(blade_a: FrozenSet[int], blade_b: FrozenSet[int]) -> Tuple[FrozenSet[int], int]:
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign

class Multivector:
    """Very small GA container supporting addition and outer product (wedge)."""
    def __init__(self, components: Dict[FrozenSet[int], float], n: int):
        self.n = int(n)
        # discard zero components
        self.components = {k: float(v) for k, v in components.items() if abs(v) > 1e-15}

    def scalar_part(self) -> float:
        return self.components.get(frozenset(), 0.0)

    def __repr__(self) -> str:
        terms = []
        for blade, coeff in sorted(self.components.items(), key=lambda x: (len(x[0]), sorted(x[0]))):
            if not blade:
                terms.append(f"{coeff:.3g}")
            else:
                basis = ''.join(f"e{i}" for i in sorted(blade))
                terms.append(f"{coeff:.3g}{basis}")
        return " + ".join(terms) if terms else "0"

    def __add__(self, other: 'Multivector') -> 'Multivector':
        if self.n != other.n:
            raise ValueError("grade mismatch")
        new_comp = self.components.copy()
        for blade, coeff in other.components.items():
            new_comp[blade] = new_comp.get(blade, 0.0) + coeff
        return Multivector(new_comp, self.n)

    def __xor__(self, other: 'Multivector') -> 'Multivector':
        """Outer (wedge) product."""
        if self.n != other.n:
            raise ValueError("grade mismatch")
        result: Dict[FrozenSet[int], float] = {}
        for blade_a, coeff_a in self.components.items():
            for blade_b, coeff_b in other.components.items():
                blade_res, sign = _multiply_blades(blade_a, blade_b)
                if blade_res in result:
                    result[blade_res] += sign * coeff_a * coeff_b
                else:
                    result[blade_res] = sign * coeff_a * coeff_b
        return Multivector(result, self.n)

# ----------------------------------------------------------------------
# Gaussian‑based utilities (Parent B)
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity I(θ) of a beam centred at *center* with *width*."""
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a single angle θ.

    F(θ) = (∂I/∂θ)² / I   where I = Gaussian beam intensity.
    """
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

# ----------------------------------------------------------------------
# Ternary router (simple sign‑based routing)
# ----------------------------------------------------------------------
def ternary_route(vector: Sequence[float]) -> List[int]:
    """Map each component to -1, 0, or +1 depending on its sign."""
    return [(-1 if v < 0 else (1 if v > 0 else 0)) for v in vector]

# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------
def assign_weighted_regions(
    points: List[Point],
    seeds: List[Point],
    beam_center: float,
    beam_width: float
) -> Dict[int, Dict[str, Any]]:
    """
    Perform deterministic Voronoi assignment, then compute for each region:

    * `centroid` – arithmetic mean of points,
    * `angle`    – polar angle of the centroid (atan2),
    * `weight`   – Gaussian beam intensity evaluated at that angle.

    Returns a mapping ``region_id -> {centroid, angle, weight, points}``.
    """
    voronoi = assign(points, seeds)
    regions: Dict[int, Dict[str, Any]] = {}
    for rid, pts in voronoi.items():
        if not pts:
            # empty region – use the seed itself as centroid
            cx, cy = seeds[rid]
        else:
            xs, ys = zip(*pts)
            cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
        angle = math.atan2(cy, cx)  # angle in radians
        weight = gaussian_beam(angle, beam_center, beam_width)
        regions[rid] = {
            "centroid": (cx, cy),
            "angle": angle,
            "weight": weight,
            "points": pts,
        }
    return regions

def region_fisher_multivector(
    region_info: Dict[str, Any],
    beam_center: float,
    beam_width: float
) -> Multivector:
    """
    Build a `Multivector` whose scalar part is the Fisher information of the
    region’s centroid angle and whose vector part encodes the centroid
    coordinates.

    The multivector lives in a 2‑dimensional GA (basis e1, e2).
    """
    cx, cy = region_info["centroid"]
    angle = region_info["angle"]
    fisher = fisher_score(angle, beam_center, beam_width)
    # components:
    # scalar (grade‑0) -> fisher
    # e1 (grade‑1)    -> cx
    # e2 (grade‑1)    -> cy
    components = {
        frozenset(): fisher,
        frozenset({1}): cx,
        frozenset({2}): cy,
    }
    return Multivector(components, n=2)

def variational_free_energy(
    mu_q: np.ndarray,
    x_obs: np.ndarray,
    sigma2: float,
    prior_mu: np.ndarray = None,
    prior_sigma2: float = 1.0
) -> float:
    """
    Compute a simple Gaussian variational free energy:

        F = 0.5 * ((mu_q - x_obs)^2 / sigma2 + log(2π sigma2))
            + 0.5 * ((mu_q - prior_mu)^2 / prior_sigma2 + log(2π prior_sigma2))

    `sigma2` is supplied by the Fisher information (larger Fisher → smaller
    variance).  If `prior_mu` is omitted, the prior term is dropped.
    """
    diff = mu_q - x_obs
    term_lik = 0.5 * (np.sum(diff ** 2) / sigma2 + np.log(2 * math.pi * sigma2))

    if prior_mu is not None:
        diff_prior = mu_q - prior_mu
        term_prior = 0.5 * (np.sum(diff_prior ** 2) / prior_sigma2 + np.log(2 * math.pi * prior_sigma2))
    else:
        term_prior = 0.0

    return term_lik + term_prior

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(0)

    # generate random points within a square
    pts = [(random.uniform(-5, 5), random.uniform(-5, 5)) for _ in range(200)]

    # choose 5 seed points
    seeds = [(random.uniform(-5, 5), random.uniform(-5, 5)) for _ in range(5)]

    beam_center = 0.0          # centre of Gaussian beam (radians)
    beam_width = 1.0           # width of Gaussian beam

    # 1️⃣ Voronoi assignment with Gaussian weighting
    regions = assign_weighted_regions(pts, seeds, beam_center, beam_width)

    # 2️⃣ Build multivectors and evaluate free energy per region
    free_energies = {}
    for rid, info in regions.items():
        mv = region_fisher_multivector(info, beam_center, beam_width)

        # Extract scalar (Fisher) as variance term (invert to get σ²)
        fisher = mv.scalar_part()
        sigma2 = 1.0 / (fisher + 1e-12)  # avoid division by zero

        # Use centroid as belief mean μ_q, observation as weighted centroid
        mu_q = np.array(info["centroid"])
        x_obs = mu_q * info["weight"]   # simple observation model

        F = variational_free_energy(mu_q, x_obs, sigma2)
        free_energies[rid] = F

    # 3️⃣ Simple ternary routing of the free‑energy vector
    fe_vector = np.array(list(free_energies.values()))
    route = ternary_route(fe_vector)

    print("Region free energies:", free_energies)
    print("Ternary route of free‑energy vector:", route)