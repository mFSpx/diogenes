# DARWIN HAMMER — match 2890, survivor 2
# gen: 2
# parent_a: fisher_localization.py (gen0)
# parent_b: hybrid_geometric_product_voronoi_partition_m4_s1.py (gen1)
# born: 2026-05-29T23:46:28Z

"""Hybrid Fisher–Geometric Voronoi algorithm.

Parent A: fisher_localization.py – provides a Gaussian beam model and the Fisher
information score for a scalar parameter (here an angle).

Parent B: hybrid_geometric_product_voronoi_partition_m4_s1.py – implements the
geometric product of multivectors in the Clifford algebra Cl(n,0) and uses it
to build a Voronoi partition of points in Euclidean space.

Mathematical bridge:
* An angle θ on the unit circle is encoded as the vector multivector  
  **v(θ) = cosθ e₁ + sinθ e₂**.
* The geometric product v(θ₁)·v(θ₂) = (v·v) + (v∧v) yields a scalar part equal
  to the ordinary dot product cos(θ₁‑θ₂).  Because the vectors are unit‑length,
  the scalar part directly gives the cosine of the angular separation, and
  the induced distance  

  d(θ₁,θ₂)=√(2‑2·cos(θ₁‑θ₂)) = 2·|sin((θ₁‑θ₂)/2)|  

  is a Euclidean metric on the circle.
* This metric is used to build a Voronoi diagram of a set of “seed” angles.
* The Fisher information score for a candidate angle is multiplied by a
  Voronoi‑derived weight w(θ) = exp(‑α·d²) (α>0) that favours candidates lying
  close to their seed region.  The hybrid score therefore fuses the
  information‑theoretic sensitivity of the Gaussian beam with the geometric‑
  algebraic spatial partitioning.

The module implements three representative hybrid operations:
1. `voronoi_assign_angles` – Voronoi assignment of candidate angles using the
   geometric‑product‑derived distance.
2. `hybrid_fisher_score` – Fisher information multiplied by a Voronoi weight.
3. `best_hybrid_angle` – selection of the optimal angle according to the
   hybrid score (ties broken by proximity to the beam centre).

All functions are pure Python and rely only on the standard library and NumPy. """

from __future__ import annotations
import math
import random
import sys
import pathlib
import numpy as np

# ----------------------------------------------------------------------
# Clifford algebra helpers (parent B)
# ----------------------------------------------------------------------
def _blade_sign(indices: list[int]) -> tuple[list[int], int]:
    """Return (sorted_blade, sign) after bubble‑sorting index list.

    Each transposition flips the sign (anti‑commutativity).  Duplicate indices
    cancel because e_i*e_i = 1.
    """
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
                # e_i * e_i = 1 -> remove both and keep sign unchanged
                del lst[j:j + 2]
                n -= 2
                i = -1  # restart outer loop because length changed
                break
            j += 1
        i += 1
    return lst, sign


def _multiply_blades(blade_a: frozenset[int], blade_b: frozenset[int]) -> tuple[frozenset[int], int]:
    """Geometric product of two basis blades.

    Returns (result_blade, sign).  The coefficient is handled outside.
    """
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


class Multivector:
    """Element of Cl(n,0) as a sparse map {blade: coefficient}.

    A blade is represented by a frozenset of basis indices, e.g. frozenset({1,2})
    for e₁∧e₂.  The scalar (grade‑0) blade is the empty frozenset().
    """

    def __init__(self, components: dict[frozenset[int], float] | None = None):
        self.components: dict[frozenset[int], float] = {}
        if components:
            # prune zero coefficients
            for b, c in components.items():
                if abs(c) > 1e-15:
                    self.components[frozenset(b)] = float(c)

    @staticmethod
    def scalar(value: float = 1.0) -> "Multivector":
        return Multivector({frozenset(): value})

    @staticmethod
    def vector(coeffs: dict[int, float]) -> "Multivector":
        """Create a pure vector (grade‑1) multivector from a dict {index: coeff}."""
        comps = {frozenset({i}): float(c) for i, c in coeffs.items() if abs(c) > 1e-15}
        return Multivector(comps)

    def __add__(self, other: "Multivector") -> "Multivector":
        result = Multivector(self.components.copy())
        for b, c in other.components.items():
            result.components[b] = result.components.get(b, 0.0) + c
            if abs(result.components[b]) < 1e-15:
                del result.components[b]
        return result

    def __neg__(self) -> "Multivector":
        return Multivector({b: -c for b, c in self.components.items()})

    def __sub__(self, other: "Multivector") -> "Multivector":
        return self + (-other)

    def __mul__(self, other: "Multivector") -> "Multivector":
        """Geometric product (bilinear, associative)."""
        result: dict[frozenset[int], float] = {}
        for ba, ca in self.components.items():
            for bb, cb in other.components.items():
                bc, sgn = _multiply_blades(ba, bb)
                coeff = ca * cb * sgn
                result[bc] = result.get(bc, 0.0) + coeff
        # prune zeros
        result = {b: c for b, c in result.items() if abs(c) > 1e-15}
        return Multivector(result)

    def scalar_part(self) -> float:
        """Return the coefficient of the scalar (grade‑0) blade."""
        return self.components.get(frozenset(), 0.0)

    def __repr__(self) -> str:
        if not self.components:
            return "0"
        terms = []
        for blade, coeff in sorted(self.components.items(), key=lambda x: (len(x[0]), sorted(x[0]))):
            if not blade:
                terms.append(f"{coeff:.3g}")
            else:
                basis = "∧".join(f"e{idx}" for idx in sorted(blade))
                terms.append(f"{coeff:.3g}{basis}")
        return " + ".join(terms)


# ----------------------------------------------------------------------
# Fisher‑information utilities (parent A)
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Normalized Gaussian intensity of a laser beam at angle theta."""
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a Gaussian beam evaluated at theta."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


# ----------------------------------------------------------------------
# Hybrid geometry‑Fisher utilities
# ----------------------------------------------------------------------
def multivector_from_angle(theta: float) -> Multivector:
    """Encode angle θ as a unit vector multivector v(θ)=cosθ·e₁+sinθ·e₂."""
    return Multivector.vector({1: math.cos(theta), 2: math.sin(theta)})


def geometric_distance(theta1: float, theta2: float) -> float:
    """Distance on the unit circle derived from the scalar part of the geometric product.

    For unit vectors v(θ₁), v(θ₂) the scalar part equals cos(θ₁‑θ₂); the induced
    Euclidean distance is sqrt(2‑2·cosΔ) = 2·|sin(Δ/2)|.
    """
    v1 = multivector_from_angle(theta1)
    v2 = multivector_from_angle(theta2)
    cos_delta = (v1 * v2).scalar_part()  # scalar part = dot product
    # Clamp due to floating‑point noise
    cos_delta = max(min(cos_delta, 1.0), -1.0)
    return math.sqrt(2.0 - 2.0 * cos_delta)


def voronoi_assign_angles(candidates: list[float], seeds: list[float]) -> dict[float, float]:
    """Assign each candidate angle to the nearest seed angle.

    Returns a mapping {candidate_angle: assigned_seed}.
    """
    if not seeds:
        raise ValueError("seed list cannot be empty")
    assignment: dict[float, float] = {}
    for theta in candidates:
        nearest = min(seeds, key=lambda s: geometric_distance(theta, s))
        assignment[theta] = nearest
    return assignment


def voronoi_weight(theta: float, seed: float, alpha: float = 1.0) -> float:
    """Weight based on distance to the assigned seed.

    w = exp(-α·d²)  ∈ (0,1], maximal at the seed.
    """
    d = geometric_distance(theta, seed)
    return math.exp(-alpha * d * d)


def hybrid_fisher_score(theta: float, center: float, width: float,
                        seed: float, alpha: float = 1.0) -> float:
    """Fisher information multiplied by a Voronoi‑derived weight."""
    return fisher_score(theta, center, width) * voronoi_weight(theta, seed, alpha)


def best_hybrid_angle(candidates: list[float], center: float, width: float,
                      seeds: list[float], alpha: float = 1.0) -> float:
    """Select the angle with the maximal hybrid score.

    Ties are broken by the smallest absolute deviation from the beam centre.
    """
    if not candidates:
        raise ValueError("candidates required")
    assignments = voronoi_assign_angles(candidates, seeds)
    best = max(
        candidates,
        key=lambda t: (
            hybrid_fisher_score(t, center, width, assignments[t], alpha),
            -abs(t - center)  # secondary criterion
        )
    )
    return best


# ----------------------------------------------------------------------
# Demonstration / smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a Gaussian beam centred at 0 rad with moderate width
    beam_center = 0.0
    beam_width = 0.3

    # Candidate angles uniformly sampled over [-π, π]
    candidates = [float(x) for x in np.linspace(-math.pi, math.pi, 200)]

    # Seed angles for the Voronoi partition (e.g. three preferred directions)
    seeds = [-math.pi / 2, 0.0, math.pi / 2]

    # Compute assignments and display a few examples
    assign = voronoi_assign_angles(candidates, seeds)
    print("Sample Voronoi assignments (candidate → seed):")
    for i in range(0, len(candidates), 40):
        th = candidates[i]
        print(f"  θ={th: .3f} → seed={assign[th]: .3f}")

    # Hybrid best angle
    optimal = best_hybrid_angle(candidates, beam_center, beam_width, seeds, alpha=2.0)
    print(f"\nOptimal angle according to hybrid score: {optimal:.6f} rad")
    print(f"Fisher score at optimal angle: {fisher_score(optimal, beam_center, beam_width):.6e}")
    print(f"Assigned seed for optimal angle: {assign[optimal]: .6f}")
    print(f"Voronoi weight at optimal angle: {voronoi_weight(optimal, assign[optimal], alpha=2.0):.6f}")