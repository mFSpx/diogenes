# DARWIN HAMMER — match 4760, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_hybrid_geomet_hybrid_hybrid_fisher_m31_s5.py (gen4)
# parent_b: hybrid_serpentina_self_righ_hybrid_hybrid_fisher_m29_s1.py (gen4)
# born: 2026-05-29T23:58:06Z

import math
import sys
from dataclasses import dataclass
from typing import Dict, FrozenSet, Tuple, Iterable, List

import numpy as np


# ----------------------------------------------------------------------
# Morphological data container
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    """Geometric dimensions of a specimen (e.g., a turtle)."""
    length: float
    width: float
    height: float
    mass: float


# ----------------------------------------------------------------------
# Helper functions for geometric algebra
# ----------------------------------------------------------------------
def _sorted_blade(blade: Iterable[int]) -> Tuple[int, ...]:
    """Return a sorted tuple representing a blade."""
    return tuple(sorted(set(blade)))


def _permutation_parity(seq: List[int]) -> int:
    """Return +1 for even, -1 for odd permutation parity of seq."""
    parity = 1
    for i in range(len(seq)):
        for j in range(i + 1, len(seq)):
            if seq[i] > seq[j]:
                parity = -parity
    return parity


def _blade_product(
    a: Tuple[int, ...], b: Tuple[int, ...]
) -> Tuple[Tuple[int, ...], int]:
    """
    Compute the exterior (wedge) product of two blades.
    Returns (resulting blade as sorted tuple, sign).
    Identical basis vectors cancel (resulting in zero blade).
    """
    # concatenate and sort while tracking swaps
    combined = list(a) + list(b)
    # Count cancellations
    counts: Dict[int, int] = {}
    for idx in combined:
        counts[idx] = counts.get(idx, 0) + 1
    # Remove even occurrences (they cancel)
    remaining = [idx for idx, cnt in counts.items() if cnt % 2 == 1]
    # Determine sign from the original ordering to the sorted ordering
    # Build the original list without cancelled pairs
    original = [idx for idx in combined if counts[idx] % 2 == 1]
    sign = _permutation_parity(original + remaining)  # parity of reordering
    return tuple(sorted(remaining)), sign


def _geometric_product_blades(
    a: Tuple[int, ...], b: Tuple[int, ...]
) -> Tuple[Tuple[int, ...], int]:
    """
    Geometric product of two blades using the metric defined by the
    morphology (Euclidean by default). The sign comes from the
    antisymmetry of the exterior part; the inner part contributes no
    additional sign in a Euclidean metric.
    """
    # Exterior part
    wedge, sign = _blade_product(a, b)

    # The inner product removes common indices; each common index contributes
    # a factor of +1 in Euclidean space, so no sign change.
    # The resulting blade is just the wedge (since we are in Euclidean metric).
    return wedge, sign


# ----------------------------------------------------------------------
# Multivector class with full geometric algebra operations
# ----------------------------------------------------------------------
class Multivector:
    """
    Simple multivector implementation for an n‑dimensional Euclidean space.
    Blades are stored as frozensets of basis indices (1‑based).
    """

    def __init__(self, components: Dict[FrozenSet[int], float], n: int):
        self.n = int(n)
        # Remove near‑zero components for stability
        self.components: Dict[FrozenSet[int], float] = {
            k: float(v) for k, v in components.items() if abs(v) > 1e-15
        }

    # ------------------------------------------------------------------
    # Basic algebraic operations
    # ------------------------------------------------------------------
    def __add__(self, other: "Multivector") -> "Multivector":
        result = dict(self.components)
        for blade, coef in other.components.items():
            result[blade] = result.get(blade, 0.0) + coef
        return Multivector(result, self.n)

    def __sub__(self, other: "Multivector") -> "Multivector":
        result = dict(self.components)
        for blade, coef in other.components.items():
            result[blade] = result.get(blade, 0.0) - coef
        return Multivector(result, self.n)

    def __neg__(self) -> "Multivector":
        return Multivector({b: -c for b, c in self.components.items()}, self.n)

    def __mul__(self, scalar: float) -> "Multivector":
        """Scalar multiplication."""
        return Multivector({b: c * scalar for b, c in self.components.items()}, self.n)

    __rmul__ = __mul__

    # ------------------------------------------------------------------
    # Geometric product (∘) using the @ operator
    # ------------------------------------------------------------------
    def __matmul__(self, other: "Multivector") -> "Multivector":
        """Geometric product."""
        result: Dict[FrozenSet[int], float] = {}
        for blade_a, coef_a in self.components.items():
            for blade_b, coef_b in other.components.items():
                a_tuple = tuple(sorted(blade_a))
                b_tuple = tuple(sorted(blade_b))
                blade_res, sign = _geometric_product_blades(a_tuple, b_tuple)
                key = frozenset(blade_res)
                result[key] = result.get(key, 0.0) + sign * coef_a * coef_b
        return Multivector(result, self.n)

    # ------------------------------------------------------------------
    # Outer (wedge) product
    # ------------------------------------------------------------------
    def wedge(self, other: "Multivector") -> "Multivector":
        """Exterior (wedge) product."""
        result: Dict[FrozenSet[int], float] = {}
        for blade_a, coef_a in self.components.items():
            for blade_b, coef_b in other.components.items():
                a_tuple = tuple(sorted(blade_a))
                b_tuple = tuple(sorted(blade_b))
                # If blades share a basis vector the wedge is zero
                if set(a_tuple) & set(b_tuple):
                    continue
                combined = a_tuple + b_tuple
                sign = _permutation_parity(list(combined))
                key = frozenset(sorted(combined))
                result[key] = result.get(key, 0.0) + sign * coef_a * coef_b
        return Multivector(result, self.n)

    # ------------------------------------------------------------------
    # Inner product (grade‑lowering)
    # ------------------------------------------------------------------
    def inner(self, other: "Multivector") -> "Multivector":
        """Euclidean inner product (contraction)."""
        result: Dict[FrozenSet[int], float] = {}
        for blade_a, coef_a in self.components.items():
            for blade_b, coef_b in other.components.items():
                set_a = set(blade_a)
                set_b = set(blade_b)
                common = set_a & set_b
                if not common:
                    continue
                # Remove common indices
                remaining = (set_a ^ set_b)
                sign = _permutation_parity(
                    list(sorted(blade_a)) + list(sorted(blade_b))
                )
                key = frozenset(sorted(remaining))
                result[key] = result.get(key, 0.0) + sign * coef_a * coef_b
        return Multivector(result, self.n)

    # ------------------------------------------------------------------
    # Reversion (grade‑reversal)
    # ------------------------------------------------------------------
    def reverse(self) -> "Multivector":
        """Reverse the order of basis vectors in each blade."""
        result: Dict[FrozenSet[int], float] = {}
        for blade, coef in self.components.items():
            k = len(blade)
            sign = 1 if (k * (k - 1) // 2) % 2 == 0 else -1
            result[blade] = sign * coef
        return Multivector(result, self.n)

    # ------------------------------------------------------------------
    # Norm and utilities
    # ------------------------------------------------------------------
    def scalar_part(self) -> float:
        """Return the scalar (grade‑0) component."""
        return self.components.get(frozenset(), 0.0)

    def norm(self) -> float:
        """Euclidean norm √⟨M·M̃⟩₀."""
        prod = (self @ self.reverse()).scalar_part()
        return math.sqrt(abs(prod))

    def grade(self, k: int) -> "Multivector":
        """Extract the grade‑k part."""
        return Multivector(
            {b: c for b, c in self.components.items() if len(b) == k}, self.n
        )

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        terms = []
        for blade, coef in sorted(self.components.items(), key=lambda x: (len(x[0]), sorted(x[0]))):
            if blade:
                basis = "e" + "^".join(str(i) for i in sorted(blade))
            else:
                basis = "1"
            terms.append(f"{coef:.3g}{basis}")
        return " + ".join(terms) if terms else "0"


# ----------------------------------------------------------------------
# Morphology‑driven geometric quantities
# ----------------------------------------------------------------------
def sphericity_index(length: float, width: float, height: float) -> float:
    """Dimensionless sphericity (ratio of volume‑equivalent sphere diameter to length)."""
    if min(length, width, height) <= 0:
        raise ValueError("All dimensions must be positive.")
    volume = length * width * height
    sphere_diameter = (6 * volume / math.pi) ** (1.0 / 3.0)
    return sphere_diameter / length


def flatness_index(length: float, width: float, height: float) -> float:
    """Flatness as the ratio of planar dimensions to thickness."""
    if min(length, width, height) <= 0:
        raise ValueError("All dimensions must be positive.")
    return (length + width) / (2.0 * height)


# ----------------------------------------------------------------------
# Gaussian beam and Fisher information expressed via multivectors
# ----------------------------------------------------------------------
def gaussian_beam_mv(
    theta: float,
    center: float,
    width: float,
    sphericity: float,
) -> Multivector:
    """
    Return a scalar multivector representing the Gaussian intensity.
    The sphericity factor scales the amplitude.
    """
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    intensity = sphericity * math.exp(-0.5 * z * z)
    return Multivector({frozenset(): intensity}, n=1)


def fisher_score_mv(
    theta: float,
    center: float,
    width: float,
    sphericity: float,
    eps: float = 1e-12,
) -> float:
    """
    Fisher information for a scalar Gaussian parameterised by theta.
    The derivative is taken analytically; the result is a scalar.
    """
    beam = gaussian_beam_mv(theta, center, width, sphericity).scalar_part()
    intensity = max(beam, eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


# ----------------------------------------------------------------------
# Structural similarity (SSIM) enriched with morphological weighting
# ----------------------------------------------------------------------
def ssim(
    x: np.ndarray,
    y: np.ndarray,
    dynamic_range: float = 255.0,
    k1: float = 0.01,
    k2: float = 0.03,
    morphology: Morphology = Morphology(1.0, 1.0, 1.0, 1.0),
) -> float:
    """
    Classic SSIM multiplied by a morphology‑derived weight.
    The weight combines sphericity and a mass‑normalised factor,
    ensuring dimensionless scaling.
    """
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x = np.std(x, ddof=0)
    sigma_y = np.std(y, ddof=0)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))

    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2

    base_ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x ** 2 + sigma_y ** 2 + c2)
    )

    sphericity = sphericity_index(
        morphology.length, morphology.width, morphology.height
    )
    mass_factor = 1.0 / (1.0 + math.exp(-morphology.mass))  # smooth [0,1] map
    return base_ssim * sphericity * mass_factor


# ----------------------------------------------------------------------
# High‑level hybrid operation exposing the deeper integration
# ----------------------------------------------------------------------
def hybrid_operation(
    theta: float,
    center: float,
    width: float,
    morphology: Morphology,
) -> Tuple[float, Multivector, float]:
    """
    Compute:
      * Fisher information (scalar)
      * Gaussian beam as a multivector (scalar grade‑0)
      * Sphericity index (dimensionless)
    The multivector enables downstream geometric‑algebraic manipulation.
    """
    sphericity = sphericity_index(
        morphology.length, morphology.width, morphology.height
    )
    fisher = fisher_score_mv(theta, center, width, sphericity)
    beam_mv = gaussian_beam_mv(theta, center, width, sphericity)
    return fisher, beam_mv, sphericity


# ----------------------------------------------------------------------
# Convenience wrappers
# ----------------------------------------------------------------------
def multivector_operation(components: Dict[FrozenSet[int], float], n: int) -> Multivector:
    """Factory for a Multivector."""
    return Multivector(components, n)


def ssim_operation(x: np.ndarray, y: np.ndarray, morphology: Morphology) -> float:
    """SSIM with morphological weighting."""
    return ssim(x, y, morphology=morphology)


# ----------------------------------------------------------------------
# Demo / simple test harness
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Example morphology (approximate turtle carapace)
    morph = Morphology(length=0.45, width=0.30, height=0.10, mass=2.5)

    # Parameter for a Gaussian beam representing a sensory field
    th = 0.37
    ctr = 0.0
    w = 0.12

    fisher_val, beam_mv, sph = hybrid_operation(th, ctr, w, morph)
    print(f"Fisher information: {fisher_val:.6g}")
    print(f"Gaussian beam multivector: {beam_mv}")
    print(f"Sphericity index: {sph:.6g}")

    # Multivector arithmetic demo
    a = multivector_operation({frozenset([1]): 2.0, frozenset(): 1.0}, n=3)
    b = multivector_operation({frozenset([2, 3]): -0.5}, n=3)
    print("a =", a)
    print("b =", b)
    print("a @ b (geometric product) =", a @ b)
    print("a.wedge(b) =", a.wedge(b))
    print("a.inner(b) =", a.inner(b))
    print("||a|| =", a.norm())

    # SSIM example
    img1 = np.array([12, 45, 78, 123, 200], dtype=float)
    img2 = np.array([10, 47, 80, 119, 198], dtype=float)
    ssim_val = ssim_operation(img1, img2, morph)
    print(f"Weighted SSIM: {ssim_val:.6g}")