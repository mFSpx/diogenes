# DARWIN HAMMER — match 1460, survivor 4
# gen: 6
# parent_a: hybrid_model_vram_scheduler_hybrid_hybrid_hybrid_m562_s1.py (gen5)
# parent_b: hybrid_hybrid_geometric_pro_hybrid_krampus_brain_m6_s1.py (gen2)
# born: 2026-05-29T23:36:32Z

"""Hybrid VRAM Scheduler & Geometric‑Curvature Module

This module fuses two parent algorithms:

* **Parent A** – a VRAM‑scheduler that builds a Gaussian prior over
  artifact memory footprints, using a curvature‑derived covariance
  (``curvature_weight``).

* **Parent B** – a geometric‑algebra engine that creates multivectors from
  data, partitions points via a Voronoi diagram and evaluates Ollivier‑Ricci
  curvature between partitions.

**Mathematical bridge**

Each artifact is mapped to a 2‑D point (e.g. via a hash).  The Voronoi
partition of these points yields regions.  For every region we build a
``Multivector`` (the geometric product of the region’s point‑vectors).  The
pairwise curvature between regions – obtained from the magnitude of the
geometric product – is used as the off‑diagonal entries of the prior
covariance matrix.  Thus the prior couples memory estimates according to
the geometric‑curvature relationship of the underlying Voronoi layout.

The resulting system can schedule VRAM while simultaneously providing a
geometric‑curvature justification for the coupling of artifacts.
"""

import math
import random
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – VRAM scheduler core
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class VramSlotPlan:
    artifact_id: str
    artifact_kind: str
    action: str
    estimated_mb: int
    reason: str
    detail: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)


def extract_evidence_features(text: str) -> Dict[str, int]:
    """Count occurrences of evidence‑related tokens in *text*."""
    matches = EVIDENCE_RE.findall(text)
    return {"evidence_count": len(matches)}


def curvature_weight(i: int, j: int, scale: float = 0.1) -> float:
    """Surrogate Ollivier‑Ricci curvature between two indices."""
    distance = abs(i - j)
    return math.exp(-scale * distance)


def build_prior(
    artifact_ids: List[str],
    base_memories: List[int],
    curvature_matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a Gaussian prior (mean vector, covariance matrix) for VRAM usage.

    *Mean* – static memory footprints (MB).
    *Covariance* – curvature‑derived coupling matrix supplied by the caller.
    """
    mean = np.array(base_memories, dtype=float)

    # Ensure covariance is positive‑semi‑definite by adding a small diagonal jitter.
    cov = curvature_matrix.copy()
    jitter = np.eye(len(artifact_ids)) * 1e-6
    cov += jitter
    return mean, cov


# ----------------------------------------------------------------------
# Parent B – Geometric product & Voronoi partitioning
# ----------------------------------------------------------------------
Point = Tuple[float, float]


def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def nearest(point: Point, seeds: List[Point]) -> int:
    if not seeds:
        raise ValueError("seeds required")
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))


def assign(points: List[Point], seeds: List[Point]) -> Dict[int, List[Point]]:
    """Voronoi assignment of *points* to the nearest *seeds*."""
    regions: Dict[int, List[Point]] = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions


def _blade_sign(indices):
    lst = list(indices)
    sign = 1
    n = len(lst)
    for i in range(n):
        for j in range(n - 1 - i):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                # duplicate index → cancel out (e_i ∧ e_i = 0)
                lst.pop(j)
                lst.pop(j)
                return lst, sign
    return lst, sign


def _multiply_blades(blade_a, blade_b):
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


class Multivector:
    """Simple exterior algebra with geometric product support."""

    def __init__(self, components: Dict[frozenset, float], n: int):
        # store only non‑zero components
        self.components = {k: float(v) for k, v in components.items() if abs(v) > 1e-12}
        self.n = int(n)

    def grade(self, k: int) -> "Multivector":
        return Multivector(
            {blade: coef for blade, coef in self.components.items() if len(blade) == k},
            self.n,
        )

    def scalar_part(self) -> float:
        return self.components.get(frozenset(), 0.0)

    def __add__(self, other: "Multivector") -> "Multivector":
        comp = self.components.copy()
        for blade, coef in other.components.items():
            comp[blade] = comp.get(blade, 0.0) + coef
        return Multivector(comp, self.n)

    def __sub__(self, other: "Multivector") -> "Multivector":
        comp = self.components.copy()
        for blade, coef in other.components.items():
            comp[blade] = comp.get(blade, 0.0) - coef
        return Multivector(comp, self.n)

    def __mul__(self, other: Any) -> "Multivector":
        """Scalar multiplication."""
        if isinstance(other, (int, float)):
            return Multivector({b: c * other for b, c in self.components.items()}, self.n)
        raise NotImplementedError("Only scalar multiplication is defined.")

    __rmul__ = __mul__

    def geometric_product(self, other: "Multivector") -> "Multivector":
        """Exterior (geometric) product using blade multiplication."""
        result: Dict[frozenset, float] = {}
        for blade_a, coef_a in self.components.items():
            for blade_b, coef_b in other.components.items():
                new_blade, sign = _multiply_blades(blade_a, blade_b)
                result[new_blade] = result.get(new_blade, 0.0) + sign * coef_a * coef_b
        return Multivector(result, self.n)

    def magnitude(self) -> float:
        """Euclidean norm of the coefficient vector."""
        return math.sqrt(sum(c * c for c in self.components.values()))

    def __repr__(self) -> str:
        if not self.components:
            return "Multivector(0)"
        terms = []
        for blade, coef in sorted(self.components.items(), key=lambda x: (len(x[0]), sorted(x[0]))):
            if blade:
                basis = "∧".join(f"e{i}" for i in sorted(blade))
                terms.append(f"{coef:.3g}{basis}")
            else:
                terms.append(f"{coef:.3g}")
        return " + ".join(terms)


def point_to_multivector(p: Point, dim: int = 2) -> Multivector:
    """Encode a 2‑D point as a grade‑1 multivector (vector)."""
    # e0 ↔ x, e1 ↔ y
    return Multivector({frozenset({0}): p[0], frozenset({1}): p[1]}, dim)


def region_multivector(region_points: List[Point]) -> Multivector:
    """Aggregate points of a Voronoi region into a single multivector."""
    if not region_points:
        return Multivector({}, 2)
    agg = Multivector({}, 2)
    for pt in region_points:
        agg = agg + point_to_multivector(pt)
    return agg


# ----------------------------------------------------------------------
# Hybrid operations – three demonstrative functions
# ----------------------------------------------------------------------
def compute_curvature_matrix_from_regions(
    regions: Dict[int, List[Point]],
) -> np.ndarray:
    """
    Build a curvature‑based matrix where entry (i, j) is
    exp(-scale * |‖G_i‖ - ‖G_j‖|) with G_i the geometric product magnitude
    of region *i*.
    """
    n = len(regions)
    mags = np.empty(n, dtype=float)
    for idx, pts in regions.items():
        mv = region_multivector(pts)
        mags[idx] = mv.magnitude()
    cov = np.zeros((n, n), dtype=float)
    scale = 0.2
    for i in range(n):
        for j in range(n):
            if i == j:
                cov[i, j] = mags[i] * 0.05  # small variance proportional to magnitude
            else:
                diff = abs(mags[i] - mags[j])
                cov[i, j] = math.exp(-scale * diff)
    return cov


def hybrid_vram_prior(
    artifact_ids: List[str],
    base_memories: List[int],
    artifact_points: List[Point],
    seed_points: List[Point],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    End‑to‑end hybrid prior construction:

    1. Voronoi partition of *artifact_points* using *seed_points*.
    2. Convert each region into a multivector and derive a curvature matrix.
    3. Feed the curvature matrix to ``build_prior``.
    """
    regions = assign(artifact_points, seed_points)
    curvature_mat = compute_curvature_matrix_from_regions(regions)
    mean, cov = build_prior(artifact_ids, base_memories, curvature_mat)
    return mean, cov


def geometric_product_demo(points: List[Point]) -> Multivector:
    """
    Demonstrate the geometric product by folding a list of point‑vectors.
    Returns the cumulative product G = v1 ⋅ v2 ⋅ … ⋅ vn.
    """
    if not points:
        return Multivector({}, 2)
    prod = point_to_multivector(points[0])
    for pt in points[1:]:
        prod = prod.geometric_product(point_to_multivector(pt))
    return prod


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Dummy artifacts
    artifact_ids = [f"art{i}" for i in range(5)]
    base_memories = [random.randint(100, 500) for _ in artifact_ids]

    # Random 2‑D points for each artifact
    random.seed(42)
    artifact_points = [(random.uniform(0, 10), random.uniform(0, 10)) for _ in artifact_ids]

    # Choose three seed points (could be a subset of artifact points)
    seed_points = artifact_points[:3]

    # 1. Evidence extraction demo
    txt = "The system recorded a log and a screenshot as proof of execution."
    print("Evidence features:", extract_evidence_features(txt))

    # 2. Geometric product demo
    gp = geometric_product_demo(artifact_points[:3])
    print("Geometric product of first three points:", gp)

    # 3. Hybrid prior construction
    mean_vec, cov_mat = hybrid_vram_prior(
        artifact_ids, base_memories, artifact_points, seed_points
    )
    print("Prior mean (MB):", mean_vec)
    print("Prior covariance matrix:\n", cov_mat)

    # Simple sanity check: covariance matrix must be symmetric
    assert np.allclose(cov_mat, cov_mat.T, atol=1e-9), "Covariance matrix not symmetric"
    print("Smoke test completed successfully.")