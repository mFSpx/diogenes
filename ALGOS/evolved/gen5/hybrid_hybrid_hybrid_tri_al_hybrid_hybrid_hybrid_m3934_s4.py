# DARWIN HAMMER — match 3934, survivor 4
# gen: 5
# parent_a: hybrid_hybrid_tri_algo_cond_hybrid_doomsday_cale_m2675_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_geomet_hybrid_hybrid_hard_t_m1242_s1.py (gen4)
# born: 2026-05-29T23:52:38Z

"""
Hybrid module integrating:

- Parent A: signal_scores (entropy‑based quality metrics) and NLMS learning‑rate adaptation.
- Parent B: geometric product of multivectors within Voronoi partitions and curvature analysis.

Mathematical bridge:
The scalar **s** derived from Parent A (the *signal_score*) is used as a universal scaling factor.
It modulates:
1. The curvature metric of multivectors (grade‑2 norm) produced by Parent B.
2. The NLMS learning rate when updating multivector coefficients, allowing data‑driven
   adaptation of geometric representations.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
import numpy as np

# ---------- Parent A – signal quality utilities ----------

def _byte_entropy(data: bytes, sample: int = 8192) -> float:
    if not data:
        return 0.0
    chunk = data[:sample]
    entropy = 0.0
    for x in set(chunk):
        p_x = chunk.count(x) / len(chunk)
        entropy += -p_x * math.log(p_x, 2)
    return entropy / 8.0  # normalize to [0,1] approx


def signal_scores(
    data: bytes,
    status_code: int | None = None,
    mime: str = "",
    keyword_hits: int = 0,
    structural_links: int = 0,
) -> tuple[float, float]:
    """Return (signal, noise) scores in [0,1] based on content characteristics."""
    size = len(data)
    entropy = _byte_entropy(data)
    status_bonus = 0.18 if status_code and 200 <= status_code < 300 else -0.10
    mime_bonus = 0.12 if any(x in (mime or "").lower() for x in ("html", "json", "text", "xml")) else 0.02
    size_bonus = min(0.22, math.log1p(size) / 60.0)
    keyword_bonus = min(0.20, keyword_hits * 0.05)
    structure_bonus = min(0.16, structural_links * 0.01)

    signal = max(
        0.0,
        min(
            1.0,
            0.20
            + status_bonus
            + mime_bonus
            + size_bonus
            + keyword_bonus
            + structure_bonus
            + 0.12 * entropy,
        ),
    )
    noise = max(
        0.0,
        min(
            1.0,
            0.58
            - 0.22 * entropy
            - keyword_bonus
            - structure_bonus
            + (0.12 if size < 64 else 0.0),
        ),
    )
    return signal, noise


# ---------- Parent B – geometric algebra utilities ----------

Point = tuple[float, float]


def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def nearest(point: Point, seeds: list[Point]) -> int:
    if not seeds:
        raise ValueError("seeds required")
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))


def assign(points: list[Point], seeds: list[Point]) -> dict[int, list[Point]]:
    """Voronoi assignment of points to the nearest seed index."""
    regions: dict[int, list[Point]] = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions


def _blade_sign(indices):
    """Canonical ordering of a blade and its sign."""
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
                # duplicate index → blade vanishes (e_i ^ e_i = 0)
                lst.pop(j)
                lst.pop(j)
                n -= 2
                i -= 1  # re‑evaluate current position
                break
            j += 1
        i += 1
    return lst, sign


def _multiply_blades(blade_a, blade_b):
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


class Multivector:
    """Simple sparse multivector in n‑dimensional Euclidean space."""

    def __init__(self, components: dict[frozenset[int], float] | None = None, n: int = 3):
        self.n = int(n)
        self.components: dict[frozenset[int], float] = {}
        if components:
            for blade, coef in components.items():
                if coef != 0.0:
                    self.components[frozenset(blade)] = float(coef)

    def __add__(self, other: "Multivector") -> "Multivector":
        res = Multivector(self.components.copy(), self.n)
        for b, c in other.components.items():
            res.components[b] = res.components.get(b, 0.0) + c
            if abs(res.components[b]) < 1e-12:
                del res.components[b]
        return res

    def __sub__(self, other: "Multivector") -> "Multivector":
        neg = Multivector({b: -c for b, c in other.components.items()}, self.n)
        return self + neg

    def __mul__(self, other: "Multivector") -> "Multivector":
        """Geometric product."""
        result = Multivector(n=self.n)
        for blade_a, coef_a in self.components.items():
            for blade_b, coef_b in other.components.items():
                blade_res, sign = _multiply_blades(blade_a, blade_b)
                result.components[blade_res] = result.components.get(blade_res, 0.0) + sign * coef_a * coef_b
        # prune near‑zero entries
        result.components = {b: c for b, c in result.components.items() if abs(c) > 1e-12}
        return result

    def grade(self, k: int) -> dict[frozenset[int], float]:
        """Return components of a specific grade."""
        return {b: c for b, c in self.components.items() if len(b) == k}

    def norm(self) -> float:
        """Euclidean norm of all components (used for curvature proxy)."""
        return math.sqrt(sum(c * c for c in self.components.values()))

    def bivector_norm(self) -> float:
        """Norm restricted to grade‑2 (bivector) part."""
        return math.sqrt(sum(c * c for b, c in self.components.items() if len(b) == 2))

    def __repr__(self) -> str:
        if not self.components:
            return "0"
        terms = []
        for blade, coef in sorted(self.components.items(), key=lambda x: (len(x[0]), sorted(x[0]))):
            if not blade:
                basis = "1"
            else:
                basis = "^".join(f"e{i}" for i in sorted(blade))
            terms.append(f"{coef:.3g}{basis}")
        return " + ".join(terms)


# ---------- Hybrid operations ----------

def hybrid_curvature(multivector: Multivector, signal: float) -> float:
    """
    Curvature metric = signal‑scaled bivector norm.
    The signal score (0‑1) acts as the universal scaling factor **s**.
    """
    biv_norm = multivector.bivector_norm()
    return signal * biv_norm


def geometric_product_of_region_vectors(
    region_points: list[Point],
    seed: Point,
) -> Multivector:
    """
    Build a simple multivector from a region:
    - scalar part = number of points
    - vector part = sum of point coordinates (treated as e1, e2)
    - bivector part = oriented area proxy (cross product of summed vectors)
    """
    scalar = len(region_points)
    vec_sum = np.array([0.0, 0.0])
    for p in region_points:
        vec_sum += np.array(p)

    # oriented area using 2‑D wedge: x1*y2 - x2*y1
    area = 0.0
    for i in range(len(region_points)):
        for j in range(i + 1, len(region_points)):
            x1, y1 = region_points[i]
            x2, y2 = region_points[j]
            area += x1 * y2 - x2 * y1

    components = {
        frozenset(): float(scalar),                 # grade‑0 (scalar)
        frozenset({0}): float(vec_sum[0]),          # e1
        frozenset({1}): float(vec_sum[1]),          # e2
        frozenset({0, 1}): float(area),             # e1^e2 (bivector)
    }
    return Multivector(components, n=2)


def hybrid_nlms_update(
    weight: np.ndarray,
    input_vec: np.ndarray,
    desired: float,
    base_lr: float,
    signal: float,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Normalized Least‑Mean‑Squares update where the learning rate μ is
    modulated by the signal score: μ = base_lr * signal.
    """
    mu = base_lr * signal
    y = float(np.dot(weight, input_vec))
    e = desired - y
    norm_factor = eps + float(np.dot(input_vec, input_vec))
    weight = weight + (mu / norm_factor) * e * input_vec
    return weight


def hybrid_process(
    data: bytes,
    points: list[Point],
    seeds: list[Point],
    base_lr: float = 0.1,
) -> dict[int, dict]:
    """
    End‑to‑end hybrid routine:
    1. Compute signal score from raw data.
    2. Partition points into Voronoi regions.
    3. Build a multivector per region and evaluate curvature (scaled by signal).
    4. Perform an NLMS weight update on a dummy linear model using the same signal.
    Returns a dict keyed by region index containing curvature and updated weight vector.
    """
    signal, _ = signal_scores(data)

    # 2. Voronoi assignment
    regions = assign(points, seeds)

    # 3. Multivector construction & curvature
    results: dict[int, dict] = {}
    for idx, pts in regions.items():
        mv = geometric_product_of_region_vectors(pts, seeds[idx])
        curv = hybrid_curvature(mv, signal)

        # 4. Simple NLMS on a synthetic regression problem:
        #    input = [scalar part, vector e1 component, vector e2 component]
        input_vec = np.array([
            mv.components.get(frozenset(), 0.0),
            mv.components.get(frozenset({0}), 0.0),
            mv.components.get(frozenset({1}), 0.0),
        ])
        # Desired output is the (scaled) bivector component
        desired = mv.components.get(frozenset({0, 1}), 0.0) * signal

        # Initialise weight if not present
        weight = np.zeros_like(input_vec)
        weight = hybrid_nlms_update(weight, input_vec, desired, base_lr, signal)

        results[idx] = {
            "multivector": mv,
            "curvature": curv,
            "weight": weight,
            "input_vec": input_vec,
            "desired": desired,
        }
    return results


# ---------- Smoke test ----------

if __name__ == "__main__":
    # Synthetic byte payload
    payload = b"The quick brown fox jumps over the lazy dog" * 10

    # Random 2‑D points
    random.seed(42)
    pts = [(random.uniform(-10, 10), random.uniform(-10, 10)) for _ in range(50)]

    # Random seeds for Voronoi partitioning
    seed_pts = [(random.uniform(-10, 10), random.uniform(-10, 10)) for _ in range(5)]

    out = hybrid_process(payload, pts, seed_pts, base_lr=0.05)

    for idx, info in out.items():
        print(f"Region {idx}:")
        print(f"  Multivector : {info['multivector']}")
        print(f"  Curvature   : {info['curvature']:.4f}")
        print(f"  Updated weight vector: {info['weight']}")
        print()
    print("Hybrid processing completed without errors.")