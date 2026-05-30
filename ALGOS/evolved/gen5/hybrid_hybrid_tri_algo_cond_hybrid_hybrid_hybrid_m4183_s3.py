# DARWIN HAMMER — match 4183, survivor 3
# gen: 5
# parent_a: hybrid_tri_algo_conduit_hybrid_geometric_pro_m1414_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_geomet_hybrid_rbf_surrogate_m409_s4.py (gen3)
# born: 2026-05-29T23:53:59Z

"""Hybrid Algorithm: tri_algo_conduit × hybrid_geometric_product × hybrid_rbf_surrogate
Parent A: hybrid_tri_algo_conduit_hybrid_geometric_pro_m1414_s0.py  
Parent B: hybrid_hybrid_hybrid_geomet_hybrid_rbf_surrogate_m409_s4.py  

Mathematical Bridge
-------------------
Both parents rely on **Multivectors** as a unifying algebraic object.  
* Parent A uses multivectors to encode the Fisher‑information‑derived
  uncertainty of edges (via Gaussian beams).  
* Parent B uses multivectors to compute geometric products that quantify
  similarity between graph nodes, which then weight a Radial Basis Function
  (RBF) surrogate.

The hybrid therefore:
1. Represents each node/edge by a Multivector.
2. Computes a **geometric product** between node‑pair multivectors to obtain a
   similarity scalar.
3. Modulates the RBF kernel with the **Fisher score** derived from the Gaussian
   beam that models measurement uncertainty.

The result is a single pipeline that maps feature vectors → similarity
→ uncertainty‑aware RBF prediction, suitable for cost/uncertainty estimation
in tree‑like structures.  
"""

import math
import random
import sys
import pathlib
from typing import Hashable, Sequence, List, Dict, Set, Tuple
import numpy as np

# ----------------------------------------------------------------------
# Multivector utilities (common core from Parent B)
# ----------------------------------------------------------------------
def _blade_sign(indices: frozenset[int]) -> Tuple[List[int], int]:
    """Return a sorted list of basis indices and the sign incurred by
    swapping to achieve canonical order. Handles duplicate indices by
    cancelling them (grade reduction)."""
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
                # duplicate basis vectors cancel (e_i ^ e_i = 0)
                lst.pop(j)
                lst.pop(j)
                n -= 2
                sign *= 1
                continue
            j += 1
        i += 1
    return lst, sign

def _multiply_blades(blade_a: frozenset[int], blade_b: frozenset[int]) -> Tuple[frozenset[int], int]:
    """Geometric product of two basis blades (ignoring metric, i.e. Euclidean)."""
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(frozenset(combined))
    return frozenset(result), sign

class Multivector:
    """Simple Euclidean Clifford algebra (grade‑agnostic) implementation."""
    def __init__(self, components: Dict[frozenset[int], float] = None):
        # components maps a blade (frozenset of basis indices) to its scalar coefficient
        self.components: Dict[frozenset[int], float] = components if components else {}

    def __add__(self, other: "Multivector") -> "Multivector":
        result = self.components.copy()
        for blade, val in other.components.items():
            result[blade] = result.get(blade, 0.0) + val
        return Multivector(result)

    def __sub__(self, other: "Multivector") -> "Multivector":
        result = self.components.copy()
        for blade, val in other.components.items():
            result[blade] = result.get(blade, 0.0) - val
        return Multivector(result)

    def __mul__(self, other):
        """Scalar multiplication or geometric product."""
        if isinstance(other, (int, float)):
            return Multivector({blade: coeff * other for blade, coeff in self.components.items()})
        if isinstance(other, Multivector):
            result: Dict[frozenset[int], float] = {}
            for blade_a, coeff_a in self.components.items():
                for blade_b, coeff_b in other.components.items():
                    blade_res, sign = _multiply_blades(blade_a, blade_b)
                    result[blade_res] = result.get(blade_res, 0.0) + sign * coeff_a * coeff_b
            return Multivector(result)
        raise TypeError("Unsupported multiplication")

    def scalar_part(self) -> float:
        """Return the grade‑0 (scalar) component, defaulting to 0."""
        return self.components.get(frozenset(), 0.0)

    def __repr__(self) -> str:
        return f"Multivector({self.components})"

# ----------------------------------------------------------------------
# Gaussian beam and Fisher information (Parent A)
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Unnormalized Gaussian intensity."""
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a Gaussian beam parameterized by theta.
    Returns the scalar Fisher information (variance inverse)."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    # For a scalar parameter the Fisher information reduces to (dlnI/dθ)^2 * I
    return (derivative * derivative) / intensity

# ----------------------------------------------------------------------
# Graph utilities (from Parent B)
# ----------------------------------------------------------------------
Point = Tuple[float, float]

def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def nearest(point: Point, seeds: List[Point]) -> int:
    if not seeds:
        raise ValueError('seeds required')
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))

def assign(points: List[Point], seeds: List[Point]) -> Dict[int, List[Point]]:
    regions: Dict[int, List[Point]] = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions

# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def feature_to_multivector(features: Sequence[float]) -> Multivector:
    """Encode a real‑valued feature vector as a multivector.
    Each feature i becomes a scalar coefficient on the basis blade e_i."""
    comps = {frozenset({i}): float(v) for i, v in enumerate(features)}
    # also store the scalar (grade‑0) part as the mean of the vector
    comps[frozenset()] = float(np.mean(features))
    return Multivector(comps)

def similarity_via_geometric_product(v1: Multivector, v2: Multivector) -> float:
    """Scalar similarity derived from the geometric product of two multivectors.
    The scalar part (grade‑0) of the product is taken as the similarity measure."""
    prod = v1 * v2
    return prod.scalar_part()

def rbf_surrogate(
    query: Sequence[float],
    centers: List[Sequence[float]],
    weights: List[float],
    epsilon: float = 1.0,
    width: float = 1.0,
    center_gauss: float = 0.0,
) -> float:
    """
    Uncertainty‑aware RBF prediction.

    * `query` – feature vector of the point to evaluate.
    * `centers` – list of prototype feature vectors.
    * `weights` – linear coefficients for each RBF term (same length as centers).
    * `epsilon` – RBF shape parameter.
    * `width` & `center_gauss` – parameters for the Gaussian beam used in the
      Fisher score that modulates each RBF contribution.

    The algorithm:
    1. Convert `query` and each `center` to multivectors.
    2. Compute a geometric‑product similarity scalar.
    3. Compute a Gaussian RBF based on Euclidean distance.
    4. Scale the RBF by the Fisher information (uncertainty) and the similarity.
    5. Accumulate the weighted sum.
    """
    if len(centers) != len(weights):
        raise ValueError("centers and weights must have the same length")
    q_mv = feature_to_multivector(query)
    result = 0.0
    for c_feat, w in zip(centers, weights):
        c_mv = feature_to_multivector(c_feat)
        sim = similarity_via_geometric_product(q_mv, c_mv)  # similarity ∈ ℝ
        # Euclidean distance in feature space
        dist = math.sqrt(sum((qc - cc) ** 2 for qc, cc in zip(query, c_feat)))
        rbf = math.exp(- (dist / epsilon) ** 2)
        # Fisher information as uncertainty weight
        fisher = fisher_score(dist, center_gauss, width)
        result += w * rbf * sim * fisher
    return result

def hybrid_tree_cost(
    points: List[Point],
    seeds: List[Point],
    width: float = 0.5,
) -> float:
    """
    Compute a synthetic “cost” for a Voronoi‑like partition of `points`
    induced by `seeds`. The cost combines:
    * Material cost: sum of distances from each point to its assigned seed.
    * Uncertainty cost: Fisher information of those distances interpreted as
      Gaussian‑beam parameters.
    """
    regions = assign(points, seeds)
    total_cost = 0.0
    for seed_idx, region_pts in regions.items():
        seed = seeds[seed_idx]
        for pt in region_pts:
            d = distance(pt, seed)
            material = d  # linear material cost
            uncertainty = fisher_score(d, 0.0, width)  # treat center at 0
            total_cost += material * uncertainty
    return total_cost

def similarity_matrix(features: List[Sequence[float]]) -> np.ndarray:
    """
    Build a symmetric similarity matrix for a set of feature vectors using
    the geometric‑product scalar part as the kernel.
    """
    n = len(features)
    mat = np.zeros((n, n))
    mv_list = [feature_to_multivector(f) for f in features]
    for i in range(n):
        for j in range(i, n):
            sim = similarity_via_geometric_product(mv_list[i], mv_list[j])
            mat[i, j] = mat[j, i] = sim
    return mat

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple synthetic data
    random.seed(42)
    np.random.seed(42)

    # Points for the hybrid_tree_cost test
    pts = [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(30)]
    seed_pts = [(2.0, 2.0), (8.0, 8.0), (5.0, 5.0)]

    cost = hybrid_tree_cost(pts, seed_pts, width=0.8)
    print(f"Hybrid tree cost: {cost:.4f}")

    # RBF surrogate test
    query_vec = [0.5, -1.2, 3.3]
    centers = [
        [0.0, 0.0, 0.0],
        [1.0, -1.0, 2.5],
        [-0.5, 0.8, 3.0],
    ]
    weights = [1.2, -0.7, 0.3]
    pred = rbf_surrogate(query_vec, centers, weights, epsilon=1.5, width=0.6, center_gauss=0.0)
    print(f"RBF surrogate prediction: {pred:.6f}")

    # Similarity matrix test
    feature_set = [
        [1.0, 2.0],
        [2.0, 1.5],
        [0.5, 3.0],
    ]
    sim_mat = similarity_matrix(feature_set)
    print("Similarity matrix:")
    print(sim_mat)