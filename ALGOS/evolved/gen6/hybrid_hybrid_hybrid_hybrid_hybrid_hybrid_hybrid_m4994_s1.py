# DARWIN HAMMER — match 4994, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_nlms_o_m1965_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_possum_filter_m1652_s0.py (gen4)
# born: 2026-05-29T23:59:11Z

"""Hybrid algorithm fusing:
- Parent A: RBF surrogate with NLMS weight adaptation.
- Parent B: Geometric‑algebra (multivector) representation and hybrid score from
  haversine distance + cosine similarity.

Mathematical bridge:
Each RBF centre is equipped with a geographic coordinate (lat,lon) and a
semantic feature vector.  The Gaussian kernel uses the Euclidean distance of the
semantic vectors, while a *hybrid score* h(i,j) = α·geo_sim + (1‑α)·cos_sim
modulates the contribution of that centre.  The NLMS update adapts the scalar
weights, and the whole weight vector is stored as a multivector (grade‑1
blades) to allow geometric‑algebraic operations if desired.  This unifies the
surrogate learning of Parent A with the distance‑semantic modulation of
Parent B."""
import math
import random
import sys
import pathlib
import numpy as np
from typing import List, Tuple, Sequence, Dict, FrozenSet

Vector = Sequence[float]
Coordinate = Tuple[float, float]  # (latitude, longitude)


def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian radial basis function."""
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: Vector, b: Vector) -> float:
    """Euclidean distance between two equal‑length vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def haversine_m(a: Coordinate, b: Coordinate) -> float:
    """Haversine distance (meters) between two lat/lon points."""
    R = 6371000.0  # Earth radius in metres
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    ha = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(min(1.0, math.sqrt(ha)))


def cosine_similarity(a: Vector, b: Vector) -> float:
    """Cosine similarity in [-1, 1]."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def hybrid_score(
    coord_a: Coordinate,
    coord_b: Coordinate,
    feat_a: Vector,
    feat_b: Vector,
    alpha: float = 0.5,
) -> float:
    """
    Combine geographic similarity (derived from haversine distance) and
    semantic similarity (cosine) into a single score in [0, 1].

    geo_sim = 1 - (d / d_max)   where d_max = π·R (half the globe)
    """
    # Geographic similarity
    d = haversine_m(coord_a, coord_b)
    d_max = math.pi * 6371000.0  # half‑circumference
    geo_sim = 1.0 - min(d / d_max, 1.0)

    # Semantic similarity
    cos_sim = (cosine_similarity(feat_a, feat_b) + 1.0) / 2.0  # map [-1,1] → [0,1]

    # Weighted blend
    return alpha * geo_sim + (1.0 - alpha) * cos_sim


def nlms_update(
    w: np.ndarray,
    x: np.ndarray,
    d: float,
    mu: float = 0.1,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Normalized Least‑Mean‑Squares (NLMS) weight update.

    w ← w + (μ / (ε + ‖x‖²)) · x · e
    where e = d – wᵀx
    """
    e = d - float(np.dot(w, x))
    norm_sq = np.dot(x, x) + eps
    step = (mu / norm_sq) * e
    w_new = w + step * x
    return w_new


# ----------------------------------------------------------------------
# Multivector utilities (grade‑1 blades only, sufficient for weight storage)
# ----------------------------------------------------------------------
def _blade_sign(indices: List[int]) -> Tuple[List[int], int]:
    """Return (sorted_blade, sign) after bubble‑sorting index list."""
    lst = list(indices)
    sign = 1
    n = len(lst)
    for i in range(n):
        for j in range(n - 1 - i):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                # duplicate index cancels out (e_i ∧ e_i = 0)
                lst.pop(j)
                lst.pop(j)  # next element shifts into position j
                return lst, sign
    return lst, sign


def _multiply_blades(blade_a: FrozenSet[int], blade_b: FrozenSet[int]) -> Tuple[FrozenSet[int], int]:
    """Multiply two basis blades (each a frozenset of indices)."""
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


class Multivector:
    """Simple grade‑1 multivector (vector) in Cl(n,0)."""

    def __init__(self, components: Dict[FrozenSet[int], float], n: int):
        self.n = int(n)
        # keep only non‑zero components
        self.components = {k: float(v) for k, v in components.items() if abs(v) > 1e-12}

    @staticmethod
    def from_array(arr: np.ndarray) -> "Multivector":
        """Create a multivector where each element corresponds to basis e_i."""
        comps = {frozenset({i}): float(v) for i, v in enumerate(arr)}
        return Multivector(comps, n=len(arr))

    def to_array(self) -> np.ndarray:
        """Extract a dense array (grade‑1 part)."""
        vec = np.zeros(self.n)
        for blade, coef in self.components.items():
            if len(blade) == 1:
                idx = next(iter(blade))
                vec[idx] = coef
        return vec

    def __add__(self, other: "Multivector") -> "Multivector":
        comps = self.components.copy()
        for blade, val in other.components.items():
            comps[blade] = comps.get(blade, 0.0) + val
        return Multivector(comps, self.n)

    def __mul__(self, scalar: float) -> "Multivector":
        comps = {blade: coef * scalar for blade, coef in self.components.items()}
        return Multivector(comps, self.n)

    __rmul__ = __mul__

    def __repr__(self) -> str:
        return f"Multivector({self.components})"


# ----------------------------------------------------------------------
# Core hybrid operations
# ----------------------------------------------------------------------
def rbf_predict(
    x_coord: Coordinate,
    x_feat: Vector,
    centers: List[Tuple[Coordinate, Vector]],
    weights: np.ndarray,
    epsilon: float = 1.0,
    alpha: float = 0.5,
) -> float:
    """
    Hybrid RBF surrogate prediction.

    For each centre i:
        r_i = Euclidean(feature_x, feature_i)
        h_i = hybrid_score(coord_x, coord_i, feature_x, feature_i, α)
        φ_i = h_i * Gaussian(r_i; ε)

    Output = Σ w_i φ_i
    """
    if len(centers) != len(weights):
        raise ValueError("centers and weights must have same length")
    phi = []
    for (c_coord, c_feat), w in zip(centers, weights):
        r = euclidean(x_feat, c_feat)
        h = hybrid_score(x_coord, c_coord, x_feat, c_feat, alpha)
        phi.append(h * gaussian(r, epsilon))
    phi_arr = np.array(phi)
    return float(np.dot(weights, phi_arr))


def hybrid_train_step(
    w_mv: Multivector,
    x_coord: Coordinate,
    x_feat: Vector,
    d: float,
    centers: List[Tuple[Coordinate, Vector]],
    mu: float = 0.1,
    epsilon: float = 1.0,
    alpha: float = 0.5,
) -> Multivector:
    """
    Perform one NLMS adaptation step using the hybrid RBF surrogate.
    Returns an updated Multivector containing the new weights.
    """
    w = w_mv.to_array()
    # Build the RBF feature vector φ(x)
    phi = []
    for c_coord, c_feat in centers:
        r = euclidean(x_feat, c_feat)
        h = hybrid_score(x_coord, c_coord, x_feat, c_feat, alpha)
        phi.append(h * gaussian(r, epsilon))
    phi_arr = np.array(phi)

    # NLMS update on the weight vector
    w_new = nlms_update(w, phi_arr, d, mu)

    return Multivector.from_array(w_new)


def hybrid_predict_mv(
    w_mv: Multivector,
    x_coord: Coordinate,
    x_feat: Vector,
    centers: List[Tuple[Coordinate, Vector]],
    epsilon: float = 1.0,
    alpha: float = 0.5,
) -> float:
    """Predict using a Multivector‑stored weight vector."""
    w = w_mv.to_array()
    return rbf_predict(x_coord, x_feat, centers, w, epsilon, alpha)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)

    # Generate synthetic centres (geographic + 3‑dim semantic)
    n_centres = 5
    centres = []
    for _ in range(n_centres):
        lat = random.uniform(-80, 80)
        lon = random.uniform(-180, 180)
        feat = np.random.randn(3).tolist()
        centres.append(((lat, lon), feat))

    # Initialise weight multivector (grade‑1)
    init_weights = np.random.randn(n_centres)
    w_mv = Multivector.from_array(init_weights)

    # Single training example
    x_coord = (10.0, 20.0)
    x_feat = np.random.randn(3).tolist()
    desired = 0.7  # target hybrid similarity

    # Perform a training step
    w_mv = hybrid_train_step(
        w_mv,
        x_coord,
        x_feat,
        desired,
        centres,
        mu=0.2,
        epsilon=1.2,
        alpha=0.6,
    )

    # Predict after adaptation
    y_pred = hybrid_predict_mv(w_mv, x_coord, x_feat, centres, epsilon=1.2, alpha=0.6)
    print(f"Predicted hybrid similarity: {y_pred:.6f}")
    # Ensure prediction is a finite number
    assert math.isfinite(y_pred)