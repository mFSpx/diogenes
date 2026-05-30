# DARWIN HAMMER — match 1907, survivor 5
# gen: 4
# parent_a: hybrid_perceptual_dedupe_hybrid_hybrid_rbf_su_m10_s7.py (gen3)
# parent_b: hybrid_hybrid_ternary_route_hybrid_voronoi_parti_m41_s2.py (gen3)
# born: 2026-05-29T23:39:49Z

"""Hybrid Perceptual‑RBF Voronoi‑Ternary Router

Parents
-------
* **Parent A** – perceptual hashing + RBF surrogate (hybrid_perceptual_dedupe…).
* **Parent B** – Voronoi partition + ternary minimum‑cost routing with a
  circuit‑breaker (hybrid_hybrid_ternary_route…).

Mathematical Bridge
-------------------
The bridge is built on the *hash space* that groups high‑dimensional feature
vectors into discrete clusters.  Each cluster is represented by a **seed**
located at the centroid of the points belonging to the cluster (using the
first two feature dimensions as a 2‑D spatial coordinate).  The set of seeds
forms a Voronoi diagram in the 2‑D plane.  Inside every Voronoi cell we run a
ternary minimum‑cost routing tree.  The edge cost combines a Euclidean term
with a Bayesian‑style failure estimate supplied by the **RBF surrogate**
trained on the points of the corresponding hash cluster:

    c(p, s) = λ·‖p – s‖₂ + μ·ĥ_s(p)

where `ĥ_s(p)` is the surrogate’s prediction for the failure probability of
seed *s* evaluated at the query feature vector *p*.  The circuit‑breaker
tracks recent failures per seed and disables a seed when its counter exceeds
a threshold.

The module therefore fuses:
* discrete Hamming‑based clustering (Parent A),
* continuous RBF regression (Parent A),
* Voronoi spatial partitioning (Parent B),
* ternary routing with Bayesian‑inspired costs (Parent B).

The public API demonstrates the hybrid workflow through three core
functions:
`compute_combined_hash`, `fit_surrogates_by_hash`, and `hybrid_route`.
"""

import math
import random
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Callable, Sequence, Any

import numpy as np

Vector = Sequence[float]
Point2D = Tuple[float, float]

# ----------------------------------------------------------------------
# Parent A – perceptual hashing utilities
# ----------------------------------------------------------------------


def compute_dhash(values: List[float]) -> int:
    """Difference hash: 1 bit per adjacent pair, 1 if decreasing."""
    bits = 0
    for i in range(len(values) - 1):
        bits = (bits << 1) | int(values[i] > values[i + 1])
    return bits


def compute_phash(values: List[float]) -> int:
    """Average hash limited to first 64 values."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits


def compute_combined_hash(values: List[float]) -> int:
    """Merge dhash (high‑order bits) and phash (low‑order bits) into one integer."""
    dh = compute_dhash(values)
    ph = compute_phash(values)
    # allocate enough bits for phash (max 64) then shift dh left
    return (dh << 64) | ph


def hamming_distance(a: int, b: int) -> int:
    """Return Hamming distance between two integers."""
    return bin(a ^ b).count("1")


# ----------------------------------------------------------------------
# Parent A – simple RBF surrogate (Gaussian kernel)
# ----------------------------------------------------------------------


class RBFSurrogate:
    """Gaussian RBF surrogate trained on (X, y)."""

    def __init__(self, X: np.ndarray, y: np.ndarray, gamma: float = 1.0, reg: float = 1e-6):
        """
        Parameters
        ----------
        X : (n_samples, n_features) array
        y : (n_samples,) array
        gamma : kernel width parameter
        reg : Tikhonov regularisation coefficient
        """
        self.gamma = gamma
        self.reg = reg
        self.X = X
        self.y = y
        self._fit()

    def _kernel(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Gaussian kernel matrix between rows of A and B."""
        sq_norms_A = np.sum(A ** 2, axis=1)[:, None]
        sq_norms_B = np.sum(B ** 2, axis=1)[None, :]
        dists = sq_norms_A + sq_norms_B - 2 * A @ B.T
        return np.exp(-self.gamma * dists)

    def _fit(self):
        K = self._kernel(self.X, self.X)
        # regularised solve for weights w: (K + reg*I) w = y
        n = K.shape[0]
        K_reg = K + self.reg * np.eye(n)
        self.alpha = np.linalg.solve(K_reg, self.y)

    def predict(self, X_new: np.ndarray) -> np.ndarray:
        """Predict for one or many new points."""
        K_new = self._kernel(X_new, self.X)
        return K_new @ self.alpha


# ----------------------------------------------------------------------
# Parent B – Voronoi / ternary router utilities
# ----------------------------------------------------------------------


def euclidean(p: Point2D, q: Point2D) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])


class CircuitBreaker:
    """Simple failure counter circuit‑breaker."""

    def __init__(self, threshold: int = 3):
        self.threshold = threshold
        self.counters: Dict[int, int] = {}

    def record_failure(self, seed_id: int):
        self.counters[seed_id] = self.counters.get(seed_id, 0) + 1

    def is_open(self, seed_id: int) -> bool:
        return self.counters.get(seed_id, 0) >= self.threshold

    def reset(self, seed_id: int):
        self.counters[seed_id] = 0


# ----------------------------------------------------------------------
# Hybrid data structures
# ----------------------------------------------------------------------


@dataclass
class Seed:
    """Represents a Voronoi seed linked to a hash cluster."""
    seed_id: int
    position: Point2D
    hash_key: int
    surrogate: RBFSurrogate
    breaker: CircuitBreaker


# ----------------------------------------------------------------------
# Core hybrid functions
# ----------------------------------------------------------------------


def fit_surrogates_by_hash(
    data: List[Tuple[List[float], float]],
    gamma: float = 1.0,
    reg: float = 1e-6,
) -> Tuple[Dict[int, RBFSurrogate], Dict[int, Point2D]]:
    """
    Cluster data by combined perceptual hash and fit an RBF surrogate per cluster.
    Also compute the 2‑D centroid (first two features) for each cluster to serve
    as a Voronoi seed position.

    Returns
    -------
    surrogates : dict mapping hash -> RBFSurrogate
    centroids  : dict mapping hash -> (x, y) centroid
    """
    clusters: Dict[int, List[Tuple[np.ndarray, float]]] = {}
    for feats, target in data:
        h = compute_combined_hash(feats)
        arr = np.asarray(feats, dtype=float)
        clusters.setdefault(h, []).append((arr, target))

    surrogates: Dict[int, RBFSurrogate] = {}
    centroids: Dict[int, Point2D] = {}

    for h, items in clusters.items():
        X = np.stack([it[0] for it in items])
        y = np.array([it[1] for it in items], dtype=float)
        surrogates[h] = RBFSurrogate(X, y, gamma=gamma, reg=reg)

        # centroid using first two dimensions (fallback to zeros if missing)
        if X.shape[1] >= 2:
            cx, cy = X[:, 0].mean(), X[:, 1].mean()
        else:
            cx = cy = 0.0
        centroids[h] = (float(cx), float(cy))

    return surrogates, centroids


def build_seeds(
    surrogates: Dict[int, RBFSurrogate],
    centroids: Dict[int, Point2D],
    breaker_threshold: int = 3,
) -> List[Seed]:
    """Create Seed objects from surrogates and centroids."""
    seeds: List[Seed] = []
    for idx, (h, surrogate) in enumerate(surrogates.items()):
        pos = centroids[h]
        breaker = CircuitBreaker(threshold=breaker_threshold)
        seeds.append(Seed(seed_id=idx, position=pos, hash_key=h, surrogate=surrogate, breaker=breaker))
    return seeds


def ternary_route(
    query_feat: List[float],
    query_pos: Point2D,
    seeds: List[Seed],
    lam: float = 1.0,
    mu: float = 1.0,
) -> List[int]:
    """
    Select up to three seed IDs with smallest hybrid cost.
    Cost = λ·EuclideanDist + μ·SurrogatePrediction.

    Returns the list of chosen seed IDs (ordered by increasing cost).
    """
    # Prepare feature vector for surrogate prediction
    x = np.asarray(query_feat, dtype=float).reshape(1, -1)

    costs: List[Tuple[float, int]] = []
    for seed in seeds:
        if seed.breaker.is_open(seed.seed_id):
            continue  # skip closed seeds
        dist = euclidean(query_pos, seed.position)
        pred = float(seed.surrogate.predict(x))  # surrogate gives failure prob estimate
        cost = lam * dist + mu * pred
        costs.append((cost, seed.seed_id))

    # sort by cost and take up to three
    costs.sort(key=lambda t: t[0])
    return [sid for _, sid in costs[:3]]


def hybrid_route(
    query_feat: List[float],
    query_pos: Point2D,
    seeds: List[Seed],
    lam: float = 1.0,
    mu: float = 1.0,
) -> List[int]:
    """
    Full hybrid operation:
    1. Compute hash of query feature vector.
    2. Find the seed whose hash is nearest (Hamming) – this seed’s surrogate
       is used as the “primary” model.
    3. Run ternary routing using *all* seeds but weighting the cost with the
       primary surrogate’s prediction (shared across seeds for simplicity).
    4. Return the selected seed IDs.
    """
    q_hash = compute_combined_hash(query_feat)

    # Find seed with minimal Hamming distance to query hash
    best_seed = min(
        seeds,
        key=lambda s: hamming_distance(q_hash, s.hash_key),
    )

    # Use the best seed’s surrogate for all cost evaluations (this mirrors the
    # idea that the nearest hash cluster provides the most relevant failure
    # estimate).
    primary_surrogate = best_seed.surrogate

    # Compute costs using the primary surrogate
    x = np.asarray(query_feat, dtype=float).reshape(1, -1)
    pred = float(primary_surrogate.predict(x))

    costs: List[Tuple[float, int]] = []
    for seed in seeds:
        if seed.breaker.is_open(seed.seed_id):
            continue
        dist = euclidean(query_pos, seed.position)
        cost = lam * dist + mu * pred
        costs.append((cost, seed.seed_id))

    costs.sort(key=lambda t: t[0])
    return [sid for _, sid in costs[:3]]


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Generate synthetic dataset
    random.seed(42)
    np.random.seed(42)

    N = 200
    dim = 5  # first two dims are spatial, rest are arbitrary
    data: List[Tuple[List[float], float]] = []
    for _ in range(N):
        feats = np.random.randn(dim).tolist()
        # target is a noisy function of the first dimension
        target = math.tanh(feats[0]) + 0.1 * np.random.randn()
        data.append((feats, float(target)))

    # Fit surrogates per hash cluster
    surrogates, centroids = fit_surrogates_by_hash(data, gamma=0.5, reg=1e-5)

    # Build seeds
    seeds = build_seeds(surrogates, centroids, breaker_threshold=5)

    # Query point
    query_feat = np.random.randn(dim).tolist()
    query_pos = (float(query_feat[0]), float(query_feat[1]))

    # Perform hybrid routing
    chosen = hybrid_route(query_feat, query_pos, seeds, lam=0.7, mu=0.3)

    print("Chosen seed IDs (hybrid route):", chosen)

    # Demonstrate ternary_route as a secondary operation
    chosen_ternary = ternary_route(query_feat, query_pos, seeds, lam=0.7, mu=0.3)
    print("Chosen seed IDs (pure ternary cost):", chosen_ternary)