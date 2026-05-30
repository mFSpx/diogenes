# DARWIN HAMMER — match 1753, survivor 5
# gen: 5
# parent_a: hybrid_hybrid_hybrid_bandit_hybrid_hybrid_hybrid_m18_s2.py (gen4)
# parent_b: hybrid_hybrid_hybrid_geomet_hybrid_rbf_surrogate_m409_s2.py (gen3)
# born: 2026-05-29T23:38:39Z

"""Hybrid Algorithm combining RBF Surrogate Bandit with Geometric‑Algebra Graph Similarity.

Parents:
- hybrid_hybrid_hybrid_bandit_hybrid_hybrid_hybrid_m18_s2.py (Bandit + RBF surrogate + social interaction)
- hybrid_hybrid_hybrid_geomet_hybrid_rbf_surrogate_m409_s2.py (Geometric algebra + RBF similarity on graphs)

Mathematical bridge:
Both parents employ Gaussian radial basis functions (RBFs) as a universal approximator.
The fused algorithm uses an RBF surrogate model both to predict expected rewards for a
bandit policy and to compute similarity weights in a geometric‑algebra‑driven
Voronoi‑like partition of a point set.  The surrogate provides a shared kernel
matrix K_{ij}=exp(-ε²‖c_i−c_j‖²) that links the two domains, allowing the bandit’s
exploration‑exploitation term to be guided by geometric relationships encoded in
Multivector objects.
"""

import math
import random
import sys
import pathlib
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence, Tuple, List, Dict, Union

import numpy as np

# ----------------------------------------------------------------------
# Core utilities (shared by both parents)
# ----------------------------------------------------------------------
Vector = Sequence[float]
Point = Tuple[float, float]

def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian RBF kernel."""
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: Vector, b: Vector) -> float:
    """Euclidean distance between two vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

# ----------------------------------------------------------------------
# RBF Surrogate (from Parent A)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class RBFSurrogate:
    centers: List[Tuple[float, ...]]
    weights: List[float]
    epsilon: float = 1.0

    def predict(self, x: Vector) -> float:
        """Predict scalar output for input vector x."""
        return sum(
            w * gaussian(euclidean(x, c), self.epsilon)
            for w, c in zip(self.weights, self.centers)
        )

# ----------------------------------------------------------------------
# Multivector geometry (from Parent B)
# ----------------------------------------------------------------------
class Multivector:
    """Simple multivector for geometric algebra (grade‑agnostic)."""
    def __init__(self, components: Dict[frozenset, float], n: int):
        # store only non‑zero components
        self.components = {k: float(v) for k, v in components.items() if v != 0.0}
        self.n = int(n)

    def grade(self, k: int) -> 'Multivector':
        """Return the grade‑k part of the multivector."""
        return Multivector(
            {blade: coef for blade, coef in self.components.items() if len(blade) == k},
            self.n,
        )

    def scalar_part(self) -> float:
        """Return the scalar (grade‑0) part."""
        return self.components.get(frozenset(), 0.0)

    def __add__(self, other: 'Multivector') -> 'Multivector':
        result = dict(self.components)
        for blade, coef in other.components.items():
            result[blade] = result.get(blade, 0.0) + coef
        return Multivector({k: v for k, v in result.items() if v != 0.0}, self.n)

    def __repr__(self) -> str:
        if not self.components:
            return "Multivector(0)"
        terms = []
        for blade, coef in sorted(self.components.items(),
                                  key=lambda x: (len(x[0]), sorted(x[0]))):
            if blade:
                label = "e" + "".join(str(i) for i in sorted(blade))
            else:
                label = "1"
            terms.append(f"{coef:+.6g}*{label}")
        return "Multivector(" + " ".join(terms) + ")"

# ----------------------------------------------------------------------
# Bandit infrastructure (Parent A)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float          # probability of selection (for logging)
    expected_reward: float     # model‑based expectation
    confidence_bound: float    # UCB term
    algorithm: str             # name of sub‑algorithm that suggested it

_POLICY: Dict[str, List[float]] = {}
_STORE: Dict[str, float] = {}

def reset_policy() -> None:
    _POLICY.clear()
    _STORE.clear()

def _reward(a: str) -> float:
    total, n = _POLICY.get(a, [0.0, 0.0])
    return total / n if n else 0.0

# ----------------------------------------------------------------------
# 1️⃣ Build an RBF surrogate from data (fusion of both parents)
# ----------------------------------------------------------------------
def build_rbf_surrogate(
    data: List[Vector],
    targets: List[float],
    epsilon: float = 1.0,
) -> RBFSurrogate:
    """
    Fit an RBF surrogate by solving K w = y, where
    K_{ij} = exp(-ε²‖c_i−c_j‖²) and c_i are the data points.
    """
    if len(data) != len(targets):
        raise ValueError("data and targets must have same length")
    n = len(data)
    K = np.empty((n, n), dtype=float)
    for i, ci in enumerate(data):
        for j, cj in enumerate(data):
            K[i, j] = gaussian(euclidean(ci, cj), epsilon)
    y = np.asarray(targets, dtype=float)
    # Regularization for numerical stability
    reg = 1e-8 * np.eye(n)
    w = np.linalg.solve(K + reg, y)
    return RBFSurrogate(centers=[tuple(p) for p in data], weights=w.tolist(), epsilon=epsilon)

# ----------------------------------------------------------------------
# 2️⃣ Social interaction term (Parent A) – now also uses geometric info
# ----------------------------------------------------------------------
def social_interaction(
    x: Vector,
    g_best: Vector,
    k: int = 1,
    r1: Union[float, None] = None,
    seed: Union[int, str, None] = None,
) -> np.ndarray:
    """
    Compute a new position for particle x attracted toward global best g_best.
    The random coefficient r ∈ [0,1] can be seeded for reproducibility.
    """
    if len(x) != len(g_best):
        raise ValueError("x and g_best must share dimension")
    if k not in (1, 2):
        raise ValueError("k is normally 1 or 2")
    rng = random.Random(seed)
    r = rng.random() if r1 is None else r1
    if not (0.0 <= r <= 1.0):
        raise ValueError("random coefficient out of bounds")
    delta = np.array(g_best) - np.array(x)
    new_pos = np.array(x) + k * r * delta
    return new_pos

# ----------------------------------------------------------------------
# 3️⃣ Geometric similarity using RBFs (Parent B) + Multivector encoding
# ----------------------------------------------------------------------
def geometric_similarity(
    points: List[Vector],
    seeds: List[Vector],
    epsilon: float = 1.0,
) -> Tuple[np.ndarray, List[Multivector]]:
    """
    Compute an RBF‑based similarity matrix S where
        S_{ij} = exp(-ε²‖p_i−s_j‖²)
    and embed each point into a multivector whose scalar part equals the
    similarity sum to all seeds and whose bivector part encodes a simple
    oriented area using the first two dimensions.
    """
    if not seeds:
        raise ValueError("seeds must be non‑empty")
    m, n = len(points), len(seeds)
    S = np.empty((m, n), dtype=float)
    mv_list: List[Multivector] = []
    for i, p in enumerate(points):
        sim_sum = 0.0
        for j, s in enumerate(seeds):
            d = euclidean(p, s)
            val = gaussian(d, epsilon)
            S[i, j] = val
            sim_sum += val
        # Build a trivial multivector:
        # scalar = total similarity, bivector = oriented area in (x,y) plane
        components: Dict[frozenset, float] = {
            frozenset(): sim_sum,                     # grade‑0
        }
        if len(p) >= 2:
            # e12 component proportional to cross product of (p - seed_mean)
            seed_mean = np.mean(seeds, axis=0)
            area = (p[0] - seed_mean[0]) * (p[1] - seed_mean[1])
            components[frozenset({1, 2})] = area
        mv_list.append(Multivector(components, n=len(p)))
    return S, mv_list

# ----------------------------------------------------------------------
# 4️⃣ Hybrid bandit step that leverages geometry
# ----------------------------------------------------------------------
def hybrid_bandit_step(
    actions: List[BanditAction],
    surrogate: RBFSurrogate,
    context: Vector,
    alpha: float = 1.0,
) -> BanditAction:
    """
    Perform one decision step:
    - Predict expected reward for each action using the surrogate (the context
      vector is concatenated with the action identifier hash to obtain a
      numeric representation).
    - Compute a confidence bound using the surrogate’s kernel similarity to
      its centers (higher similarity → lower uncertainty).
    - Return the action with maximal Upper Confidence Bound (UCB).
    """
    if not actions:
        raise ValueError("no actions provided")
    best_ucb = -math.inf
    best_action = None
    for act in actions:
        # Encode action_id into a numeric vector via simple hash
        aid_vec = [float(ord(c)) for c in act.action_id[:len(context)]]
        # Pad / truncate to match context length
        if len(aid_vec) < len(context):
            aid_vec += [0.0] * (len(context) - len(aid_vec))
        else:
            aid_vec = aid_vec[:len(context)]
        x = [c + a for c, a in zip(context, aid_vec)]
        mu = surrogate.predict(x)
        # similarity to nearest center (proxy for variance)
        dists = [euclidean(x, c) for c in surrogate.centers]
        min_dist = min(dists) if dists else 0.0
        sigma = math.sqrt(min_dist)  # larger distance → larger uncertainty
        ucb = mu + alpha * sigma
        if ucb > best_ucb:
            best_ucb = ucb
            best_action = BanditAction(
                action_id=act.action_id,
                propensity=act.propensity,
                expected_reward=mu,
                confidence_bound=ucb,
                algorithm="HybridRBFBandit",
            )
    if best_action is None:
        raise RuntimeError("failed to select an action")
    return best_action

# ----------------------------------------------------------------------
# 5️⃣ Auxiliary graph utilities (Parent B)
# ----------------------------------------------------------------------
def distance(a: Point, b: Point) -> float:
    """Euclidean distance for 2‑D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def nearest(point: Point, seeds: List[Point]) -> int:
    """Return index of the nearest seed to point."""
    if not seeds:
        raise ValueError('seeds required')
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))

def assign(points: List[Point], seeds: List[Point]) -> Dict[int, List[Point]]:
    """
    Voronoi‑like assignment of each point to its nearest seed.
    Returns a dict mapping seed index → list of assigned points.
    """
    regions: Dict[int, List[Point]] = {i: [] for i in range(len(seeds))}
    for p in points:
        idx = nearest(p, seeds)
        regions[idx].append(p)
    return regions

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 1️⃣ Create synthetic data for the surrogate
    rng = np.random.default_rng(42)
    data = rng.uniform(-5, 5, size=(10, 3)).tolist()
    targets = (rng.normal(0, 1, size=10) * np.array([1, 2, 3])).tolist()
    surrogate = build_rbf_surrogate(data, targets, epsilon=0.8)

    # 2️⃣ Define a simple context vector (e.g., user features)
    context = [0.5, -1.2, 3.3]

    # 3️⃣ Define a set of candidate actions
    candidate_actions = [
        BanditAction("A", propensity=0.2, expected_reward=0.0, confidence_bound=0.0, algorithm="base"),
        BanditAction("B", propensity=0.5, expected_reward=0.0, confidence_bound=0.0, algorithm="base"),
        BanditAction("C", propensity=0.3, expected_reward=0.0, confidence_bound=0.0, algorithm="base"),
    ]

    # 4️⃣ Run a hybrid bandit decision
    chosen = hybrid_bandit_step(candidate_actions, surrogate, context, alpha=0.5)
    print("Chosen action:", chosen)

    # 5️⃣ Geometric similarity demo
    points = [tuple(p[:2]) for p in data]  # use first two dimensions as 2‑D points
    seeds = [(0.0, 0.0), (2.5, -2.5), (-3.0, 3.0)]
    S, mv = geometric_similarity(points, seeds, epsilon=0.6)
    print("Similarity matrix shape:", S.shape)
    print("First multivector:", mv[0])

    # 6️⃣ Social interaction example
    x = [1.0, 2.0, -1.0]
    g_best = [0.0, 0.0, 0.0]
    new_x = social_interaction(x, g_best, k=2, seed=123)
    print("New position after social interaction:", new_x)

    # 7️⃣ Voronoi assignment sanity check
    regions = assign(points, seeds)
    for idx, pts in regions.items():
        print(f"Seed {idx} has {len(pts)} assigned points")