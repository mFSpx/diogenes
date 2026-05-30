# DARWIN HAMMER — match 3236, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_rbf_su_m2551_s0.py (gen5)
# parent_b: hybrid_hybrid_geometric_pro_hybrid_hybrid_hybrid_m1375_s4.py (gen5)
# born: 2026-05-29T23:48:43Z

import numpy as np
import math
import random
import hashlib
from dataclasses import dataclass
from typing import Dict, Tuple, List, Iterable

# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
Blade = Tuple[int, ...]                     # Sorted tuple of basis indices
Multivector = Dict[Blade, float]           # Sparse representation
HyperVector = np.ndarray                    # Bipolar vector (±1)

# ----------------------------------------------------------------------
# Utility functions for blade algebra
# ----------------------------------------------------------------------
def _blade_mul(b1: Blade, b2: Blade) -> Tuple[Blade, int]:
    """
    Multiply two blades using the exterior algebra rules.
    Returns (resulting_blade, sign) where sign is +1 or -1.
    Duplicate indices cancel (e_i * e_i = 1).
    """
    result: List[int] = list(b1)
    sign = 1
    for idx in b2:
        if idx in result:
            # cancel the duplicate basis vector
            result.remove(idx)
        else:
            # insertion sort to keep result ordered and count swaps
            pos = 0
            while pos < len(result) and result[pos] < idx:
                pos += 1
            # every shift of an existing element past idx flips the sign
            sign *= (-1) ** (len(result) - pos)
            result.insert(pos, idx)
    return tuple(result), sign


def geometric_product(mv1: Multivector, mv2: Multivector) -> Multivector:
    """
    Compute the geometric product of two multivectors.
    The result is a new sparse multivector.
    """
    result: Multivector = {}
    for b1, a in mv1.items():
        for b2, b in mv2.items():
            blade, s = _blade_mul(b1, b2)
            coeff = a * b * s
            if blade in result:
                result[blade] += coeff
                if abs(result[blade]) < 1e-12:
                    del result[blade]          # prune near‑zero entries
            else:
                result[blade] = coeff
    return result


def euclidean_mv(mv1: Multivector, mv2: Multivector) -> float:
    """
    Euclidean distance between two multivectors interpreted as vectors
    of their scalar coefficients (blades that are missing are treated as 0).
    """
    all_blades = set(mv1) | set(mv2)
    diff_sq = 0.0
    for b in all_blades:
        diff = mv1.get(b, 0.0) - mv2.get(b, 0.0)
        diff_sq += diff * diff
    return math.sqrt(diff_sq)


# ----------------------------------------------------------------------
# Hyperdimensional primitives
# ----------------------------------------------------------------------
def _hash_to_seed(value: float) -> int:
    """Deterministic hash of a float to an integer seed."""
    h = hashlib.sha256(str(value).encode("utf-8")).hexdigest()
    return int(h[:16], 16)


def scalar_to_hypervector(scalar: float, dim: int = 10_000) -> HyperVector:
    """
    Produce a deterministic bipolar hypervector from a scalar.
    The same scalar always yields the same hypervector.
    """
    rng = np.random.default_rng(_hash_to_seed(scalar))
    vec = rng.integers(0, 2, size=dim, endpoint=False)
    return np.where(vec == 0, -1, 1).astype(np.int8)


def bind(a: HyperVector, b: HyperVector) -> HyperVector:
    """Binding via element‑wise multiplication (XOR for bipolar vectors)."""
    return a * b


def bundle(vectors: Iterable[HyperVector]) -> HyperVector:
    """
    Bundling by summation followed by a sign‑threshold.
    The resulting vector is again bipolar.
    """
    summed = np.sum(np.stack(list(vectors)), axis=0)
    return np.where(summed >= 0, 1, -1).astype(np.int8)


# ----------------------------------------------------------------------
# Bandit component (epsilon‑greedy)
# ----------------------------------------------------------------------
@dataclass
class EpsilonGreedyBandit:
    n_actions: int
    epsilon: float = 0.1
    q_estimates: np.ndarray = None   # estimated value for each action

    def __post_init__(self) -> None:
        self.q_estimates = np.zeros(self.n_actions, dtype=float)

    def select_action(self) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        return int(np.argmax(self.q_estimates))

    def update(self, action: int, reward: float, alpha: float = 0.1) -> None:
        """Incremental update of the estimated value."""
        self.q_estimates[action] += alpha * (reward - self.q_estimates[action])


# ----------------------------------------------------------------------
# Core hybrid algorithm
# ----------------------------------------------------------------------
def hybrid_operation(
    feature_vec: List[float],
    bandit: EpsilonGreedyBandit,
    dim: int = 10_000,
) -> float:
    """
    Perform a deeper integration of geometric algebra and hyperdimensional
    computing within a contextual bandit loop.

    Steps
    -----
    1. Build a scalar multivector from the feature vector.
    2. Compute pairwise Euclidean distances between each scalar and the rest.
    3. Convert each distance to a deterministic hypervector.
    4. Bind each distance‑hypervector with a category‑specific hypervector.
    5. Bundle all bound vectors into a single context hypervector H.
    6. Choose an action via the bandit, evaluate reward as a sigmoid of the
       similarity between H and a pre‑generated action hypervector.
    7. Update the bandit with the observed reward.
    """
    # 1. scalar multivector (all scalars live on the grade‑0 blade ())
    mv: Multivector = {(): float(v) for v in feature_vec}

    # 2. pairwise distances (leave‑one‑out)
    distances: List[float] = []
    for i, val in enumerate(feature_vec):
        # multivector that contains only the i‑th scalar
        mv_i: Multivector = {(): val}
        # multivector of the remaining scalars
        mv_rest: Multivector = {(): sum(feature_vec) - val}
        distances.append(euclidean_mv(mv_i, mv_rest))

    # 3. deterministic hypervectors for each distance
    distance_hvs: List[HyperVector] = [
        scalar_to_hypervector(d, dim=dim) for d in distances
    ]

    # 4. category hypervectors (one per feature index)
    #    they are generated once and cached globally
    if not hasattr(hybrid_operation, "_category_hvs"):
        rng = np.random.default_rng(42)
        hybrid_operation._category_hvs = [
            rng.choice([-1, 1], size=dim).astype(np.int8) for _ in feature_vec
        ]
    category_hvs = hybrid_operation._category_hvs

    bound_vectors = [
        bind(dhv, chv) for dhv, chv in zip(distance_hvs, category_hvs)
    ]

    # 5. bundle into a single context hypervector
    context_hv = bundle(bound_vectors)

    # 6. bandit action selection
    action = bandit.select_action()

    # pre‑generated action hypervectors (deterministic, cached)
    if not hasattr(hybrid_operation, "_action_hvs"):
        rng = np.random.default_rng(999)
        hybrid_operation._action_hvs = [
            rng.choice([-1, 1], size=dim).astype(np.int8)
            for _ in range(bandit.n_actions)
        ]
    action_hv = hybrid_operation._action_hvs[action]

    # similarity as normalized dot product
    similarity = float(np.dot(context_hv, action_hv) / dim)

    # 7. reward via sigmoid (smooth probability)
    reward = 1.0 / (1.0 + math.exp(-similarity))

    # 8. update bandit
    bandit.update(action, reward)

    return reward


# ----------------------------------------------------------------------
# Simple smoke test
# ----------------------------------------------------------------------
def smoke_test() -> None:
    feature_vec = [0.5, 0.3, 0.2, 0.1]
    bandit = EpsilonGreedyBandit(n_actions=4, epsilon=0.2)

    for step in range(10):
        r = hybrid_operation(feature_vec, bandit)
        print(f"step {step:2d} | reward = {r:.4f} | q = {bandit.q_estimates}")

if __name__ == "__main__":
    smoke_test()