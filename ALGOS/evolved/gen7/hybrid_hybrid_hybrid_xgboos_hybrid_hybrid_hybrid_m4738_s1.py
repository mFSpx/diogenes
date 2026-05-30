# DARWIN HAMMER — match 4738, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_xgboost_objec_hybrid_semantic_neig_m2287_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_sketch_m1264_s3.py (gen6)
# born: 2026-05-29T23:57:49Z

"""Hybrid Algorithm: XGBoost‑Endpoint‑NLMS‑Semantic‑Recovery + Sphericity‑Modulated Count‑Min Sketch with Tropical Broadcast

Parents:
- hybrid_hybrid_xgboost_objec_hybrid_semantic_neig_m2287_s1.py
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_sketch_m1264_s3.py

Mathematical Bridge:
Both parents expose a scalar “quality” factor that modulates downstream computations.
The XGBoost side uses *endpoint_health* × *recovery_priority* as a regularizer,
while the sketch‑bandit side uses a *sphericity_index* derived from object morphology
to adjust dimensionality reduction and leader‑election thresholds.

The bridge is built by treating the sphericity index as a universal scaling term
`σ`.  In the hybrid objective it multiplies the regularization factor
(endpoint_health·recovery_priority) and also scales the width of the
Count‑Min sketch, the Hoeffding bound threshold, and the simulated‑annealing
acceptance probability.  This yields a single unified system that simultaneously
learns tree leaf weights, reduces streaming data, and performs leader election
under a common geometric “shape” metric.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Core data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str

@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float

# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def sigmoid(x: float) -> float:
    """Standard logistic sigmoid."""
    return 1.0 / (1.0 + math.exp(-x))

def sphericity_index(morph: Morphology) -> float:
    """
    Simple geometric sphericity: volume divided by mass^(2/3).
    The exponent makes the index dimensionless.
    """
    volume = morph.length * morph.width * morph.height
    denom = max(morph.mass ** (2.0 / 3.0), 1e-9)
    return volume / denom

def tropical_max_plus(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Max‑plus matrix multiplication: (A ⊗ B)[i,j] = max_k (A[i,k] + B[k,j])
    """
    n, m = A.shape
    _, p = B.shape
    result = np.full((n, p), -np.inf)
    for i in range(n):
        for k in range(m):
            result[i, :] = np.maximum(result[i, :], A[i, k] + B[k, :])
    return result

def tropical_broadcast(adj: np.ndarray, steps: int = 5) -> np.ndarray:
    """
    Repeated max‑plus multiplication of the adjacency matrix with itself,
    then sum rows to obtain a broadcast strength vector.
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")
    power = adj.copy()
    for _ in range(steps - 1):
        power = tropical_max_plus(power, adj)
    # broadcast strength per node = max over reachable path weight
    return np.max(power, axis=1)

# ----------------------------------------------------------------------
# Count‑Min Sketch with sphericity‑modulated width
# ----------------------------------------------------------------------
class CountMinSketch:
    """
    Simple Count‑Min sketch where the width is scaled by the sphericity index.
    Depth is kept fixed (3) for simplicity.
    """
    def __init__(self, base_width: int, depth: int, sigma: float):
        self.depth = depth
        self.width = max(1, int(base_width * sigma))
        self.tables = np.zeros((depth, self.width), dtype=np.int64)
        random.seed(0)
        self.hash_seeds = [random.randint(1, 2**31 - 1) for _ in range(depth)]

    def _hash(self, item: int, seed: int) -> int:
        return (hash((item, seed)) % self.width)

    def update(self, item: int, increment: int = 1) -> None:
        for d, seed in enumerate(self.hash_seeds):
            idx = self._hash(item, seed)
            self.tables[d, idx] += increment

    def estimate(self, item: int) -> int:
        return min(
            self.tables[d, self._hash(item, self.hash_seeds[d])]
            for d in range(self.depth)
        )

# ----------------------------------------------------------------------
# Hybrid XGBoost objective components (scaled by sphericity)
# ----------------------------------------------------------------------
def binary_logistic_grad_hess(
    y_true: np.ndarray,
    margin: np.ndarray,
    endpoint_health: np.ndarray,
    recovery_priority: np.ndarray,
    sigma: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gradient and Hessian of binary logistic loss, scaled by
    endpoint_health × recovery_priority × sigma.
    """
    p = np.vectorize(sigmoid)(margin)
    g = p - y_true
    h = p * (1.0 - p)
    scale = endpoint_health * recovery_priority * sigma
    return g * scale, h * scale

def optimal_leaf_weight(
    gradient_sum: float,
    hessian_sum: float,
    reg_lambda: float = 1.0,
    endpoint_health: float = 1.0,
    recovery_priority: float = 1.0,
    sigma: float = 1.0,
) -> float:
    """
    Closed‑form optimal leaf weight for XGBoost, multiplied by the
    composite scaling factor.
    """
    base = -gradient_sum / (hessian_sum + reg_lambda)
    return base * endpoint_health * recovery_priority * sigma

def split_gain(
    left_gradient: float,
    left_hessian: float,
    right_gradient: float,
    right_hessian: float,
    *,
    reg_lambda: float = 1.0,
    gamma: float = 0.0,
    endpoint_health: float = 1.0,
    recovery_priority: float = 1.0,
    sigma: float = 1.0,
) -> float:
    """
    Gain of a candidate split, scaled by the same composite factor.
    """
    gain = (
        0.5
        * (
            (left_gradient ** 2) / (left_hessian + reg_lambda)
            + (right_gradient ** 2) / (right_hessian + reg_lambda)
        )
        - 0.5
        * ((left_gradient + right_gradient) ** 2)
        / (left_hessian + right_hessian + reg_lambda)
    )
    return (gain - gamma) * endpoint_health * recovery_priority * sigma

# ----------------------------------------------------------------------
# Hoeffding bound based leader election (modulated by sigma)
# ----------------------------------------------------------------------
def hoeffding_split_test(
    gains: np.ndarray,
    n_observations: int,
    delta: float = 0.05,
    sigma: float = 1.0,
) -> np.ndarray:
    """
    Returns a boolean mask indicating which nodes have statistically
    significant gains.  The bound threshold is divided by sigma,
    making higher sphericity more permissive.
    """
    epsilon = math.sqrt(math.log(1.0 / delta) / (2.0 * n_observations))
    threshold = epsilon / max(sigma, 1e-9)
    return gains > threshold

# ----------------------------------------------------------------------
# Simulated annealing acceptance (sigma‑aware)
# ----------------------------------------------------------------------
def anneal_accept(delta_E: float, temperature: float, sigma: float) -> bool:
    """
    Metropolis acceptance with temperature scaled by sigma.
    Higher sigma => higher effective temperature => more exploration.
    """
    if delta_E <= 0:
        return True
    effective_temp = temperature * max(sigma, 1e-9)
    prob = math.exp(-delta_E / effective_temp)
    return random.random() < prob

# ----------------------------------------------------------------------
# Bandit parameter update (sigma‑aware)
# ----------------------------------------------------------------------
def update_bandit(
    actions: List[BanditAction],
    updates: List[BanditUpdate],
    sigma: float,
) -> List[BanditAction]:
    """
    Simple Thompson‑sampling‑like update: increase expected reward for
    actions that received positive reward, scaled by sigma.
    """
    reward_map = {u.action_id: u.reward for u in updates}
    new_actions = []
    for a in actions:
        reward = reward_map.get(a.action_id, 0.0)
        # Scale the reward impact by sigma
        new_expected = a.expected_reward + sigma * reward
        # Shrink confidence bound proportional to sqrt of observations
        new_conf = max(0.0, a.confidence_bound - sigma * 0.1)
        new_actions.append(
            BanditAction(
                action_id=a.action_id,
                propensity=a.propensity,
                expected_reward=new_expected,
                confidence_bound=new_conf,
                algorithm=a.algorithm,
            )
        )
    return new_actions

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Morphology → sphericity
    morph = Morphology(length=2.0, width=1.5, height=1.0, mass=3.0)
    sigma = sphericity_index(morph)
    print(f"Sphericity σ = {sigma:.4f}")

    # 2. Tropical broadcast on a tiny graph
    adj = np.array(
        [
            [0.0, 1.2, -np.inf],
            [-np.inf, 0.0, 0.8],
            [0.5, -np.inf, 0.0],
        ]
    )
    b_vec = tropical_broadcast(adj, steps=4)
    print(f"Tropical broadcast strengths: {b_vec}")

    # Normalise broadcast to obtain endpoint health (0‑1)
    endpoint_health = (b_vec - b_vec.min()) / (b_vec.ptp() + 1e-9)
    recovery_priority = np.clip(1.0 - endpoint_health, 0.0, 1.0)  # opposite for demo

    # 3. Dummy XGBoost data
    y_true = np.array([0, 1, 0], dtype=float)
    margin = np.array([0.2, -0.1, 0.5], dtype=float)

    g, h = binary_logistic_grad_hess(
        y_true, margin, endpoint_health, recovery_priority, sigma
    )
    print(f"Gradients: {g}")
    print(f"Hessians: {h}")

    leaf_w = optimal_leaf_weight(
        gradient_sum=g.sum(),
        hessian_sum=h.sum(),
        reg_lambda=1.0,
        endpoint_health=endpoint_health.mean(),
        recovery_priority=recovery_priority.mean(),
        sigma=sigma,
    )
    print(f"Optimal leaf weight: {leaf_w:.4f}")

    gain = split_gain(
        left_gradient=g[0],
        left_hessian=h[0],
        right_gradient=g[1] + g[2],
        right_hessian=h[1] + h[2],
        reg_lambda=1.0,
        gamma=0.1,
        endpoint_health=endpoint_health.mean(),
        recovery_priority=recovery_priority.mean(),
        sigma=sigma,
    )
    print(f"Split gain: {gain:.4f}")

    # 4. Count‑Min sketch
    cms = CountMinSketch(base_width=100, depth=3, sigma=sigma)
    for item in [10, 20, 10, 30, 20, 10]:
        cms.update(item)
    est_10 = cms.estimate(10)
    est_20 = cms.estimate(20)
    print(f"CMS estimates – 10: {est_10}, 20: {est_20}")

    # 5. Hoeffding test on broadcast gains
    n_obs = 50
    significant = hoeffding_split_test(b_vec, n_obs, delta=0.05, sigma=sigma)
    print(f"Hoeffding significant nodes: {significant}")

    # 6. Simulated annealing acceptance
    accept = anneal_accept(delta_E=0.7, temperature=1.0, sigma=sigma)
    print(f"Annealing acceptance: {accept}")

    # 7. Bandit update
    actions = [
        BanditAction("a1", 0.5, 0.2, 0.1, "hybrid"),
        BanditAction("a2", 0.5, 0.5, 0.2, "hybrid"),
    ]
    updates = [
        BanditUpdate("ctx1", "a1", reward=1.0, propensity=0.5),
        BanditUpdate("ctx1", "a2", reward=-0.5, propensity=0.5),
    ]
    new_actions = update_bandit(actions, updates, sigma)
    print("Updated bandit actions:")
    for a in new_actions:
        print(asdict(a))

    sys.exit(0)