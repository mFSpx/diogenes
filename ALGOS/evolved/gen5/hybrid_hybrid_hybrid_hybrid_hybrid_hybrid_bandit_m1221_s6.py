# DARWIN HAMMER — match 1221, survivor 6
# gen: 5
# parent_a: hybrid_hybrid_hybrid_bandit_hybrid_hybrid_doomsd_m988_s0.py (gen4)
# parent_b: hybrid_hybrid_bandit_router_hybrid_hybrid_bandit_m534_s1.py (gen4)
# born: 2026-05-29T23:34:30Z

"""Hybrid Bandit‑Schoolfield‑Geometric Algorithm

Parents:
- hybrid_hybrid_hybrid_bandit_hybrid_hybrid_doomsd_m988_s0 (bandit action selection & policy tracking)
- hybrid_hybrid_bandit_router_hybrid_hybrid_bandit_m534_s1 (Schoolfield temperature model & geometric operations)

Mathematical bridge:
The expected reward of each bandit arm is first enriched by a linear geometric projection
`score = context·W[:,a]` (matrix multiplication from the geometric parent).  
This score is then modulated by the temperature‑dependent developmental rate
`ρ(T)` from the Schoolfield model, producing a temperature‑scaled utility

    U_a = ρ(T)·( reward̂_a + context·W[:,a] )

which is finally used by the bandit selector (ε‑greedy / LinUCB‑like) to pick an action.
Thus the bandit policy, the geometric projection, and the thermodynamic scaling are
fused into a single decision‑making pipeline.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
import numpy as np

# ----------------------------------------------------------------------
# Core data structures (shared by both parents)
# ----------------------------------------------------------------------
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

@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0               # rate at 25 °C (298.15 K)
    delta_h_activation: float = 12_000.0
    t_low: float = 283.15
    t_high: float = 307.15
    delta_h_low: float = -45_000.0
    delta_h_high: float = 65_000.0
    r_cal: float = 1.987

# ----------------------------------------------------------------------
# Bandit policy storage (parent A)
# ----------------------------------------------------------------------
_POLICY: dict[str, list[float]] = {}

def reset_policy() -> None:
    """Clear all accumulated reward statistics."""
    _POLICY.clear()

def update_policy(updates: list[BanditUpdate]) -> None:
    """Accumulate raw rewards for each action."""
    for u in updates:
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0])  # [total_reward, count]
        stats[0] += float(u.reward)
        stats[1] += 1.0

def _raw_reward(action: str) -> float:
    """Mean reward observed for an action (0 if never selected)."""
    total, n = _POLICY.get(action, [0.0, 0.0])
    return total / n if n else 0.0

# ----------------------------------------------------------------------
# Schoolfield temperature model (parent B)
# ----------------------------------------------------------------------
def c_to_k(celsius: float) -> float:
    """Convert Celsius to Kelvin."""
    return celsius + 273.15

def developmental_rate(temp_k: float, params: SchoolfieldParams) -> float:
    """Schoolfield equation – returns a temperature‑dependent scaling factor."""
    if temp_k <= 0:
        raise ValueError("Kelvin temperature must be positive")
    # Numerator (Arrhenius term)
    num = params.rho_25 * (temp_k / 298.15) * math.exp(
        (params.delta_h_activation / params.r_cal) * (1 / 298.15 - 1 / temp_k)
    )
    # Denominator (low‑ and high‑temperature deactivation)
    denom = (1 + math.exp(
        (params.delta_h_low / params.r_cal) * (1 / params.t_low - 1 / temp_k)
    ) + math.exp(
        (params.delta_h_high / params.r_cal) * (1 / params.t_high - 1 / temp_k)
    ))
    return num / denom

# ----------------------------------------------------------------------
# Geometric projection utilities (derived from the geometric parent)
# ----------------------------------------------------------------------
def init_weight_matrix(num_features: int, num_actions: int, seed: int | None = 42) -> np.ndarray:
    """Create a random weight matrix W ∈ ℝ^{num_features×num_actions}."""
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0, scale=0.1, size=(num_features, num_actions))

def context_vector(context: dict[str, float], feature_order: list[str]) -> np.ndarray:
    """Transform a dict of features into a fixed‑order numpy column vector."""
    return np.array([float(context.get(k, 0.0)) for k in feature_order], dtype=float)

def geometric_scores(
    ctx_vec: np.ndarray,
    W: np.ndarray,
    actions: list[str],
    action_index: dict[str, int]
) -> dict[str, float]:
    """
    Compute the linear geometric projection for each action:
        s_a = ctx_vecᵀ · W[:,a]
    """
    scores = {}
    for a in actions:
        idx = action_index[a]
        scores[a] = float(ctx_vec @ W[:, idx])
    return scores

# ----------------------------------------------------------------------
# Hybrid bandit selector (core fusion)
# ----------------------------------------------------------------------
def select_action_hybrid(
    context: dict[str, float],
    actions: list[str],
    W: np.ndarray,
    params: SchoolfieldParams,
    temperature_c: float,
    algorithm: str = 'linucb',
    epsilon: float = 0.1,
    seed: int | str | None = 7
) -> BanditAction:
    """
    Choose an action using a temperature‑scaled utility that mixes:
        - empirical mean reward from the bandit policy,
        - geometric projection of the current context,
        - temperature scaling via the Schoolfield model.
    """
    if not actions:
        raise ValueError('actions required')
    rng = random.Random(seed)

    # Determine feature ordering from the weight matrix shape
    num_features = W.shape[0]
    feature_order = list(context.keys())[:num_features]  # truncate/pad as needed
    if len(feature_order) < num_features:
        # pad missing features with zero entries
        feature_order += [f"__pad_{i}" for i in range(num_features - len(feature_order))]

    ctx_vec = context_vector(context, feature_order)

    # Map actions to column indices of W
    action_index = {a: i for i, a in enumerate(actions)}
    # Geometric contribution
    geo_scores = geometric_scores(ctx_vec, W, actions, action_index)

    # Temperature scaling factor
    temp_factor = developmental_rate(c_to_k(temperature_c), params)

    # Compute temperature‑scaled utilities
    utilities = {}
    for a in actions:
        base = _raw_reward(a) + geo_scores[a]
        utilities[a] = temp_factor * base

    # Exploration‑exploitation policy
    if algorithm == 'epsilon_greedy' and rng.random() < epsilon:
        chosen = rng.choice(actions)
    elif algorithm == 'thompson':
        # Simple beta‑posterior sampling using observed successes/failures
        a_alpha = 1 + max(0.0, _raw_reward(chosen))
        b_beta = 1 + max(0.0, 1 - _raw_reward(chosen))
        chosen = max(actions, key=lambda a: rng.betavariate(a_alpha, b_beta))
    else:  # default LinUCB‑like scoring
        # Confidence term inversely proportional to sqrt(selection count)
        def conf(a: str) -> float:
            n = _POLICY.get(a, [0.0, 0.0])[1]
            return 0.1 / math.sqrt(1 + n)
        chosen = max(actions, key=lambda a: utilities[a] + conf(a))

    # Assemble return object
    prop = 1.0 / len(actions)
    exp_rw = _raw_reward(chosen)
    conf_bound = 1.0 / math.sqrt(1 + _POLICY.get(chosen, [0.0, 0.0])[1])
    return BanditAction(chosen, prop, exp_rw, conf_bound, algorithm)

# ----------------------------------------------------------------------
# Additional geometric helper – Voronoi assignment (illustrative)
# ----------------------------------------------------------------------
def voronoi_assign(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """
    Assign each point to the nearest center (Euclidean distance).
    Returns an array of region indices of shape (len(points),).
    """
    if points.ndim != 2 or centers.ndim != 2:
        raise ValueError("points and centers must be 2‑D arrays")
    # Compute squared Euclidean distances efficiently
    diff = points[:, None, :] - centers[None, :, :]   # shape (P, C, D)
    dists = np.einsum('pcd,pcd->pc', diff, diff)     # shape (P, C)
    return np.argmin(dists, axis=1)

# ----------------------------------------------------------------------
# Example usage / smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a tiny problem
    actions = ["A", "B", "C"]
    context_example = {"f1": 0.7, "f2": -1.2, "f3": 0.3}
    W = init_weight_matrix(num_features=3, num_actions=len(actions), seed=123)

    # Initialise policy with some synthetic updates
    reset_policy()
    synthetic_updates = [
        BanditUpdate("ctx1", "A", reward=1.0, propensity=1/3),
        BanditUpdate("ctx2", "B", reward=0.2, propensity=1/3),
        BanditUpdate("ctx3", "C", reward=0.5, propensity=1/3),
        BanditUpdate("ctx4", "A", reward=0.8, propensity=1/3),
    ]
    update_policy(synthetic_updates)

    # Choose an action at 22 °C using the hybrid selector
    params = SchoolfieldParams()
    chosen = select_action_hybrid(
        context=context_example,
        actions=actions,
        W=W,
        params=params,
        temperature_c=22.0,
        algorithm='linucb',
        epsilon=0.1,
        seed=42
    )
    print(f"Chosen action: {chosen}")

    # Demonstrate Voronoi assignment on random points
    pts = np.random.rand(5, 2)
    ctr = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]])
    regions = voronoi_assign(pts, ctr)
    print("Voronoi regions:", regions)