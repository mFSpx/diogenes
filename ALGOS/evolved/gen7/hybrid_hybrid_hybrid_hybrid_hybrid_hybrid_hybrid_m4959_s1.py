# DARWIN HAMMER — match 4959, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_ollivier_ricci_curva_m1848_s0.py (gen6)
# parent_b: hybrid_hybrid_hybrid_bandit_hybrid_hybrid_rbf_su_m971_s0.py (gen4)
# born: 2026-05-29T23:59:03Z

"""Hybrid Algorithm integrating geometric algebra‑based Ollivier‑Ricci curvature with
temperature‑scaled RBF surrogate bandit decision making.

Parents:
- hybrid_hybrid_hybrid_hybrid_ollivier_ricci_curva_m1848_s0 (geometric product,
  lazy random walk, Ollivier‑Ricci curvature)
- hybrid_hybrid_hybrid_bandit_hybrid_hybrid_rbf_su_m971_s0 (Schoolfield temperature
  model, RBF surrogate, contextual bandit)

Mathematical bridge:
The geometric‑algebra distance between nodes supplies a metric space on which the
Ollivier‑Ricci curvature is evaluated.  The curvature values form a feature vector
that feeds an RBF surrogate.  The surrogate’s kernel width ε is modulated by the
Schoolfield developmental‑rate function, i.e. ε = ρ(T).  The surrogate predicts
expected rewards for bandit actions, which are then selected via an Upper‑Confidence‑Bound
policy.  Thus curvature → temperature‑scaled RBF → bandit creates a single unified
pipeline.
"""

import math
import random
import sys
import pathlib
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, FrozenSet

import numpy as np

# ----------------------------------------------------------------------
# Geometric‑Algebra utilities (from Parent A)
# ----------------------------------------------------------------------
def _blade_sign(indices: Tuple[int, ...]) -> Tuple[List[int], int]:
    lst = list(indices)
    sign = 1
    n = len(lst)
    for i in range(n):
        for j in range(n - 1 - i):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                # cancel duplicate index (e_i * e_i = 1)
                lst.pop(j)
                lst.pop(j)
                return lst, sign
    return lst, sign

def _multiply_blades(blade_a: FrozenSet[int], blade_b: FrozenSet[int]) -> Tuple[FrozenSet[int], int]:
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(tuple(combined))
    return frozenset(result), sign

def geometric_product_distance(blade_a: FrozenSet[int], blade_b: FrozenSet[int]) -> float:
    """A simple metric derived from the geometric product:
    distance = |blade_a Δ blade_b| (size of symmetric difference)."""
    return float(len(blade_a.symmetric_difference(blade_b)))

# ----------------------------------------------------------------------
# Random‑walk distribution (Parent A)
# ----------------------------------------------------------------------
def lazy_rw_distribution(adj: Dict[Any, List[Any]], node: Any, alpha: float = 0.5) -> Dict[Any, float]:
    neighbours = adj.get(node, [])
    deg = len(neighbours)
    dist = {node: alpha}
    if deg > 0:
        spread = (1.0 - alpha) / deg
        for nb in neighbours:
            dist[nb] = dist.get(nb, 0.0) + spread
    return dist

# ----------------------------------------------------------------------
# Ollivier‑Ricci curvature (Parent A) using the geometric distance metric
# ----------------------------------------------------------------------
def compute_ollivier_ricci_curvature(
    adj: Dict[Any, List[Any]],
    node_blades: Dict[Any, FrozenSet[int]],
) -> Dict[Tuple[Any, Any], float]:
    """Return curvature for each undirected edge (u, v)."""
    curvature: Dict[Tuple[Any, Any], float] = {}
    nodes = list(adj.keys())
    # pre‑compute lazy walk distributions
    walks = {n: lazy_rw_distribution(adj, n) for n in nodes}
    # pre‑compute pairwise geometric distances
    geo_dist = {
        (u, v): geometric_product_distance(node_blades[u], node_blades[v])
        for u in nodes for v in nodes
    }
    for u in nodes:
        for v in adj[u]:
            if (v, u) in curvature:
                continue  # already processed
            edge_len = geo_dist[(u, v)]
            if edge_len == 0:
                curvature[(u, v)] = 0.0
                continue
            # Wasserstein‑1 distance approximated by L1 on the walk measures
            w1 = 0.0
            for w in nodes:
                mu_u = walks[u].get(w, 0.0)
                mu_v = walks[v].get(w, 0.0)
                w1 += abs(mu_u - mu_v) * geo_dist[(u, w)]
            curvature[(u, v)] = 1.0 - (w1 / edge_len)
    return curvature

# ----------------------------------------------------------------------
# Schoolfield temperature model (Parent B)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0
    delta_h_activation: float = 12_000.0
    t_low: float = 283.15
    t_high: float = 307.15
    delta_h_low: float = -45_000.0
    delta_h_high: float = 65_000.0
    r_cal: float = 1.987

def c_to_k(celsius: float) -> float:
    return celsius + 273.15

def developmental_rate(temp_k: float, params: SchoolfieldParams = SchoolfieldParams()) -> float:
    """Schoolfield temperature‑dependent rate ρ(T)."""
    if temp_k <= 0 or params.rho_25 < 0:
        raise ValueError("temperature must be Kelvin‑positive and rho_25 non‑negative")
    numerator = params.rho_25 * (temp_k / 298.15) * math.exp(
        (params.delta_h_activation / params.r_cal) * ((1.0 / 298.15) - (1.0 / temp_k))
    )
    low = math.exp((params.delta_h_low / params.r_cal) * ((1.0 / params.t_low) - (1.0 / temp_k)))
    high = math.exp((params.delta_h_high / params.r_cal) * ((1.0 / params.t_high) - (1.0 / temp_k)))
    return numerator / (low * high)

# ----------------------------------------------------------------------
# RBF surrogate (Parent B) with temperature‑scaled kernel width
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class RBFSurrogate:
    centers: List[np.ndarray]          # each center is a curvature feature vector
    weights: List[float]               # linear weights for each kernel
    epsilon_base: float = 1.0          # base kernel width (scaled by temperature)

def gaussian_kernel(r: float, epsilon: float) -> float:
    return math.exp(-((epsilon * r) ** 2))

def rbf_surrogate_predict(
    x: np.ndarray,
    surrogate: RBFSurrogate,
    temp_celsius: float,
) -> float:
    """Predict a scalar value from curvature vector x."""
    temp_k = c_to_k(temp_celsius)
    eps = surrogate.epsilon_base * developmental_rate(temp_k)
    preds = 0.0
    for center, w in zip(surrogate.centers, surrogate.weights):
        r = np.linalg.norm(x - center)
        preds += w * gaussian_kernel(r, eps)
    return preds

# ----------------------------------------------------------------------
# Bandit structures (Parent B)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "Hybrid"

def ucb_score(action: BanditAction) -> float:
    """Upper‑Confidence‑Bound score used for selection."""
    return action.expected_reward + action.confidence_bound

def select_action(actions: List[BanditAction]) -> BanditAction:
    """Select the action with the highest UCB score."""
    return max(actions, key=ucb_score)

def update_action(
    action: BanditAction,
    reward: float,
    learning_rate: float = 0.1,
) -> BanditAction:
    """Return a new BanditAction with updated expected reward and confidence."""
    new_exp = (1 - learning_rate) * action.expected_reward + learning_rate * reward
    # shrink confidence bound as we gather more evidence (simple heuristic)
    new_conf = max(0.0, action.confidence_bound * (1 - learning_rate))
    return BanditAction(
        action_id=action.action_id,
        propensity=action.propensity,
        expected_reward=new_exp,
        confidence_bound=new_conf,
        algorithm=action.algorithm,
    )

# ----------------------------------------------------------------------
# Hybrid pipeline
# ----------------------------------------------------------------------
def hybrid_step(
    adj: Dict[Any, List[Any]],
    node_blades: Dict[Any, FrozenSet[int]],
    actions: List[BanditAction],
    surrogate: RBFSurrogate,
    temp_celsius: float,
) -> Tuple[BanditAction, Dict[Tuple[Any, Any], float]]:
    """
    1. Compute Ollivier‑Ricci curvature on the graph using geometric‑algebra distances.
    2. Assemble a curvature feature vector (mean curvature per edge).
    3. Predict a reward for each action via the temperature‑scaled RBF surrogate.
    4. Choose an action with UCB and return the updated action together with curvature map.
    """
    # 1. Curvature
    curvature_map = compute_ollivier_ricci_curvature(adj, node_blades)

    # 2. Feature vector: use mean and std of curvature values
    curv_vals = np.array(list(curvature_map.values()))
    feature_vec = np.array([curv_vals.mean(), curv_vals.std()])

    # 3. Predict a base reward for each action (same feature for all actions)
    base_reward = rbf_surrogate_predict(feature_vec, surrogate, temp_celsius)

    # 4. Inject the predicted reward into actions (simple additive model)
    enriched_actions = []
    for act in actions:
        enriched_reward = act.expected_reward + base_reward
        enriched = BanditAction(
            action_id=act.action_id,
            propensity=act.propensity,
            expected_reward=enriched_reward,
            confidence_bound=act.confidence_bound,
            algorithm=act.algorithm,
        )
        enriched_actions.append(enriched)

    # 5. Select and update
    chosen = select_action(enriched_actions)
    # Simulate observed reward (here we just reuse the predicted one plus noise)
    observed = chosen.expected_reward + random.gauss(0, 0.1)
    updated = update_action(chosen, observed)

    return updated, curvature_map

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple triangle graph
    adjacency = {
        "A": ["B", "C"],
        "B": ["A", "C"],
        "C": ["A", "B"],
    }

    # Assign each node a random blade (subset of {1,2,3,4})
    possible_basis = [1, 2, 3, 4]
    node_blades = {
        n: frozenset(random.sample(possible_basis, k=random.randint(1, 3)))
        for n in adjacency
    }

    # Initialise three dummy actions
    actions = [
        BanditAction(action_id="act1", propensity=0.33, expected_reward=0.0, confidence_bound=0.5),
        BanditAction(action_id="act2", propensity=0.33, expected_reward=0.0, confidence_bound=0.5),
        BanditAction(action_id="act3", propensity=0.34, expected_reward=0.0, confidence_bound=0.5),
    ]

    # Dummy RBF surrogate: two centers (mean/std pairs) with arbitrary weights
    surrogate = RBFSurrogate(
        centers=[np.array([0.0, 0.0]), np.array([0.5, 0.2])],
        weights=[1.2, -0.7],
        epsilon_base=0.8,
    )

    temperature_c = 25.0  # Celsius

    updated_action, curv_map = hybrid_step(
        adjacency,
        node_blades,
        actions,
        surrogate,
        temperature_c,
    )

    print("Updated action:", updated_action)
    print("Curvature map:", curv_map)