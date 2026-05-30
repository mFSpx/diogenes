# DARWIN HAMMER — match 4198, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_bandit_m1221_s6.py (gen5)
# parent_b: hybrid_hybrid_hybrid_minimu_semantic_neighbors_m298_s3.py (gen3)
# born: 2026-05-29T23:54:05Z

"""Hybrid Bandit‑Tree‑Semantic Algorithm

Parents:
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_bandit_m1221_s6.py (Bandit with
  context projection and Schoolfield temperature scaling)
- hybrid_hybrid_hybrid_minimu_semantic_neighbors_m298_s3.py (Semantic
  similarity weighted minimum‑cost tree with Bayesian edge updates)

Mathematical Bridge
------------------
For each bandit arm *a* we define a context‑driven linear score
`c_a = context · W[:,a]`.  The Schoolfield model provides a temperature‑dependent
developmental rate ρ(T).  In the tree parent, every arm corresponds to a node
in a semantic graph; the edge probability *p_e* is obtained by a Bayesian
update that incorporates a semantic similarity weight
`sim(u,v) = cosine(u_vec, v_vec)`.  The final utility used by the bandit
selector is

    U_a = ρ(T) · ( μ_a + c_a ) · Π_{e∈path(root→a)} p_e

where μ_a is the current reward estimate for arm *a* and the product runs over
all edges on the path from the root node to the arm’s node.  This fuses the
linear‑geometric projection, the thermodynamic scaling, and the semantic‑tree
probabilities into a single decision‑making pipeline.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Core data structures (shared)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str                     # node identifier in the semantic tree
    expected_reward: float = 0.0      # μ_a, running mean of observed rewards
    propensity: float = 0.0           # probability of being selected (for logging)
    confidence_bound: float = 0.0    # optional UCB term
    algorithm: str = "HybridBanditTree"

@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float

@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0                # rate at 25 °C (298.15 K)
    delta_h_activation: float = 12_000.0
    t_low: float = 283.15
    t_high: float = 307.15
    delta_h_low: float = -45_000.0
    delta_h_high: float = 65_000.0
    r_cal: float = 1.987               # gas constant cal·K⁻¹·mol⁻¹

# ----------------------------------------------------------------------
# Tree structures for the semantic side
# ----------------------------------------------------------------------
NodeID = str
Edge = Tuple[NodeID, NodeID]          # (parent, child)

@dataclass
class TreeEdgeData:
    """Holds the Bayesian probability of an edge being “relevant”. """
    prior: float = 0.5                # initial belief
    likelihood: float = 0.5           # model likelihood (updated by similarity)
    false_positive: float = 0.1       # false‑positive rate
    posterior: float = field(init=False)

    def __post_init__(self):
        self.posterior = self.prior

# ----------------------------------------------------------------------
# Global storages (in‑memory)
# ----------------------------------------------------------------------
_POLICY: Dict[NodeID, List[float]] = {}           # reward statistics per arm
_EDGE_DATA: Dict[Edge, TreeEdgeData] = {}        # Bayesian edge states
_NODE_EMBEDDINGS: Dict[NodeID, np.ndarray] = {}  # semantic vectors per node
_W: np.ndarray = np.empty((0, 0))                # geometric projection matrix

# ----------------------------------------------------------------------
# Utility functions from Parent B
# ----------------------------------------------------------------------
def semantic_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two semantic vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """Marginal probability for Bayesian update."""
    return likelihood * prior + false_positive * (1.0 - prior)

def bayes_update(prior: float, likelihood: float, false_positive: float) -> float:
    """Posterior after observing evidence with given likelihood."""
    marginal = bayes_marginal(prior, likelihood, false_positive)
    if marginal <= 0:
        raise ValueError("Invalid marginal probability")
    return prior * likelihood / marginal

# ----------------------------------------------------------------------
# Schoolfield temperature scaling (Parent A)
# ----------------------------------------------------------------------
def schoolfield_rate(T: float, p: SchoolfieldParams = SchoolfieldParams()) -> float:
    """
    Compute the temperature‑dependent rate ρ(T) using the Schoolfield model.

    ρ(T) = ρ25 * exp(-ΔH_a / R * (1/T - 1/T_ref)) /
           (1 + exp(ΔH_low / R * (1/T - 1/T_low)) + exp(ΔH_high / R * (1/T - 1/T_high)))

    T is absolute temperature in Kelvin.
    """
    inv_T = 1.0 / T
    inv_Tref = 1.0 / 298.15
    inv_Tlow = 1.0 / p.t_low
    inv_Thigh = 1.0 / p.t_high

    num = math.exp(-p.delta_h_activation / p.r_cal * (inv_T - inv_Tref))
    denom = (
        1.0
        + math.exp(p.delta_h_low / p.r_cal * (inv_T - inv_Tlow))
        + math.exp(p.delta_h_high / p.r_cal * (inv_T - inv_Thigh))
    )
    return p.rho_25 * num / denom

# ----------------------------------------------------------------------
# Core hybrid operations
# ----------------------------------------------------------------------
def compute_context_score(context: np.ndarray, arm_id: NodeID) -> float:
    """Linear geometric projection of the context onto the arm's weight vector."""
    if _W.size == 0:
        raise RuntimeError("Projection matrix _W is not initialized")
    if arm_id not in _W.shape[1] * [None]:  # placeholder sanity check
        pass
    col_index = int(arm_id) if arm_id.isdigit() else None
    if col_index is None or col_index >= _W.shape[1]:
        raise KeyError(f"Arm {arm_id} not represented in projection matrix")
    return float(context @ _W[:, col_index])

def path_edge_product(node_id: NodeID, root_id: NodeID = "root") -> float:
    """
    Multiply posterior probabilities of edges along the unique path from the root
    to the given node.  The tree is assumed to be a directed acyclic graph where each
    node (except the root) has exactly one parent.
    """
    prod = 1.0
    current = node_id
    # Build parent map on first call if not present
    parent_map = {child: parent for (parent, child) in _EDGE_DATA.keys()}
    while current != root_id:
        parent = parent_map.get(current)
        if parent is None:
            raise KeyError(f"No parent found for node {current}")
        edge = (parent, current)
        prod *= _EDGE_DATA[edge].posterior
        current = parent
    return prod

def compute_arm_utility(
    context: np.ndarray,
    arm: BanditAction,
    temperature_K: float,
    root_id: NodeID = "root"
) -> float:
    """
    Unified utility:
        U_a = ρ(T) * ( μ_a + c_a ) * Π_edge posterior
    """
    rho = schoolfield_rate(temperature_K)
    c_a = compute_context_score(context, arm.action_id)
    path_factor = path_edge_product(arm.action_id, root_id)
    return rho * (arm.expected_reward + c_a) * path_factor

def update_edge_posteriors_from_similarity() -> None:
    """
    For every edge, compute a semantic similarity between the parent and child
    node embeddings and treat it as a likelihood.  Then perform a Bayesian update
    of the edge's posterior probability.
    """
    for (parent, child), data in _EDGE_DATA.items():
        vec_parent = _NODE_EMBEDDINGS.get(parent)
        vec_child = _NODE_EMBEDDINGS.get(child)
        if vec_parent is None or vec_child is None:
            continue  # skip edges without embeddings
        likelihood = semantic_similarity(vec_parent, vec_child)
        # Clamp likelihood to a reasonable interval to avoid degenerate updates
        likelihood = max(0.01, min(0.99, likelihood))
        new_posterior = bayes_update(data.prior, likelihood, data.false_positive)
        data.prior = data.posterior  # shift current posterior to prior for next step
        data.posterior = new_posterior

def select_action(
    context: np.ndarray,
    actions: List[BanditAction],
    temperature_K: float,
    epsilon: float = 0.1,
    root_id: NodeID = "root"
) -> BanditAction:
    """
    ε‑greedy selection based on the hybrid utility.
    With probability ε a random action is chosen; otherwise the action with the
    highest utility is returned.
    """
    if random.random() < epsilon:
        return random.choice(actions)

    utilities = [
        (compute_arm_utility(context, a, temperature_K, root_id), a) for a in actions
    ]
    _, best_action = max(utilities, key=lambda tup: tup[0])
    return best_action

def apply_bandit_update(updates: List[BanditUpdate]) -> None:
    """
    Update the running reward statistics for each arm based on observed rewards.
    Simple incremental mean is used.
    """
    for upd in updates:
        stats = _POLICY.setdefault(upd.action_id, [0.0, 0])  # [sum, count]
        stats[0] += upd.reward
        stats[1] += 1
        # Refresh expected_reward in the global actions list (if needed)
        # This function is deliberately lightweight; higher‑level code can rebuild
        # BanditAction objects from _POLICY when required.

# ----------------------------------------------------------------------
# Initialization helpers (for the smoke test)
# ----------------------------------------------------------------------
def _init_demo_environment(num_arms: int = 5, dim_context: int = 3) -> Tuple[np.ndarray, List[BanditAction]]:
    """
    Create a synthetic environment:
      - Random projection matrix _W (dim_context × num_arms)
      - Random semantic embeddings for each node (including root)
      - A simple star‑shaped tree (root → each arm)
      - Initial BanditAction objects with zero reward estimate
    Returns a dummy context vector and the list of actions.
    """
    global _W, _NODE_EMBEDDINGS, _EDGE_DATA

    _W = np.random.randn(dim_context, num_arms)

    # Create embeddings
    for i in range(num_arms):
        node_id = str(i)
        _NODE_EMBEDDINGS[node_id] = np.random.rand(dim_context)
        _NODE_EMBEDDINGS["root"] = np.random.rand(dim_context)  # ensure root exists

        # Edge from root to node
        edge = ("root", node_id)
        _EDGE_DATA[edge] = TreeEdgeData(prior=0.5, likelihood=0.5, false_positive=0.1)

    # Build actions
    actions = [
        BanditAction(action_id=str(i), expected_reward=0.0) for i in range(num_arms)
    ]

    # Dummy context
    context = np.random.randn(dim_context)

    return context, actions

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Initialise a tiny demo scenario
    ctx, acts = _init_demo_environment(num_arms=4, dim_context=3)

    # Simulate a single decision step at 298 K
    chosen = select_action(context=ctx, actions=acts, temperature_K=298.15, epsilon=0.0)
    print(f"Chosen action: {chosen.action_id}")

    # Pretend we observed a reward
    reward = random.random()
    upd = BanditUpdate(context_id="demo", action_id=chosen.action_id, reward=reward, propensity=1.0)
    apply_bandit_update([upd])

    # Update edge posteriors using semantic similarity
    update_edge_posteriors_from_similarity()

    # Show updated edge posteriors
    for edge, data in _EDGE_DATA.items():
        print(f"Edge {edge} posterior={data.posterior:.4f}")

    sys.exit(0)