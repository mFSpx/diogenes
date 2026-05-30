# DARWIN HAMMER — match 1120, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_bandit_router_hybrid_hybrid_krampu_m9_s4.py (gen3)
# parent_b: hybrid_hybrid_hybrid_bandit_hybrid_hybrid_hybrid_m185_s2.py (gen4)
# born: 2026-05-29T23:32:53Z

"""Hybrid Algorithm: Bandit‑Router + Honeybee‑Store + Schoolfield Temperature + Graph Curvature

Parents:
- hybrid_hybrid_bandit_router_hybrid_hybrid_krampu_m9_s4.py (Bandit selector uses node‑wise Ollivier‑Ricci curvature as expected reward; honeybee store updates edge weights linearly)
- hybrid_hybrid_hybrid_bandit_hybrid_hybrid_hybrid_m185_s2.py (Schoolfield temperature model provides a temperature‑dependent learning‑rate λ_T; store provides a sigmoid scaling σ_S)

Mathematical Bridge:
1. Compute a curvature proxy ϰ_i for each node i from the adjacency matrix A.
2. The curvature serves as the raw expected reward for the bandit.
3. The honeybee store value S_i is transformed by σ_S = 1/(1+exp(‑S_i)) and multiplies the curvature, yielding a temperature‑aware expected reward ȓ_i = ϰ_i·σ_S.
4. The Schoolfield developmental rate λ_T(T) (temperature T in °C) scales the learning‑rate η = η₀·λ_T·σ_S.
5. After an action (node) a is chosen, the store is updated S_a ← S_a + r_a – baseline and the adjacency edges incident to a are linearly updated:
   A[a, j] ← A[a, j]·(1 + η·ΔS_a)  for all neighbours j.
Thus the bandit’s reward estimate, the store dynamics, the temperature‑dependent learning‑rate, and the linear test‑time training of the graph are tightly coupled in a single unified system.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Shared data structures (identical to parents)
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


# ----------------------------------------------------------------------
# Schoolfield temperature model (Parent B)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0               # rate at 25 °C (298.15 K)
    delta_h_activation: float = 12_000.0   # J mol⁻¹
    t_low: float = 283.15            # K
    t_high: float = 307.15           # K
    delta_h_low: float = -45_000.0   # J mol⁻¹
    delta_h_high: float = 65_000.0   # J mol⁻¹
    r_cal: float = 1.987             # cal mol⁻¹ K⁻¹ (≈ 8.314 J mol⁻¹ K⁻¹)


def celsius_to_kelvin(c: float) -> float:
    """Convert Celsius to Kelvin."""
    return c + 273.15


def schoolfield_rate(temp_c: float,
                    params: SchoolfieldParams = SchoolfieldParams()) -> float:
    """
    Schoolfield developmental rate λ_T(T) for a temperature given in Celsius.
    Implements the classic Schoolfield equation (simplified, without low/high terms).
    """
    T = celsius_to_kelvin(temp_c)
    # Convert r_cal to J mol⁻¹ K⁻¹
    R = params.r_cal * 4.184
    num = params.rho_25 * math.exp(-params.delta_h_activation / (R * (1.0 / T - 1.0 / 298.15)))
    # Low‑temperature term (optional, omitted for brevity)
    return num


# ----------------------------------------------------------------------
# Graph curvature proxy (Parent A)
# ----------------------------------------------------------------------
def curvature_proxy(adj: np.ndarray) -> np.ndarray:
    """
    Simple node‑wise curvature proxy.
    For node i: ϰ_i = 1 - (sum_j A_ij / max_degree)
    where max_degree is the maximum weighted degree in the graph.
    """
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError("Adjacency matrix must be square.")
    weighted_degree = adj.sum(axis=1)
    max_deg = weighted_degree.max() if weighted_degree.max() > 0 else 1.0
    curvature = 1.0 - (weighted_degree / max_deg)
    return curvature  # values in [0,1]


# ----------------------------------------------------------------------
# Bandit core (UCB style, uses curvature as expected reward)
# ----------------------------------------------------------------------
class BanditUCB:
    """
    LinUCB‑like bandit where the expected reward for node i is
        μ_i = curvature_i * σ_S
    and the confidence bound follows the classic UCB formula.
    """
    def __init__(self, n_actions: int, base_lr: float = 0.1):
        self.n = n_actions
        self.counts = np.zeros(n_actions, dtype=int)          # N_i
        self.total_counts = 0                                 # Σ_i N_i
        self.sum_rewards = np.zeros(n_actions, dtype=float)   # Σ rewards per action
        self.base_lr = base_lr

    def select(self,
               curvature: np.ndarray,
               sigma_s: np.ndarray,
               eta: float) -> Tuple[int, float]:
        """
        Select an action index.
        Returns (action_index, estimated_reward_before_update)
        """
        # Expected reward μ_i
        mu = curvature * sigma_s

        # Confidence bound
        total = max(self.total_counts, 1)
        conf = np.sqrt(2 * np.log(total) / np.maximum(self.counts, 1))

        # UCB score
        ucb = mu + conf

        # Break ties randomly
        max_val = ucb.max()
        candidates = np.where(ucb == max_val)[0]
        action = int(random.choice(candidates))

        return action, mu[action]

    def update(self, action: int, reward: float) -> None:
        self.counts[action] += 1
        self.total_counts += 1
        self.sum_rewards[action] += reward

    def estimated_reward(self, action: int) -> float:
        n = self.counts[action]
        return self.sum_rewards[action] / n if n > 0 else 0.0


# ----------------------------------------------------------------------
# Store dynamics (honeybee store, shared by both parents)
# ----------------------------------------------------------------------
def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def update_store(store: Dict[int, float],
                 action: int,
                 reward: float,
                 baseline: float = 0.0) -> float:
    """
    Update the scalar store for a given node.
    ΔS = reward - baseline
    Returns the new store value for the selected node.
    """
    delta = reward - baseline
    old = store.get(action, 0.0)
    new = old + delta
    store[action] = new
    return delta  # we also need the delta for adjacency update


def sigma_from_store(store: Dict[int, float]) -> np.ndarray:
    """σ_S = sigmoid(S_i) for all nodes i."""
    max_key = max(store.keys()) if store else -1
    # Ensure array length matches number of actions
    sigma = np.zeros(max_key + 1, dtype=float)
    for k, v in store.items():
        sigma[k] = sigmoid(v)
    return sigma


# ----------------------------------------------------------------------
# Linear adjacency (test‑time training) update (Parent A)
# ----------------------------------------------------------------------
def linear_adj_update(adj: np.ndarray,
                     action: int,
                     eta: float,
                     delta_store: float) -> None:
    """
    Perform a linear update on edges incident to the selected node.
    A[a, j] ← A[a, j]·(1 + η·ΔS)   for all j
    The matrix remains symmetric.
    """
    if delta_store == 0.0:
        return
    factor = 1.0 + eta * delta_store
    # Update row and column
    adj[action, :] *= factor
    adj[:, action] *= factor
    # Preserve symmetry (numerical errors may accumulate)
    adj[:] = (adj + adj.T) / 2.0


# ----------------------------------------------------------------------
# High‑level hybrid step combining all pieces
# ----------------------------------------------------------------------
def hybrid_step(context_id: str,
                adj: np.ndarray,
                store: Dict[int, float],
                temperature_c: float,
                bandit: BanditUCB,
                baseline_reward: float = 0.0) -> BanditUpdate:
    """
    Execute one hybrid iteration:
    1. Compute curvature proxy.
    2. Compute temperature‑dependent learning‑rate η.
    3. Compute σ_S from the store.
    4. Select a node via the bandit.
    5. Treat curvature as raw reward, apply store scaling → actual reward.
    6. Update store, adjacency, and bandit statistics.
    Returns a BanditUpdate dataclass for logging/analysis.
    """
    # 1. Curvature
    curv = curvature_proxy(adj)

    # 2. Learning‑rate scaling
    lambda_t = schoolfield_rate(temperature_c)          # λ_T(T)
    sigma_s = sigma_from_store(store)                  # σ_S per node
    # Ensure sigma_s length matches number of actions
    if sigma_s.shape[0] < bandit.n:
        sigma_s = np.pad(sigma_s, (0, bandit.n - sigma_s.shape[0]), constant_values=0.5)
    eta = bandit.base_lr * lambda_t * sigma_s.mean()   # a single scalar η for the update

    # 3. Bandit selection
    action, mu_pre = bandit.select(curv, sigma_s, eta)

    # 4. Reward calculation (curvature scaled by store sigmoid)
    raw_reward = curv[action]
    reward = raw_reward * sigma_s[action]

    # 5. Store update
    delta_store = update_store(store, action, reward, baseline=baseline_reward)

    # 6. Linear adjacency update
    linear_adj_update(adj, action, eta, delta_store)

    # 7. Bandit statistics update
    bandit.update(action, reward)

    # 8. Assemble BanditUpdate record
    update = BanditUpdate(
        context_id=context_id,
        action_id=str(action),
        reward=reward,
        propensity=1.0 / bandit.n  # uniform propensity for illustration
    )
    return update


# ----------------------------------------------------------------------
# Example usage / smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a small random weighted graph (5 nodes)
    np.random.seed(42)
    n_nodes = 5
    A = np.random.rand(n_nodes, n_nodes)
    A = (A + A.T) / 2.0          # make symmetric
    np.fill_diagonal(A, 0.0)    # no self‑loops

    # Initialise store (empty → all zeros)
    store_dict: Dict[int, float] = {}

    # Initialise bandit
    bandit = BanditUCB(n_actions=n_nodes, base_lr=0.05)

    # Run a few hybrid steps at different temperatures
    for step in range(10):
        temp = random.uniform(10.0, 30.0)   # Celsius
        upd = hybrid_step(
            context_id=f"step_{step}",
            adj=A,
            store=store_dict,
            temperature_c=temp,
            bandit=bandit,
            baseline_reward=0.0
        )
        print(f"{upd.context_id}: action={upd.action_id}, reward={upd.reward:.4f}, temp={temp:.1f}°C")
    print("\nFinal adjacency matrix:\n", np.round(A, 3))
    print("\nFinal store values:", {k: round(v, 3) for k, v in store_dict.items()})