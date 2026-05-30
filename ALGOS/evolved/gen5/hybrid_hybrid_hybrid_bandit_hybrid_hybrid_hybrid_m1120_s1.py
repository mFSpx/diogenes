# DARWIN HAMMER — match 1120, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_bandit_router_hybrid_hybrid_krampu_m9_s4.py (gen3)
# parent_b: hybrid_hybrid_hybrid_bandit_hybrid_hybrid_hybrid_m185_s2.py (gen4)
# born: 2026-05-29T23:32:53Z

"""Hybrid Algorithm: Curvature‑Bandit‑Temperature‑Store Fusion

Parents:
- hybrid_hybrid_bandit_router_hybrid_hybrid_krampu_m9_s4.py (A)
- hybrid_hybrid_hybrid_bandit_hybrid_hybrid_hybrid_m185_s2.py (B)

Mathematical Bridge
-------------------
1. **Node curvature** `κ_i` is derived from a similarity graph `W`.  
   It serves as the *raw expected reward* for a bandit arm (node).

2. **Temperature factor** `λ_T` is obtained from the Schoolfield developmental‑rate
   model `ρ(T)`. It modulates a *global learning‑rate* `η`.

3. **Honeybee store** `S_k` is a scalar memory per context key.  
   Its sigmoid transform `σ_S = 1/(1+exp(-S_k))` scales the learning‑rate
   further, yielding the *effective learning‑rate*  

   `η_eff = η_0 · λ_T · σ_S`.

4. **Hybrid update** – after selecting node `i`:
   * The store is updated with the observed reward.
   * Edge weights incident to `i` are linearly adjusted:

   `W_{ij} ← W_{ij}·(1 + η_eff·Δ)`  where `Δ` is the reward delta.

Thus curvature, temperature, and store dynamics are mathematically intertwined
into a single unified decision‑learning loop.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Shared data structures (identical in both parents)
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
# Store (honeybee virtual store)
# ----------------------------------------------------------------------
_STORE: Dict[str, float] = {}  # key → stored scalar


def reset_store() -> None:
    """Clear the virtual store."""
    _STORE.clear()


def update_store(key: str, reward: float) -> None:
    """Add reward to the store entry for *key*."""
    _STORE[key] = _STORE.get(key, 0.0) + reward


def store_scaling(key: str) -> float:
    """Sigmoid scaling σ_S = 1/(1+exp(-S_k))."""
    s = _STORE.get(key, 0.0)
    return 1.0 / (1.0 + math.exp(-s))


# ----------------------------------------------------------------------
# Temperature model (Schoolfield) – Parent B
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


def c_to_k(celsius: float) -> float:
    """Convert Celsius to Kelvin."""
    return celsius + 273.15


def schoolfield_rate(temp_k: float,
                    params: SchoolfieldParams = SchoolfieldParams()) -> float:
    """
    Simplified Schoolfield developmental rate ρ(T).

    The full model contains low‑ and high‑temperature deactivation terms.
    For the purpose of this hybrid we keep the activation term and a
    temperature‑dependent denominator that mimics the original shape.
    """
    # activation term
    act = math.exp(-params.delta_h_activation /
                   (params.r_cal * temp_k))
    # low‑temperature deactivation (logistic)
    low = 1.0 / (1.0 + math.exp((params.t_low - temp_k) /
                               (params.r_cal * 0.1)))
    # high‑temperature deactivation (logistic)
    high = 1.0 / (1.0 + math.exp((temp_k - params.t_high) /
                                (params.r_cal * 0.1)))
    return params.rho_25 * act * low * high


def temperature_factor(celsius: float) -> float:
    """Convenient wrapper returning λ_T for a Celsius temperature."""
    return schoolfield_rate(c_to_k(celsius))


# ----------------------------------------------------------------------
# Graph curvature proxy – Parent A
# ----------------------------------------------------------------------
def node_curvature(adj: np.ndarray) -> np.ndarray:
    """
    Very lightweight Ollivier‑Ricci‑like proxy.

    For node i:
        κ_i = 1 - ( Σ_j w_ij / degree_i )
    where degree_i = Σ_j w_ij + ε to avoid division by zero.
    """
    eps = 1e-12
    degree = adj.sum(axis=1) + eps
    neighbor_sum = adj.sum(axis=1)
    curvature = 1.0 - (neighbor_sum / degree)
    return curvature  # shape (n_nodes,)


# ----------------------------------------------------------------------
# Bandit selection using curvature, temperature, and store scaling
# ----------------------------------------------------------------------
def select_action(curvature: np.ndarray,
                  propensities: np.ndarray,
                  store_key: str,
                  base_lr: float,
                  temp_c: float) -> Tuple[int, float]:
    """
    Compute an effective expected reward for each node and pick the best arm.

    expected_reward_i = κ_i * η_eff
    where η_eff = base_lr * λ_T * σ_S
    """
    λ_T = temperature_factor(temp_c)
    σ_S = store_scaling(store_key)
    η_eff = base_lr * λ_T * σ_S

    # combine curvature with propensity (propensity can be used as a bias)
    scores = curvature * η_eff * propensities
    chosen = int(np.argmax(scores))
    return chosen, η_eff


# ----------------------------------------------------------------------
# Linear test‑time training on the adjacency matrix
# ----------------------------------------------------------------------
def linear_adjacency_update(adj: np.ndarray,
                            node: int,
                            reward_delta: float,
                            η_eff: float) -> np.ndarray:
    """
    Perform a linear update on edges incident to *node*:

        W_ij ← W_ij * (1 + η_eff * Δ)

    The update is symmetric; only rows/cols of *node* are touched.
    """
    factor = 1.0 + η_eff * reward_delta
    # broadcast factor to the incident row and column
    adj[node, :] *= factor
    adj[:, node] *= factor
    # keep diagonal zero (no self‑loops)
    np.fill_diagonal(adj, 0.0)
    return adj


# ----------------------------------------------------------------------
# High‑level hybrid step
# ----------------------------------------------------------------------
def hybrid_step(adj: np.ndarray,
                propensities: np.ndarray,
                context_id: str,
                base_lr: float,
                temp_c: float) -> Tuple[np.ndarray, int, float]:
    """
    Execute one hybrid iteration:
    1. Compute node curvatures.
    2. Select an action (node) using the fused bandit rule.
    3. Observe a synthetic reward (here: curvature value plus noise).
    4. Update the store and compute the reward delta.
    5. Apply the linear adjacency update.
    Returns the updated adjacency, the chosen node, and the observed reward.
    """
    κ = node_curvature(adj)
    chosen, η_eff = select_action(κ, propensities,
                                  store_key=context_id,
                                  base_lr=base_lr,
                                  temp_c=temp_c)

    # synthetic reward: curvature plus Gaussian noise
    noise = random.gauss(0.0, 0.01)
    reward = float(κ[chosen] + noise)

    # store handling
    prev_store = _STORE.get(context_id, 0.0)
    update_store(context_id, reward)
    reward_delta = reward - prev_store  # incremental change

    # adjacency linear update
    adj = linear_adjacency_update(adj, chosen, reward_delta, η_eff)

    return adj, chosen, reward


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # reproducible demo
    random.seed(42)
    np.random.seed(42)

    # create a small undirected similarity graph (5 nodes)
    n = 5
    # random symmetric weight matrix with zeros on diagonal
    raw = np.random.rand(n, n)
    adj = (raw + raw.T) / 2.0
    np.fill_diagonal(adj, 0.0)

    # uniform propensities for simplicity
    propensities = np.ones(n)

    # reset store
    reset_store()

    # run a few hybrid steps
    for step in range(3):
        adj, node, rew = hybrid_step(adj,
                                     propensities,
                                     context_id="demo",
                                     base_lr=0.05,
                                     temp_c=22.0)  # 22 °C
        print(f"Step {step+1}: chosen node={node}, reward={rew:.4f}")
        print(f"Adjacency matrix after step {step+1}:\n{adj}\n")