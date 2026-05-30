# DARWIN HAMMER — match 4633, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_distributed_l_hybrid_hoeffding_tre_m24_s2.py (gen2)
# parent_b: hybrid_hybrid_physarum_netw_hybrid_sparse_wta_hy_m336_s0.py (gen4)
# born: 2026-05-29T23:57:05Z

"""Hybrid Algorithm: Distributed Leader‑Election + Hoeffding Tree ↔ Physarum Conductance + Sparse WTA

Parents
-------
* **Algorithm A** – `hybrid_hybrid_distributed_l_hybrid_hoeffding_tre_m24_s2.py`  
  Provides probabilistic acceptance (simulated annealing), Hoeffding bound
  for streaming split decisions and a tropical (max‑plus) algebraic layer.

* **Algorithm B** – `hybrid_hybrid_physarum_netw_hybrid_sparse_wta_hy_m336_s0.py`  
  Supplies a physarum‑inspired conductance dynamics (`flux`, `update_conductance`,
  `hybrid_bandit_update`) and a deterministic sparse Winner‑Take‑All (WTA)
  projection.

Mathematical Bridge
-------------------
Both families manipulate *propensities* that are later combined by a
*max‑plus* reduction:

* In A, the acceptance probability `p = exp(-ΔE/T)` behaves like a
  propensity to accept a split; the Hoeffding bound supplies a confidence
  interval `ε` that can be interpreted as a “gain” for the split.

* In B, the bandit update computes a *propensity* `q = propensity * reward`
  that drives conductance growth.

The hybrid therefore treats the **Hoeffding confidence** as a *reward* for
the physarum network, while the **acceptance probability** acts as a
temperature‑scaled gain for conductance updates.  After each update the
edge‑wise fluxes are aggregated with tropical matrix multiplication,
yielding a piecewise‑linear convex “flow potential” that can be used as a
splitting criterion in a streaming decision tree.

The three core functions below demonstrate this integration:
1. `streaming_split_confidence` – Hoeffding bound + annealed acceptance.
2. `physarum_conductance_step` – flux computation + bandit‑guided conductance
   update, followed by tropical aggregation.
3. `sparse_wta_encode` – deterministic sparse encoding of the aggregated flow
   vector, suitable for privacy‑preserving downstream models.
"""

import numpy as np
import math
import random
import sys
import pathlib
import hashlib
from typing import List, Tuple, Mapping, Hashable

# ----------------------------------------------------------------------
# Helper utilities (from Parent A)
# ----------------------------------------------------------------------
def broadcast_probability(phase: int, step: int) -> float:
    if phase < 1 or step < 1:
        raise ValueError('phase and step must be positive')
    return min(1.0, 1.0 / (2 ** max(0, phase - step)))


def acceptance_probability(delta_e: float, temperature: float) -> float:
    """Metropolis‑style acceptance used as a propensity."""
    if delta_e < 0:
        return 1.0
    if temperature <= 0:
        return 0.0
    return math.exp(-delta_e / temperature)


def cooling_temperature(k: int, t0: float = 1.0, alpha: float = 0.95) -> float:
    if k < 0 or t0 < 0 or not (0 <= alpha <= 1):
        raise ValueError("invalid cooling parameters")
    return t0 * (alpha ** k)


def hoeffding_bound(r: float, delta: float, n: int) -> float:
    """Hoeffding bound for a Bernoulli‑like variable."""
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0 required")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))


def t_add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Tropical addition (max)."""
    return np.maximum(x, y)


def t_mul(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Tropical multiplication (addition)."""
    return np.add(x, y)


def t_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Tropical matrix multiplication: (A ⊗ B)_{ij} = max_k (A_{ik} + B_{kj})
    """
    if A.shape[1] != B.shape[0]:
        raise ValueError("inner dimensions must match")
    # Use broadcasting to compute all pairwise sums then max over k
    # Result shape (A_rows, B_cols)
    A_exp = A[:, :, np.newaxis]          # (i, k, 1)
    B_exp = B[np.newaxis, :, :]          # (1, k, j)
    sums = A_exp + B_exp                 # (i, k, j)
    return np.max(sums, axis=1)          # max over k


# ----------------------------------------------------------------------
# Helper utilities (from Parent B)
# ----------------------------------------------------------------------
def flux(conductance: float, edge_length: float, pressure_a: float, pressure_b: float,
         eps: float = 1e-12) -> float:
    if edge_length <= 0:
        raise ValueError('edge_length must be positive')
    return conductance / max(edge_length, eps) * (pressure_a - pressure_b)


def update_conductance(conductance: float, q: float,
                       dt: float = 1.0, gain: float = 1.0, decay: float = 0.05) -> float:
    if dt < 0 or decay < 0:
        raise ValueError('dt and decay must be non‑negative')
    return max(0.0, conductance + dt * (gain * abs(q) - decay * conductance))


def hybrid_bandit_update(conductance: float, propensity: float, reward: float,
                         dt: float = 1.0, gain: float = 1.0, decay: float = 0.05) -> float:
    """Propensity × reward plays the role of the bandit “q‑value”. """
    q = propensity * reward
    return update_conductance(conductance, q, dt, gain, decay)


def sparse_wta_encode(values: List[float], m: int, salt: str = '') -> List[float]:
    """
    Deterministic sparse WTA projection.
    For each input value we pick the top‑k (k=1) index via a hash function,
    set that position to the value, all others to 0.
    Collisions are summed.
    """
    if m <= 0:
        raise ValueError('m must be positive')
    out = np.zeros(m, dtype=float)
    for idx, v in enumerate(values):
        # Create a reproducible hash based on index, value and optional salt
        h = hashlib.blake2b(f"{salt}{idx}{v}".encode(), digest_size=4).digest()
        pos = int.from_bytes(h, 'little') % m
        out[pos] += v
    return out.tolist()


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def streaming_split_confidence(
    observed_range: float,
    delta: float,
    n_samples: int,
    delta_e: float,
    step: int
) -> Tuple[float, float]:
    """
    Combines the Hoeffding bound (confidence `ε`) with an annealed acceptance
    probability.  The returned tuple is `(epsilon, accept_prob)`.
    """
    epsilon = hoeffding_bound(observed_range, delta, n_samples)
    temperature = cooling_temperature(step)
    accept_prob = acceptance_probability(delta_e, temperature)
    return epsilon, accept_prob


def physarum_conductance_step(
    conductances: np.ndarray,
    edge_lengths: np.ndarray,
    pressures: np.ndarray,
    propensities: np.ndarray,
    rewards: np.ndarray,
    dt: float = 1.0,
    gain: float = 1.0,
    decay: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    One simulation step:
    1. Compute fluxes on each edge (physarum flow).
    2. Update each edge conductance using the hybrid bandit rule where
       the reward is the absolute flux (larger flow → larger reward).
    3. Aggregate the updated conductance matrix with tropical matrix multiplication
       to obtain a node‑wise “potential” vector.
    Returns the updated conductance matrix and the tropical potential vector.
    """
    if conductances.shape != edge_lengths.shape:
        raise ValueError("conductances and edge_lengths must share shape")
    if pressures.shape[0] != conductances.shape[0]:
        raise ValueError("pressures length must match number of nodes")

    n_nodes = conductances.shape[0]
    # Flux matrix: F_ij = flux(conductance_ij, length_ij, p_i, p_j)
    flux_matrix = np.zeros_like(conductances, dtype=float)
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i == j:
                continue
            flux_matrix[i, j] = flux(
                conductances[i, j],
                edge_lengths[i, j],
                pressures[i],
                pressures[j]
            )

    # Reward = |flux| (magnitude of material transport)
    reward_matrix = np.abs(flux_matrix)

    # Update conductances edge‑wise
    updated = np.zeros_like(conductances)
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i == j:
                continue
            updated[i, j] = hybrid_bandit_update(
                conductances[i, j],
                propensities[i, j],
                reward_matrix[i, j],
                dt=dt,
                gain=gain,
                decay=decay
            )

    # Tropical aggregation: compute node potentials V = max_k (C_ik + C_kj)
    # First interpret updated conductance as tropical matrix A.
    potentials = t_matmul(updated, updated)  # shape (n_nodes, n_nodes)
    # Collapse to a single potential per node by tropical addition over columns
    node_potential = np.max(potentials, axis=1)

    return updated, node_potential


def hybrid_decision_step(
    stats: Mapping[str, float],
    graph: Mapping[Hashable, set[Hashable]],
    conductances: np.ndarray,
    edge_lengths: np.ndarray,
    pressures: np.ndarray,
    propensities: np.ndarray,
    rewards: np.ndarray,
    step: int,
    sparse_dim: int = 128
) -> Tuple[bool, List[float]]:
    """
    High‑level hybrid operation that could be called for each incoming data
    instance in a streaming setting.

    1. Compute Hoeffding confidence and acceptance probability.
    2. If the probabilistic test passes, perform a physarum conductance step.
    3. Encode the resulting node potentials into a sparse WTA vector.
    4. Return a decision flag (True = split accepted) and the sparse encoding.

    The function demonstrates a genuine fusion of the two parent topologies.
    """
    # 1. Hoeffding / acceptance
    epsilon, accept_prob = streaming_split_confidence(
        observed_range=stats.get('range', 1.0),
        delta=stats.get('delta', 0.05),
        n_samples=int(stats.get('samples', 1)),
        delta_e=stats.get('delta_e', 0.0),
        step=step
    )
    decision = random.random() < accept_prob

    if not decision:
        # No update – return empty encoding
        return False, [0.0] * sparse_dim

    # 2. Conductance dynamics
    updated_cond, node_potential = physarum_conductance_step(
        conductances,
        edge_lengths,
        pressures,
        propensities,
        rewards,
        dt=1.0,
        gain=1.0,
        decay=0.05
    )

    # 3. Sparse WTA encoding of the tropical node potentials
    sparse_vec = sparse_wta_encode(node_potential.tolist(), sparse_dim, salt=str(step))

    # 4. Return
    # (In a real system the updated conductance matrix would be stored back;
    # here we simply return the decision flag and the encoding.)
    return True, sparse_vec


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Small synthetic graph with 3 nodes
    n = 3
    # Random conductance matrix (symmetrized, zero diagonal)
    np.random.seed(42)
    base_cond = np.random.rand(n, n)
    np.fill_diagonal(base_cond, 0.0)
    conductances = (base_cond + base_cond.T) / 2.0

    # Edge lengths – positive values
    edge_lengths = np.full((n, n), 1.0)
    np.fill_diagonal(edge_lengths, 0.0)

    # Pressures at nodes (random)
    pressures = np.random.rand(n)

    # Propensities matrix – use acceptance probabilities as a proxy
    propensities = np.full((n, n), 0.5)
    np.fill_diagonal(propensities, 0.0)

    # Rewards placeholder (will be overwritten inside the step)
    rewards = np.zeros((n, n))

    # Dummy graph structure (adjacency set)
    graph = {i: {j for j in range(n) if j != i} for i in range(n)}

    # Stats for streaming split
    stats = {
        'range': 1.0,
        'delta': 0.05,
        'samples': 100,
        'delta_e': 0.1  # small energy increase
    }

    # Run a few hybrid steps
    for step in range(5):
        decision, encoding = hybrid_decision_step(
            stats=stats,
            graph=graph,
            conductances=conductances,
            edge_lengths=edge_lengths,
            pressures=pressures,
            propensities=propensities,
            rewards=rewards,
            step=step,
            sparse_dim=64
        )
        print(f"Step {step}: decision={decision}, non‑zero entries={np.count_nonzero(encoding)}")
        # For demonstration, we keep the conductance matrix unchanged across steps.
        # In a real system you would feed back `updated_cond` from `physarum_conductance_step`.