# DARWIN HAMMER — match 2901, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_bandit_router_hybrid_hybrid_krampu_m9_s4.py (gen3)
# parent_b: hybrid_hybrid_hybrid_ternar_hybrid_fisher_locali_m1102_s0.py (gen4)
# born: 2026-05-29T23:46:40Z

import json
import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Data structures
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
# Bayesian store for failure probability (Beta prior)
# ----------------------------------------------------------------------
class FailurePosterior:
    """Beta(α, β) posterior for Bernoulli failure events per seed/action."""
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        self.alpha = alpha
        self.beta = beta

    def update(self, success: bool) -> None:
        if success:
            self.alpha += 1.0
        else:
            self.beta += 1.0

    @property
    def mean(self) -> float:
        """Posterior mean of failure probability."""
        return self.beta / (self.alpha + self.beta)


# ----------------------------------------------------------------------
# Global in‑memory stores
# ----------------------------------------------------------------------
_POLICY: Dict[str, Tuple[float, int]] = {}          # action_id → (cumulative_reward, count)
_FAILURE_POSTERIOR: Dict[str, FailurePosterior] = {}  # action_id → posterior


def reset_policy() -> None:
    _POLICY.clear()
    _FAILURE_POSTERIOR.clear()


def update_policy(updates: List[BanditUpdate]) -> None:
    """Incrementally update reward statistics and failure posteriors."""
    for u in updates:
        # reward stats
        total, cnt = _POLICY.get(u.action_id, (0.0, 0))
        _POLICY[u.action_id] = (total + float(u.reward), cnt + 1)

        # failure posterior – we treat reward==0 as a failure
        post = _FAILURE_POSTERIOR.setdefault(u.action_id, FailurePosterior())
        post.update(success=bool(u.reward))


def _reward_mean(action: str) -> float:
    total, cnt = _POLICY.get(action, (0.0, 0))
    return total / cnt if cnt else 0.0


def _failure_mean(action: str) -> float:
    return _FAILURE_POSTERIOR.get(action, FailurePosterior()).mean


# ----------------------------------------------------------------------
# Graph curvature (Ollivier‑Ricci proxy)
# ----------------------------------------------------------------------
def compute_node_curvature(adj: np.ndarray, node_idx: int) -> float:
    """
    Very lightweight Ollivier‑Ricci proxy:
        κ(i) = 1 - (W1(m_i, m_j) / d(i,j))
    where m_i is the uniform distribution over i's neighbours.
    For speed we approximate W1 by the L1 distance of degree vectors.
    """
    if adj.shape[0] != adj.shape[1]:
        raise ValueError("Adjacency matrix must be square.")
    if not (0 <= node_idx < adj.shape[0]):
        raise IndexError("node_idx out of bounds.")

    deg_i = adj[node_idx, :]
    # neighbours of i (including self‑loop for stability)
    neigh_i = np.where(deg_i > 0)[0]
    if len(neigh_i) == 0:
        return 0.0

    # average curvature to neighbours
    curv_sum = 0.0
    for j in neigh_i:
        if i == j:
            continue
        deg_j = adj[j, :]
        # L1 distance between the two degree profiles
        w1 = np.sum(np.abs(deg_i / deg_i.sum() - deg_j / deg_j.sum()))
        # shortest‑path distance approximated by 1 (edge weight) for unweighted graphs
        d_ij = 1.0
        curv_sum += 1.0 - w1 / d_ij
    return curv_sum / len(neigh_i)


def curvature_vector(adj: np.ndarray) -> np.ndarray:
    """Vectorised curvature for all nodes."""
    n = adj.shape[0]
    curv = np.empty(n, dtype=float)
    for i in range(n):
        curv[i] = compute_node_curvature(adj, i)
    return curv


# ----------------------------------------------------------------------
# Fisher information (quadratic form) – kept simple but vectorised
# ----------------------------------------------------------------------
def fisher_score(x: np.ndarray) -> float:
    """Quadratic Fisher information proxy: ‖x‖²."""
    return float(np.dot(x, x))


# ----------------------------------------------------------------------
# Weighted Voronoi / Ternary routing
# ----------------------------------------------------------------------
def weighted_cost(
    point: np.ndarray,
    seed: np.ndarray,
    curvature: float,
    failure_mean: float,
    lam: float = 1.0,
    mu: float = 1.0,
) -> float:
    """
    c(p, s) = λ·‖p‑s‖₂ + μ·curvature(s)·failure_mean(s)·fisher_score(p‑s)
    """
    eucl = np.linalg.norm(point - seed)
    fisher = fisher_score(point - seed)
    return lam * eucl + mu * curvature * failure_mean * fisher


def hybrid_routing(
    points: np.ndarray,
    seeds: np.ndarray,
    adj: np.ndarray,
    lam: float = 1.0,
    mu: float = 1.0,
) -> List[int]:
    """
    Returns the index of the chosen seed for each point.
    The curvature and failure means are pre‑computed for each seed
    (seed index coincides with node index in the adjacency matrix).
    """
    if seeds.shape[0] != adj.shape[0]:
        raise ValueError("Number of seeds must match number of graph nodes.")

    curv_vec = curvature_vector(adj)                     # κ(s) for each seed
    fail_vec = np.array([_failure_mean(str(i)) for i in range(seeds.shape[0])])

    assignments = []
    for p in points:
        costs = np.empty(seeds.shape[0], dtype=float)
        for s_idx, s in enumerate(seeds):
            costs[s_idx] = weighted_cost(
                p,
                s,
                curvature=curv_vec[s_idx],
                failure_mean=fail_vec[s_idx],
                lam=lam,
                mu=mu,
            )
        assignments.append(int(np.argmin(costs)))
    return assignments


# ----------------------------------------------------------------------
# Policy‑aware graph update (deep integration)
# ----------------------------------------------------------------------
def hybrid_update_policy(
    updates: List[BanditUpdate],
    adj: np.ndarray,
    seeds: np.ndarray,
    points: np.ndarray,
    lam: float = 1.0,
    mu: float = 1.0,
) -> None:
    """
    1. Update reward statistics & failure posteriors.
    2. Re‑compute node curvatures (expensive – done once per batch).
    3. Adjust adjacency weights using the routing costs as a soft‑feedback signal.
       Edge (i, j) is increased proportionally to how often seed i is selected for
       points that are close to seed j.
    """
    update_policy(updates)

    # 2. recompute curvature once – used later
    curv = curvature_vector(adj)

    # 3. soft‑feedback edge update
    # Build a histogram of (point → chosen seed) using current curvatures & failures
    assignments = hybrid_routing(points, seeds, adj, lam=lam, mu=mu)

    # Count how many points each seed serves
    serve_counts = np.bincount(assignments, minlength=seeds.shape[0])

    # Normalise to obtain a probability‑like influence matrix
    influence = serve_counts.astype(float) / max(1, serve_counts.sum())

    # Update adjacency: stronger influence → slightly larger edge weight
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            # Symmetric update; keep matrix stochastic by renormalising later
            delta = lam * influence[i] * influence[j] * (1.0 + mu * curv[i] * curv[j])
            adj[i, j] = max(0.0, adj[i, j] + delta)

    # Re‑normalise rows to keep a proper transition matrix (optional)
    row_sums = adj.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    adj[:] = adj / row_sums


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(0)

    # synthetic data
    points = np.random.rand(20, 2)          # 20 requests
    seeds = np.random.rand(7, 2)            # 7 service nodes (must match graph size)
    adj = np.random.rand(7, 7)              # adjacency for the 7 seeds
    adj = (adj + adj.T) / 2.0                # make it symmetric
    adj /= adj.sum(axis=1, keepdims=True)   # stochastic rows

    # generate some bandit updates (action_id matches seed index as string)
    updates = [
        BanditUpdate("ctx1", "0", 1.0, 0.5),
        BanditUpdate("ctx1", "0", 0.0, 0.5),
        BanditUpdate("ctx2", "3", 1.0, 0.7),
        BanditUpdate("ctx2", "5", 0.0, 0.3),
    ]

    # run the fused update
    hybrid_update_policy(updates, adj, seeds, points)

    # routing after the update
    routing = hybrid_routing(points, seeds, adj)
    print("Assignments:", routing)
    print("Adjacency after update (first 3 rows):")
    print(adj[:3])