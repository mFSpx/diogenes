# DARWIN HAMMER — match 612, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_hybrid_doomsd_hybrid_hybrid_hybrid_m192_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_decisi_hybrid_hybrid_bandit_m67_s1.py (gen4)
# born: 2026-05-29T23:30:12Z

import datetime as dt
import math
import random
import re
import sys
from pathlib import Path
from collections import Counter
from typing import List, Tuple, Iterable

import numpy as np

# ----------------------------------------------------------------------
# Parent A utilities
# ----------------------------------------------------------------------
def doomsday_numpy(years: np.ndarray, months: np.ndarray, days: np.ndarray) -> np.ndarray:
    """Vectorised weekday calculation (Sun=0 … Sat=6)."""
    dates = np.stack([years, months, days], axis=-1).astype("datetime64[D]")
    flat = dates.ravel()
    py_weekday = np.fromiter(
        (
            dt.datetime.utcfromtimestamp(int(d.astype("datetime64[s]"))).weekday()
            for d in flat
        ),
        dtype=np.int8,
        count=flat.size,
    )
    py_weekday = py_weekday.reshape(dates.shape[:-1])
    # shift: Monday=0 → Sunday=0
    return (py_weekday + 1) % 7


def weekday_counts(dates: Iterable[dt.date]) -> np.ndarray:
    """Return a length‑7 array with counts of each weekday (Sun=0 … Sat=6)."""
    counts = np.zeros(7, dtype=int)
    for d in dates:
        if isinstance(d, dt.datetime):
            d = d.date()
        counts[d.weekday() % 7] += 1
    # shift to match doomsday_numpy convention (Sun=0)
    return np.roll(counts, 1)


def gini_coefficient(values: np.ndarray) -> float:
    """Gini coefficient for a 1‑D non‑negative array."""
    if values.size == 0:
        return 0.0
    sorted_vals = np.sort(values.astype(float))
    n = values.size
    cumulative = np.cumsum(sorted_vals)
    gini = (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n
    return float(gini)


def compute_health(
    reconstruction_risk_score: float,
    failures: int,
    failure_threshold: int,
    recovery_priority: float,
) -> float:
    """
    health = (1 - (reconstruction_risk_score * failure_rate)) * (1 - recovery_priority)

    where failure_rate = failures / failure_threshold.
    """
    failure_rate = min(1.0, failures / max(1, failure_threshold))
    health = (1.0 - (reconstruction_risk_score * failure_rate)) * (1.0 - recovery_priority)
    return max(0.0, min(1.0, health))


# ----------------------------------------------------------------------
# Parent B utilities
# ----------------------------------------------------------------------
def shannon_entropy(probs: np.ndarray) -> float:
    """Compute Shannon entropy (base e) for a probability vector."""
    probs = probs.astype(float)
    probs = probs[probs > 0]  # avoid log(0)
    return -float(np.sum(probs * np.log(probs)))


def curvature_proxy(adj: np.ndarray, node: int) -> float:
    """
    Simple Ollivier‑Ricci curvature proxy.
    curvature = (avg_neighbor_degree - degree_i) / max(degree_i, 1)
    """
    deg_i = float(np.sum(adj[node]))
    neighbors = np.where(adj[node] > 0)[0]
    if neighbors.size == 0:
        return 0.0
    neighbor_degs = np.sum(adj[neighbors], axis=1)
    avg_neighbor_deg = float(np.mean(neighbor_degs))
    curvature = (avg_neighbor_deg - deg_i) / max(deg_i, 1.0)
    return curvature


# ----------------------------------------------------------------------
# Hybrid bandit engine
# ----------------------------------------------------------------------
class HybridBandit:
    """UCB‑style bandit where the expected reward blends health, entropy and curvature."""

    def __init__(self, actions: List[str]):
        self.actions = actions
        self.counts = np.zeros(len(actions), dtype=int)
        self.values = np.zeros(len(actions), dtype=float)  # running average reward

    def update(self, action_idx: int, reward: float) -> None:
        """Incremental update of average reward."""
        self.counts[action_idx] += 1
        n = self.counts[action_idx]
        self.values[action_idx] += (reward - self.values[action_idx]) / n

    def select(self, health: float, entropy: float, curvature: float, gini: float) -> Tuple[int, str]:
        """
        Compute UCB scores using the hybrid reward model:
            hybrid_reward = health * (1 - entropy) * (1 + curvature) * gini

        The UCB term adds exploration pressure.
        """
        total_counts = max(1, self.counts.sum())
        hybrid_reward = health * (1.0 - entropy) * (1.0 + curvature) * gini

        # Blend the hybrid reward with the learned value for each arm
        blended = self.values + hybrid_reward

        # UCB exploration term
        exploration = np.sqrt(2 * np.log(total_counts) / (self.counts + 1e-6))
        ucb = blended + exploration

        idx = int(np.argmax(ucb))
        return idx, self.actions[idx]

    def __repr__(self) -> str:
        return f"<HybridBandit actions={self.actions} counts={self.counts.tolist()} values={self.values.tolist()}>"


# ----------------------------------------------------------------------
# High‑level hybrid function
# ----------------------------------------------------------------------
def hybrid_decision_pipeline(
    dates: List[dt.date],
    reconstruction_risk_score: float,
    failures: int,
    failure_threshold: int,
    recovery_priority: float,
    feature_probs: np.ndarray,
    adjacency: np.ndarray,
    bandit: HybridBandit,
) -> str:
    """
    End‑to‑end pipeline:
    1. Derive health from risk/failure metrics.
    2. Compute Gini of weekday distribution (contextual signal).
    3. Evaluate entropy of feature probabilities.
    4. Compute curvature proxy for the currently most promising node.
    5. Use the hybrid bandit to select an action.
    """
    # 1. Health
    health = compute_health(
        reconstruction_risk_score, failures, failure_threshold, recovery_priority
    )

    # 2. Gini 
    counts = weekday_counts(dates)
    gini = gini_coefficient(counts)

    # Weighted Gini 
    weighted_gini = gini * health

    # 3. Entropy
    entropy = shannon_entropy(feature_probs)

    # 4. Curvature – pick a random node as placeholder for “current context”
    node = random.randrange(adjacency.shape[0])
    curvature = curvature_proxy(adjacency, node)

    # 5. Bandit selection
    idx, action = bandit.select(health, entropy, curvature, weighted_gini)
    # Simulated reward based on the same hybrid formula
    reward = health * (1.0 - entropy) * (1.0 + curvature) * weighted_gini
    bandit.update(idx, reward)

    return action


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Sample dates spanning a week
    today = dt.date.today()
    sample_dates = [today + dt.timedelta(days=i) for i in range(14)]

    # Risk / failure parameters
    recon_risk = 0.27
    failures = 3
    failure_thresh = 10
    recovery_prio = 0.15

    # Feature probability vector (must sum to 1)
    probs = np.array([0.2, 0.15, 0.25, 0.1, 0.3])

    # Simple undirected graph (5 nodes)
    adj_matrix = np.array(
        [
            [0, 1, 0, 0, 1],
            [1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 1, 0, 1],
            [1, 0, 0, 1, 0],
        ],
        dtype=float,
    )

    # Initialise bandit with placeholder actions
    actions = ["ALERT", "MITIGATE", "IGNORE", "ESCALATE"]
    bandit = HybridBandit(actions)

    for _ in range(10):
        action = hybrid_decision_pipeline(
            sample_dates,
            recon_risk,
            failures,
            failure_thresh,
            recovery_prio,
            probs,
            adj_matrix,
            bandit,
        )
        print(action)