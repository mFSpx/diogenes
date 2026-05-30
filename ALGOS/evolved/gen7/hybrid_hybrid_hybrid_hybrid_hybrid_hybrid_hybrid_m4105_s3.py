# DARWIN HAMMER — match 4105, survivor 3
# gen: 7
# parent_a: hybrid_hybrid_hybrid_doomsd_hybrid_hybrid_bandit_m48_s3.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_sheaf__m2428_s0.py (gen6)
# born: 2026-05-29T23:53:34Z

"""Hybrid Fusion Module
Parents:
- hybrid_hybrid_hybrid_doomsd_hybrid_hybrid_bandit_m48_s3 (Algorithm A)
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_sheaf__m2428_s0 (Algorithm B)

Mathematical Bridge:
Both parents manipulate vectors/matrices and associate scalar scores with entities.
Algorithm A provides scalar scores (expected_reward, confidence_bound) for
bandit actions; Algorithm B builds a weighted tree where edge weights are
Euclidean distances. The fusion treats the expected_reward of a bandit
action as a multiplicative factor that scales the Euclidean edge length,
producing a *reward‑scaled edge metric*. The distribution of these scaled
weights is summarised by the Gini coefficient (from A) which then drives
the pruning probability (from B). This creates a single unified system
where temporal context (weekday), bandit statistics, and tree geometry
interact mathematically.

The module implements:
1. Vectorised weekday extraction for date contexts.
2. Construction of a reward‑scaled tree from node positions and bandit
   actions.
3. Computation of hybrid metrics (adjacency, scaled edge lengths,
   root‑to‑node distances) together with the Gini coefficient of the
   scaled edges.
4. Pruning of the tree based on a probability that depends on the Gini
   coefficient (higher inequality → higher pruning chance).

All functions are pure NumPy / standard‑library operations and can be
used independently or as a pipeline.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
import numpy as np

# ----------------------------------------------------------------------
# Re‑implemented pieces from Parent A
# ----------------------------------------------------------------------
def weekday_sakamoto(years: np.ndarray, months: np.ndarray, days: np.ndarray) -> np.ndarray:
    """Vectorised Sakamoto algorithm – returns weekday index (0=Sunday)."""
    y = years.astype(np.int64)
    m = months.astype(np.int64)
    d = days.astype(np.int64)

    m_adj = np.where(m < 3, m + 12, m)
    y_adj = np.where(m < 3, y - 1, y)

    t = np.array([0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4], dtype=np.int64)
    w = (y_adj + y_adj // 4 - y_adj // 100 + y_adj // 400 + t[m_adj - 1] + d) % 7
    return w.astype(np.int8)


def gini_coefficient(values: np.ndarray) -> float:
    """Gini coefficient for a 1‑D non‑negative array."""
    if values.ndim != 1:
        raise ValueError("values must be a 1‑D array")
    x = np.sort(values.astype(np.float64))
    if x.size == 0 or np.isclose(x.sum(), 0.0):
        return 0.0
    if np.any(x < 0):
        raise ValueError("values must be non‑negative")
    n = x.size
    i = np.arange(1, n + 1, dtype=np.float64)
    numerator = np.sum((2 * i - n - 1) * x)
    denominator = n * x.sum()
    return float(numerator / denominator)


@dataclass(frozen=True)
class BanditAction:
    """Immutable description of a bandit arm."""
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str


# ----------------------------------------------------------------------
# Re‑implemented pieces from Parent B
# ----------------------------------------------------------------------
def euclidean_length(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Euclidean distance between two 2‑D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def prune_probability(t: float, lam: float = 1.0, alpha: float = 0.2) -> float:
    """Pruning probability as defined in Parent B."""
    if t < 0 or lam < 0 or alpha < 0:
        raise ValueError("t, lam and alpha must be non‑negative")
    return min(1.0, lam * math.exp(-alpha * t))


# ----------------------------------------------------------------------
# Hybrid Functions
# ----------------------------------------------------------------------
def build_reward_scaled_tree(
    nodes: dict[str, tuple[float, float]],
    edges: list[tuple[str, str]],
    bandit_actions: list[BanditAction],
) -> tuple[dict[str, list[str]], dict[tuple[str, str], float], dict[str, float]]:
    """
    Build a tree where each edge length is scaled by the *average* expected
    reward of the two incident nodes' associated bandit actions.

    Parameters
    ----------
    nodes: mapping node_id → (x, y) coordinates.
    edges: list of (parent_id, child_id) tuples (assumed to form a tree).
    bandit_actions: list of BanditAction objects.  The ``action_id`` must
        match a node identifier; actions without a matching node are ignored.

    Returns
    -------
    adjacency: dict node_id → list of neighbour ids.
    scaled_lengths: dict (parent, child) → float (scaled edge length).
    root_distances: dict node_id → cumulative scaled distance from the root.
    """
    # Map node_id → expected_reward (default 1.0 if missing)
    reward_map = {ba.action_id: ba.expected_reward for ba in bandit_actions}
    default_reward = 1.0

    adjacency: dict[str, list[str]] = {nid: [] for nid in nodes}
    scaled_lengths: dict[tuple[str, str], float] = {}

    for parent, child in edges:
        if parent not in nodes or child not in nodes:
            raise KeyError(f"Edge ({parent},{child}) references unknown node.")
        adjacency[parent].append(child)
        adjacency[child].append(parent)

        # Euclidean length
        raw_len = euclidean_length(nodes[parent], nodes[child])

        # Scale by mean expected reward of the two endpoints
        r1 = reward_map.get(parent, default_reward)
        r2 = reward_map.get(child, default_reward)
        scale = (r1 + r2) / 2.0
        scaled_len = raw_len * scale
        scaled_lengths[(parent, child)] = scaled_len
        scaled_lengths[(child, parent)] = scaled_len  # undirected access

    # Compute root‑to‑node distances (choose arbitrary root: first key)
    root = next(iter(nodes))
    root_distances: dict[str, float] = {nid: math.inf for nid in nodes}
    root_distances[root] = 0.0
    queue = [root]
    visited = {root}
    while queue:
        cur = queue.pop(0)
        for neigh in adjacency[cur]:
            if neigh in visited:
                continue
            edge_key = (cur, neigh)
            root_distances[neigh] = root_distances[cur] + scaled_lengths[edge_key]
            visited.add(neigh)
            queue.append(neigh)

    return adjacency, scaled_lengths, root_distances


def hybrid_prune_tree(
    nodes: dict[str, tuple[float, float]],
    edges: list[tuple[str, str]],
    bandit_actions: list[BanditAction],
    t: float,
    lam: float = 1.0,
    alpha: float = 0.2,
    seed: int | None = None,
) -> list[tuple[str, str]]:
    """
    Prune edges probabilistically.  The pruning probability is modulated by
    the Gini coefficient of the *scaled* edge lengths (higher inequality →
    higher chance to prune).

    Parameters
    ----------
    t, lam, alpha : pruning hyper‑parameters (see Parent B).
    seed : optional random seed for reproducibility.

    Returns
    -------
    List of edges that survive the pruning step.
    """
    rng = random.Random(seed)

    # Build the reward‑scaled tree to obtain scaled lengths
    _, scaled_lengths, _ = build_reward_scaled_tree(nodes, edges, bandit_actions)

    # Extract the unique undirected lengths
    unique_lengths = np.array(
        list({frozenset(k) for k in scaled_lengths.keys()})
    )
    # Convert frozenset back to a length value
    lengths = np.array([scaled_lengths[tuple(edge)] for edge in scaled_lengths if edge[0] < edge[1]])

    # Compute Gini of the length distribution
    gini = gini_coefficient(lengths)

    # Adjust lambda by the Gini coefficient (more inequality → larger λ)
    adjusted_lam = lam * (1.0 + gini)

    p = prune_probability(t, adjusted_lam, alpha)

    # Keep an edge if a uniform draw exceeds p
    kept_edges = [e for e in edges if rng.random() > p]
    return kept_edges


def evaluate_hybrid(
    years: np.ndarray,
    months: np.ndarray,
    days: np.ndarray,
    nodes: dict[str, tuple[float, float]],
    edges: list[tuple[str, str]],
    bandit_actions: list[BanditAction],
    t: float,
) -> dict[str, float]:
    """
    End‑to‑end evaluation that:
    1. Computes weekday context indices.
    2. Builds a reward‑scaled tree.
    3. Computes Gini of scaled edges.
    4. Prunes the tree using the Gini‑adjusted probability.
    5. Returns a dictionary of summary statistics.

    The weekday vector is included to illustrate the temporal bridge
    (it could be used downstream for context‑aware bandit updates).

    Returns
    -------
    dict with keys:
        'weekday_mode' – most common weekday index.
        'gini_before' – Gini of scaled edges before pruning.
        'edges_before' – number of edges before pruning.
        'edges_after'  – number of edges after pruning.
        'prune_prob'   – probability used for pruning.
    """
    # 1. Weekday computation (vectorised)
    weekdays = weekday_sakamoto(years, months, days)
    weekday_mode = int(np.bincount(weekdays).argmax())

    # 2. Build tree and obtain scaled lengths
    adjacency, scaled_lengths, _ = build_reward_scaled_tree(nodes, edges, bandit_actions)

    # 3. Gini of the scaled edge lengths (undirected unique set)
    unique_edges = {frozenset((u, v)) for u, v in scaled_lengths}
    lengths = np.array([scaled_lengths[tuple(edge)] for edge in scaled_lengths if edge[0] < edge[1]])
    gini_before = gini_coefficient(lengths)

    # 4. Prune
    kept_edges = hybrid_prune_tree(nodes, edges, bandit_actions, t, seed=42)
    prune_prob = prune_probability(t, lam=1.0 * (1.0 + gini_before), alpha=0.2)

    # 5. Assemble results
    return {
        "weekday_mode": weekday_mode,
        "gini_before": gini_before,
        "edges_before": len(edges),
        "edges_after": len(kept_edges),
        "prune_prob": prune_prob,
    }


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic date vectors (10 random dates)
    rng = np.random.default_rng(0)
    years = rng.integers(2000, 2025, size=10)
    months = rng.integers(1, 13, size=10)
    days = rng.integers(1, 28, size=10)

    # Create a tiny tree of 5 nodes placed on a grid
    nodes = {
        "A": (0.0, 0.0),
        "B": (1.0, 0.0),
        "C": (1.0, 1.0),
        "D": (0.0, 1.0),
        "E": (0.5, 1.5),
    }
    edges = [
        ("A", "B"),
        ("B", "C"),
        ("C", "D"),
        ("D", "A"),
        ("C", "E"),
    ]

    # Bandit actions – associate each node with a random expected reward
    bandit_actions = [
        BanditAction(action_id=n, propensity=0.2, expected_reward=float(rng.uniform(0.5, 2.0)),
                     confidence_bound=0.1, algorithm="demo")
        for n, r in zip(nodes.keys(), rng.random(len(nodes)))
    ]

    # Run the hybrid evaluation
    summary = evaluate_hybrid(years, months, days, nodes, edges, bandit_actions, t=2.0)
    for k, v in summary.items():
        print(f"{k}: {v}")

    # Ensure the pruning function returns a plausible edge list
    pruned = hybrid_prune_tree(nodes, edges, bandit_actions, t=2.0, seed=123)
    assert isinstance(pruned, list) and all(isinstance(e, tuple) for e in pruned)
    print("Pruned edges:", pruned)