# DARWIN HAMMER — match 2869, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2709_s0.py (gen6)
# parent_b: hybrid_hybrid_omni_chaotic__hybrid_hybrid_hybrid_m474_s1.py (gen4)
# born: 2026-05-29T23:46:20Z

"""Hybrid Algorithm integrating Bayesian tree analysis with ternary bandit routing

Parents:
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m442_s0.py
- hybrid_hybrid_omni_chaotic__hybrid_hybrid_hybrid_m474_s1.py

Mathematical bridge:
The edge‑length distribution of a rooted tree provides a numeric representation **v**.
From parent A we obtain Bayesian posterior probabilities (bayes_update) based on
epistemic certainty flags. From parent B we have a ternary bandit router that
chooses an action according to a similarity score between **v** and a reference
vector **r** (a simple SSIM‑like metric).  
The hybrid therefore:

1. Computes edge lengths and cumulative root‑to‑node distances (tree_metrics).
2. Derives a weighted Gini coefficient of the edge‑length distribution, using
   epistemic flags as weights.
3. Forms a representation vector **v** = [Gini, mean length, max length].
4. Evaluates similarity S(v, r) with a fixed reference **r**.
5. Uses S as a reward to scale Bayesian posterior updates for three actions.
6. Performs a Thompson‑sampling style ternary bandit routing based on the
   updated posteriors.

The result is a single probabilistic decision that respects both the
statistical certainty of the tree model and the adaptive routing of the bandit
router.
"""

import math
import random
import sys
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Shared constants and simple utilities (from Parent A)
# ----------------------------------------------------------------------
EPISTEMIC_FLAGS: Tuple[str, ...] = (
    "FACT",
    "PROBABLE",
    "POSSIBLE",
    "BULLSHIT",
    "SURE_MAYBE",
)

FLAG_WEIGHTS: Dict[str, float] = {
    "FACT": 1.0,
    "PROBABLE": 0.8,
    "POSSIBLE": 0.5,
    "BULLSHIT": 0.1,
    "SURE_MAYBE": 0.6,
}


def length(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """Marginal probability for a Bayesian update."""
    if not all(0.0 <= x <= 1.0 for x in (prior, likelihood, false_positive)):
        raise ValueError("probabilities must be in [0, 1]")
    return likelihood * prior + false_positive * (1.0 - prior)


def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    """Standard Bayesian posterior."""
    if marginal == 0.0:
        return prior
    return (likelihood * prior) / marginal


# ----------------------------------------------------------------------
# Tree analysis (core of Parent A)
# ----------------------------------------------------------------------
def tree_metrics(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
) -> Tuple[Dict[str, List[str]], Dict[Tuple[str, str], float], Dict[str, float]]:
    """
    Build adjacency, compute Euclidean edge lengths and root‑to‑node distances.

    Returns
    -------
    adj : dict mapping node → list of neighbours
    edge_len : dict mapping edge (as supplied) → length
    dist : dict mapping node → cumulative distance from root
    """
    adj: Dict[str, List[str]] = defaultdict(list)
    edge_len: Dict[Tuple[str, str], float] = {}

    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        edge_len[(u, v)] = length(nodes[u], nodes[v])
        edge_len[(v, u)] = edge_len[(u, v)]

    # BFS for root‑to‑node distances
    dist: Dict[str, float] = {root: 0.0}
    visited = {root}
    queue = deque([root])
    while queue:
        cur = queue.popleft()
        for nb in adj[cur]:
            if nb not in visited:
                dist[nb] = dist[cur] + edge_len[(cur, nb)]
                visited.add(nb)
                queue.append(nb)

    return dict(adj), edge_len, dist


def weighted_gini(edge_lengths: List[float], flags: List[str]) -> float:
    """
    Compute a Gini coefficient weighted by epistemic flags.
    """
    if len(edge_lengths) != len(flags):
        raise ValueError("Lengths and flags must be same size")
    # Apply weights
    weighted = [l * FLAG_WEIGHTS.get(f, 0.5) for l, f in zip(edge_lengths, flags)]
    sorted_vals = np.sort(weighted)
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    cumvals = np.cumsum(sorted_vals)
    gini = (n + 1 - 2 * np.sum(cumvals) / cumvals[-1]) / n
    return float(gini)


# ----------------------------------------------------------------------
# Similarity metric (inspired by Parent B's SSIM idea)
# ----------------------------------------------------------------------
def ssim_like(x: np.ndarray, y: np.ndarray) -> float:
    """
    Very small SSIM‑like similarity between two 1‑D vectors.
    Returns a value in [0, 1] where 1 means identical.
    """
    C1 = 1e-4
    C2 = 9e-4
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x = np.var(x)
    sigma_y = np.var(y)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))

    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    return float(numerator / denominator) if denominator != 0 else 0.0


# ----------------------------------------------------------------------
# Ternary bandit routing (core of Parent B) with Bayesian scaling
# ----------------------------------------------------------------------
def ternary_bandit_routing(
    priors: Tuple[float, float, float],
    reward: float,
) -> int:
    """
    Perform a Thompson‑sampling style selection among three arms.
    The reward scales the likelihood term of a Bayesian update for each arm.
    Returns the index of the chosen arm (0, 1, or 2).
    """
    # Treat reward as a likelihood factor (clipped to [0,1])
    likelihood = max(0.0, min(1.0, reward))

    posteriors = []
    for prior in priors:
        marginal = bayes_marginal(prior, likelihood, false_positive=1 - likelihood)
        posterior = bayes_update(prior, likelihood, marginal)
        posteriors.append(posterior)

    # Sample from Beta(1+posterior, 1+(1-posterior)) for each arm
    samples = [
        random.betavariate(1 + p, 1 + (1 - p)) for p in posteriors
    ]
    return int(np.argmax(samples))


# ----------------------------------------------------------------------
# Hybrid algorithm bringing everything together
# ----------------------------------------------------------------------
def hybrid_algorithm(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
    epistemic_assignments: Dict[Tuple[str, str], str],
    prior_actions: Tuple[float, float, float],
    false_positive: float = 0.1,
) -> Tuple[int, Dict[str, float]]:
    """
    Execute the fused workflow:

    1. Compute tree metrics.
    2. Build a compact representation vector v = [Gini, mean_len, max_len].
    3. Compare v with a fixed reference r using ssim_like → similarity S.
    4. Use S as a reward to update action posteriors via ternary_bandit_routing.
    5. Return chosen action index and a dict with intermediate diagnostics.
    """
    # 1. Tree analysis
    adj, edge_len_dict, dist = tree_metrics(nodes, edges, root)

    # Gather edge lengths and associated epistemic flags
    lengths = []
    flags = []
    for (u, v) in edges:
        lengths.append(edge_len_dict[(u, v)])
        flags.append(epistemic_assignments.get((u, v), "POSSIBLE"))

    # 2. Representation vector
    gini = weighted_gini(lengths, flags)
    mean_len = float(np.mean(lengths)) if lengths else 0.0
    max_len = float(np.max(lengths)) if lengths else 0.0
    v = np.array([gini, mean_len, max_len])

    # Reference vector (chosen arbitrarily as a neutral baseline)
    r = np.array([0.3, 1.0, 2.0])

    # 3. Similarity as reward
    similarity = ssim_like(v, r)

    # 4. Bandit routing
    chosen = ternary_bandit_routing(prior_actions, reward=similarity)

    diagnostics = {
        "gini": gini,
        "mean_len": mean_len,
        "max_len": max_len,
        "similarity": similarity,
        "posteriors_before": prior_actions,
    }

    return chosen, diagnostics


# ----------------------------------------------------------------------
# Simple smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Construct a tiny tree
    nodes_example = {
        "A": (0.0, 0.0),
        "B": (1.0, 0.0),
        "C": (0.0, 1.0),
        "D": (1.0, 1.0),
    }
    edges_example = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")]
    root_example = "A"

    # Assign epistemic flags to each edge
    epistemic_example = {
        ("A", "B"): "FACT",
        ("A", "C"): "PROBABLE",
        ("B", "D"): "POSSIBLE",
        ("C", "D"): "SURE_MAYBE",
    }

    # Prior probabilities for three actions (must sum to 1 for a proper prior)
    prior_actions_example = (0.33, 0.33, 0.34)

    action, info = hybrid_algorithm(
        nodes=nodes_example,
        edges=edges_example,
        root=root_example,
        epistemic_assignments=epistemic_example,
        prior_actions=prior_actions_example,
        false_positive=0.05,
    )

    print(f"Chosen action index: {action}")
    for k, v in info.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")