# DARWIN HAMMER — match 8, survivor 3
# gen: 3
# parent_a: hybrid_hybrid_minimum_cost__hybrid_decision_hygi_m7_s0.py (gen2)
# parent_b: model_vram_scheduler.py (gen0)
# born: 2026-05-29T23:25:05Z

"""Hybrid VRAM Scheduler & Minimum‑Cost Tree with Bayesian Decision Hygiene.

Parents:
- hybrid_hybrid_minimum_cost__hybrid_decision_hygi_m7_s0.py (tree metrics, Bayesian update, entropy)
- model_vram_scheduler.py (VRAM budgeting for base model, embeddings, LoRA adapters)

Mathematical bridge:
The VRAM budget is treated as a “cost” on a rooted tree whose vertices are model
artifacts (base model, embedding, adapters).  Edge lengths are the memory
footprint of each artifact, so the root‑to‑leaf distance equals cumulative VRAM
usage.  A Bayesian marginal‑posterior update quantifies the probability that the
observed usage fits within the budget given measurement uncertainty (false‑positive
rate).  The Shannon entropy of the resulting probability distribution provides a
decision‑hygiene score that drives the final admission/rejection policy.
"""

import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Core utilities borrowed from Parent A
# ----------------------------------------------------------------------


def _euclidean_length(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Euclidean distance between two points (unused for VRAM but kept for completeness)."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


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
    edge_len : dict mapping edge (ordered as supplied) → length
    dist : dict mapping node → distance from *root* (sum of edge lengths along the unique path)
    """
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    edge_len: Dict[Tuple[str, str], float] = {}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        edge_len[(a, b)] = _euclidean_length(nodes[a], nodes[b])

    dist: Dict[str, float] = {root: 0.0}
    stack = [root]
    while stack:
        cur = stack.pop()
        for nxt in adj[cur]:
            if nxt not in dist:
                edge_key = (cur, nxt) if (cur, nxt) in edge_len else (nxt, cur)
                dist[nxt] = dist[cur] + edge_len[edge_key]
                stack.append(nxt)

    return adj, edge_len, dist


def bayes_marginal(prior: np.ndarray, likelihood: np.ndarray, false_positive: float) -> np.ndarray:
    """
    Vectorised marginal:  P(E) = L·p + FP·(1‑p)
    """
    return likelihood * prior + false_positive * (1.0 - prior)


def bayes_update(prior: np.ndarray, likelihood: np.ndarray, marginal: np.ndarray) -> np.ndarray:
    """
    Vectorised posterior:  P(H|E) = (L·p) / P(E)
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        posterior = np.where(marginal != 0, (likelihood * prior) / marginal, 0.0)
    return posterior


def shannon_entropy(prob_dist: np.ndarray) -> float:
    """Standard base‑2 Shannon entropy, ignoring zero probabilities."""
    prob = prob_dist[prob_dist > 0]
    return -float(np.sum(prob * np.log2(prob)))


# ----------------------------------------------------------------------
# VRAM‑specific constants (mirroring Parent B)
# ----------------------------------------------------------------------
DEFAULT_BASE_MODEL_MB = 1800
DEFAULT_EMBEDDING_MB = 384
DEFAULT_ADAPTER_MB = 128
DEFAULT_BUDGET_MB = 4096
DEFAULT_RESERVE_MB = 768  # safety headroom


# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------


def build_vram_tree(
    base_mb: int = DEFAULT_BASE_MODEL_MB,
    embedding_mb: int = DEFAULT_EMBEDDING_MB,
    adapters_mb: List[int] | None = None,
) -> Tuple[Dict[str, Tuple[float, float]], List[Tuple[str, str]], str]:
    """
    Construct a minimal tree where each node represents a VRAM consumer.
    The geometric coordinates are synthetic; only the edge list matters for
    the cost aggregation performed later.

    Returns
    -------
    nodes : mapping name → (x, y) placeholder coordinates
    edges : list of (parent, child) tuples
    root  : name of the root node ('system')
    """
    if adapters_mb is None:
        adapters_mb = [DEFAULT_ADAPTER_MB, DEFAULT_ADAPTER_MB]

    # Dummy coordinates – they are irrelevant for the memory‑cost view.
    nodes = {
        "system": (0.0, 0.0),
        "base_model": (1.0, 0.0),
        "embedding": (0.0, 1.0),
    }
    edges = [("system", "base_model"), ("system", "embedding")]

    for i, mb in enumerate(adapters_mb):
        name = f"adapter_{i}"
        nodes[name] = (1.0 + i, 1.0)  # arbitrary
        edges.append(("system", name))

    # Store memory usage as a hidden attribute on the node dict for later lookup.
    # We attach a parallel dict instead of mutating the tuple.
    nodes["_mem_mb"] = {
        "system": 0,
        "base_model": base_mb,
        "embedding": embedding_mb,
    }
    for i, mb in enumerate(adapters_mb):
        nodes["_mem_mb"][f"adapter_{i}"] = mb

    return nodes, edges, "system"


def cumulative_vram_usage(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
) -> Dict[str, float]:
    """
    Compute cumulative VRAM usage from the root to each node.
    The cost of traversing an edge equals the memory footprint of the child node.
    """
    # Build adjacency for traversal
    adj: Dict[str, List[str]] = {n: [] for n in nodes if not n.startswith("_")}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)

    mem_map = nodes["_mem_mb"]
    cum: Dict[str, float] = {root: 0.0}
    stack = [root]
    while stack:
        cur = stack.pop()
        for nxt in adj[cur]:
            if nxt not in cum:
                cum[nxt] = cum[cur] + mem_map.get(nxt, 0)
                stack.append(nxt)
    return cum


def fit_probability(
    total_usage_mb: float,
    budget_mb: int = DEFAULT_BUDGET_MB,
    reserve_mb: int = DEFAULT_RESERVE_MB,
    false_positive: float = 0.05,
) -> Tuple[float, float]:
    """
    Estimate the probability that the current VRAM usage fits within the
    available budget (budget – reserve).  A simple Bayesian model is used:

    * prior  = 0.5 (uninformed)
    * likelihood = 1 if usage ≤ effective_budget else 0
    * false_positive models systematic under‑estimation of usage.

    Returns (posterior, entropy) where entropy is the Shannon entropy of the
    binary distribution [posterior, 1‑posterior].
    """
    effective_budget = budget_mb - reserve_mb
    prior = np.array([0.5, 0.5])  # [fit, not_fit]
    likelihood = np.array([1.0, 0.0]) if total_usage_mb <= effective_budget else np.array([0.0, 1.0])

    marginal = bayes_marginal(prior, likelihood, false_positive)
    posterior = bayes_update(prior, likelihood, marginal)

    prob_fit = float(posterior[0])
    entropy = shannon_entropy(posterior)
    return prob_fit, entropy


def hybrid_vram_decision(
    base_mb: int = DEFAULT_BASE_MODEL_MB,
    embedding_mb: int = DEFAULT_EMBEDDING_MB,
    adapters_mb: List[int] | None = None,
    budget_mb: int = DEFAULT_BUDGET_MB,
    reserve_mb: int = DEFAULT_RESERVE_MB,
    false_positive: float = 0.05,
    entropy_threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Full hybrid pipeline:
    1. Build a VRAM‑resource tree.
    2. Compute cumulative usage.
    3. Estimate fit probability via Bayesian update.
    4. Compute decision‑hygiene entropy.
    5. Return a structured plan dict.
    """
    nodes, edges, root = build_vram_tree(base_mb, embedding_mb, adapters_mb)
    cum_usage = cumulative_vram_usage(nodes, edges, root)
    total_usage = max(cum_usage.values())  # leaf with greatest cumulative cost

    prob_fit, entropy = fit_probability(
        total_usage,
        budget_mb=budget_mb,
        reserve_mb=reserve_mb,
        false_positive=false_positive,
    )

    decision = "ACCEPT" if (prob_fit > 0.5 and entropy < entropy_threshold) else "REJECT"

    plan = {
        "total_vram_mb": total_usage,
        "effective_budget_mb": budget_mb - reserve_mb,
        "probability_fit": prob_fit,
        "entropy_bits": entropy,
        "decision": decision,
        "components": {
            name: {"cumulative_mb": cum_usage[name], "individual_mb": nodes["_mem_mb"][name]}
            for name in cum_usage
            if name != root
        },
    }
    return plan


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Example with three adapters of varying size
    adapters = [120, 140, 160]
    result = hybrid_vram_decision(
        base_mb=1800,
        embedding_mb=384,
        adapters_mb=adapters,
        budget_mb=4096,
        reserve_mb=768,
        false_positive=0.07,
        entropy_threshold=0.6,
    )
    print("Hybrid VRAM Decision Plan:")
    for k, v in result.items():
        if k == "components":
            print("  components:")
            for comp, details in v.items():
                print(f"    {comp}: {details}")
        else:
            print(f"  {k}: {v}")