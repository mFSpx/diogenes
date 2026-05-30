# DARWIN HAMMER — match 2723, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_hybrid_minimu_hybrid_hybrid_hybrid_m405_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_privac_hybrid_hybrid_endpoi_m16_s1.py (gen3)
# born: 2026-05-29T23:43:44Z

"""
Hybrid Algorithm Fusion of:
- PARENT ALGORITHM A (minimum‑cost tree Bayesian bandit‑router + hybrid privacy health metric)
- PARENT ALGORITHM B (privacy reconstruction risk scoring + endpoint work‑share allocation)

Mathematical Bridge
------------------
Algorithm A produces a *health* score for each tree node using a log‑count statistic
derived from the bandit‑router (`log(count)`) together with a reconstruction‑risk term.
Algorithm B defines a *model health* as `(1‑reconstruction_risk)*(1‑recovery_priority)`.
The fusion replaces the plain product in Algorithm B with the richer
log‑count‑weighted term from Algorithm A, yielding a unified health:

    health_node = (1 - (reconstruction_risk * failure_rate)) *
                  (1 - recovery_priority) *
                  log(count + 1)

where

    failure_rate      = failures / failure_threshold
    reconstruction_risk = unique_quasi_identifiers / total_records
    recovery_priority   = morphology‑driven right‑ing time (here supplied as a scalar)

The resulting node healths are then normalised and used as *work‑share*
fractions for a set of model tiers (Algorithm B).  This creates a single
system that simultaneously respects tree topology, Bayesian bandit statistics,
privacy risk, and VRAM‑aware model scheduling.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Iterable

import numpy as np

# ----------------------------------------------------------------------
# Types shared by both parents
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Edge = Tuple[str, str]

# ----------------------------------------------------------------------
# Core utilities from Algorithm A
# ----------------------------------------------------------------------
def length(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def tree_metrics(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
) -> Tuple[Dict[str, List[str]], Dict[Edge, float], Dict[str, float]]:
    """
    Build adjacency, compute Euclidean edge lengths and root‑to‑node distances.

    Returns
    -------
    adj : dict mapping node → list of neighbours
    edge_len : dict mapping edge (ordered a, b) → length
    node_dist : dict mapping node → root‑to‑node distance
    """
    from collections import defaultdict, deque

    adj: Dict[str, List[str]] = defaultdict(list)
    edge_len: Dict[Edge, float] = {}
    node_dist: Dict[str, float] = {}

    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)  # undirected for distance propagation
        edge_len[(a, b)] = length(nodes[a], nodes[b])
        edge_len[(b, a)] = edge_len[(a, b)]

    # BFS to compute distance from root
    visited = set([root])
    node_dist[root] = 0.0
    q = deque([root])
    while q:
        cur = q.popleft()
        for nb in adj[cur]:
            if nb not in visited:
                visited.add(nb)
                node_dist[nb] = node_dist[cur] + edge_len[(cur, nb)]
                q.append(nb)

    return dict(adj), edge_len, node_dist

# ----------------------------------------------------------------------
# Privacy helpers from Algorithm B
# ----------------------------------------------------------------------
def reconstruction_risk_score(unique_quasi_identifiers: int, total_records: int) -> float:
    """Simple proportion‑based reconstruction risk in [0,1]."""
    if total_records <= 0:
        return 0.0
    return max(0.0, min(1.0, unique_quasi_identifiers / total_records))

def health_score_simple(reconstruction_risk: float, recovery_priority: float) -> float:
    """Legacy health from Algorithm B (kept for comparison)."""
    return (1 - reconstruction_risk) * (1 - recovery_priority)

# ----------------------------------------------------------------------
# Model tier definition (Algorithm B)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str
    vram_mb: int

# Example tiers (could be extended by the user)
TIER_T1_QWEN_0_5B = ModelTier("qwen-0.5b", 512, "T1", 1024)
TIER_T2_REASONING = ModelTier("reasoning-t2", 3000, "T2", 2048)
TIER_T2_TOOL = ModelTier("tool-t2", 2600, "T2", 2048)
TIER_T3_QWEN_7B = ModelTier("qwen-7b", 7000, "T3", 4096)

# ----------------------------------------------------------------------
# Fusion Functions
# ----------------------------------------------------------------------
def compute_node_health(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
    *,
    unique_quasi_identifiers: Dict[str, int],
    total_records: int,
    failures: Dict[str, int],
    failure_threshold: int,
    recovery_priority: Dict[str, float],
    bandit_counts: Dict[Edge, int],
) -> Dict[str, float]:
    """
    Compute the unified health metric for every node.

    Parameters
    ----------
    unique_quasi_identifiers : mapping node → number of unique quasi‑identifiers
    failures                : mapping node → observed failure count
    recovery_priority       : mapping node → morphology‑driven priority in [0,1]
    bandit_counts           : mapping edge → count statistic from the bandit router

    Returns
    -------
    health : dict node → health value (non‑negative float)
    """
    # Prepare adjacency and edge lengths (needed for traversal)
    adj, edge_len, _ = tree_metrics(nodes, edges, root)

    # Propagate edge counts to nodes by simple averaging of incident edges
    node_counts: Dict[str, float] = {}
    for node in nodes:
        incident = adj[node]
        if not incident:
            node_counts[node] = 0.0
            continue
        total = sum(bandit_counts.get((node, nb), bandit_counts.get((nb, node), 0)) for nb in incident)
        node_counts[node] = total / len(incident)

    health: Dict[str, float] = {}
    for node in nodes:
        recon_risk = reconstruction_risk_score(
            unique_quasi_identifiers.get(node, 0), total_records
        )
        fail_rate = (
            failures.get(node, 0) / failure_threshold
            if failure_threshold > 0 else 0.0
        )
        rp = recovery_priority.get(node, 0.0)
        count = max(node_counts.get(node, 0), 0) + 1  # +1 to avoid log(0)

        h = (1 - (recon_risk * fail_rate)) * (1 - rp) * math.log(count)
        health[node] = max(0.0, h)  # enforce non‑negative

    return health

def allocate_workshare_to_models(
    models: List[ModelTier],
    node_health: Dict[str, float],
    *,
    vram_budget_mb: int,
) -> Dict[ModelTier, float]:
    """
    Convert node healths into a normalized work‑share vector and map it onto
    the supplied model tiers respecting a total VRAM budget.

    The algorithm:
    1. Sum all node healths → H_total.
    2. For each model, compute a raw share proportional to the average health of
       nodes whose name contains the model identifier (simple heuristic).
    3. Normalise the raw shares so that they sum to 1.
    4. Clip shares that would exceed the VRAM budget and renormalise.

    Returns
    -------
    dict mapping ModelTier → final work‑share fraction (in [0,1]).
    """
    if not models:
        raise ValueError("Model list cannot be empty")
    if vram_budget_mb <= 0:
        raise ValueError("VRAM budget must be positive")

    # 1. Global health total
    H_total = sum(node_health.values()) or 1.0

    # 2. Raw share per model (heuristic: sum health of nodes whose name contains model name)
    raw_shares: Dict[ModelTier, float] = {}
    for model in models:
        related_health = sum(
            h for name, h in node_health.items() if model.name.split("-")[0] in name.lower()
        )
        raw_shares[model] = related_health / H_total

    # Ensure every model gets at least a tiny share to avoid division by zero later
    epsilon = 1e-6
    for m in models:
        raw_shares[m] = max(raw_shares[m], epsilon)

    # 3. Normalise
    total_raw = sum(raw_shares.values())
    norm_shares = {m: s / total_raw for m, s in raw_shares.items()}

    # 4. Enforce VRAM budget: allocate VRAM proportionally, then clip
    vram_alloc = {m: norm_shares[m] * vram_budget_mb for m in models}
    for m, alloc in vram_alloc.items():
        if alloc > m.vram_mb:
            # Clip and redistribute excess
            excess = alloc - m.vram_mb
            vram_alloc[m] = m.vram_mb
            # Distribute excess uniformly among others that are not at their limit
            eligible = [x for x in models if x != m and vram_alloc[x] < x.vram_mb]
            if eligible:
                share = excess / len(eligible)
                for e in eligible:
                    vram_alloc[e] = min(e.vram_mb, vram_alloc[e] + share)

    # Final normalisation back to fractions
    total_vram = sum(vram_alloc.values()) or 1.0
    final_shares = {m: vram_alloc[m] / total_vram for m in models}
    return final_shares

def update_tree_with_workshare(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
    workshare: Dict[ModelTier, float],
) -> Dict[str, float]:
    """
    Demonstrates a downstream effect: each node receives a *load factor* derived
    from the work‑share of the model most associated with it.  The association
    uses the same heuristic as in `allocate_workshare_to_models`.

    Returns
    -------
    dict node → load factor (float in [0,1]).
    """
    load_factor: Dict[str, float] = {}
    for node in nodes:
        # Find the model with the highest workshare whose identifier appears in the node name
        candidate_shares = [
            workshare[m] for m in workshare
            if m.name.split("-")[0] in node.lower()
        ]
        load_factor[node] = max(candidate_shares) if candidate_shares else 0.0
    return load_factor

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple synthetic tree
    nodes = {
        "root": (0.0, 0.0),
        "a": (1.0, 0.0),
        "b": (0.0, 1.0),
        "c_qwen": (1.0, 1.0),
        "d_reasoning": (2.0, 0.0),
    }
    edges = [("root", "a"), ("root", "b"), ("a", "c_qwen"), ("a", "d_reasoning")]
    root = "root"

    # Synthetic auxiliary data
    unique_qi = {"root": 5, "a": 3, "b": 2, "c_qwen": 8, "d_reasoning": 6}
    total_records = 100
    failures = {"root": 0, "a": 1, "b": 0, "c_qwen": 2, "d_reasoning": 1}
    failure_threshold = 5
    recovery_priority = {"root": 0.1, "a": 0.2, "b": 0.15, "c_qwen": 0.05, "d_reasoning": 0.08}
    bandit_counts = {("root", "a"): 30, ("root", "b"): 20, ("a", "c_qwen"): 50, ("a", "d_reasoning"): 40}

    # Compute unified health per node
    node_health = compute_node_health(
        nodes,
        edges,
        root,
        unique_quasi_identifiers=unique_qi,
        total_records=total_records,
        failures=failures,
        failure_threshold=failure_threshold,
        recovery_priority=recovery_priority,
        bandit_counts=bandit_counts,
    )
    print("Node healths:", node_health)

    # Allocate workshare across models given a VRAM budget
    models = [TIER_T1_QWEN_0_5B, TIER_T2_REASONING, TIER_T2_TOOL, TIER_T3_QWEN_7B]
    workshare = allocate_workshare_to_models(models, node_health, vram_budget_mb=8192)
    print("Model workshare fractions:", {m.name: f for m, f in workshare.items()})

    # Propagate workshare back onto the tree (load factor per node)
    load_factor = update_tree_with_workshare(nodes, edges, root, workshare)
    print("Node load factors:", load_factor)

    # Verify that fractions sum to 1
    assert math.isclose(sum(workshare.values()), 1.0, rel_tol=1e-6), "Workshare does not sum to 1"
    print("Smoke test completed successfully.")