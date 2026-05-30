# DARWIN HAMMER — match 1913, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hard_truth_ma_m318_s2.py (gen4)
# parent_b: hybrid_hybrid_hybrid_regret_regret_engine_m822_s1.py (gen4)
# born: 2026-05-29T23:39:41Z

"""Hybrid Algorithm integrating:
- Parent A: LSM vector text representation and geometric edge costs (minimum‑cost tree).
- Parent B: Regret‑Weighted Strategy with Bandit Router.

Mathematical bridge:
Each action (or graph edge) carries a textual identifier.  The LSM vector of this
identifier provides a scalar weight w_lsm = Σ_k v_k (term‑frequency).  The
regret‑weighted strategy yields a probability distribution p_regret over the
same actions.  We fuse them multiplicatively:
    p_hybrid = p_regret * w_lsm
The hybrid probability re‑weights the geometric edge length ℓ(i,j) when building
the Minimum‑Cost Spanning Tree (MST):
    cost_hybrid(i,j) = ℓ(i,j) / p_hybrid
Thus the MST respects both the geometry of the points and the learned
regret‑adjusted, text‑driven importance of each edge.  The resulting system
produces:
1. LSM vectors from text.
2. Regret‑weighted probabilities.
3. A hybrid MST whose edges are selected by a bandit‑style propensity
   derived from the product of the two weights.
"""

import math
import random
import sys
import pathlib
from collections import Counter
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A utilities (text → LSM vector, geometric helpers)
# ----------------------------------------------------------------------
def length(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Euclidean distance between two 2‑D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def words(text: str) -> List[str]:
    """Lower‑case word tokenisation (alphabetic + optional apostrophe)."""
    import re

    return [w for w in re.findall(r"[a-z]+(?:'[a-z]+)?", (text or "").lower())]


def lsm_vector(text: str) -> Dict[str, float]:
    """
    Simple term‑frequency LSM vector for a piece of text.
    Returns a dictionary mapping each word to its relative frequency.
    """
    ws = words(text)
    total = max(1, len(ws))
    cnt = Counter(ws)
    return {word: cnt[word] / total for word in cnt}


def lsm_scalar_weight(text: str) -> float:
    """
    Collapse an LSM vector to a single positive scalar.
    We use the L1 norm (sum of absolute frequencies) which equals 1 for non‑empty text,
    but for empty strings we return a small epsilon to avoid division by zero.
    """
    vec = lsm_vector(text)
    if not vec:
        return 1e-6
    return sum(abs(v) for v in vec.values())


# ----------------------------------------------------------------------
# Parent B utilities (regret‑weighted strategy & bandit router)
# ----------------------------------------------------------------------
from dataclasses import dataclass, field


@dataclass(frozen=True)
class MathAction:
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True)
class MathCounterfactual:
    action_id: str
    outcome_value: float
    probability: float = 1.0


@dataclass(frozen=True)
class BanditAction:
    """Result of an action selection."""
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "HybridRegretLSM"


def compute_regret_weighted_strategy(
    actions: List[MathAction], counterfactuals: List[MathCounterfactual]
) -> Dict[str, float]:
    """
    Regret‑weighted soft‑max over (expected_value – cost – risk + counterfactual).
    Returns a normalized probability distribution over action ids.
    """
    if not actions:
        return {}
    cf = {c.action_id: c.outcome_value * c.probability for c in counterfactuals}
    vals = {
        a.id: a.expected_value - a.cost - a.risk + cf.get(a.id, 0.0) for a in actions
    }
    best = max(vals.values())
    # Numerically stable soft‑max using subtraction of max
    w = {k: math.exp(v - best) for k, v in vals.items()}
    total = sum(w.values()) or 1.0
    return {k: v / total for k, v in w.items()}


def hybrid_bandit_actions(
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual],
) -> List[BanditAction]:
    """
    Produce BanditAction objects whose propensity is the product of:
        p_regret  – from the regret‑weighted strategy
        w_lsm     – scalar LSM weight derived from the action id string
    The confidence bound is a simple heuristic: sqrt(p_hybrid) * expected_value.
    """
    regret_probs = compute_regret_weighted_strategy(actions, counterfactuals)
    bandit_list: List[BanditAction] = []
    for a in actions:
        w_lsm = lsm_scalar_weight(a.id)
        p_hybrid = regret_probs.get(a.id, 0.0) * w_lsm
        # Guard against zero propensity
        p_hybrid = max(p_hybrid, 1e-9)
        confidence = math.sqrt(p_hybrid) * a.expected_value
        bandit_list.append(
            BanditAction(
                action_id=a.id,
                propensity=p_hybrid,
                expected_reward=a.expected_value,
                confidence_bound=confidence,
            )
        )
    # Normalise propensities to sum to 1 for downstream use
    total = sum(b.propensity for b in bandit_list) or 1.0
    bandit_list = [
        BanditAction(
            action_id=b.action_id,
            propensity=b.propensity / total,
            expected_reward=b.expected_reward,
            confidence_bound=b.confidence_bound,
            algorithm=b.algorithm,
        )
        for b in bandit_list
    ]
    return bandit_list


# ----------------------------------------------------------------------
# Hybrid core: geometry + regret‑LSM propensities → Minimum‑Cost Spanning Tree
# ----------------------------------------------------------------------
Edge = Tuple[int, int, float, str]  # (node_i, node_j, geometric_length, label)


def build_complete_edge_list(
    points: List[Tuple[float, float]], labels: List[str]
) -> List[Edge]:
    """
    Construct a complete undirected graph.
    Each edge receives a label formed by concatenating the identifiers of its two endpoints.
    """
    n = len(points)
    edges: List[Edge] = []
    for i in range(n):
        for j in range(i + 1, n):
            geom_len = length(points[i], points[j])
            label = f"{labels[i]}_{labels[j]}"
            edges.append((i, j, geom_len, label))
    return edges


def compute_hybrid_edge_cost(edge: Edge, propensity: float) -> float:
    """
    Hybrid cost = geometric length divided by the hybrid propensity.
    Larger propensity (more important edge) reduces its contribution to the total cost.
    """
    _, _, geom_len, _ = edge
    # Avoid division by zero
    return geom_len / max(propensity, 1e-9)


def prim_mst_hybrid(
    points: List[Tuple[float, float]],
    labels: List[str],
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual],
) -> List[Edge]:
    """
    Prim's algorithm on the complete graph where each edge cost is the hybrid cost.
    The hybrid propensities are obtained from the regret‑LSM bandit router.
    Returns the list of edges belonging to the MST (using original geometric length).
    """
    # Step 1 – obtain hybrid propensities for each action (edge)
    bandit_actions = hybrid_bandit_actions(actions, counterfactuals)
    propensity_map = {b.action_id: b.propensity for b in bandit_actions}

    # Step 2 – build edge list with hybrid costs
    all_edges = build_complete_edge_list(points, labels)
    hybrid_costs = {
        (e[0], e[1]): compute_hybrid_edge_cost(e, propensity_map.get(e[3], 1e-6))
        for e in all_edges
    }

    n = len(points)
    in_mst = [False] * n
    key = [float("inf")] * n
    parent = [-1] * n

    key[0] = 0.0

    for _ in range(n):
        # Pick the vertex with minimum key value not yet in MST
        u = min((idx for idx in range(n) if not in_mst[idx]), key=lambda idx: key[idx])
        in_mst[u] = True

        # Update keys of adjacent vertices
        for v in range(n):
            if in_mst[v] or u == v:
                continue
            i, j = (min(u, v), max(u, v))
            cost = hybrid_costs.get((i, j), float("inf"))
            if cost < key[v]:
                key[v] = cost
                parent[v] = u

    # Reconstruct MST edge list (using original geometric lengths)
    mst_edges: List[Edge] = []
    for v in range(1, n):
        u = parent[v]
        if u == -1:
            continue
        # Find the original edge data
        label = f"{labels[min(u, v)]}_{labels[max(u, v)]}"
        geom_len = length(points[u], points[v])
        mst_edges.append((u, v, geom_len, label))
    return mst_edges


# ----------------------------------------------------------------------
# Demonstration functions
# ----------------------------------------------------------------------
def generate_random_points(n: int, seed: int = 42) -> List[Tuple[float, float]]:
    random.seed(seed)
    return [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(n)]


def create_actions_from_edges(edges: List[Edge]) -> List[MathAction]:
    """
    Each edge becomes a MathAction where the expected value is the geometric length.
    """
    return [
        MathAction(id=e[3], expected_value=e[2], cost=0.0, risk=0.0) for e in edges
    ]


def demo_hybrid_mst() -> None:
    """Run a small demo: build a hybrid MST and print its total geometric length."""
    points = generate_random_points(6)
    labels = [f"node{i}" for i in range(len(points))]

    # Build complete edge list to generate actions
    all_edges = build_complete_edge_list(points, labels)
    actions = create_actions_from_edges(all_edges)
    counterfactuals: List[MathCounterfactual] = []  # empty for this demo

    mst = prim_mst_hybrid(points, labels, actions, counterfactuals)
    total_geom = sum(e[2] for e in mst)
    print("Hybrid MST edges (node_i, node_j, geometric_length, label):")
    for e in mst:
        print(e)
    print(f"Total geometric length of MST: {total_geom:.3f}")


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    demo_hybrid_mst()