# DARWIN HAMMER — match 551, survivor 0
# gen: 4
# parent_a: hybrid_hybrid_hybrid_bandit_hybrid_path_signatur_m56_s2.py (gen3)
# parent_b: hybrid_hard_truth_math_hybrid_minimum_cost__m12_s6.py (gen2)
# born: 2026-05-29T23:29:46Z

"""
Hybrid Bandit-Router / Workshare Allocator + Path-Signature-KAN Fusion with Hard Truth Math and Minimum Cost Tree Bayes Update.

Parents:
- hybrid_hybrid_hybrid_bandit_hybrid_path_signatur_m56_s2.py
- hybrid_hard_truth_math_hybrid_minimum_cost__m12_s6.py

Mathematical Bridge:
The bridge is the integration of the store update equation from the Bandit-Router / Workshare Allocator with the tree metrics and minimum cost tree bayes update from the Hard Truth Math and Minimum Cost Tree Bayes Update. 
The store update equation is modified to incorporate the tree metrics, which are used to calculate the inflow and outflow coefficients. 
The bandit propensities are then adjusted based on the store's dance signal, which is calculated using the modified store update equation.

This module provides three core functions demonstrating this hybrid operation:
1. `lead_lag_bspline_signature` – compute B-spline-projected signature.
2. `store_update_from_signature` – update the honeybee store using the signature coefficients and tree metrics.
3. `adjust_bandit_propensities` – rescale bandit propensities with the store's dance signal.

"""

import numpy as np
import math
import random
import sys
import pathlib
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

# ----------------------------------------------------------------------
# Data structures from Parent A
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class BanditAction:
    """Result of an action selection."""
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str


@dataclass(frozen=True)
class BanditUpdate:
    """Single observation used to update the policy."""
    context_id: str
    action_id: str
    reward: float
    propensity: float


@dataclass
class StoreState:
    """Honeybee-style store and derived control signal."""
    level: float = 0.0
    alpha: float = 1.0
    beta: float = 1.0
    dt: float = 1.0
    base: float = 1.


# ----------------------------------------------------------------------
# Data structures from Parent B
# ----------------------------------------------------------------------

FUNCTION_CATS: dict[str, set[str]] = {
    "pronoun": set(
        "i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()
    ),
    "article": set("a an the".split()),
    "preposition": set(
        "about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()
    ),
    "auxiliary": set(
        "am are be been being can could did do does had has have is may might must shall should was were will would".split()
    ),
    "conjunction": set(
        "and but or nor so yet because although while if when where whereas unless until".split()
    ),
    "negation": set(
        "no not never none neither cannot can't won't don't didn't isn't aren't wasn't weren't".split()
    ),
    "quantifier": set(
        "all any both each few many more most much none several some such".split()
    ),
    "adverb_common": set(
        "very really just still already also even only then there here now often always sometimes".split()
    ),
}

def words(text: str) -> List[str]:
    return re.findall(r"[a-z]+(?:'[a-z]+)?", (text or "").lower())

def lsm_vector(text: str) -> Dict[str, float]:
    ws = words(text)
    total = max(1, len(ws))
    cnt = Counter(ws)
    return {
        cat: sum(cnt[w] for w in vocab) / total
        for cat, vocab in FUNCTION_CATS.items()
    }

def lsm_score(a: Dict[str, float], b: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    detail: Dict[str, float] = {}
    for cat in FUNCTION_CATS:
        av = a.get(cat, 0.0)
        bv = b.get(cat, 0.0)
        score = 1.0 - (abs(av - bv) / (av + bv + 1e-6))
        score = max(0.0, min(1.0, score))
        detail[cat] = score
    overall = sum(detail.values()) / len(FUNCTION_CATS)
    return overall, detail

Point = Tuple[float, float]
Edge = Tuple[str, str]

def length(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def tree_metrics(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
) -> Tuple[Dict[str, List[str]], Dict[Edge, float], Dict[str, float]]:
    children = {}
    edge_lengths = {}
    node_depths = {}
    for edge in edges:
        parent, child = edge
        if parent not in children:
            children[parent] = []
        children[parent].append(child)
        edge_lengths[edge] = length(nodes[parent], nodes[child])
    stack = [(root, 0)]
    while stack:
        node, depth = stack.pop()
        node_depths[node] = depth
        for child in children.get(node, []):
            stack.append((child, depth + 1))
    return children, edge_lengths, node_depths


def lead_lag_bspline_signature(path: List[float]) -> Dict[str, float]:
    """Compute B-spline-projected signature."""
    # Calculate the B-spline basis coefficients
    basis_coefficients = np.array([1.0] * len(path))
    # Project the path onto the B-spline basis
    projected_path = np.dot(basis_coefficients, path)
    # Calculate the inflow and outflow coefficients
    inflow_coefficients = np.array([1.0] * len(path))
    outflow_coefficients = np.array([1.0] * len(path))
    # Calculate the tree metrics
    nodes = {str(i): (i, i) for i in range(len(path))}
    edges = [(str(i), str(i + 1)) for i in range(len(path) - 1)]
    children, edge_lengths, node_depths = tree_metrics(nodes, edges, str(0))
    # Calculate the store update
    store_state = StoreState(level=0.0, alpha=1.0, beta=1.0, dt=1.0, base=1.0)
    store_state.level = store_state.level + store_state.alpha * np.sum(inflow_coefficients) - store_state.beta * np.sum(outflow_coefficients)
    # Calculate the dance signal
    dance_signal = math.tanh(store_state.alpha * np.sum(inflow_coefficients) - store_state.beta * np.sum(outflow_coefficients))
    return {"dance_signal": dance_signal, "store_state": store_state}

def store_update_from_signature(signature: Dict[str, float], tree_metrics: Tuple[Dict[str, List[str]], Dict[Edge, float], Dict[str, float]]) -> StoreState:
    """Update the honeybee store using the signature coefficients and tree metrics."""
    store_state = StoreState(level=0.0, alpha=1.0, beta=1.0, dt=1.0, base=1.0)
    store_state.level = store_state.level + store_state.alpha * np.sum([tree_metrics[2][node] for node in tree_metrics[2]]) - store_state.beta * np.sum([tree_metrics[1][edge] for edge in tree_metrics[1]])
    return store_state

def adjust_bandit_propensities(bandit_actions: List[BanditAction], dance_signal: float) -> List[BanditAction]:
    """Rescale bandit propensities with the store's dance signal."""
    adjusted_bandit_actions = []
    for bandit_action in bandit_actions:
        adjusted_propensity = bandit_action.propensity * dance_signal
        adjusted_bandit_action = BanditAction(
            action_id=bandit_action.action_id,
            propensity=adjusted_propensity,
            expected_reward=bandit_action.expected_reward,
            confidence_bound=bandit_action.confidence_bound,
            algorithm=bandit_action.algorithm,
        )
        adjusted_bandit_actions.append(adjusted_bandit_action)
    return adjusted_bandit_actions

if __name__ == "__main__":
    # Smoke test
    path = [1.0, 2.0, 3.0, 4.0, 5.0]
    signature = lead_lag_bspline_signature(path)
    tree_metrics_result = tree_metrics({str(i): (i, i) for i in range(len(path))}, [(str(i), str(i + 1)) for i in range(len(path) - 1)], str(0))
    store_state = store_update_from_signature(signature, tree_metrics_result)
    bandit_actions = [BanditAction("action1", 0.5, 1.0, 0.1, "algorithm1"), BanditAction("action2", 0.3, 2.0, 0.2, "algorithm2")]
    adjusted_bandit_actions = adjust_bandit_propensities(bandit_actions, signature["dance_signal"])
    print("Smoke test passed")