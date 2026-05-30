# DARWIN HAMMER — match 551, survivor 2
# gen: 4
# parent_a: hybrid_hybrid_hybrid_bandit_hybrid_path_signatur_m56_s2.py (gen3)
# parent_b: hybrid_hard_truth_math_hybrid_minimum_cost__m12_s6.py (gen2)
# born: 2026-05-29T23:29:46Z

"""Hybrid LSM‑Tree‑Store‑Bandit Fusion
===================================

Parent A: *hybrid_hybrid_hybrid_bandit_hybrid_path_signatur_m56_s2.py* – provides a
store dynamics equation that consumes “inflow” and “outflow” vectors and emits a
scalar “dance” signal which rescales bandit propensities.

Parent B: *hybrid_hard_truth_math_hybrid_minimum_cost__m12_s6.py* – supplies a
lexical‑function‑category (LSM) representation of text and a similarity score,
together with a generic tree‑metric routine that yields edge lengths.

**Mathematical bridge**

* Inflow is taken as the *overall LSM similarity* between two sentences (a
  scalar in [0, 1]).
* Outflow is taken as the *sum of edge lengths* of a tree that encodes structural
  information (e.g. a parse tree).  
* The store update equation from Parent A  

\[
\Delta = \alpha\sum\text{inflow} - \beta\sum\text{outflow},\qquad
\text{level}_{t+1}= \max(0,\;\text{level}_t + \Delta\cdot dt)
\]

is applied with the above inflow/outflow.  
* The resulting “dance’’ signal  

\[
\text{dance}= \tanh(\gamma\Delta)
\]

modulates the raw bandit propensities exactly as in Parent A.

Thus the two topologies are mathematically fused into a single closed‑loop
system that can be used for text‑driven decision making.

The module implements three core functions demonstrating the hybrid operation
and ends with a tiny smoke test.
"""

import math
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent‑A data structures (trimmed)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    """Result of an action selection."""
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str


@dataclass
class StoreState:
    """Honeybee‑style store and derived control signal."""
    level: float = 0.0
    alpha: float = 1.0   # inflow gain
    beta: float = 1.0    # outflow gain
    dt: float = 1.0
    base: float = 1.0    # unused but kept for compatibility
    gamma: float = 1.0   # gain for dance non‑linearity


# ----------------------------------------------------------------------
# Parent‑B lexical‑function‑category machinery
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
    """Tokenise a string into lower‑case words (simple regex)."""
    return re.findall(r"[a-z]+(?:'[a-z]+)?", (text or "").lower())


def lsm_vector(text: str) -> Dict[str, float]:
    """
    Compute a lexical‑function‑category (LSM) vector for *text*.
    The entry for a category is the normalised frequency of words belonging
    to that category.
    """
    ws = words(text)
    total = max(1, len(ws))
    cnt = Counter(ws)
    return {
        cat: sum(cnt[w] for w in vocab) / total
        for cat, vocab in FUNCTION_CATS.items()
    }


def lsm_score(a: Dict[str, float], b: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    """
    Compute a similarity score between two LSM vectors.
    For each category we use

        s = 1 - |a - b| / (a + b + ε)

    and clamp to [0, 1].  The overall score is the mean over categories.
    """
    detail: Dict[str, float] = {}
    for cat in FUNCTION_CATS:
        av = a.get(cat, 0.0)
        bv = b.get(cat, 0.0)
        score = 1.0 - (abs(av - bv) / (av + bv + 1e-6))
        score = max(0.0, min(1.0, score))
        detail[cat] = score
    overall = sum(detail.values()) / len(FUNCTION_CATS)
    return overall, detail


# ----------------------------------------------------------------------
# Simple tree utilities (from Parent B)
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Edge = Tuple[str, str]


def length(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def tree_metrics(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
) -> Tuple[Dict[str, List[str]], Dict[Edge, float], Dict[str, float]]:
    """
    Produce three structures for a rooted tree:

    * adjacency list,
    * edge‑length map,
    * depth (cumulative distance from root) for each node.
    """
    adjacency: Dict[str, List[str]] = {k: [] for k in nodes}
    edge_len: Dict[Edge, float] = {}
    for u, v in edges:
        adjacency[u].append(v)
        adjacency[v].append(u)  # undirected for simplicity
        edge_len[(u, v)] = edge_len[(v, u)] = length(nodes[u], nodes[v])

    depth: Dict[str, float] = {root: 0.0}
    visited = {root}
    stack = [root]
    while stack:
        cur = stack.pop()
        for nb in adjacency[cur]:
            if nb in visited:
                continue
            visited.add(nb)
            depth[nb] = depth[cur] + edge_len[(cur, nb)]
            stack.append(nb)

    return adjacency, edge_len, depth


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def compute_inflow(text_a: str, text_b: str) -> float:
    """
    Inflow is defined as the overall LSM similarity between two sentences.
    The result lies in [0, 1] and is interpreted as a scalar flow.
    """
    vec_a = lsm_vector(text_a)
    vec_b = lsm_vector(text_b)
    overall, _ = lsm_score(vec_a, vec_b)
    return overall


def compute_outflow(nodes: Dict[str, Point], edges: List[Edge]) -> float:
    """
    Outflow is defined as the sum of all edge lengths in the supplied tree.
    This provides a positive scalar that grows with structural complexity.
    """
    _, edge_len, _ = tree_metrics(nodes, edges, root=list(nodes)[0])
    return sum(edge_len.values())


def store_update_from_signals(
    store: StoreState,
    inflow: float,
    outflow: float,
) -> None:
    """
    Apply the Parent‑A store dynamics using the hybrid inflow/outflow.
    The function mutates *store* in‑place.
    """
    delta = store.alpha * inflow - store.beta * outflow
    store.level = max(0.0, store.level + delta * store.dt)
    # store.dance is a derived signal used by the bandit adjustment step
    store.dance = math.tanh(store.gamma * delta)


def adjust_bandit_propensities(
    actions: List[BanditAction],
    store: StoreState,
) -> List[BanditAction]:
    """
    Rescale each action's propensity by (1 + dance).  The higher the store
    activity, the more exploratory the bandit becomes.
    Returns a new list of actions with updated propensities.
    """
    factor = 1.0 + store.dance
    adjusted = []
    for a in actions:
        new_prop = max(0.0, a.propensity * factor)
        adjusted.append(
            BanditAction(
                action_id=a.action_id,
                propensity=new_prop,
                expected_reward=a.expected_reward,
                confidence_bound=a.confidence_bound,
                algorithm=a.algorithm,
            )
        )
    return adjusted


def lead_lag_bspline_signature(
    path: np.ndarray,
    degree: int = 3,
    n_knots: int = 10,
) -> np.ndarray:
    """
    Very lightweight stand‑in for the full lead‑lag → B‑spline signature
    pipeline from Parent A.  It builds a lead‑lag matrix, constructs an
    equally‑spaced knot vector and returns the projection coefficients obtained
    by solving a least‑squares problem.

    The implementation is deliberately simple to stay within the standard‑library
    constraint (no SciPy).  It suffices to illustrate that the hybrid system can
    also accept genuine path‑signature data as an alternative inflow source.
    """
    if path.ndim != 2 or path.shape[1] != 2:
        raise ValueError("path must be (T, 2) array representing a 2‑D trajectory")
    # Lead‑lag transform: concatenate (x, y, Δx, Δy)
    diffs = np.diff(path, axis=0, prepend=path[0:1])
    lead_lag = np.hstack([path, diffs])
    T, D = lead_lag.shape

    # Build uniform B‑spline basis matrix B of shape (T, n_knots)
    knots = np.linspace(0, 1, n_knots - degree + 1)
    # Simple hat functions as a proxy for B‑splines
    B = np.zeros((T, n_knots))
    t_vals = np.linspace(0, 1, T)
    for i, k in enumerate(knots):
        B[:, i] = np.maximum(0, 1 - np.abs(t_vals - k) * (n_knots - 1))
    # Least‑squares projection
    coeffs, _, _, _ = np.linalg.lstsq(B, lead_lag, rcond=None)
    return coeffs  # shape (n_knots, D)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":

    # ---- 1. Textual inflow -------------------------------------------------
    txt1 = "The quick brown fox jumps over the lazy dog."
    txt2 = "A fast dark-colored fox leapt above a sleepy canine."
    inflow = compute_inflow(txt1, txt2)
    print(f"Inflow (LSM similarity): {inflow:.4f}")

    # ---- 2. Structural outflow --------------------------------------------
    # Simple binary tree in the plane
    nodes_example = {
        "root": (0.0, 0.0),
        "left": (-1.0, -1.0),
        "right": (1.0, -1.0),
        "leaf1": (-1.5, -2.0),
        "leaf2": (-0.5, -2.0),
        "leaf3": (0.5, -2.0),
        "leaf4": (1.5, -2.0),
    }
    edges_example = [
        ("root", "left"),
        ("root", "right"),
        ("left", "leaf1"),
        ("left", "leaf2"),
        ("right", "leaf3"),
        ("right", "leaf4"),
    ]
    outflow = compute_outflow(nodes_example, edges_example)
    print(f"Outflow (sum of edge lengths): {outflow:.4f}")

    # ---- 3. Store dynamics -------------------------------------------------
    store = StoreState(level=0.5, alpha=2.0, beta=0.5, dt=0.1, gamma=3.0)
    store_update_from_signals(store, inflow, outflow)
    print(f"Updated store level: {store.level:.4f}, dance signal: {store.dance:.4f}")

    # ---- 4. Bandit propensities adjustment ---------------------------------
    raw_actions = [
        BanditAction("A", propensity=0.2, expected_reward=1.0, confidence_bound=0.1, algorithm="UCB"),
        BanditAction("B", propensity=0.5, expected_reward=0.8, confidence_bound=0.2, algorithm="UCB"),
        BanditAction("C", propensity=0.3, expected_reward=0.6, confidence_bound=0.15, algorithm="UCB"),
    ]
    adjusted = adjust_bandit_propensities(raw_actions, store)
    for a in adjusted:
        print(f"Action {a.action_id}: adjusted propensity = {a.propensity:.4f}")

    # ---- 5. Optional: feed a synthetic path into the signature routine -------
    # Demonstrates that the hybrid can also accept a genuine signature as inflow.
    t = np.linspace(0, 1, 100)
    synthetic_path = np.column_stack((np.sin(2 * np.pi * t), np.cos(2 * np.pi * t)))
    coeffs = lead_lag_bspline_signature(synthetic_path, degree=3, n_knots=12)
    print(f"Signature coefficient matrix shape: {coeffs.shape}")

    print("Smoke test completed without errors.")