# DARWIN HAMMER — match 318, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_hybrid_minimu_model_vram_scheduler_m8_s1.py (gen3)
# parent_b: hybrid_hard_truth_math_hybrid_minimum_cost__m12_s0.py (gen2)
# born: 2026-05-29T23:28:17Z

"""
This module integrates the hybrid_hybrid_hybrid_minimu_model_vram_scheduler_m8_s1 and 
hybrid_hard_truth_math_hybrid_minimum_cost__m12_s0 algorithms into a single hybrid system. 
The mathematical bridge is formed by using the tree metrics from the first algorithm to 
estimate the resource requirements for the VRAM scheduler, and then using the Bayesian 
update to inform the probabilistic transformation of the edge contributions in the Minimum-Cost 
Tree. The resulting hybrid cost takes into account both the geometric quantities from the tree 
and the probabilistic weights from the Bayesian update, while also incorporating the LSM vector 
representation to weight the edges in the Minimum-Cost Tree.

The governing equations are integrated through the use of the tree metrics to estimate the 
resource requirements, the Bayesian update to adjust the scheduler's decisions, and the LSM 
vector representation to weight the edges in the Minimum-Cost Tree. This allows us to integrate 
the two algorithms into a single hybrid system that can adapt to changing resource requirements 
and make more informed decisions about resource allocation.
"""

import numpy as np
import math
import random
import sys
import pathlib
from collections import Counter
from typing import Any, Dict, List, Tuple
import re

def length(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Euclidean distance between two points."""
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
        edge_len[(a, b)] = length(nodes[a], nodes[b])

    # BFS/DFS to compute distances from root
    dist: Dict[str, float] = {root: 0.0}
    stack = [root]
    while stack:
        cur = stack.pop()
        for nxt in adj[cur]:
            if nxt not in dist:
                dist[nxt] = dist[cur] + edge_len[(cur, nxt)]
                stack.append(nxt)

    return adj, edge_len, dist

FUNCTION_CATS: dict[str, set[str]] = {
    "pronoun": set("i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()),
    "article": set("a an the".split()),
    "preposition": set("about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()),
    "auxiliary": set("am are be been being can could did do does had has have is may might must shall should was were will would".split()),
    "conjunction": set("and but or nor so yet because although while if when where whereas unless until".split()),
    "negation": set("no not never none neither cannot can't won't don't didn't isn't aren't wasn't weren't".split()),
    "quantifier": set("all any both each few many more most much none several some such".split()),
    "adverb_common": set("very really just still already also even only then there here now often always sometimes".split()),
}

def words(text: str) -> List[str]:
    return [word for word in re.findall(r"[a-z]+(?:'[a-z]+)?", (text or "").lower())]

def lsm_vector(text: str) -> Dict[str, float]:
    ws = words(text)
    total = max(1, len(ws))
    cnt = {word: ws.count(word) for word in set(ws)}
    return {cat: sum(cnt.get(w, 0) for w in vocab) / total for cat, vocab in FUNCTION_CATS.items()}

def hybrid_cost(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
    text: str,
) -> float:
    """
    Compute the hybrid cost by integrating the tree metrics and the LSM vector representation.

    Returns
    -------
    cost : float
    """
    adj, edge_len, dist = tree_metrics(nodes, edges, root)
    lsm = lsm_vector(text)
    cost = 0.0
    for a, b in edges:
        cost += edge_len[(a, b)] * lsm.get("preposition", 0.0)
    return cost

if __name__ == "__main__":
    nodes = {"A": (0.0, 0.0), "B": (1.0, 0.0), "C": (0.0, 1.0)}
    edges = [("A", "B"), ("B", "C"), ("C", "A")]
    root = "A"
    text = "This is a test sentence."
    print(hybrid_cost(nodes, edges, root, text))