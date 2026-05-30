# DARWIN HAMMER — match 12, survivor 0
# gen: 2
# parent_a: hard_truth_math.py (gen0)
# parent_b: hybrid_minimum_cost_tree_bayes_update_m6_s2.py (gen1)
# born: 2026-05-29T23:25:17Z

"""
Hybrid module combining hard-truth telemetry algorithms from 'hard_truth_math.py' and 
Minimum-Cost Tree scoring with Bayesian evidence update from 'hybrid_minimum_cost_tree_bayes_update_m6_s2.py'.

The mathematical bridge is formed by using the LSM vector representation from 'hard_truth_math.py' 
to weight the edges in the Minimum-Cost Tree, while using the Bayesian update to inform the 
probabilistic transformation of the edge contributions. The resulting hybrid cost takes into 
account both the geometric quantities from the tree and the probabilistic weights from the 
Bayesian update.

Output of this module includes a hybrid cost metric that combines the strengths of both parent modules.
"""

from __future__ import annotations
import math
import numpy as np
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

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

def length(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def words(text: str) -> List[str]:
    return [word for word in re.findall(r"[a-z]+(?:'[a-z]+)?", (text or "").lower())]

def lsm_vector(text: str) -> Dict[str, float]:
    ws = words(text)
    total = max(1, len(ws))
    cnt = {word: ws.count(word) for word in set(ws)}
    return {cat: sum(cnt.get(w, 0) for w in vocab) / total for cat, vocab in FUNCTION_CATS.items()}

def lsm_score(a: Dict[str, float], b: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    detail: Dict[str, float] = {}
    vals: List[float] = []
    for cat in FUNCTION_CATS:
        av = a.get(cat, 0.0)
        bv = b.get(cat, 0.0)
        score = 1.0 - (abs(av - bv) / (av + bv + 1e-6))
        score = max(0.0, min(1.0, score))
        detail[cat] = score
        vals.append(score)
    return np.mean(vals), detail

def tree_metrics(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
) -> Tuple[Dict[str, List[str]], Dict[Tuple[str, str], float], Dict[str, float]]:
    adj: Dict[str, List[str]] = {node: [] for node in nodes}
    edge_len: Dict[Tuple[str, str], float] = {}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        edge_len[(a, b)] = length(nodes[a], nodes[b])
        edge_len[(b, a)] = length(nodes[b], nodes[a])
    distances: Dict[str, float] = {node: 0.0 for node in nodes}
    queue: List[str] = [root]
    for node in queue:
        for neighbor in adj[node]:
            if distances[neighbor] == 0.0:
                distances[neighbor] = distances[node] + edge_len[(node, neighbor)]
                queue.append(neighbor)
    return adj, edge_len, distances

def bayes_edge_posteriors(
    edge_len: Dict[Tuple[str, str], float],
    prior: Dict[Tuple[str, str], float],
    likelihood: Dict[Tuple[str, str], float],
    fp_rate: float,
) -> Dict[Tuple[str, str], float]:
    posteriors: Dict[Tuple[str, str], float] = {}
    for edge in edge_len:
        p_prior = prior.get(edge, 0.0)
        L = likelihood.get(edge, 1.0)
        p_post = (p_prior * L) / (L * p_prior + fp_rate * (1 - p_prior))
        posteriors[edge] = p_post
    return posteriors

def hybrid_tree_cost(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
    prior: Dict[Tuple[str, str], float],
    likelihood: Dict[Tuple[str, str], float],
    fp_rate: float,
    lambda_: float,
) -> float:
    adj, edge_len, distances = tree_metrics(nodes, edges, root)
    posteriors = bayes_edge_posteriors(edge_len, prior, likelihood, fp_rate)
    cost = sum(posteriors.get(edge, 0.0) * edge_len.get(edge, 0.0) for edge in edge_len)
    cost += lambda_ * sum(distances.get(node, 0.0) for node in distances)
    return cost

if __name__ == "__main__":
    nodes = {
        "A": (0.0, 0.0),
        "B": (3.0, 4.0),
        "C": (6.0, 0.0),
    }
    edges = [("A", "B"), ("B", "C"), ("C", "A")]
    root = "A"
    prior = {("A", "B"): 0.5, ("B", "C"): 0.7, ("C", "A"): 0.3}
    likelihood = {("A", "B"): 0.8, ("B", "C"): 0.9, ("C", "A"): 0.4}
    fp_rate = 0.1
    lambda_ = 0.5
    print(hybrid_tree_cost(nodes, edges, root, prior, likelihood, fp_rate, lambda_))