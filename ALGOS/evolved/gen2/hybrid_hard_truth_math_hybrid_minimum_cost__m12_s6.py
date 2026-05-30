# DARWIN HAMMER — match 12, survivor 6
# gen: 2
# parent_a: hard_truth_math.py (gen0)
# parent_b: hybrid_minimum_cost_tree_bayes_update_m6_s2.py (gen1)
# born: 2026-05-29T23:25:17Z

import math
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np

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
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    edge_len: Dict[Edge, float] = {}

    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        l = length(nodes[u], nodes[v])
        edge_len[(u, v)] = l
        edge_len[(v, u)] = l

    root_dist: Dict[str, float] = {root: 0.0}
    visited = {root}
    frontier = [root]
    while frontier:
        cur = frontier.pop()
        for nb in adj[cur]:
            if nb not in visited:
                visited.add(nb)
                root_dist[nb] = root_dist[cur] + edge_len[(cur, nb)]
                frontier.append(nb)

    return adj, edge_len, root_dist

def bayes_edge_posteriors(
    priors: Dict[Edge, float],
    likelihoods: Dict[Edge, float],
    fp_rate: float,
) -> Dict[Edge, float]:
    post: Dict[Edge, float] = {}
    for e, p_prior in priors.items():
        L = likelihoods.get(e, 0.0)
        numerator = p_prior * L
        denominator = L * p_prior + fp_rate * (1.0 - p_prior)
        post[e] = numerator / denominator if denominator != 0 else 0.0
    return post

def node_beliefs(
    adj: Dict[str, List[str]],
    edge_post: Dict[Edge, float],
) -> Dict[str, float]:
    q: Dict[str, float] = {}
    for v, neigh in adj.items():
        if not neigh:
            q[v] = 0.0
            continue
        total = sum(edge_post.get((v, nb), 0.0) for nb in neigh)
        q[v] = total / len(neigh)
    return q

def hybrid_tree_cost(
    edge_len: Dict[Edge, float],
    root_dist: Dict[str, float],
    edge_post: Dict[Edge, float],
    node_belief: Dict[str, float],
    lam: float = 0.5,
) -> float:
    edge_term = sum(edge_post[e] * l for e, l in edge_len.items() if e in edge_post)
    node_term = lam * sum(node_belief[v] * d for v, d in root_dist.items())
    return edge_term + node_term

def build_lsm_vectors(texts: Dict[str, str]) -> Dict[str, Dict[str, float]]:
    return {nid: lsm_vector(txt) for nid, txt in texts.items()}

def edge_likelihoods_from_lsm(
    lsm_vectors: Dict[str, Dict[str, float]],
    edges: List[Edge],
) -> Dict[Edge, float]:
    likelihoods: Dict[Edge, float] = {}
    for u, v in edges:
        score, _ = lsm_score(lsm_vectors[u], lsm_vectors[v])
        likelihoods[(u, v)] = score
        likelihoods[(v, u)] = score
    return likelihoods

def improved_hybrid_tree_cost(
    texts: Dict[str, str],
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
    priors: Dict[Edge, float],
    fp_rate: float,
    lam: float = 0.5,
) -> float:
    lsm_vectors = build_lsm_vectors(texts)
    likelihoods = edge_likelihoods_from_lsm(lsm_vectors, edges)
    adj, edge_len, root_dist = tree_metrics(nodes, edges, root)
    edge_post = bayes_edge_posteriors(priors, likelihoods, fp_rate)
    node_belief = node_beliefs(adj, edge_post)
    return hybrid_tree_cost(edge_len, root_dist, edge_post, node_belief, lam)

# Example usage:
texts = {
    "A": "This is a sample text.",
    "B": "This is another sample text.",
    "C": "This is yet another sample text.",
}
nodes = {
    "A": (0.0, 0.0),
    "B": (1.0, 0.0),
    "C": (0.5, 1.0),
}
edges = [("A", "B"), ("B", "C"), ("C", "A")]
root = "A"
priors = {
    ("A", "B"): 0.5,
    ("B", "C"): 0.5,
    ("C", "A"): 0.5,
}
fp_rate = 0.1
lam = 0.5

cost = improved_hybrid_tree_cost(texts, nodes, edges, root, priors, fp_rate, lam)
print(cost)