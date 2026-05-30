# DARWIN HAMMER — match 12, survivor 5
# gen: 2
# parent_a: hard_truth_math.py (gen0)
# parent_b: hybrid_minimum_cost_tree_bayes_update_m6_s2.py (gen1)
# born: 2026-05-29T23:25:17Z

from __future__ import annotations

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
    """Extract lower‑case alphabetic tokens."""
    return re.findall(r"[a-z]+(?:'[a-z]+)?", (text or "").lower())


def lsm_vector(text: str) -> Dict[str, float]:
    """Return a sparse LSM vector: proportion of each function category."""
    ws = words(text)
    total = max(1, len(ws))
    cnt = Counter(ws)
    return {
        cat: sum(cnt[w] for w in vocab) / total
        for cat, vocab in FUNCTION_CATS.items()
    }


def lsm_score(a: Dict[str, float], b: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    """
    Deterministic similarity between two LSM vectors.
    Returns (overall_score, per‑category detail).
    """
    detail: Dict[str, float] = {}
    for cat in FUNCTION_CATS:
        av = a.get(cat, 0.0)
        bv = b.get(cat, 0.0)
        # harmonic‑like similarity bounded in [0,1]
        score = 1.0 - (abs(av - bv) / (av + bv + 1e-6))
        score = max(0.0, min(1.0, score))
        detail[cat] = score
    overall = sum(detail.values()) / len(FUNCTION_CATS)
    return overall, detail


def length(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def tree_metrics(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
) -> Tuple[Dict[str, List[str]], Dict[Tuple[str, str], float], Dict[str, float]]:
    """
    Build adjacency, compute Euclidean edge lengths, and root‑to‑node distances.

    Returns
    -------
    adj : node → list of neighbours
    edge_len : ordered edge → length
    root_dist : node → distance from root (sum of edge lengths along the unique path)
    """
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    edge_len: Dict[Tuple[str, str], float] = {}

    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        # store both orientations for easy lookup
        l = length(nodes[u], nodes[v])
        edge_len[(u, v)] = l
        edge_len[(v, u)] = l

    # BFS to compute root distances
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
    priors: Dict[Tuple[str, str], float],
    likelihoods: Dict[Tuple[str, str], float],
    fp_rate: float,
) -> Dict[Tuple[str, str], float]:
    """
    Vectorised Bayesian update for all edges.

    p_post = (p_prior·L) / (L·p_prior + FP·(1‑p_prior))

    Parameters
    ----------
    priors : edge → prior probability (0..1)
    likelihoods : edge → likelihood (0..1) derived from LSM similarity
    fp_rate : false‑positive rate (0..1)

    Returns
    -------
    posteriors : edge → posterior probability
    """
    post: Dict[Tuple[str, str], float] = {}
    for e, p_prior in priors.items():
        L = likelihoods.get(e, 0.0)
        numerator = p_prior * L
        denominator = L * p_prior + fp_rate * (1.0 - p_prior)
        post[e] = numerator / denominator if denominator != 0 else 0.0
    return post


def node_beliefs(
    adj: Dict[str, List[str]],
    edge_post: Dict[Tuple[str, str], float],
) -> Dict[str, float]:
    """
    Compute node belief q_v as the average posterior of incident edges.
    """
    q: Dict[str, float] = {}
    for v, neigh in adj.items():
        if not neigh:
            q[v] = 0.0
            continue
        total = sum(edge_post.get((v, nb), 0.0) for nb in neigh) + sum(edge_post.get((nb, v), 0.0) for nb in neigh)
        q[v] = total / (2 * len(neigh)) if neigh else 0.0
    return q


def hybrid_tree_cost(
    edge_len: Dict[Tuple[str, str], float],
    root_dist: Dict[str, float],
    edge_post: Dict[Tuple[str, str], float],
    node_belief: Dict[str, float],
    lam: float = 0.5,
) -> float:
    """
    Evaluate the hybrid cost C_h = Σ_e p_e·ℓ(e) + λ Σ_v q_v·d(v).

    Parameters
    ----------
    edge_len   : ordered edge → Euclidean length (from LSM vector space)
    root_dist  : node → distance from root
    edge_post  : edge → posterior weight p_e
    node_belief: node → belief q_v
    lam        : path‑weight λ

    Returns
    -------
    Hybrid cost as a float.
    """
    edge_term = sum(edge_post[e] * l for e, l in edge_len.items() if e in edge_post)
    node_term = lam * sum(node_belief[v] * d for v, d in root_dist.items())
    return edge_term + node_term


def build_lsm_vectors(texts: Dict[str, str]) -> Dict[str, Dict[str, float]]:
    """Map node identifier → LSM vector."""
    return {nid: lsm_vector(txt) for nid, txt in texts.items()}


def edge_likelihoods_from_lsm(
    lsm_vectors: Dict[str, Dict[str, float]],
    edges: List[Tuple[str, str]],
) -> Dict[Tuple[str, str], float]:
    """
    Compute LSM similarity‑based likelihood for each edge.

    Returns
    -------
    edge_likelihood : edge → likelihood (0..1)
    """
    edge_likelihood: Dict[Tuple[str, str], float] = {}
    for u, v in edges:
        lsm_u = lsm_vectors[u]
        lsm_v = lsm_vectors[v]
        score, _ = lsm_score(lsm_u, lsm_v)
        edge_likelihood[(u, v)] = score
        edge_likelihood[(v, u)] = score
    return edge_likelihood


def main():
    texts = {
        "A": "The quick brown fox jumps over the lazy dog",
        "B": "The dog runs quickly, but the fox is lazy",
        "C": "The sun shines brightly in the clear blue sky",
    }
    lsm_vectors = build_lsm_vectors(texts)

    edges = [("A", "B"), ("B", "C"), ("C", "A")]
    edge_likelihood = edge_likelihoods_from_lsm(lsm_vectors, edges)

    nodes = {
        "A": (0.0, 0.0),
        "B": (1.0, 0.0),
        "C": (0.5, 1.0),
    }
    adj, edge_len, root_dist = tree_metrics(nodes, edges, "A")

    priors = {e: 0.1 for e in edges}
    fp_rate = 0.05
    edge_post = bayes_edge_posteriors(priors, edge_likelihood, fp_rate)
    node_belief = node_beliefs(adj, edge_post)

    lam = 0.5
    cost = hybrid_tree_cost(edge_len, root_dist, edge_post, node_belief, lam)
    print(f"Hybrid tree cost: {cost:.4f}")


if __name__ == "__main__":
    main()