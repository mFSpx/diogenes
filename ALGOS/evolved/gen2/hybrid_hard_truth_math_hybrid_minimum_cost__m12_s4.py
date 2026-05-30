# DARWIN HAMMER — match 12, survivor 4
# gen: 2
# parent_a: hard_truth_math.py (gen0)
# parent_b: hybrid_minimum_cost_tree_bayes_update_m6_s2.py (gen1)
# born: 2026-05-29T23:25:17Z

#!/usr/bin/env python3
"""Hybrid LSM‑Bayesian Tree Module

Parent A – *hard_truth_math.py* provides linguistic LSM (function‑category) vectors
and a deterministic similarity score.  
Parent B – *hybrid_minimum_cost_tree_bayes_update_m6_s2.py* supplies a tree
metric (edge lengths, root‑to‑node distances) and a Bayesian posterior update.

Mathematical bridge
-------------------
* Edge geometry* is derived from the Euclidean distance between LSM vectors of the
  two endpoint nodes:

      ℓ(e) = ‖v_u – v_v‖₂                                            (1)

* Likelihood* for an edge is the LSM similarity score `lsm_score` between the two
  node texts, normalised to a probability‑like quantity `L_e ∈ [0,1]`.

* Bayesian posterior* for each edge follows the parent‑B update:

      p_e = (p_prior·L_e) / (L_e·p_prior + FP·(1‑p_prior))          (2)

* Node belief* `q_v` is the mean posterior of all incident edges:

      q_v = (1/deg(v)) Σ_{e∈inc(v)} p_e                             (3)

* Hybrid cost* finally fuses the deterministic metric with the probabilistic
  weights (parent B equation 3):

      C_h = Σ_e p_e·ℓ(e) + λ Σ_v q_v·d(v)                           (4)

The code below implements the three core functions needed to realise (4) and
provides a small smoke test.  All operations rely only on the Python standard
library and NumPy."""

from __future__ import annotations

import math
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – LSM utilities
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


# ----------------------------------------------------------------------
# Parent B – Tree utilities (adapted)
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
    Build adjacency, compute Euclidean edge lengths, and root‑to‑node distances.

    Returns
    -------
    adj : node → list of neighbours
    edge_len : ordered edge → length
    root_dist : node → distance from root (sum of edge lengths along the unique path)
    """
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    edge_len: Dict[Edge, float] = {}

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


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def bayes_edge_posteriors(
    priors: Dict[Edge, float],
    likelihoods: Dict[Edge, float],
    fp_rate: float,
) -> Dict[Edge, float]:
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
    """
    Compute node belief q_v as the average posterior of incident edges.
    """
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


# ----------------------------------------------------------------------
# Demonstration helpers
# ----------------------------------------------------------------------
def build_lsm_vectors(texts: Dict[str, str]) -> Dict[str, Dict[str, float]]:
    """Map node identifier → LSM vector."""
    return {nid: lsm_vector(txt) for nid, txt in texts.items()}


def edge_likelihoods_from_lsm(
    edges: List[Edge],
    lsm_vecs: Dict[str, Dict[str, float]],
) -> Dict[Edge, float]:
    """
    Derive a likelihood L_e for each edge from the LSM similarity of its endpoints.
    Normalise the raw similarity to [0,1] (already bounded) and use it directly.
    """
    L: Dict[Edge, float] = {}
    for u, v in edges:
        sim, _ = lsm_score(lsm_vecs[u], lsm_vecs[v])
        L[(u, v)] = sim
        L[(v, u)] = sim  # symmetric
    return L


def random_prior(edges: List[Edge], seed: int = 42) -> Dict[Edge, float]:
    """Assign a uniform random prior in (0,1) to each directed edge."""
    random.seed(seed)
    return {(u, v): random.random() for u, v in edges for _ in (0, 1)}


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Sample texts for three nodes
    sample_texts = {
        "A": "I love the bright sunrise over the mountains, it feels very alive.",
        "B": "You should not ignore the dark clouds; they can be very dangerous.",
        "C": "We are gathering together to celebrate the new beginning, very happy!",
    }

    # Fixed 2‑D coordinates (could be anything; we will replace them with LSM‑derived geometry)
    node_coords: Dict[str, Point] = {
        "A": (0.0, 0.0),
        "B": (1.0, 0.0),
        "C": (0.5, 1.0),
    }

    edges = [("A", "B"), ("B", "C"), ("C", "A")]
    root = "A"

    # 1️⃣ Build LSM vectors
    lsm_vecs = build_lsm_vectors(sample_texts)

    # 2️⃣ Replace Euclidean geometry with LSM‑space distances
    #    (edge_len is computed from vectors, not from node_coords)
    def vector_to_array(vec: Dict[str, float]) -> np.ndarray:
        """Convert sparse LSM dict to dense numpy array ordered by FUNCTION_CATS keys."""
        return np.array([vec.get(k, 0.0) for k in FUNCTION_CATS.keys()], dtype=float)

    # compute edge lengths as Euclidean distance between dense vectors
    edge_len: Dict[Edge, float] = {}
    for u, v in edges:
        vu = vector_to_array(lsm_vecs[u])
        vv = vector_to_array(lsm_vecs[v])
        dist = float(np.linalg.norm(vu - vv))
        edge_len[(u, v)] = dist
        edge_len[(v, u)] = dist

    # 3️⃣ Root distances – we can reuse tree_metrics but with dummy coordinates;
    #    we only need the topology, so we pass node_coords and ignore their values.
    adj, _, root_dist = tree_metrics(node_coords, edges, root)

    # 4️⃣ Bayesian ingredients
    priors = random_prior(edges)
    likelihoods = edge_likelihoods_from_lsm(edges, lsm_vecs)
    fp_rate = 0.1
    posteriors = bayes_edge_posteriors(priors, likelihoods, fp_rate)

    # 5️⃣ Node beliefs
    q = node_beliefs(adj, posteriors)

    # 6️⃣ Hybrid cost
    lam = 0.3
    cost = hybrid_tree_cost(edge_len, root_dist, posteriors, q, lam)

    print("Hybrid tree cost:", cost)
    print("Edge posteriors:", {f"{e}": f"{p:.3f}" for e, p in posteriors.items()})
    print("Node beliefs:", {v: f"{b:.3f}" for v, b in q.items()})