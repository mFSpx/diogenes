# DARWIN HAMMER — match 12, survivor 2
# gen: 2
# parent_a: hard_truth_math.py (gen0)
# parent_b: hybrid_minimum_cost_tree_bayes_update_m6_s2.py (gen1)
# born: 2026-05-29T23:25:17Z

#!/usr/bin/env python3
"""
Hybrid module combining the hard-truth telemetry algorithms of hard_truth_math.py 
and the hybrid minimum-cost tree Bayesian update of hybrid_minimum_cost_tree_bayes_update_m6_s2.py.

The mathematical bridge between the two parents lies in the use of the expected value 
of edge contributions in the hybrid minimum-cost tree scoring, which can be 
analogously applied to the word frequencies in the hard-truth telemetry algorithms. 
By replacing the deterministic word frequencies in the hard-truth telemetry algorithms 
with their expected values under the posterior word belief, we can fuse the governing 
equations of both parents.

Mathematical interface: 
The expected value of edge contributions in the hybrid minimum-cost tree scoring 
is used to compute the expected word frequencies in the hard-truth telemetry algorithms.
The posterior word belief is derived from the incident edge posteriors in the hybrid 
minimum-cost tree scoring.

This module implements:
* `words` – extracts words from a given text.
* `lsm_vector` – computes the word frequency vector for a given text.
* `hybrid_lsm_score` – evaluates the hybrid score using the posterior word belief.
"""

from __future__ import annotations

import math
import re
import sys
from collections import Counter
from typing import Dict, List, Tuple
from pathlib import Path
import numpy as np
import random

# Constants
FUNCTION_CATS: Dict[str, set[str]] = {
    "pronoun": set("i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()),
    "article": set("a an the".split()),
    "preposition": set("about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()),
    "auxiliary": set("am are be been being can could did do does had has have is may might must shall should was were will would".split()),
    "conjunction": set("and but or nor so yet because although while if when where whereas unless until".split()),
    "negation": set("no not never none neither cannot can't won't don't didn't isn't aren't wasn't weren't".split()),
    "quantifier": set("all any both each few many more most much none several some such".split()),
    "adverb_common": set("very really just still already also even only then there here now often always sometimes".split()),
}

PUNCT = "!?;:,.—-()[]{}\"'`/\\|@#$%^&*+=~"

# Types
Point = Tuple[float, float]
Edge = Tuple[str, str]

def words(text: str) -> List[str]:
    """
    Extracts words from a given text.

    Args
    ----
    text : str
        The input text.

    Returns
    -------
    words : List[str]
        A list of extracted words.
    """
    return re.findall(r"[a-z]+(?:'[a-z]+)?", (text or "").lower())

def lsm_vector(text: str) -> Dict[str, float]:
    """
    Computes the word frequency vector for a given text.

    Args
    ----
    text : str
        The input text.

    Returns
    -------
    lsm_vector : Dict[str, float]
        A dictionary of word frequencies.
    """
    ws = words(text)
    total = max(1, len(ws))
    cnt = Counter(ws)
    return {cat: sum(cnt[w] for w in vocab) / total for cat, vocab in FUNCTION_CATS.items()}

def length(a: Point, b: Point) -> float:
    """
    Euclidean distance between two points.

    Args
    ----
    a : Point
        The first point.
    b : Point
        The second point.

    Returns
    -------
    length : float
        The Euclidean distance between the two points.
    """
    return math.hypot(a[0] - b[0], a[1] - b[1])

def tree_metrics(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
) -> Tuple[Dict[str, List[str]], Dict[Edge, float], Dict[str, float]]:
    """
    Build adjacency, compute Euclidean edge lengths and root-to-node distances.

    Args
    ----
    nodes : Dict[str, Point]
        A dictionary of node points.
    edges : List[Edge]
        A list of edges.
    root : str
        The root node.

    Returns
    -------
    adj : Dict[str, List[str]]
        A dictionary mapping node to list of neighbours.
    edge_len : Dict[Edge, float]
        A dictionary mapping edge to Euclidean length.
    root_dist : Dict[str, float]
        A dictionary mapping node to root distance.
    """
    adj = {node: [] for node in nodes}
    for edge in edges:
        adj[edge[0]].append(edge[1])
        adj[edge[1]].append(edge[0])

    edge_len = {edge: length(nodes[edge[0]], nodes[edge[1]]) for edge in edges}
    root_dist = {node: length(nodes[node], nodes[root]) for node in nodes}

    return adj, edge_len, root_dist

def bayes_edge_posteriors(
    edge_len: Dict[Edge, float],
    edge_prior: Dict[Edge, float],
    likelihood: float,
    fp_rate: float,
) -> Dict[Edge, float]:
    """
    Compute the posterior edge probabilities using Bayesian update.

    Args
    ----
    edge_len : Dict[Edge, float]
        A dictionary mapping edge to Euclidean length.
    edge_prior : Dict[Edge, float]
        A dictionary mapping edge to prior probability.
    likelihood : float
        The likelihood of an edge.
    fp_rate : float
        The false positive rate.

    Returns
    -------
    edge_post : Dict[Edge, float]
        A dictionary mapping edge to posterior probability.
    """
    edge_post = {}
    for edge, prior in edge_prior.items():
        post = (prior * likelihood) / (likelihood * prior + fp_rate * (1 - prior))
        edge_post[edge] = post

    return edge_post

def hybrid_lsm_score(
    text_a: str,
    text_b: str,
    edge_len: Dict[Edge, float],
    edge_prior: Dict[Edge, float],
    likelihood: float,
    fp_rate: float,
) -> Tuple[float, Dict[str, float]]:
    """
    Evaluate the hybrid score using the posterior word belief.

    Args
    ----
    text_a : str
        The first text.
    text_b : str
        The second text.
    edge_len : Dict[Edge, float]
        A dictionary mapping edge to Euclidean length.
    edge_prior : Dict[Edge, float]
        A dictionary mapping edge to prior probability.
    likelihood : float
        The likelihood of an edge.
    fp_rate : float
        The false positive rate.

    Returns
    -------
    score : float
        The hybrid score.
    detail : Dict[str, float]
        A dictionary of detailed scores.
    """
    lsm_a = lsm_vector(text_a)
    lsm_b = lsm_vector(text_b)

    edge_post = bayes_edge_posteriors(edge_len, edge_prior, likelihood, fp_rate)
    score = 0.0
    detail = {}

    for cat in FUNCTION_CATS:
        av = lsm_a.get(cat, 0.0)
        bv = lsm_b.get(cat, 0.0)
        post_av = av * edge_post.get((cat, cat), 0.0)
        post_bv = bv * edge_post.get((cat, cat), 0.0)
        score += 1.0 - (abs(post_av - post_bv) / (post_av + post_bv + 1e-6))
        detail[cat] = score

    return score, detail

if __name__ == "__main__":
    text_a = "This is a test text."
    text_b = "This is another test text."
    nodes = {"A": (0.0, 0.0), "B": (1.0, 1.0)}
    edges = [("A", "B")]
    root = "A"
    edge_prior = {("A", "B"): 0.5}
    likelihood = 0.8
    fp_rate = 0.2

    adj, edge_len, root_dist = tree_metrics(nodes, edges, root)
    score, detail = hybrid_lsm_score(text_a, text_b, edge_len, edge_prior, likelihood, fp_rate)
    print("Hybrid score:", score)
    print("Detailed scores:", detail)