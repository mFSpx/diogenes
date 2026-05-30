# DARWIN HAMMER — match 385, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_hard_truth_ma_kan_m27_s4.py (gen2)
# parent_b: hybrid_hybrid_hybrid_minimu_model_vram_scheduler_m8_s4.py (gen3)
# born: 2026-05-29T23:28:39Z

"""Hybrid Algorithm: Fusing Stylometry Features with Bayesian Tree Cost Integration.

This module fuses two parent algorithms:
- **hybrid_hybrid_hard_truth_ma_kan_m27_s4.py** – provides stylometry features
  and language model metrics.
- **hybrid_hybrid_hybrid_minimu_model_vram_scheduler_m8_s4.py** – defines
  Bayesian tree cost integration and VRAM budgeting.

The mathematical bridge is the *probabilistic weighting of stylometry features*
using a Bayesian update.  We integrate the language model metrics with the
Bayesian tree cost integration to obtain a unified system that can advise
whether a given text fits within a stylometry-constrained VRAM budget.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
from collections import Counter

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
        "no not never none neither cannot can't won't don't didn't isn't aren't was wasn't weren't".split()
    ),
    "quantifier": set(
        "all any both each few many more most much none several some such".split()
    ),
    "adverb_common": set(
        "very really just still already also even only then there here now often always sometimes".split()
    ),
}

PUNCT = "!?;:,.—-()[]{}\"'`/\\|@#$%^&*+=~"

def words(text: str) -> List[str]:
    return [word for word in (text or "").lower().split() if word.isalpha()]

def lsm_vector(text: str) -> dict[str, float]:
    ws = words(text)
    total = max(1, len(ws))
    cnt = Counter(ws)
    return {
        cat: sum(cnt[w] for w in vocab) / total
        for cat, vocab in FUNCTION_CATS.items()
    }

def stable_hash(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)

def stylometry_features(text: str, dim: int = 96) -> np.ndarray:
    ws = words(text)
    total_words = max(1, len(ws))
    total_chars = max(1, len(text or ""))

    handcrafted = [
        total_words / 500.0,
        sum(len(w) for w in ws) / total_words / 12.0,
        (text.count("\n") + 1) / 200.0,
        sum(text.count(p) for p in "!?") / total_chars,
        sum(text.count(p) for p in ";:") / total_chars,
        sum(text.count(p) for p in "-—") / total_chars,
        sum(1 for ch in text if ch.isupper()) / total_chars,
    ]
    return np.array(handcrafted)

def length(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Euclidean distance between two 2‑D points."""
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
    adjacency: Dict[str, List[str]]
    edge_lengths: Dict[Tuple[str, str], float]
    distances: Dict[str, float]
    """
    adjacency = {node: [] for node in nodes}
    edge_lengths = {}
    distances = {root: 0.0}

    for u, v in edges:
        adjacency[u].append(v)
        adjacency[v].append(u)
        edge_lengths[(u, v)] = length(nodes[u], nodes[v])
        edge_lengths[(v, u)] = edge_lengths[(u, v)]

    stack = [root]
    while stack:
        node = stack.pop()
        for neighbor in adjacency[node]:
            if neighbor not in distances:
                distances[neighbor] = distances[node] + edge_lengths[(node, neighbor)]
                stack.append(neighbor)

    return adjacency, edge_lengths, distances

@dataclass
class Node:
    size: float
    prior: float

def bayesian_update(nodes: List[Node], likelihood: List[float]) -> List[float]:
    posterior = [node.prior * likelihood[i] for i, node in enumerate(nodes)]
    posterior = [p / sum(posterior) for p in posterior]
    return posterior

def expected_vram(posterior: List[float], nodes: List[Node]) -> float:
    return sum(p * node.size for p, node in zip(posterior, nodes))

def hybrid_stylometry_vram(text: str, nodes: List[Node], likelihood: List[float]) -> Tuple[np.ndarray, float]:
    stylometry = stylometry_features(text)
    posterior = bayesian_update(nodes, likelihood)
    expected = expected_vram(posterior, nodes)
    return stylometry, expected

if __name__ == "__main__":
    nodes = [Node(100.0, 0.5), Node(200.0, 0.3), Node(50.0, 0.2)]
    likelihood = [0.6, 0.3, 0.1]
    text = "This is a sample text."
    stylometry, expected = hybrid_stylometry_vram(text, nodes, likelihood)
    print(stylometry)
    print(expected)