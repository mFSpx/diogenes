# DARWIN HAMMER — match 385, survivor 3
# gen: 4
# parent_a: hybrid_hybrid_hard_truth_ma_kan_m27_s4.py (gen2)
# parent_b: hybrid_hybrid_hybrid_minimu_model_vram_scheduler_m8_s4.py (gen3)
# born: 2026-05-29T23:28:39Z

import hashlib
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict

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

@dataclass
class TreeNode:
    name: str
    size: int
    prior_probability: float

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
    return np.array(handcrafted[:dim])

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
        adjacency list representation of the tree
    edge_lengths: Dict[Tuple[str, str], float]
        Euclidean length of each edge
    node_distances: Dict[str, float]
        distance from the root to each node
    """
    adjacency = {node: [] for node in nodes}
    edge_lengths = {}
    node_distances = {root: 0.0}

    for edge in edges:
        adjacency[edge[0]].append(edge[1])
        adjacency[edge[1]].append(edge[0])
        edge_lengths[edge] = length(nodes[edge[0]], nodes[edge[1]])
        edge_lengths[(edge[1], edge[0])] = edge_lengths[edge]

    queue = [root]
    while queue:
        node = queue.pop(0)
        for neighbor in adjacency[node]:
            if neighbor not in node_distances:
                node_distances[neighbor] = node_distances[node] + edge_lengths[(node, neighbor)]
                queue.append(neighbor)

    return adjacency, edge_lengths, node_distances

def bayesian_update(node: TreeNode, likelihood: float, prior: float = 0.5) -> float:
    """
    Update the probability of a node given the likelihood of the text characteristics.

    Returns
    -------

    posterior_probability: float
        updated probability of the node
    """
    prior_probability = node.prior_probability
    posterior_probability = (prior_probability * likelihood) / (prior_probability * likelihood + (1 - prior_probability) * (1 - likelihood))
    return posterior_probability

def expected_vram(nodes: List[TreeNode], likelihoods: List[float], prior: float = 0.5) -> float:
    """
    Calculate the expected VRAM consumption based on the node sizes and probabilities.

    Returns
    -------

    expected_vram: float
        expected VRAM consumption
    """
    expected_vram = sum(node.size * bayesian_update(node, likelihood, prior) for node, likelihood in zip(nodes, likelihoods))
    return expected_vram

def stylometry_informed_bayesian_update(text: str, node: TreeNode, likelihood: float) -> float:
    lsm = lsm_vector(text)
    stylometry_features_text = stylometry_features(text)
    prior_probability = node.prior_probability
    posterior_probability = (prior_probability * likelihood) / (prior_probability * likelihood + (1 - prior_probability) * (1 - likelihood))
    stylometry_informed_posterior = posterior_probability * (1 + np.mean(stylometry_features_text)) / 2
    return stylometry_informed_posterior

def stylometry_informed_expected_vram(text: str, nodes: List[TreeNode], likelihoods: List[float]) -> float:
    expected_vram = sum(node.size * stylometry_informed_bayesian_update(text, node, likelihood) for node, likelihood in zip(nodes, likelihoods))
    return expected_vram

if __name__ == "__main__":
    text = "This is a sample text for testing the stylometry features."
    stylometry_features_text = stylometry_features(text)
    print(stylometry_features_text)

    nodes = [TreeNode("node1", 100, 0.5), TreeNode("node2", 200, 0.3)]
    likelihoods = [0.7, 0.4]
    expected_vram_consumption = stylometry_informed_expected_vram(text, nodes, likelihoods)
    print(expected_vram_consumption)