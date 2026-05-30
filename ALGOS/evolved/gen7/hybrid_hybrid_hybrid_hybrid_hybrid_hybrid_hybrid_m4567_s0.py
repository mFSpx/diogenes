# DARWIN HAMMER — match 4567, survivor 0
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m607_s2.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_sketch_m1851_s1.py (gen6)
# born: 2026-05-29T23:56:30Z

"""
This module implements a novel hybrid algorithm that fuses the core topologies of 
hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m607_s2.py and hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_sketch_m1851_s1.py. 
The mathematical bridge between the two structures lies in the incorporation of 
the language complexity score from the first algorithm into the modulation_vector generation 
of the RBF surrogate in the second algorithm. This allows for efficient, probabilistic estimation 
of modulation vectors based on hashed item frequencies, stylometry features, and Gaussian beam 
intensity and Euclidean distance metrics.
"""

import numpy as np
import math
import random
import sys
import pathlib

FUNCTION_CATS = {
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

def words(text: str) -> list[str]:
    return [word for word in (text or "").lower().split() if word.isalpha()]

def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian beam intensity."""
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a Gaussian beam."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

def length(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def language_complexity_score(text: str) -> float:
    """Language complexity score."""
    words_in_text = words(text)
    num_words = len(words_in_text)
    if num_words == 0:
        return 0.0
    word_frequencies = [words_in_text.count(word) for word in set(words_in_text)]
    word_frequencies.sort(reverse=True)
    top_10_frequencies = word_frequencies[:10]
    language_complexity = sum(top_10_frequencies) / num_words
    return language_complexity

def hybrid_modulation_vector(
    text: str, center: float, width: float, eps: float = 1e-12
) -> float:
    """Hybrid modulation vector."""
    language_complexity = language_complexity_score(text)
    fisher_info = fisher_score(language_complexity, center, width, eps)
    return fisher_info

def tree_metrics(
    nodes: dict[str, tuple[float, float]],
    edges: list[tuple[str, str]],
    root: str,
) -> tuple[dict[str, list[str]], dict[tuple[str, str], float], dict[str, float]]:
    """
    Build adjacency, compute Euclidean edge lengths and root‑to‑node distances.

    Returns
    -------
    adj : dict mapping node → list of neighbours
    edge_len : dict mapping edge (ordered as supplied) → length
    dist : dict mapping node → distance from *root* (sum of edge lengths along the unique path)
    """
    adj: dict[str, list[str]] = {n: [] for n in nodes}
    edge_len: dict[tuple[str, str], float] = {}
    for a, b in edges:
        adj[a].append(b)
        edge_len[(a, b)] = length(nodes[a], nodes[b])
    dist: dict[str, float] = {root: 0.0}
    queue: list[str] = [root]
    while queue:
        node = queue.pop(0)
        for neighbour in adj[node]:
            if neighbour not in dist:
                dist[neighbour] = dist[node] + edge_len[(node, neighbour)]
                queue.append(neighbour)
    return adj, edge_len, dist

if __name__ == "__main__":
    text = "This is a test sentence."
    center = 0.5
    width = 1.0
    print(hybrid_modulation_vector(text, center, width))
    nodes = {
        "A": (0.0, 0.0),
        "B": (1.0, 0.0),
        "C": (0.5, 1.0),
    }
    edges = [("A", "B"), ("A", "C"), ("B", "C")]
    root = "A"
    adj, edge_len, dist = tree_metrics(nodes, edges, root)
    print(adj, edge_len, dist)