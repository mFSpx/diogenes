# DARWIN HAMMER — match 393, survivor 0
# gen: 4
# parent_a: hybrid_hybrid_hard_truth_ma_hybrid_hybrid_geomet_m18_s0.py (gen3)
# parent_b: ollivier_ricci_curvature.py (gen0)
# born: 2026-05-29T23:28:34Z

"""
This module implements a novel hybrid algorithm, fusing the stylometry analysis from 
'hybrid_hard_truth_math_model_pool_m8_s5.py' with the Ollivier-Ricci curvature on weighted graphs 
from 'ollivier_ricci_curvature.py'. 
The mathematical bridge between these two structures lies in the representation of 
text data as graph vertices, where the stylometry features are used as edge weights. 
The Ollivier-Ricci curvature is then applied to estimate the curvature of the graph at each vertex.
"""

import numpy as np
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from collections import Counter, OrderedDict


# ----------------------------------------------------------------------
# Parent A – stylometry / LSM utilities
# ----------------------------------------------------------------------
FUNCTION_CATS: Dict[str, set[str]] = {
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
    """Return a list of lowercase words (ASCII letters + optional apostrophe)."""
    return re.findall(r'\b\w+\b', text.lower())


def stylometry_features(text: str) -> Dict[str, int]:
    """Return a dictionary of stylometry features (word counts) for the given text."""
    words_list = words(text)
    features = {}
    for word in words_list:
        if word not in features:
            features[word] = 1
        else:
            features[word] += 1
    return features


def lazy_rw_distribution(adj, node, alpha=0.5):
    """Lazy random walk distribution centred at *node*.

    Parameters
    ----------
    adj   : dict mapping node_id -> list of neighbour node_ids
    node  : the source node
    alpha : mass kept at the node itself (laziness parameter)

    Returns
    -------
    dict mapping node_id -> float probability
    """
    neighbours = adj.get(node, [])
    deg = len(neighbours)
    dist = {node: alpha}
    if deg > 0:
        spread = (1.0 - alpha) / deg
        for nb in neighbours:
            dist[nb] = dist.get(nb, 0.0) + spread
    return dist


def bfs_distances(adj):
    """All-pairs shortest-path distances via BFS.

    Parameters
    ----------
    adj : dict mapping node_id -> list of neighbour node_ids

    Returns
    -------
    dict mapping node_id -> list of shortest-path distances to all other nodes
    """
    distances = {}
    for node in adj:
        distances[node] = {}
        visited = set()
        queue = deque([node])
        while queue:
            current_node = queue.popleft()
            if current_node not in visited:
                visited.add(current_node)
                for neighbour in adj[current_node]:
                    if neighbour not in visited:
                        queue.append(neighbour)
                        distances[node][neighbour] = 1
                    elif neighbour in distances[node]:
                        distances[node][neighbour] += 1
    return distances


def ollivier_ricci_curvature(adj, alpha=0.5):
    """Ollivier-Ricci curvature on weighted graphs.

    Parameters
    ----------
    adj : dict mapping node_id -> list of neighbour node_ids
    alpha : mass kept at the node itself (laziness parameter)

    Returns
    -------
    dict mapping edge (u, v) -> float Ollivier-Ricci curvature
    """
    curvatures = {}
    for u in adj:
        for v in adj[u]:
            if u < v:  # avoid double counting
                curvatures[(u, v)] = 1 - lazy_rw_distribution(adj, u, alpha)[v] / bfs_distances(adj)[u][v]
    return curvatures


def hybrid_stylometry_curvature(text: str) -> Dict[Tuple[int, int], float]:
    """Return a dictionary of stylometry features and Ollivier-Ricci curvatures for the given text."""
    words_list = words(text)
    features = stylometry_features(text)
    adj = {}
    for i in range(len(words_list)):
        for j in range(i + 1, len(words_list)):
            adj[(i, j)] = []
            adj[(j, i)] = []
    for u in adj:
        for v in adj[u]:
            if words_list[u] not in features or words_list[v] not in features:
                adj[u].append(v)
                adj[v].append(u)
    curvatures = ollivier_ricci_curvature(adj)
    for edge in curvatures:
        if edge[0] < edge[1]:
            continue
        curvatures[edge] = curvatures[(edge[1], edge[0])]
    return {edge: curvatures[edge] for edge in adj if edge in curvatures}


# Smoke test
if __name__ == "__main__":
    text = "This is a test text with multiple words and a variety of styles."
    print(hybrid_stylometry_curvature(text))