# DARWIN HAMMER — match 2122, survivor 0
# gen: 3
# parent_a: hybrid_possum_filter_hybrid_privacy_model_m53_s2.py (gen2)
# parent_b: hybrid_hybrid_minimum_cost__hybrid_decision_hygi_m7_s1.py (gen2)
# born: 2026-05-29T23:40:48Z

"""
Hybrid filter-selection algorithm merging Possum-style spatial-signature filtering
(PARENT ALGORITHM A) with the privacy-aware model-resource linear formulation
(PARENT ALGORITHM B) and the decision hygiene scoring system from PARENT ALGORITHM B.

Mathematical bridge:
* Entities and models are combined into a single decision graph using a weighted
  graph structure, where the weights represent the spatial and privacy loads.
* The decision hygiene scores are used as weights for the entities and models in
  the graph, and the expected cost of the decision tree is used as a weight for the
  decision hygiene scores.
* The Shannon entropy of the weighted scores is calculated to gain insights into
  the complexity and uncertainty of the decision-making process.
* The linear constraints from PARENT ALGORITHM A are used to constrain the selection
  process.
"""

import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

import numpy as np

# ---------- Parent A: spatial-signature utilities ----------
@dataclass(frozen=True)
class Entity:
    id: str
    lat: float
    lon: float
    category: str
    score: float = 0.0
    address_signature: str = ""

def haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle distance in metres between two (lat,lon) points."""
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    return 2 * 6_371_000

# ---------- Parent B: decision hygiene utilities ----------
def length(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def tree_metrics(
    nodes: Dict[str, tuple[float, float]],
    edges: List[tuple[str, str]],
    root: str,
) -> Tuple[Dict[str, List[str]], Dict[tuple[str, str], float], Dict[str, float]]:
    """
    Build adjacency, compute Euclidean edge lengths and root-to-node distances.

    Returns
    -------
    adj : dict mapping node → list of neighbours
    edge_len : dict mapping edge (ordered as supplied) → length
    dist : dict mapping node → distance from *root* (sum of edge lengths along the unique path)
    """
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    edge_len: Dict[tuple[str, str], float] = {}
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
                # identify the next node
                dist[nxt] = dist[cur] + edge_len[(cur, nxt)]
                stack.append(nxt)

    return adj, edge_len, dist

# ---------- Hybrid algorithm ----------
def calculate_expected_cost(adj: Dict[str, List[str]], edge_len: Dict[tuple[str, str], float], dist: Dict[str, float], nodes: Dict[str, tuple[float, float]]) -> Dict[str, float]:
    """
    Calculate the expected cost of the decision tree.

    Returns
    -------
    expected_cost : dict mapping node → expected cost
    """
    expected_cost: Dict[str, float] = {}
    for node in nodes:
        # calculate the expected cost as the sum of edge lengths along the unique path from the root
        expected_cost[node] = dist[node]

    return expected_cost

def calculate_shannon_entropy(expected_cost: Dict[str, float], decision_hygiene_scores: Dict[str, float]) -> float:
    """
    Calculate the Shannon entropy of the weighted scores.

    Returns
    -------
    shannon_entropy : float
    """
    # calculate the weighted scores
    weighted_scores: List[float] = [expected_cost[node] * decision_hygiene_scores[node] for node in expected_cost]

    # calculate the Shannon entropy
    shannon_entropy = -sum([score / sum(weighted_scores) * math.log(score / sum(weighted_scores)) for score in weighted_scores])

    return shannon_entropy

def hybrid_selection(entities: List[Entity], models: List[Dict[str, float]], spatial_budget: float, privacy_budget: float) -> List[str]:
    """
    Select a subset of entities and models based on the linear constraints.

    Returns
    -------
    selected_entities : list of selected entity IDs
    selected_models : list of selected model IDs
    """
    # calculate the resource vectors
    e_vectors: List[List[float]] = [[haversine_m((entity.lat, entity.lon), (0, 0)), 1 if entity.address_signature in [model['address_signature'] for model in models] else 0] for entity in entities]
    m_vectors: List[List[float]] = [[model['ram'], model['tau'] * model['mu']] for model in models]

    # stack the vectors
    A = np.vstack((e_vectors, m_vectors))

    # calculate the binary indicator
    x = np.zeros(len(entities) + len(models))
    for i in range(len(entities) + len(models)):
        if sum(A[i]) <= spatial_budget and sum(A[i]) <= privacy_budget:
            x[i] = 1

    # return the selected entities and models
    selected_entities = [entity.id for i, entity in enumerate(entities) if x[i + len(models)] == 1]
    selected_models = [model['id'] for i, model in enumerate(models) if x[i] == 1]

    return selected_entities, selected_models

# smoke test
if __name__ == "__main__":
    entities = [
        Entity('e1', 37.7749, -122.4194, 'entity1'),
        Entity('e2', 34.0522, -118.2437, 'entity2'),
        Entity('e3', 40.7128, -74.0060, 'entity3'),
    ]
    models = [
        {'id': 'm1', 'ram': 1024, 'tau': 1, 'mu': 0.5},
        {'id': 'm2', 'ram': 2048, 'tau': 2, 'mu': 0.5},
        {'id': 'm3', 'ram': 4096, 'tau': 3, 'mu': 0.5},
    ]
    spatial_budget = 10000
    privacy_budget = 1000

    selected_entities, selected_models = hybrid_selection(entities, models, spatial_budget, privacy_budget)
    print(selected_entities)
    print(selected_models)