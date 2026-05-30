# DARWIN HAMMER — match 1918, survivor 4
# gen: 5
# parent_a: hybrid_hybrid_semantic_neig_hybrid_hybrid_krampu_m540_s1.py (gen3)
# parent_b: hybrid_hybrid_hybrid_minimu_hybrid_hybrid_hybrid_m405_s0.py (gen4)
# born: 2026-05-29T23:39:45Z

"""
Hybrid Algorithm combining:
- Parent A: hybrid_semantic_neighbors + pheromone + entropy + Ollivier‑Ricci curvature.
- Parent B: minimum‑cost tree metrics + Bayesian log‑count reward estimation + privacy health scores.

Mathematical bridge:
Edge costs of the tree are modulated by *semantic pheromone* levels derived from
node feature vectors (Parent A) and by an *Ollivier‑Ricci curvature* term that
measures similarity of neighbouring feature distributions.  The curvature‑adjusted
cost Cₑ is

    Cₑ = ℓₑ · (1 – κₑ) / (π_u · π_v) ,

where ℓₑ is Euclidean length (Parent B), κₑ ∈ [0,1] is the Ricci curvature
computed from cosine similarities of node feature vectors, and π_u, π_v are
pheromone probabilities of the incident nodes (Parent A).  These costs feed a
Bayesian log‑count estimator that yields expected rewards for actions on the
tree, while a privacy‑aware health score (from Parent B) can be injected as a
regulariser.  The resulting system jointly exploits semantic neighbourhood
information, curvature‑aware geometry, and Bayesian reward learning.
"""

import math
import random
import sys
import pathlib
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Types
Point = Tuple[float, float]
Edge = Tuple[str, str]

# ----------------------------------------------------------------------
# Parent A utilities (semantic / pheromone / entropy)

def _cos(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two vectors."""
    den = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return 0.0 if den == 0 else sum(x * y for x, y in zip(a, b)) / den


def pheromone_probabilities(pheromones: List[float]) -> List[float]:
    """Normalise a list of pheromone strengths into a probability distribution."""
    total = sum(pheromones)
    if total == 0:
        raise ValueError("Pheromone vector must contain positive mass.")
    return [p / total for p in pheromones]


def entropy(probabilities: List[float], eps: float = 1e-12) -> float:
    """Shannon entropy of a probability distribution."""
    total = sum(probabilities)
    if total <= 0:
        raise ValueError("positive probability mass required")
    return -sum((p / total) * math.log(max(p / total, eps)) for p in probabilities if p > 0)


def extract_full_features(text: str) -> Dict[str, float]:
    """
    Deterministic pseudo‑feature extraction from a string.
    The function uses a hash‑based RNG to produce reproducible floating values.
    """
    rnd = random.Random(hash(text))
    keys = [
        "operator_visceral_ratio", "operator_tech_ratio",
        "operator_legal_osint_ratio", "operator_ledger_density",
        "operator_recursion_score", "operator_directive_ratio",
        "operator_target_density", "psyche_forensic_shield_ratio",
        "psyche_poetic_entropy", "psyche_dissociative_index",
        "psyche_wrath_velocity", "resilience_bureaucratic_weaponization_index",
        "resilience_resource_exhaustion_metric", "resilience_swarm_orchestration_density",
        "resilience_logic_crucifixion_index", "resilience_conspiracy_grounding_ratio",
        "resilience_chaotic_good_tax", "rainmaker_corporate_grit_tension",
        "rainmaker_countdown_density", "rainmaker_asset_structuring_weight",
        "rainmaker_pitch_formatting_ratio"
    ]
    return {k: rnd.random() for k in keys}


def feature_vector(features: Dict[str, float]) -> List[float]:
    """Convert feature dict to ordered list (stable order)."""
    return [features[k] for k in sorted(features.keys())]


# ----------------------------------------------------------------------
# Parent B utilities (tree metrics, Bayesian log‑count, health score)

def length(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def tree_metrics(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
) -> Tuple[Dict[str, List[str]], Dict[Edge, float], Dict[str, float]]:
    """
    Build adjacency list, compute Euclidean edge lengths and root‑to‑node distances.
    """
    adj: Dict[str, List[str]] = defaultdict(list)
    edge_len: Dict[Edge, float] = {}
    node_dist: Dict[str, float] = {}

    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        l = length(nodes[a], nodes[b])
        edge_len[(a, b)] = l
        edge_len[(b, a)] = l

    # BFS for root‑to‑node distances
    queue: List[Tuple[str, float]] = [(root, 0.0)]
    visited = set()
    while queue:
        node, dist = queue.pop(0)
        if node in visited:
            continue
        visited.add(node)
        node_dist[node] = dist
        for nb in adj[node]:
            if nb not in visited:
                queue.append((nb, dist + edge_len[(node, nb)]))

    return adj, edge_len, node_dist


def log_count_update(log_counts: Dict[str, float], node: str, reward: float, alpha: float = 0.1) -> None:
    """
    Bayesian log‑count update for expected reward of a node.
    log_counts stores log(N) where N is the pseudo‑count of observations.
    """
    # Convert to count, add reward (treated as pseudo‑observation), then back to log.
    count = math.exp(log_counts.get(node, 0.0))
    count = (1 - alpha) * count + alpha * reward
    log_counts[node] = math.log(count + 1e-12)  # avoid log(0)


def privacy_health_score(node: str) -> float:
    """
    Placeholder privacy health score (0 … 1).  In a real system this would
    depend on data sensitivity, reconstruction risk, etc.
    """
    rnd = random.Random(hash(node))
    return rnd.random()


# ----------------------------------------------------------------------
# Hybrid core functions (the mathematical bridge)

def compute_curvature(
    node_u: str,
    node_v: str,
    features: Dict[str, List[float]],
) -> float:
    """
    Approximate Ollivier‑Ricci curvature κₑ for edge (u,v) using cosine similarity
    of the feature vectors of the two endpoints.
    κₑ = (1 + sim) / 2  ∈ [0,1]   (sim ∈ [‑1,1])
    """
    vec_u = features[node_u]
    vec_v = features[node_v]
    sim = _cos(vec_u, vec_v)          # ∈ [‑1,1]
    return (1.0 + sim) / 2.0          # map to [0,1]


def hybrid_edge_costs(
    edge_len: Dict[Edge, float],
    pheromones: Dict[str, float],
    curvature: Dict[Edge, float],
) -> Dict[Edge, float]:
    """
    Compute curvature‑adjusted, pheromone‑scaled edge costs:
        Cₑ = ℓₑ · (1 – κₑ) / (π_u · π_v)
    where π_* are pheromone probabilities.
    """
    # Normalise pheromones to probabilities
    pi_vals = pheromone_probabilities(list(pheromones.values()))
    pi_map = dict(zip(pheromones.keys(), pi_vals))

    costs: Dict[Edge, float] = {}
    for (u, v), l in edge_len.items():
        kappa = curvature.get((u, v), 0.0)
        pi_u = max(pi_map.get(u, 1e-6), 1e-6)
        pi_v = max(pi_map.get(v, 1e-6), 1e-6)
        cost = l * (1.0 - kappa) / (pi_u * pi_v)
        costs[(u, v)] = cost
    return costs


def hybrid_action_selection(
    current: str,
    adj: Dict[str, List[str]],
    edge_costs: Dict[Edge, float],
    log_counts: Dict[str, float],
    temperature: float = 1.0,
) -> str:
    """
    Choose the next node from neighbours of `current` using a Boltzmann‑like
    distribution that blends cost (lower is better) and Bayesian expected reward
    (higher is better).  The probability of neighbour v is proportional to

        exp( (E[reward_v] – C_{current,v}) / T )

    where E[reward_v] = exp(log_counts[v]) (pseudo‑expected reward).
    """
    neighbours = adj[current]
    if not neighbours:
        raise RuntimeError(f"No neighbours to move from node {current}")

    scores = []
    for nb in neighbours:
        cost = edge_costs.get((current, nb), 1e6)
        reward_est = math.exp(log_counts.get(nb, 0.0))
        score = math.exp((reward_est - cost) / max(temperature, 1e-6))
        scores.append(score)

    total = sum(scores)
    probs = [s / total for s in scores]
    chosen = random.choices(neighbours, weights=probs, k=1)[0]
    return chosen


def hybrid_step(
    current: str,
    adj: Dict[str, List[str]],
    edge_costs: Dict[Edge, float],
    log_counts: Dict[str, float],
    pheromones: Dict[str, float],
    curvature: Dict[Edge, float],
    temperature: float = 1.0,
) -> Tuple[str, Dict[str, float]]:
    """
    Perform one hybrid iteration:
    1. Select next node using `hybrid_action_selection`.
    2. Simulate a stochastic reward (here: negative cost + entropy bonus).
    3. Update Bayesian log‑counts.
    4. Update pheromone levels (simple evaporation + deposit).
    Returns the new node and the updated pheromone map.
    """
    next_node = hybrid_action_selection(current, adj, edge_costs, log_counts, temperature)

    # Reward model: higher entropy of outgoing pheromones + negative cost
    out_edges = [(current, nb) for nb in adj[current]]
    out_costs = [edge_costs.get(e, 1e6) for e in out_edges]
    avg_cost = sum(out_costs) / max(len(out_costs), 1)

    # Entropy of pheromone distribution over neighbours
    neigh_pher = [pheromones.get(nb, 0.0) for nb in adj[current]]
    try:
        neigh_entropy = entropy(neigh_pher)
    except ValueError:
        neigh_entropy = 0.0

    reward = -avg_cost + 0.1 * neigh_entropy  # tunable scaling

    # Bayesian update
    log_count_update(log_counts, next_node, reward)

    # Pheromone dynamics (evaporation + deposit proportional to reward)
    evaporation_rate = 0.05
    for node in pheromones:
        pheromones[node] *= (1.0 - evaporation_rate)
    pheromones[next_node] += max(reward, 0.0)  # only positive reward reinforces

    return next_node, pheromones


# ----------------------------------------------------------------------
# Helper to initialise the hybrid system from raw inputs

def initialise_hybrid_system(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
    node_texts: Dict[str, str],
) -> Tuple[Dict[str, List[str]], Dict[Edge, float], Dict[str, float],
           Dict[str, float], Dict[Edge, float], Dict[str, float]]:
    """
    Build all structures required for the hybrid algorithm:
    - adjacency, Euclidean edge lengths, root distances (Parent B)
    - feature vectors per node (Parent A)
    - pheromone strengths (initialised from a chosen feature)
    - curvature per edge (bridge)
    - edge costs (bridge)
    - Bayesian log‑counts (initially zero)
    """
    # Tree metrics
    adj, edge_len, _ = tree_metrics(nodes, edges, root)

    # Feature extraction
    raw_features = {nid: extract_full_features(txt) for nid, txt in node_texts.items()}
    vecs = {nid: feature_vector(f) for nid, f in raw_features.items()}

    # Initialise pheromones from a single scalar feature (e.g., operator_visceral_ratio)
    pheromones = {nid: raw_features[nid]["operator_visceral_ratio"] for nid in nodes}

    # Curvature per directed edge
    curvature = {}
    for (u, v) in edge_len:
        curvature[(u, v)] = compute_curvature(u, v, vecs)

    # Edge costs using the bridge formula
    edge_costs = hybrid_edge_costs(edge_len, pheromones, curvature)

    # Bayesian log‑counts start at log(1) = 0 for every node
    log_counts = {nid: 0.0 for nid in nodes}

    return adj, edge_len, pheromones, curvature, edge_costs, log_counts


# ----------------------------------------------------------------------
# Smoke test

if __name__ == "__main__":
    # Simple synthetic tree
    nodes = {
        "A": (0.0, 0.0),
        "B": (1.0, 0.0),
        "C": (0.0, 1.0),
        "D": (1.0, 1.0)
    }
    edges = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")]
    root = "A"

    # Dummy texts for feature extraction
    node_texts = {
        "A": "alpha",
        "B": "bravo",
        "C": "charlie",
        "D": "delta"
    }

    adj, edge_len, pheromones, curvature, edge_costs, log_counts = initialise_hybrid_system(
        nodes, edges, root, node_texts
    )

    current_node = root
    print(f"Start at node {current_node}")

    for step in range(5):
        current_node, pheromones = hybrid_step(
            current_node,
            adj,
            edge_costs,
            log_counts,
            pheromones,
            curvature,
            temperature=0.5
        )
        print(f"Step {step+1}: moved to {current_node}")

    print("Final pheromone levels:", pheromones)
    print("Final log‑counts:", log_counts)