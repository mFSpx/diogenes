# DARWIN HAMMER — match 1383, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_regret_engine_hybrid_hybrid_bandit_m83_s1.py (gen3)
# parent_b: hybrid_hybrid_hybrid_distri_hybrid_bayes_claim_k_m592_s1.py (gen3)
# born: 2026-05-29T23:35:52Z

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Iterable
import math
import sys


# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathAction:
    """Immutable description of an action."""
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True)
class MathCounterfactual:
    """Outcome that would have occurred for a given action."""
    action_id: str
    outcome_value: float
    probability: float = 1.0


# ----------------------------------------------------------------------
# Core utilities
# ----------------------------------------------------------------------
def compute_regret_weighted_strategy(
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual],
) -> Dict[str, float]:
    """
    Return a probability distribution over actions derived from a regret‑weighted
    exponential weighting scheme.

    The weight for an action a is
        w_a = exp( V_a - max(V) )
    where V_a = expected_value - cost - risk + counterfactual_correction.
    """
    if not actions:
        return {}

    # map counterfactual contributions (outcome * probability) by action id
    cf_contrib = {
        cf.action_id: cf.outcome_value * cf.probability for cf in counterfactuals
    }

    # raw values before exponentiation
    raw_vals = {
        a.id: a.expected_value - a.cost - a.risk + cf_contrib.get(a.id, 0.0)
        for a in actions
    }

    max_val = max(raw_vals.values())
    exp_weights = {k: math.exp(v - max_val) for k, v in raw_vals.items()}
    total = sum(exp_weights.values()) or 1.0
    return {k: v / total for k, v in exp_weights.items()}


def _perceptual_hash(values: List[float]) -> int:
    """64‑bit perceptual hash based on average comparison (first 64 values)."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits


def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two 64‑bit integers."""
    return (a ^ b).bit_count()


def build_perceptual_graph(
    feature_vectors: List[List[float]],
    max_hamming: int = 4,
) -> Dict[str, Set[str]]:
    """
    Construct an undirected similarity graph where each node corresponds to a
    feature vector (identified by its index as a string). Two nodes are linked
    when the Hamming distance between their perceptual hashes does not exceed
    ``max_hamming``.
    """
    n = len(feature_vectors)
    hashes = [_perceptual_hash(vec) for vec in feature_vectors]

    graph: Dict[str, Set[str]] = {str(i): set() for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if hamming_distance(hashes[i], hashes[j]) <= max_hamming:
                graph[str(i)].add(str(j))
                graph[str(j)].add(str(i))
    return graph


# ----------------------------------------------------------------------
# Koopman‑based kinetic scoring
# ----------------------------------------------------------------------
def _koopman_operator(
    adjacency: Dict[str, Set[str]],
) -> Tuple[np.ndarray, List[str]]:
    """
    Build a stochastic Koopman operator from an adjacency dict.

    The operator is a row‑stochastic matrix P where
        P_ij = 1 / deg(i)   if (i, j) is an edge,
        P_ij = 0           otherwise.
    The returned order list maps matrix rows/cols to node identifiers.
    """
    nodes = sorted(adjacency.keys())
    idx = {node: i for i, node in enumerate(nodes)}
    size = len(nodes)
    P = np.zeros((size, size), dtype=float)

    for node, neighbours in adjacency.items():
        i = idx[node]
        deg = len(neighbours) or 1  # avoid division by zero for isolated nodes
        for nb in neighbours:
            j = idx[nb]
            P[i, j] = 1.0 / deg

    # rows already sum to 1 (or 0 for isolated nodes, which we keep as 0)
    return P, nodes


def _kinetic_scores(
    koopman: np.ndarray,
    regret_weights: np.ndarray,
) -> np.ndarray:
    """
    Apply the Koopman operator to the regret‑weighted distribution.
    The result is a new distribution that incorporates the dynamics of the
    similarity graph.
    """
    return koopman @ regret_weights


# ----------------------------------------------------------------------
# Leader election using Bayesian update
# ----------------------------------------------------------------------
def elect_leaders_via_bayesian_graph(
    graph: Dict[str, Set[str]],
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual],
    posterior_threshold: float = 0.5,
) -> List[str]:
    """
    Compute a Bayesian posterior over actions using a Koopman‑transformed
    regret‑weighted strategy and return the identifiers of actions whose
    posterior probability exceeds ``posterior_threshold``.
    """
    # ------------------------------------------------------------------
    # 1. Regret‑weighted base distribution (prior)
    # ------------------------------------------------------------------
    regret_dist = compute_regret_weighted_strategy(actions, counterfactuals)

    # Align graph nodes with action identifiers.
    # If the graph was built on external feature vectors, we map the first
    # ``len(actions)`` nodes to the actions in order; otherwise we fall back
    # to using the action ids directly if they already appear as nodes.
    node_order: List[str] = []
    if set(regret_dist.keys()).issubset(graph.keys()):
        node_order = sorted(regret_dist.keys())
    else:
        # Assume graph nodes are numeric indices and actions are ordered
        node_order = [str(i) for i in range(len(actions))]

    # Build the Koopman operator on the induced sub‑graph of relevant nodes.
    subgraph = {node: graph[node] & set(node_order) for node in node_order}
    koopman, ordered_nodes = _koopman_operator(subgraph)

    # ------------------------------------------------------------------
    # 2. Kinetic scoring via Koopman dynamics
    # ------------------------------------------------------------------
    weight_vec = np.array([regret_dist.get(node, 0.0) for node in ordered_nodes])
    kinetic_vec = _kinetic_scores(koopman, weight_vec)

    # ------------------------------------------------------------------
    # 3. Bayesian posterior (Dirichlet‑like update with uniform prior)
    # ------------------------------------------------------------------
    uniform_prior = np.full_like(kinetic_vec, 1.0 / len(kinetic_vec))
    posterior_vec = kinetic_vec + uniform_prior
    posterior_vec /= posterior_vec.sum() or 1.0

    # ------------------------------------------------------------------
    # 4. Leader selection
    # ------------------------------------------------------------------
    leaders = [
        node
        for node, prob in zip(ordered_nodes, posterior_vec)
        if prob >= posterior_threshold
    ]
    return leaders


# ----------------------------------------------------------------------
# Example usage (executed only when run as a script)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a small toy problem
    actions = [
        MathAction("action0", 10.0, cost=1.0, risk=0.5),
        MathAction("action1", 20.0, cost=2.0, risk=1.0),
        MathAction("action2", 30.0, cost=1.5, risk=0.2),
    ]

    counterfactuals = [
        MathCounterfactual("action0", 5.0, probability=0.8),
        MathCounterfactual("action1", 10.0, probability=0.6),
        MathCounterfactual("action2", 15.0, probability=0.9),
    ]

    # Feature vectors could be anything; here we use the raw expected values.
    feature_vectors = [
        [a.expected_value, a.cost, a.risk] for a in actions
    ]

    # Build similarity graph
    similarity_graph = build_perceptual_graph(feature_vectors, max_hamming=2)

    # Elect leaders
    elected = elect_leaders_via_bayesian_graph(
        similarity_graph,
        actions,
        counterfactuals,
        posterior_threshold=0.4,
    )
    print("Elected leaders:", elected)