# DARWIN HAMMER — match 3320, survivor 0
# gen: 3
# parent_a: hybrid_minimum_cost_tree_bayes_update_m6_s2.py (gen1)
# parent_b: hybrid_hybrid_distributed_l_hybrid_hoeffding_tre_m24_s4.py (gen2)
# born: 2026-05-29T23:49:11Z

from __future__ import annotations
import random
import math
import sys
import pathlib
import numpy as np

# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Edge = Tuple[str, str]
Node = str

# ----------------------------------------------------------------------
# Algorithm A – deterministic tree utilities
# ----------------------------------------------------------------------
def length(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def tree_metrics(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
) -> Tuple[Dict[str, List[str]], Dict[Edge, float], Dict[str, float]]:
    """
    Build adjacency, compute Euclidean edge lengths and root‑to‑node distances.

    Returns
    -------
    adj : dict mapping node → list of neighbours
    edge_len : dict mapping edge (ordered a
    """
    adj = {}
    edge_len = {}
    root_dist = {root: 0}
    queue = [(root, 0)]
    while queue:
        node, dist = queue.pop(0)
        adj[node] = []
        for edge in edges:
            if edge[0] == node:
                neighbor = edge[1]
                if neighbor not in root_dist:
                    root_dist[neighbor] = dist + 1
                    queue.append((neighbor, dist + 1))
                edge_len[edge] = length(nodes[node], nodes[neighbor])
                adj[node].append(neighbor)
    return adj, edge_len, root_dist

# ----------------------------------------------------------------------
# Algorithm B – probabilistic primitives
# ----------------------------------------------------------------------
def broadcast_probability(total_phases: int, current_phase: int) -> float:
    """Probability that a node broadcasts in the current phase."""
    if total_phases < 1 or current_phase < 1:
        raise ValueError('phases and phase must be positive')
    return min(1.0, 1.0 / (2 ** max(0, total_phases - current_phase)))

def hoeffding_bound(
    successes: float,
    total: float,
    confidence: float = 0.95,
) -> float:
    """
    Hoeffding bound for a binomial distribution.

    Parameters
    ----------
    successes : float
        Number of successes
    total : float
        Total number of trials
    confidence : float, optional
        Confidence level (default: 0.95)
    """
    z = math.sqrt(2 * math.log(2 / confidence))
    return (z ** 2 / (2 * total)) * successes

# ----------------------------------------------------------------------
# Hybrid module combining Minimum‑Cost Tree scoring and Bayesian evidence update
# ----------------------------------------------------------------------
def hybrid_tree_cost(
    adj: Dict[str, List[str]],
    edge_len: Dict[Edge, float],
    root_dist: Dict[str, float],
    nodes: Dict[str, Point],
    bayes_edge_posteriors: np.ndarray,
    bayes_node_beliefs: np.ndarray,
    path_weight: float,
) -> float:
    """
    Evaluates the hybrid tree cost using the Bayesian posteriors.

    Parameters
    ----------
    adj : dict mapping node → list of neighbours
    edge_len : dict mapping edge (ordered a
    root_dist : dict mapping node → root-to-node distance
    nodes : dict mapping node → point coordinates
    bayes_edge_posteriors : numpy array of Bayesian edge posteriors
    bayes_node_beliefs : numpy array of Bayesian node beliefs
    path_weight : float
        Path-weight (default: 1.0)
    """
    hybrid_cost = 0
    for edge in edge_len:
        hybrid_cost += bayes_edge_posteriors[edge] * edge_len[edge]
    for node in adj:
        hybrid_cost += path_weight * bayes_node_beliefs[node] * root_dist[node]
    return hybrid_cost

def bayes_edge_posteriors(
    adj: Dict[str, List[str]],
    edge_len: Dict[Edge, float],
    root_dist: Dict[str, float],
    nodes: Dict[str, Point],
    bayes_edge_likelihood: np.ndarray,
    bayes_edge_prior: np.ndarray,
    false_positive_rate: float,
) -> np.ndarray:
    """
    Evaluates the Bayesian edge posteriors.

    Parameters
    ----------
    adj : dict mapping node → list of neighbours
    edge_len : dict mapping edge (ordered a
    root_dist : dict mapping node → root-to-node distance
    nodes : dict mapping node → point coordinates
    bayes_edge_likelihood : numpy array of Bayesian edge likelihoods
    bayes_edge_prior : numpy array of Bayesian edge priors
    false_positive_rate : float
        False positive rate (default: 1.0)
    """
    posterior = (bayes_edge_likelihood * bayes_edge_prior) / (
        bayes_edge_likelihood * bayes_edge_prior +
        false_positive_rate * (1 - bayes_edge_prior)
    )
    return posterior

def bayes_node_beliefs(
    adj: Dict[str, List[str]],
    edge_len: Dict[Edge, float],
    root_dist: Dict[str, float],
    nodes: Dict[str, Point],
    bayes_edge_posteriors: np.ndarray,
) -> np.ndarray:
    """
    Evaluates the Bayesian node beliefs.

    Parameters
    ----------
    adj : dict mapping node → list of neighbours
    edge_len : dict mapping edge (ordered a
    root_dist : dict mapping node → root-to-node distance
    nodes : dict mapping node → point coordinates
    bayes_edge_posteriors : numpy array of Bayesian edge posteriors
    """
    node_beliefs = np.zeros(len(adj))
    for node in adj:
        for neighbor in adj[node]:
            node_beliefs[node] += bayes_edge_posteriors[(node, neighbor)] * edge_len[(node, neighbor)]
        node_beliefs[node] /= len(adj[node])
    return node_beliefs

def hybrid_hoeffding_tree(
    adj: Dict[str, List[str]],
    edge_len: Dict[Edge, float],
    root_dist: Dict[str, float],
    nodes: Dict[str, Point],
    hoeffding_bound: float,
    tropical_gain: float,
    temperature: float,
) -> bool:
    """
    Evaluates the hybrid Hoeffding Tree cost using the Bayesian posteriors.

    Parameters
    ----------
    adj : dict mapping node → list of neighbours
    edge_len : dict mapping edge (ordered a
    root_dist : dict mapping node → root-to-node distance
    nodes : dict mapping node → point coordinates
    hoeffding_bound : float
        Hoeffding bound (default: 0.0)
    tropical_gain : float
        Tropical gain (default: 0.0)
    temperature : float
        Temperature (default: 1.0)
    """
    hybrid_cost = hoeffding_bound - tropical_gain
    acceptance_probability = math.exp(-hybrid_cost / temperature)
    return random.random() < acceptance_probability

# ----------------------------------------------------------------------
# Main program
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Smoke test
    nodes = {
        'A': (0, 0),
        'B': (1, 0),
        'C': (1, 1),
        'D': (0, 1),
    }
    edges = [('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'D')]
    adj, edge_len, root_dist = tree_metrics(nodes, edges, 'A')
    bayes_edge_likelihood = np.array([0.5, 0.5, 0.5, 0.5])
    bayes_edge_prior = np.array([0.5, 0.5, 0.5, 0.5])
    false_positive_rate = 0.1
    bayes_edge_posteriors = bayes_edge_posteriors(adj, edge_len, root_dist, nodes, bayes_edge_likelihood, bayes_edge_prior, false_positive_rate)
    bayes_node_beliefs = bayes_node_beliefs(adj, edge_len, root_dist, nodes, bayes_edge_posteriors)
    path_weight = 1.0
    hybrid_cost = hybrid_tree_cost(adj, edge_len, root_dist, nodes, bayes_edge_posteriors, bayes_node_beliefs, path_weight)
    print(hybrid_cost)
    hoeffding_bound_value = hoeffding_bound(2, 4)
    tropical_gain_value = 0.5
    temperature_value = 1.0
    hybrid_hoeffding_tree_result = hybrid_hoeffding_tree(adj, edge_len, root_dist, nodes, hoeffding_bound_value, tropical_gain_value, temperature_value)
    print(hybrid_hoeffding_tree_result)