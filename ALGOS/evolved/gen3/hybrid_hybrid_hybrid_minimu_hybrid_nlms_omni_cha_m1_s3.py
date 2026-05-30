# DARWIN HAMMER — match 1, survivor 3
# gen: 3
# parent_a: hybrid_hybrid_minimum_cost__epistemic_certainty_m48_s0.py (gen2)
# parent_b: hybrid_nlms_omni_chaotic_sprint_m59_s5.py (gen1)
# born: 2026-05-29T23:26:25Z

import math
import numpy as np
from typing import Dict, List, Tuple

Point = tuple[float, float]
Edge = tuple[str, str]
NodeId = str
EdgeWithImpedance = Tuple[NodeId, NodeId, int]

EPISTEMIC_FLAGS: tuple[str, ...] = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")

def length(a: Point, b: Point) -> float:
    """Calculate the Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """Compute the marginal probability for Bayesian update."""
    if not all(0 <= x <= 1 for x in (prior, likelihood, false_positive)):
        raise ValueError("probabilities must be in [0,1]")
    return likelihood * prior + false_positive * (1.0 - prior)

def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    """Perform Bayesian update on the prior probability."""
    if marginal <= 0:
        raise ValueError("P(E) must be > 0")
    return prior * likelihood / marginal

def certainty(
    label: str,
    *,
    confidence_bps: int,
    authority_class: str,
    rationale: str,
    evidence_refs: tuple[str, ...] = (),
) -> dict[str, str]:
    return {
        "label": label,
        "confidence_bps": str(confidence_bps),
        "authority_class": authority_class,
        "rationale": rationale,
        "evidence_refs": evidence_refs,
    }

def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Return the dot‑product prediction w·x."""
    return float(weights @ x)

def nlms_batch_update(
    weights: np.ndarray,
    X: np.ndarray,
    targets: np.ndarray,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform a *batch* NLMS update.

    Parameters
    ----------
    weights : np.ndarray
        Current weight vector (shape: (d,)).
    X : np.ndarray
        Input matrix where each row is an input vector x_i (shape: (N, d)).
    targets : np.ndarray
        Desired scalar outputs v_i (shape: (N,)).
    mu : float
        Learning rate in (0, 2).
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    new_weights : np.ndarray
        Updated weight vector.
    errors : np.ndarray
        Prediction errors for each sample.
    """
    if not (0.0 < mu < 2.0):
        raise ValueError("mu must be in the interval (0, 2)")

    # Predictions and errors
    preds = X @ weights
    errors = targets - preds

    # Normalized step for each sample
    powers = np.sum(X * X, axis=1) + eps  # shape (N,)
    steps = (mu * errors / powers)[:, None] * X   # shape (N, d)

    # Aggregate the per‑sample steps
    delta_w = steps.sum(axis=0)
    new_weights = weights + delta_w
    return new_weights, errors

def certainty_modified_nlms_update(
    weights: np.ndarray,
    X: np.ndarray,
    targets: np.ndarray,
    certainty_flags: list[dict[str, str]],
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform a *batch* certainty-modified NLMS update.

    Parameters
    ----------
    weights : np.ndarray
        Current weight vector (shape: (d,)).
    X : np.ndarray
        Input matrix where each row is an input vector x_i (shape: (N, d)).
    targets : np.ndarray
        Desired scalar outputs v_i (shape: (N,)).
    certainty_flags : list[dict[str, str]]
        List of certainty flags for each sample.
    mu : float
        Learning rate in (0, 2).
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    new_weights : np.ndarray
        Updated weight vector.
    errors : np.ndarray
        Prediction errors for each sample.
    """
    if not (0.0 < mu < 2.0):
        raise ValueError("mu must be in the interval (0, 2)")

    # Predictions and errors
    preds = X @ weights
    errors = targets - preds

    # Certainty-modified step for each sample
    modified_steps = []
    for i, (x, error, flag) in enumerate(zip(X, errors, certainty_flags)):
        confidence_bps = int(flag["confidence_bps"])
        # Map certainty to modified step size
        modified_mu = mu * np.exp(confidence_bps / 100)  # exponential mapping
        power = np.sum(x * x) + eps
        step = (modified_mu * error / power) * x
        modified_steps.append(step)
    delta_w = np.sum(modified_steps, axis=0)
    new_weights = weights + delta_w
    return new_weights, np.array(errors)

class DisjointSet:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            if self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            elif self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            else:
                self.parent[root_y] = root_x
                self.rank[root_x] += 1

def hybrid_tree_cost(
    nodes: dict[str, Point],
    edges: list[Edge],
    root: str,
    prior_probabilities: dict[str, float],
    likelihoods: dict[Edge, float],
    false_positives: dict[Edge, float],
    certainty_flags: dict[Edge, dict[str, str]],
):
    # Initialize minimum spanning tree
    mst = []
    # Kruskal's algorithm with certainty flags
    edges_with_certainty = []
    for edge in edges:
        node1, node2 = edge
        # Calculate edge weight with certainty
        weight = length(nodes[node1], nodes[node2])
        flag = certainty_flags[edge]
        confidence_bps = int(flag["confidence_bps"])
        weight *= np.exp(-confidence_bps / 100)  # exponential mapping
        edges_with_certainty.append((weight, edge))
    edges_with_certainty.sort()

    disjoint_set = DisjointSet(len(nodes))
    for weight, edge in edges_with_certainty:
        node1, node2 = edge
        if disjoint_set.find(list(nodes.keys()).index(node1)) != disjoint_set.find(list(nodes.keys()).index(node2)):
            disjoint_set.union(list(nodes.keys()).index(node1), list(nodes.keys()).index(node2))
            mst.append(edge)

    return calculate_tree_cost(mst, nodes)

def calculate_tree_cost(mst: list[Edge], nodes: dict[str, Point]) -> float:
    # Calculate the total cost of the minimum spanning tree
    total_cost = 0
    for edge in mst:
        node1, node2 = edge
        total_cost += length(nodes[node1], nodes[node2])
    return total_cost

if __name__ == "__main__":
    # Smoke test
    nodes = {"A": (0, 0), "B": (3, 4), "C": (6, 8)}
    edges = [("A", "B"), ("B", "C"), ("C", "A")]
    root = "A"
    prior_probabilities = {"A": 0.5, "B": 0.3, "C": 0.2}
    likelihoods = {("A", "B"): 0.7, ("B", "C"): 0.6, ("C", "A"): 0.8}
    false_positives = {("A", "B"): 0.1, ("B", "C"): 0.2, ("C", "A"): 0.3}
    certainty_flags = {
        ("A", "B"): certainty("edge1", confidence_bps=50, authority_class="high", rationale="test"),
        ("B", "C"): certainty("edge2", confidence_bps=30, authority_class="medium", rationale="test"),
        ("C", "A"): certainty("edge3", confidence_bps=70, authority_class="high", rationale="test"),
    }
    cost = hybrid_tree_cost(nodes, edges, root, prior_probabilities, likelihoods, false_positives, certainty_flags)
    print(f"Hybrid tree cost: {cost}")

    # NLMS test
    weights = np.array([0.5, 0.5])
    X = np.array([[1, 2], [3, 4]])
    targets = np.array([2, 4])
    certainty_flags = [
        certainty("sample1", confidence_bps=50, authority_class="high", rationale="test"),
        certainty("sample2", confidence_bps=30, authority_class="medium", rationale="test"),
    ]
    new_weights, errors = certainty_modified_nlms_update(weights, X, targets, certainty_flags)
    print(f"Updated weights: {new_weights}")