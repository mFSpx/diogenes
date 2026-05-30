# DARWIN HAMMER — match 2107, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_pherom_hybrid_bayes_claim_k_m9_s2.py (gen4)
# parent_b: hybrid_nlms_omni_chaotic_sprint_m59_s4.py (gen1)
# born: 2026-05-29T23:40:47Z

"""
Hybrid algorithm merging:

- Parent A: pheromone probability estimation, decision hygiene scoring,
  Shannon entropy, Bayesian update, and minimum‑cost tree evaluation.
- Parent B: Normalised Least‑Mean‑Squares (NLMS) adaptive filtering,
  synthetic graph generation, and seismic wavefront velocity propagation.

Mathematical bridge:
The pheromone probability vector for each node is treated as a
probability distribution. Its Shannon entropy quantifies uncertainty.
That entropy is used as the normalising evidence term in a Bayesian
update that rescales the NLMS weight vector after each adaptive step.
Thus the NLMS error‑driven delta and the Bayesian scaling jointly drive
the weight adaptation. The resulting weights are then employed as
edge‑impedance modifiers in the tree‑cost computation, linking the
graph‑based cost model of Parent A with the velocity field of Parent B.
"""

import math
import random
import sys
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
Point = Tuple[float, float]
NodeId = str
Edge = Tuple[NodeId, NodeId, int]          # (src, dst, impedance)
Adjacency = Dict[NodeId, List[Tuple[NodeId, int]]]

# ----------------------------------------------------------------------
# Parent‑A building blocks
# ----------------------------------------------------------------------
def calculate_pheromone_probabilities(surface_key: str, limit: int, db_url: str) -> List[float]:
    """Simulated pheromone probabilities calculation."""
    random.seed(hash(surface_key) % (2**32))
    return [random.random() for _ in range(limit)]

def shannon_entropy(probabilities: List[float]) -> float:
    """Compute the Shannon entropy of a probability distribution."""
    return -sum(p * math.log2(p) for p in probabilities if p > 0)

def bayesian_update(prior: float, likelihood: float, evidence: float) -> float:
    """Perform a Bayesian update given prior, likelihood, and evidence."""
    if evidence == 0:
        return prior
    return (prior * likelihood) / evidence

def tree_cost(
    adjacency: Adjacency,
    root: NodeId,
    edge_modifier: Dict[Tuple[NodeId, NodeId], float],
    base_path_weight: float = 0.2,
) -> float:
    """Minimum‑cost tree cost from root using modified edge weights."""
    visited = set()
    total_cost = 0.0
    queue = deque([root])
    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        for neighbor, imp in adjacency.get(node, []):
            if neighbor in visited:
                continue
            # base impedance plus a modifier derived from NLMS weights
            mod = edge_modifier.get((node, neighbor), 1.0)
            cost = (imp * base_path_weight) * mod
            total_cost += cost
            queue.append(neighbor)
    return total_cost

# ----------------------------------------------------------------------
# Parent‑B building blocks
# ----------------------------------------------------------------------
def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    return float(weights @ x)

def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, float]:
    if not (0.0 < mu < 2.0):
        raise ValueError("mu must be in the interval (0, 2)")
    y = nlms_predict(weights, x)
    error = target - y
    power = float(x @ x) + eps
    delta = mu * error * x / power
    new_weights = weights + delta
    return new_weights, error

def generate_synthetic_graph(
    num_nodes: int,
    avg_degree: int = 3,
) -> Tuple[Adjacency, np.ndarray]:
    random.seed(42)
    nodes = [f"n{i}" for i in range(num_nodes)]
    adjacency: Adjacency = {n: [] for n in nodes}
    edges: List[Edge] = []

    # chain backbone
    for i in range(num_nodes - 1):
        impedance = random.choice([1, 5, 10, 20])
        edges.append((nodes[i], nodes[i + 1], impedance))

    # extra random edges to reach desired average degree
    extra_edges = num_nodes * avg_degree // 2 - (num_nodes - 1)
    while extra_edges > 0:
        a, b = random.sample(nodes, 2)
        if any(nb == b for nb, _ in adjacency[a]):
            continue
        impedance = random.choice([1, 5, 10, 20])
        edges.append((a, b, impedance))
        extra_edges -= 1

    for u, v, imp in edges:
        adjacency[u].append((v, imp))
        adjacency[v].append((u, imp))

    # feature matrix (node embeddings)
    feature_dim = 4
    features = np.random.randn(num_nodes, feature_dim)

    return adjacency, features

def seismic_wavefront_velocities(
    adjacency: Adjacency,
    root: NodeId,
    max_visits: int = 10_000,
) -> Dict[NodeId, float]:
    visited: set[NodeId] = set()
    velocities: Dict[NodeId, float] = {}
    queue: deque[Tuple[NodeId, int]] = deque([(root, 0)])
    visits = 0
    while queue and visits < max_visits:
        node, stress = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        velocities[node] = 1.0 / max(float(stress), 1.0)
        visits += 1
        for neighbor, _ in adjacency.get(node, []):
            if neighbor not in visited:
                queue.append((neighbor, stress + 1))
    return velocities

# ----------------------------------------------------------------------
# Hybrid components
# ----------------------------------------------------------------------
def hybrid_graph_with_pheromones(
    num_nodes: int,
    avg_degree: int = 3,
    pheromone_limit: int = 5,
) -> Tuple[Adjacency, np.ndarray, Dict[NodeId, List[float]]]:
    """
    Generates a synthetic graph and attaches a pheromone probability
    vector to each node.
    """
    adjacency, features = generate_synthetic_graph(num_nodes, avg_degree)
    pheromones: Dict[NodeId, List[float]] = {}
    for node in adjacency.keys():
        probs = calculate_pheromone_probabilities(node, pheromone_limit, db_url="")
        # Normalise to a proper distribution
        total = sum(probs) or 1.0
        pheromones[node] = [p / total for p in probs]
    return adjacency, features, pheromones

def hybrid_nlms_step(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    pheromone_dist: List[float],
) -> Tuple[np.ndarray, float]:
    """
    Performs an NLMS adaptation step and then rescales the resulting
    weight vector with a Bayesian factor derived from the pheromone
    distribution (entropy as evidence).
    """
    # NLMS core update
    new_weights, error = nlms_update(weights, x, target)

    # Bayesian scaling
    prior = float(np.mean(new_weights))
    likelihood = float(np.mean(pheromone_dist))
    evidence = shannon_entropy(pheromone_dist) + 1e-9  # avoid zero division
    scale = bayesian_update(prior, likelihood, evidence)

    # Apply scaling element‑wise (preserve sign)
    scaled_weights = new_weights * np.sign(new_weights) * abs(scale)

    return scaled_weights, error

def hybrid_tree_cost(
    adjacency: Adjacency,
    root: NodeId,
    weights: np.ndarray,
    velocities: Dict[NodeId, float],
) -> float:
    """
    Computes a cost for the spanning tree rooted at `root`.
    Edge modifiers are derived from NLMS weights (mapped to edges) and
    further weighted by the inverse of the seismic velocities (higher
    stress → higher cost).
    """
    # Map each edge to a modifier using the weight vector (repeat if needed)
    edge_modifier: Dict[Tuple[NodeId, NodeId], float] = {}
    flat_weights = np.abs(weights)
    w_iter = iter(flat_weights.tolist() * 10)  # repeat to cover all edges

    for src, neighs in adjacency.items():
        for dst, _ in neighs:
            if (src, dst) in edge_modifier or (dst, src) in edge_modifier:
                continue
            edge_modifier[(src, dst)] = next(w_iter)

    # Base tree cost
    base_cost = tree_cost(adjacency, root, edge_modifier)

    # Velocity‑based regularisation
    vel_factor = sum(1.0 / (velocities.get(node, 1.0) + 1e-9) for node in adjacency)
    total_cost = base_cost * (1.0 + vel_factor / len(adjacency))

    return total_cost

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Parameters
    NODES = 10
    ROOT = "n0"

    # Build hybrid graph with pheromones
    adj, feats, pheros = hybrid_graph_with_pheromones(NODES)

    # Initialise NLMS weights (dimension matches feature size)
    weight_dim = feats.shape[1]
    w = np.zeros(weight_dim)

    # Perform a few adaptive steps over random targets
    for step in range(5):
        idx = random.randrange(NODES)
        x_vec = feats[idx]
        target_val = random.random()
        node_id = f"n{idx}"
        w, err = hybrid_nlms_step(w, x_vec, target_val, pheros[node_id])
        print(f"Step {step}: error={err:.4f}, weight_norm={np.linalg.norm(w):.4f}")

    # Propagate seismic velocities
    vel = seismic_wavefront_velocities(adj, ROOT)

    # Compute final hybrid cost
    final_cost = hybrid_tree_cost(adj, ROOT, w, vel)
    print(f"Hybrid tree cost: {final_cost:.4f}")