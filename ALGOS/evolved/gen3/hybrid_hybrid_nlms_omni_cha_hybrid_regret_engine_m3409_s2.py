# DARWIN HAMMER — match 3409, survivor 2
# gen: 3
# parent_a: hybrid_nlms_omni_chaotic_sprint_m59_s4.py (gen1)
# parent_b: hybrid_regret_engine_hybrid_doomsday_cale_m19_s1.py (gen2)
# born: 2026-05-29T23:49:54Z

"""Hybrid NLMS-Regret Engine

This module fuses the adaptive filtering core of *parent algorithm A* (NLMS weight
update) with the regret‑weighted decision logic of *parent algorithm B*.
The mathematical bridge is the use of the NLMS predicted values as the
“expected values” in the regret engine.  After each NLMS update the distribution
of predicted values is quantified by the Gini coefficient, which is then used
to scale the exponential regret weights.  Thus the adaptive filter supplies a
dynamic, data‑driven estimate of action utilities, while the regret engine
provides a probabilistic policy that is regularised by the inequality measure
(Gini).

The main public entry point is `hybrid_nlms_regret_strategy`.
"""

import math
import random
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
NodeId = str
Edge = Tuple[NodeId, NodeId, int]  # (src, dst, impedance)


# ----------------------------------------------------------------------
# Parent A – NLMS core (adapted)
# ----------------------------------------------------------------------
def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Linear prediction y = wᵀx."""
    return float(weights @ x)


def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, float]:
    """
    Normalised Least‑Mean‑Squares weight update.

    Returns the new weight vector and the instantaneous error.
    """
    if not (0.0 < mu < 2.0):
        raise ValueError("mu must be in the interval (0, 2)")
    y = nlms_predict(weights, x)
    error = target - y
    power = float(x @ x) + eps
    delta = mu * error * x / power
    new_weights = weights + delta
    return new_weights, error


def generate_synthetic_graph(num_nodes: int, avg_degree: int = 3) -> Tuple[Dict[NodeId, List[Tuple[NodeId, int]]], np.ndarray]:
    """
    Creates an undirected graph with random impedances and a random feature
    matrix (num_nodes × feature_dim).  The feature matrix will be used as the
    NLMS input vectors.
    """
    random.seed(42)
    nodes = [f"n{i}" for i in range(num_nodes)]
    adjacency: Dict[NodeId, List[Tuple[NodeId, int]]] = {n: [] for n in nodes}
    edges: List[Edge] = []

    # Build a backbone chain to guarantee connectivity
    for i in range(num_nodes - 1):
        impedance = random.choice([1, 5, 10, 20])
        edges.append((nodes[i], nodes[i + 1], impedance))

    # Add extra random edges to reach the desired average degree
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

    feature_dim = 4
    features = np.random.randn(num_nodes, feature_dim)

    return adjacency, features


# ----------------------------------------------------------------------
# Parent B – Regret & Gini core (adapted)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathAction:
    """Action identifier together with its ground‑truth expected value."""
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


def gini_coefficient(values: List[float]) -> float:
    """Standard Gini coefficient for a non‑negative list."""
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0.0:
        return 0.0
    if xs[0] < 0:
        raise ValueError("values must be non‑negative")
    n = len(xs)
    cumulative = sum((2 * i - n - 1) * x for i, x in enumerate(xs, 1))
    return cumulative / (n * sum(xs))


def regret_weighted_distribution(actions: List[MathAction], gini_scale: float = 1.0) -> Dict[str, float]:
    """
    Compute a softmax‑like distribution over actions, optionally scaled by a
    Gini factor.  The distribution sums to one.
    """
    if not actions:
        return {}
    # Base value = expected – cost – risk
    base_vals = {a.id: a.expected_value - a.cost - a.risk for a in actions}
    best = max(base_vals.values())
    # Exponential weighting centred at the best value
    raw = {k: math.exp(v - best) * gini_scale for k, v in base_vals.items()}
    total = sum(raw.values()) or 1.0
    return {k: v / total for k, v in raw.items()}


# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------
def hybrid_predict_actions(
    weights: np.ndarray,
    features: np.ndarray,
    actions: List[MathAction],
) -> List[MathAction]:
    """
    Use the NLMS predictor to generate *estimated* expected values for each
    action.  The node identifier of an action must match the row index in the
    feature matrix.
    """
    updated_actions = []
    for act in actions:
        # Assume action.id is of the form "n{i}" matching feature row i
        idx = int(act.id[1:])  # strip leading 'n'
        x = features[idx]
        pred = nlms_predict(weights, x)
        updated_actions.append(
            MathAction(
                id=act.id,
                expected_value=pred,
                cost=act.cost,
                risk=act.risk,
            )
        )
    return updated_actions


def hybrid_nlms_regret_strategy(
    adjacency: Dict[NodeId, List[Tuple[NodeId, int]]],
    features: np.ndarray,
    actions: List[MathAction],
    mu: float = 0.5,
    epochs: int = 3,
) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Core hybrid algorithm:

    1. Initialise NLMS weights (zero vector).
    2. For a number of epochs iterate over actions:
       * Predict a value with the current weights.
       * Treat the *ground‑truth* expected_value of the action as the target.
       * Perform an NLMS update.
    3. After training, compute a regret‑weighted distribution where the Gini
       coefficient of the predicted values modulates the exponential weights.

    Returns the final probability distribution over actions and the learned
    weight vector.
    """
    feature_dim = features.shape[1]
    weights = np.zeros(feature_dim)

    # Training loop (NLMS adaptation)
    for _ in range(epochs):
        for act in actions:
            idx = int(act.id[1:])
            x = features[idx]
            target = act.expected_value  # ground‑truth
            weights, _ = nlms_update(weights, x, target, mu=mu)

    # Predict with the learned weights
    predicted_actions = hybrid_predict_actions(weights, features, actions)

    # Gini of the predicted values (used as a scaling factor)
    pred_vals = [a.expected_value for a in predicted_actions]
    gini = gini_coefficient(pred_vals)

    # Regret distribution scaled by Gini
    distribution = regret_weighted_distribution(predicted_actions, gini_scale=gini)

    return distribution, weights


def seismic_wavefront_velocities(
    adjacency: Dict[NodeId, List[Tuple[NodeId, int]]],
    root: NodeId,
    max_visits: int = 10_000,
) -> Dict[NodeId, float]:
    """
    Simple BFS‑like traversal that assigns a velocity inversely proportional to
    accumulated stress (sum of impedances).  This function is retained from the
    original NLMS parent to illustrate that graph‑level information can be
    combined with the hybrid strategy if desired.
    """
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
        for neighbor, impedance in adjacency.get(node, []):
            if neighbor not in visited:
                queue.append((neighbor, stress + impedance))

    return velocities


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Build a tiny synthetic graph
    adj, feats = generate_synthetic_graph(num_nodes=8, avg_degree=2)

    # Create one action per node with a random ground‑truth expected value
    rng = np.random.default_rng(123)
    actions = [
        MathAction(id=f"n{i}", expected_value=float(rng.normal()), cost=0.0, risk=0.0)
        for i in range(feats.shape[0])
    ]

    # Run the hybrid algorithm
    policy, final_weights = hybrid_nlms_regret_strategy(
        adjacency=adj,
        features=feats,
        actions=actions,
        mu=0.7,
        epochs=5,
    )

    print("Learned NLMS weights:", final_weights)
    print("Hybrid regret policy (probabilities):")
    for aid, prob in sorted(policy.items()):
        print(f"  {aid}: {prob:.4f}")

    # Optional: compute wavefront velocities from node 'n0'
    velocities = seismic_wavefront_velocities(adj, root="n0")
    print("\nSample wavefront velocities (first 5 nodes):")
    for nid, vel in list(velocities.items())[:5]:
        print(f"  {nid}: {vel:.3f}")