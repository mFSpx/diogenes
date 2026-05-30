# DARWIN HAMMER — match 1263, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hard_truth_ma_m551_s2.py (gen4)
# parent_b: hybrid_hybrid_hybrid_nlms_o_hybrid_hybrid_percep_m598_s2.py (gen5)
# born: 2026-05-29T23:34:49Z

"""Hybrid LSM‑Bandit‑NLMS Fusion
================================

Parent A (hybrid_hybrid_hybrid_hybrid_hybrid_hard_truth_ma_m551_s2.py) provides a
store dynamics where an *inflow* scalar and an *outflow* scalar drive a level
update and a “dance’’ control signal that rescales bandit propensities.

Parent B (hybrid_hybrid_hybrid_nlms_o_hybrid_hybrid_percep_m598_s2.py) supplies a
normalized‑least‑mean‑squares (NLMS) adaptive filter, a graph built from the
weight vector (pairwise similarity edges) and utilities for similarity
matrices and a minimum‑cost‑tree (MST) extraction.

**Mathematical bridge**

* The *inflow* for the store is taken as the LSM‑style similarity between two
  weight vectors `w_a` and `w_b`.  We reuse the Gaussian kernel from Parent B:
  `inflow = exp(- (epsilon * euclidean(w_a, w_b))**2)` which lies in (0,1].

* The *outflow* is the total edge length of the MST built on the similarity
  graph derived from a single weight vector (here we use `w_a`).  Edge length
  is defined as `1 - similarity`, where similarity follows Parent B’s
  construction.

* The store update equation from Parent A  

  
  Δ = α·inflow - β·outflow
  level_{t+1} = max(0, level_t + Δ·dt)
  dance = tanh(γ·Δ)
  

  is applied with the above inflow/outflow.  The resulting `dance` rescales
  each bandit propensity multiplicatively, completing the fusion.

The module defines three public functions that showcase the hybrid operation
and finishes with a tiny smoke test."""

import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent‑A structures (trimmed)
# ----------------------------------------------------------------------
@dataclass
class StoreState:
    """Store dynamics state."""
    level: float = 0.0
    alpha: float = 1.0   # inflow gain
    beta: float = 1.0    # outflow gain
    dt: float = 1.0
    gamma: float = 1.0   # non‑linearity gain


@dataclass(frozen=True)
class BanditAction:
    """Result of a bandit selection."""
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str


# ----------------------------------------------------------------------
# Parent‑B utilities (NLMS, graph, similarity, MST)
# ----------------------------------------------------------------------
def predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Linear prediction."""
    return float(np.dot(weights, x))


def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, float]:
    """Normalized LMS weight update returning new weights and error."""
    y = predict(weights, x)
    error = target - y
    power = float(np.dot(x, x) + eps)
    next_weights = weights + mu * error * x / power
    return next_weights, error


def construct_graph(weights: np.ndarray) -> Dict[int, List[Tuple[int, float]]]:
    """
    Build a fully‑connected similarity graph.
    Edge weight = similarity ∈ [0,1].
    """
    n = len(weights)
    graph: Dict[int, List[Tuple[int, float]]] = {}
    for i in range(n):
        graph[i] = []
        for j in range(n):
            if i == j:
                continue
            diff = abs(weights[i] - weights[j])
            similarity = 1.0 - diff / (1.0 + diff)  # same formula as Parent B
            graph[i].append((j, similarity))
    return graph


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    """L2 distance."""
    return float(np.linalg.norm(a - b))


def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian kernel used as inflow."""
    return math.exp(-((epsilon * r) ** 2))


def mst_total_length(graph: Dict[int, List[Tuple[int, float]]]) -> float:
    """
    Prim's algorithm on the *edge length* = 1 - similarity.
    Returns the sum of edge lengths in the minimum spanning tree.
    """
    if not graph:
        return 0.0
    start = next(iter(graph))
    visited = {start}
    edges = [
        (1.0 - w, start, nb) for nb, w in graph[start]
    ]  # (length, src, dst)
    total = 0.0
    while len(visited) < len(graph):
        # pick smallest edge crossing the cut
        edges = [(l, s, d) for l, s, d in edges if d not in visited]
        if not edges:
            break  # disconnected (should not happen)
        length, src, dst = min(edges, key=lambda x: x[0])
        total += length
        visited.add(dst)
        # add new frontier edges
        for nb, sim in graph[dst]:
            if nb not in visited:
                edges.append((1.0 - sim, dst, nb))
    return total


# ----------------------------------------------------------------------
# Hybrid core: bridge between A and B
# ----------------------------------------------------------------------
def compute_inflow_outflow(
    w_a: np.ndarray,
    w_b: np.ndarray,
    epsilon: float = 1.0,
) -> Tuple[float, float]:
    """
    Compute the scalar inflow (Gaussian of Euclidean distance between two weight
    vectors) and outflow (MST total length of the graph built from w_a).
    """
    inflow = gaussian(euclidean(w_a, w_b), epsilon)
    graph = construct_graph(w_a)
    outflow = mst_total_length(graph)
    return inflow, outflow


def update_store(state: StoreState, inflow: float, outflow: float) -> float:
    """
    Apply Parent A's store equation and return the dance signal.
    """
    delta = state.alpha * inflow - state.beta * outflow
    state.level = max(0.0, state.level + delta * state.dt)
    dance = math.tanh(state.gamma * delta)
    return dance


def modulate_bandits(bandits: List[BanditAction], dance: float) -> List[BanditAction]:
    """
    Rescale each bandit propensity by (1 + dance) and return new actions.
    """
    factor = 1.0 + dance
    return [
        BanditAction(
            action_id=b.action_id,
            propensity=b.propensity * factor,
            expected_reward=b.expected_reward,
            confidence_bound=b.confidence_bound,
            algorithm=b.algorithm,
        )
        for b in bandits
    ]


# ----------------------------------------------------------------------
# Demonstration functions
# ----------------------------------------------------------------------
def hybrid_step(
    state: StoreState,
    weights_a: np.ndarray,
    weights_b: np.ndarray,
    bandits: List[BanditAction],
) -> Tuple[StoreState, List[BanditAction]]:
    """
    One hybrid iteration:
      1. Compute inflow/outflow from the two weight vectors.
      2. Update the store and obtain the dance signal.
      3. Modulate the bandit propensities.
    Returns the updated state and the new bandit list.
    """
    inflow, outflow = compute_inflow_outflow(weights_a, weights_b)
    dance = update_store(state, inflow, outflow)
    new_bandits = modulate_bandits(bandits, dance)
    return state, new_bandits


def random_bandits(n: int) -> List[BanditAction]:
    """Generate a list of n dummy bandit actions for testing."""
    return [
        BanditAction(
            action_id=f"act_{i}",
            propensity=random.random(),
            expected_reward=random.random(),
            confidence_bound=random.random(),
            algorithm="hybrid",
        )
        for i in range(n)
    ]


def random_weights(dim: int) -> np.ndarray:
    """Create a random weight vector."""
    return np.random.rand(dim)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Initialise components
    state = StoreState(level=0.5, alpha=1.2, beta=0.8, dt=0.1, gamma=2.0)
    w1 = random_weights(8)
    w2 = random_weights(8)
    bandits = random_bandits(5)

    # Perform a few hybrid steps
    for step in range(3):
        state, bandits = hybrid_step(state, w1, w2, bandits)
        print(f"Step {step+1}: level={state.level:.4f}, dance signal applied")
        for b in bandits:
            print(f"  {b.action_id}: propensity={b.propensity:.4f}")

    print("Smoke test completed without errors.")