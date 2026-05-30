# DARWIN HAMMER — match 4242, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_nlms_o_hybrid_hybrid_percep_m598_s4.py (gen5)
# parent_b: hybrid_hybrid_hybrid_rlct_g_hybrid_hybrid_hybrid_m2070_s0.py (gen5)
# born: 2026-05-29T23:54:26Z

"""Hybrid Algorithm integrating RBF‑NLMS learning (Parent A) with RLCT‑based complexity scaling
and epistemic certainty flags (Parent B).

Mathematical bridge:
- The RLCT (Rate of Learning Curve Transition) is estimated from a log‑log fit of
  training loss versus sample size. It quantifies model complexity.
- RLCT is used to *scale* the NLMS step‑size μ (μ_eff = μ / (1 + |RLCT|)) and to
  re‑weight similarity scores in the graph constructed from the learned RBF output
  weights (edge weight ← similarity^(1+|RLCT|)).
- The instantaneous prediction error is mapped to an epistemic flag (FACT,
  PROBABLE, …) providing a qualitative certainty measure for each output.

The resulting system performs adaptive RBF prediction, updates its weights with a
complexity‑aware NLMS rule, builds a similarity graph whose topology reflects both
weight similarity and RLCT‑derived scaling, and finally extracts a Minimum
Spanning Tree (MST) from that graph. Three core functions demonstrate the hybrid
behaviour: `hybrid_adapt`, `compute_epistemic_flag`, and `build_scaled_similarity_graph`.
"""

import math
import random
import sys
import pathlib
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – RBF network with NLMS adaptation
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Node:
    id: int
    weight: float

def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return np.linalg.norm(a - b)

def rbf_activation(x: np.ndarray, centers: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian RBF activations for a single input vector x."""
    dists = np.linalg.norm(centers - x, axis=1)
    return np.exp(- (dists ** 2) / (2 * sigma ** 2))

class RBFNLMS:
    """RBF network whose output weights are adapted with the Normalised LMS rule."""
    def __init__(self, centers: np.ndarray,
                 sigma: float = 1.0,
                 mu: float = 0.5,
                 eps: float = 1e-9):
        self.centers = centers                     # shape (M, D)
        self.sigma = sigma
        self.base_mu = mu
        self.eps = eps
        self.weights = np.random.rand(centers.shape[0])  # one weight per RBF centre

    def predict(self, x: np.ndarray) -> float:
        phi = rbf_activation(x, self.centers, self.sigma)   # shape (M,)
        return float(np.dot(self.weights, phi))

    def adapt(self, x: np.ndarray, target: float, rlct: float = 0.0) -> float:
        """
        Perform one NLMS update using a learning rate scaled by the RLCT.
        Returns the instantaneous error.
        """
        phi = rbf_activation(x, self.centers, self.sigma)
        y = float(np.dot(self.weights, phi))
        error = target - y

        # Complexity‑aware step size
        mu_eff = self.base_mu / (1.0 + abs(rlct))
        power = float(np.dot(phi, phi) + self.eps)
        self.weights += (mu_eff * error / power) * phi
        return error

# ----------------------------------------------------------------------
# Parent B – RLCT estimation and epistemic certainty
# ----------------------------------------------------------------------
EPISTEMIC_FLAGS = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")

def estimate_rlct_from_losses(train_losses_per_n: List[float],
                              n_values: List[int]) -> float:
    """
    Estimate the RLCT (slope) from a log‑log plot of loss vs. sample size.
    Uses ordinary least squares on (log n, log loss).
    """
    losses = np.asarray(train_losses_per_n, dtype=np.float64)
    ns = np.asarray(n_values, dtype=np.float64)

    if np.any(ns <= math.e):
        raise ValueError("All n_values must be > e for log(log(n)) to be positive")

    if losses.shape != ns.shape:
        raise ValueError("train_losses_per_n and n_values must have the same length")

    # Guard against non‑positive losses
    y = np.log(np.maximum(losses, 1e-10))
    z = np.log(ns)
    # Linear fit y = a*z + b ; a is the RLCT
    a, _ = np.polyfit(z, y, 1)
    return float(a)

def compute_epistemic_flag(error: float,
                           error_scale: float = 1.0) -> str:
    """
    Map a signed error to an epistemic certainty flag.
    The magnitude of the error relative to `error_scale` determines the flag:
        |e| < 0.1·scale   → FACT
        |e| < 0.3·scale   → PROBABLE
        |e| < 0.6·scale   → POSSIBLE
        |e| < 1.0·scale   → BULLSHIT
        otherwise        → SURE_MAYBE
    """
    mag = abs(error) / (error_scale + 1e-12)
    if mag < 0.1:
        return EPISTEMIC_FLAGS[0]
    if mag < 0.3:
        return EPISTEMIC_FLAGS[1]
    if mag < 0.6:
        return EPISTEMIC_FLAGS[2]
    if mag < 1.0:
        return EPISTEMIC_FLAGS[3]
    return EPISTEMIC_FLAGS[4]

# ----------------------------------------------------------------------
# Hybrid utilities – graph construction, MST, and combined operation
# ----------------------------------------------------------------------
def construct_similarity_graph(weights: np.ndarray,
                               rlct: float = 0.0) -> Dict[int, List[Tuple[int, float]]]:
    """
    Build a fully‑connected undirected graph where edge weights are similarity scores.
    The base similarity is 1 - |wi - wj| / (1 + |wi - wj|) (as in Parent A).
    The RLCT scales the similarity by raising it to the power (1 + |rlct|),
    thereby sharpening or flattening the graph depending on model complexity.
    """
    n = len(weights)
    graph: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(n)}
    scale = 1.0 + abs(rlct)

    for i in range(n):
        for j in range(i + 1, n):
            base_sim = 1.0 - abs(weights[i] - weights[j]) / (1.0 + abs(weights[i] - weights[j]))
            scaled_sim = base_sim ** scale
            graph[i].append((j, scaled_sim))
            graph[j].append((i, scaled_sim))
    return graph

def prim_mst(graph: Dict[int, List[Tuple[int, float]]]) -> List[Tuple[int, int, float]]:
    """
    Compute the Minimum Spanning Tree (MST) of an undirected weighted graph
    using Prim's algorithm. Returns a list of edges (u, v, weight).
    """
    if not graph:
        return []

    start = next(iter(graph))
    visited = {start}
    edges = []

    # Candidate edges as (weight, u, v)
    import heapq
    candidate: List[Tuple[float, int, int]] = []
    for v, w in graph[start]:
        heapq.heappush(candidate, (w, start, v))

    while candidate and len(visited) < len(graph):
        w, u, v = heapq.heappop(candidate)
        if v in visited:
            continue
        visited.add(v)
        edges.append((u, v, w))
        for nxt, wgt in graph[v]:
            if nxt not in visited:
                heapq.heappush(candidate, (wgt, v, nxt))

    return edges

def hybrid_adapt(model: RBFNLMS,
                 x: np.ndarray,
                 target: float,
                 rlct: float,
                 error_scale: float = 1.0) -> Tuple[float, str]:
    """
    Perform a single hybrid adaptation step:
        1. Predict and adapt the RBFNLMS model using RLCT‑scaled learning rate.
        2. Compute an epistemic flag from the resulting error.
    Returns (error, flag).
    """
    error = model.adapt(x, target, rlct=rlct)
    flag = compute_epistemic_flag(error, error_scale=error_scale)
    return error, flag

def build_scaled_similarity_graph(model: RBFNLMS,
                                  rlct: float) -> Tuple[Dict[int, List[Tuple[int, float]]], List[Tuple[int, int, float]]]:
    """
    Convenience wrapper that builds the RLCT‑scaled similarity graph from the model's
    current output weights and then extracts its MST.
    Returns (graph, mst_edges).
    """
    graph = construct_similarity_graph(model.weights, rlct=rlct)
    mst = prim_mst(graph)
    return graph, mst

# ----------------------------------------------------------------------
# Simple smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Seed for reproducibility
    np.random.seed(42)
    random.seed(42)

    # Synthetic RBF centres (5 centres, 2‑D input)
    centres = np.random.randn(5, 2)

    # Instantiate the hybrid model
    model = RBFNLMS(centers=centres, sigma=1.0, mu=0.4)

    # Generate a toy dataset
    X = np.random.randn(200, 2)
    true_weights = np.random.randn(5)
    y_true = X @ np.random.randn(2) * 0.5 + np.random.randn(200) * 0.1  # noisy linear target

    # Simulate training to collect loss history
    losses = []
    ns = []
    for i, (x, y) in enumerate(zip(X, y_true), start=1):
        pred = model.predict(x)
        err = y - pred
        losses.append(err ** 2)
        ns.append(i)
        # Adapt using a placeholder RLCT (0) during this preliminary phase
        model.adapt(x, y, rlct=0.0)

    # Estimate RLCT from the collected loss curve
    rlct_est = estimate_rlct_from_losses(losses, ns)
    print(f"Estimated RLCT: {rlct_est:.4f}")

    # Perform a few hybrid adaptation steps with RLCT‑aware learning rate
    for step in range(5):
        idx = random.randint(0, len(X) - 1)
        x, y = X[idx], y_true[idx]
        error, flag = hybrid_adapt(model, x, y, rlct=rlct_est, error_scale=np.mean(losses) ** 0.5)
        print(f"Step {step+1}: error={error:.4f}, flag={flag}")

    # Build graph and MST using the final weights and RLCT scaling
    graph, mst = build_scaled_similarity_graph(model, rlct=rlct_est)
    total_mst_weight = sum(w for _, _, w in mst)
    print(f"MST contains {len(mst)} edges, total weight = {total_mst_weight:.4f}")

    # Verify that MST spans all nodes
    assert len(mst) == len(model.weights) - 1, "MST does not span all vertices"
    print("Smoke test completed successfully.")