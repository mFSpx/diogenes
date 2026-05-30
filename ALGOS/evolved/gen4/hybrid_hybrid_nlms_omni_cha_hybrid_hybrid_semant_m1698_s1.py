# DARWIN HAMMER — match 1698, survivor 1
# gen: 4
# parent_a: hybrid_nlms_omni_chaotic_sprint_m59_s5.py (gen1)
# parent_b: hybrid_hybrid_semantic_neig_hybrid_hybrid_krampu_m540_s0.py (gen3)
# born: 2026-05-29T23:38:25Z

"""Hybrid NLMS‑Graph‑Semantic Algorithm
====================================

This module fuses the two parent algorithms:

* **Parent A** – Normalised Least‑Mean‑Squares (NLMS) batch update together with a
  synthetic graph generator.
* **Parent B** – Semantic feature extraction, pheromone probability handling and
  entropy utilities.

The mathematical bridge is the *per‑sample learning‑rate modulation*:

* Each graph node supplies an input vector **xᵢ** (its semantic feature vector).
* For every node we compute a curvature‑derived factor **cᵢ** from the
  Ollivier‑Ricci‑like curvature of its incident edges (impedance‑based).
* A pheromone vector **p** assigns a probability **πᵢ** to each node.
* The effective NLMS step size for sample *i* becomes  

  ``μᵢ = μ₀ · cᵢ · πᵢ``  

  where ``μ₀`` is the global base learning rate.

Thus the classic NLMS update  

``w ← w + Σ ( μ·eᵢ / (‖xᵢ‖²+ε) )·xᵢ``  

is transformed into a *curvature‑ and pheromone‑aware* update  

``w ← w + Σ ( μᵢ·eᵢ / (‖xᵢ‖²+ε) )·xᵢ``  

which simultaneously exploits graph geometry, semantic content and
pheromone‑driven exploration.

The module provides three high‑level functions that illustrate this hybrid
behaviour:

1. ``compute_ollivier_ricci_curvature`` – approximates curvature from edge
   impedances.
2. ``semantic_feature_matrix`` – builds a feature matrix Φ from raw text
   using the parent‑B feature extractor.
3. ``hybrid_nlms_update`` – performs the NLMS batch update with per‑sample
   learning rates derived from curvature and pheromones.

A minimal smoke test is executed when the file is run as a script."""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – NLMS core utilities
# ----------------------------------------------------------------------
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
    Classic NLMS batch update (single global step size).
    """
    if not (0.0 < mu < 2.0):
        raise ValueError("mu must be in the interval (0, 2)")

    preds = X @ weights
    errors = targets - preds
    powers = np.sum(X * X, axis=1) + eps
    steps = (mu * errors / powers)[:, None] * X
    new_weights = weights + steps.sum(axis=0)
    return new_weights, errors


# ----------------------------------------------------------------------
# Parent B – Semantic feature extraction & pheromone utilities
# ----------------------------------------------------------------------
def _cos(a: List[float], b: List[float]) -> float:
    den = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return 0.0 if den == 0 else sum(x * y for x, y in zip(a, b)) / den


def pheromone_probabilities(pheromones: List[float]) -> List[float]:
    total = sum(pheromones)
    if total == 0:
        raise ValueError("sum of pheromones must be > 0")
    return [p / total for p in pheromones]


def entropy(probabilities: List[float], eps: float = 1e-12) -> float:
    total = sum(probabilities)
    if total <= 0:
        raise ValueError("positive probability mass required")
    return -sum(
        (p / total) * math.log(max(p / total, eps))
        for p in probabilities
        if p > 0
    )


def expected_entropy(p_hit: float, hit_state: List[float], miss_state: List[float]) -> float:
    if not 0 <= p_hit <= 1:
        raise ValueError("p_hit must be in [0,1]")
    return p_hit * entropy(hit_state) + (1.0 - p_hit) * entropy(miss_state)


def extract_full_features(text: str) -> Dict[str, float]:
    """
    Deterministic pseudo‑semantic feature extractor.
    The same text always yields the same vector because the random generator
    is seeded with ``hash(text)``.
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
        "rainmaker_countdown_friction", "rainmaker_malware_apt_score",
    ]
    return {k: rnd.random() for k in keys}


# ----------------------------------------------------------------------
# Graph utilities (partial parent A)
# ----------------------------------------------------------------------
NodeId = str
Edge = Tuple[NodeId, NodeId, int]  # (src, dst, impedance)


def generate_synthetic_graph(num_nodes: int, avg_degree: int = 3) -> Tuple[Dict[NodeId, List[Tuple[NodeId, int]]], np.ndarray]:
    """
    Create a random undirected graph with integer impedances and a random
    feature matrix Φ (shape: num_nodes × feature_dim).
    """
    adjacency: Dict[NodeId, List[Tuple[NodeId, int]]] = {}
    nodes = [f"N{i}" for i in range(num_nodes)]
    for n in nodes:
        adjacency[n] = []

    # Random edges respecting avg_degree
    possible = [(i, j) for i in range(num_nodes) for j in range(i + 1, num_nodes)]
    random.shuffle(possible)
    target_edges = max(num_nodes * avg_degree // 2, 1)
    for i, j in possible[:target_edges]:
        impedance = random.randint(1, 10)
        src, dst = nodes[i], nodes[j]
        adjacency[src].append((dst, impedance))
        adjacency[dst].append((src, impedance))

    # Feature matrix: each node gets a random 10‑dim vector
    feature_dim = 10
    Phi = np.random.randn(num_nodes, feature_dim).astype(np.float32)

    return adjacency, Phi


# ----------------------------------------------------------------------
# Fusion – curvature, semantic matrix, hybrid NLMS
# ----------------------------------------------------------------------
def compute_ollivier_ricci_curvature(adjacency: Dict[NodeId, List[Tuple[NodeId, int]]]) -> Dict[NodeId, float]:
    """
    Approximate Ollivier‑Ricci curvature for each node using edge impedances.
    For an undirected edge (u,v) with impedance w, define edge curvature
    ``k_uv = 1 / (1 + w)``.  Node curvature is the average of incident edge curvatures.
    """
    curvature: Dict[NodeId, float] = {}
    for node, neigh in adjacency.items():
        if not neigh:
            curvature[node] = 0.0
            continue
        curv_sum = 0.0
        for nbr, imp in neigh:
            curv_sum += 1.0 / (1.0 + imp)
        curvature[node] = curv_sum / len(neigh)
    return curvature


def semantic_feature_matrix(texts: List[str]) -> np.ndarray:
    """
    Convert a list of raw texts into a feature matrix Φ (N × D) where each row
    corresponds to the vector of extracted semantic features.
    Missing keys are filled with zeros to keep a consistent dimensionality.
    """
    feature_dicts = [extract_full_features(t) for t in texts]
    # Determine a stable ordering of keys across all samples
    all_keys = sorted({k for d in feature_dicts for k in d})
    D = len(all_keys)
    N = len(texts)
    Phi = np.zeros((N, D), dtype=np.float32)
    for i, d in enumerate(feature_dicts):
        for j, k in enumerate(all_keys):
            Phi[i, j] = d.get(k, 0.0)
    return Phi


def hybrid_nlms_update(
    weights: np.ndarray,
    adjacency: Dict[NodeId, List[Tuple[NodeId, int]]],
    X: np.ndarray,
    targets: np.ndarray,
    base_mu: float = 0.5,
    pheromones: List[float] | None = None,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform a NLMS batch update where the per‑sample step size μᵢ is modulated
    by graph curvature and pheromone probabilities.

    Parameters
    ----------
    weights : np.ndarray (d,)
        Current weight vector.
    adjacency : dict
        Graph adjacency with impedance information.
    X : np.ndarray (N, d)
        Input feature matrix (semantic vectors for each node).
    targets : np.ndarray (N,)
        Desired scalar outputs.
    base_mu : float
        Global base learning rate (0 < base_mu < 2).
    pheromones : list of float, optional
        Pheromone strength for each node; if omitted, uniform probabilities are used.
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    new_weights : np.ndarray
        Updated weight vector.
    errors : np.ndarray
        Prediction errors for each sample.
    """
    if not (0.0 < base_mu < 2.0):
        raise ValueError("base_mu must be in the interval (0, 2)")

    N = X.shape[0]
    if pheromones is None:
        pheromones = [1.0] * N
    if len(pheromones) != N:
        raise ValueError("pheromones length must match number of samples")

    # 1️⃣ Curvature per node (average of incident edge curvatures)
    node_list = list(adjacency.keys())
    if len(node_list) != N:
        raise ValueError("Number of graph nodes must equal number of samples")
    curvature_map = compute_ollivier_ricci_curvature(adjacency)
    curvature_vec = np.array([curvature_map[n] for n in node_list], dtype=np.float32)

    # 2️⃣ Pheromone probabilities
    pi = np.array(pheromone_probabilities(pheromones), dtype=np.float32)

    # 3️⃣ Effective per‑sample learning rates
    mu_i = base_mu * curvature_vec * pi  # shape (N,)

    # NLMS core calculations using the individual μᵢ
    preds = X @ weights
    errors = targets - preds
    powers = np.sum(X * X, axis=1) + eps  # (N,)

    # Broadcast μᵢ * errors / powers to (N, d)
    coeff = (mu_i * errors) / powers
    steps = coeff[:, None] * X
    new_weights = weights + steps.sum(axis=0)

    return new_weights, errors


# ----------------------------------------------------------------------
# Demonstration / smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 1️⃣ Generate a tiny synthetic graph
    adj, _ = generate_synthetic_graph(num_nodes=5, avg_degree=2)

    # 2️⃣ Create dummy texts for each node
    texts = [
        "alpha beta gamma",
        "delta epsilon zeta",
        "eta theta iota",
        "kappa lambda mu",
        "nu xi omicron",
    ]

    # 3️⃣ Build semantic feature matrix Φ
    X = semantic_feature_matrix(texts)          # shape (5, D)

    # 4️⃣ Random target vector
    rng = np.random.default_rng(42)
    targets = rng.normal(size=5).astype(np.float32)

    # 5️⃣ Initialise weights (matching feature dimension)
    d = X.shape[1]
    w = rng.normal(size=d).astype(np.float32)

    # 6️⃣ Random pheromone strengths
    pheromones = [rng.random() for _ in range(5)]

    # 7️⃣ Perform hybrid NLMS update
    new_w, errs = hybrid_nlms_update(
        weights=w,
        adjacency=adj,
        X=X,
        targets=targets,
        base_mu=0.7,
        pheromones=pheromones,
    )

    # Simple sanity prints
    print("Old weights (first 5):", w[:5])
    print("New weights (first 5):", new_w[:5])
    print("Errors:", errs)
    print("Update successful.")