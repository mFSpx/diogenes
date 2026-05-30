# DARWIN HAMMER — match 293, survivor 4
# gen: 3
# parent_a: hybrid_nlms_omni_chaotic_sprint_m59_s2.py (gen1)
# parent_b: hybrid_hybrid_krampus_brain_hybrid_krampus_brain_m74_s1.py (gen2)
# born: 2026-05-29T23:28:15Z

"""Hybrid NLMS‑Ollivier‑Ricci Graph Engine
========================================
This module fuses the two parent algorithms:

* **Parent A – nlms.py** – a Normalized Least‑Mean‑Squares adaptive filter that
  updates a global weight vector **w** with the rule  

  ``w ← w + μ·e·x / (‖x‖²+ε)``  

  where ``e = target – w·x`` and ``x`` is an input feature vector.

* **Parent B – krampus_brainmap…py** – a feature‑extraction pipeline that builds a
  high‑dimensional *master vector* from textual data and, in the original
  research, applies Ollivier‑Ricci curvature to graph connections.

**Mathematical Bridge**

For a graph ``G = (V,E)`` let

* ``imp_ij`` be the impedance weight of edge ``(i,j)`` (parent A).
* ``κ_ij`` be the Ollivier‑Ricci curvature of edge ``(i,j)`` (parent B).
* ``φ_j`` be the master‑vector of node ``j`` obtained from the text‑feature
  extractor of parent B.

We construct a *curvature‑weighted neighbourhood vector* for every node ``i``


x_i = Σ_{j∈N(i)} (imp_ij · κ_ij) · φ_j


The NLMS predictor ``ŷ_i = w·x_i`` attempts to model the *wavefront velocity*
``v_i = 1 / max(stress_i, 1)`` produced by the seismic‑propagation engine of
parent A.  The error ``e_i = v_i – ŷ_i`` drives a global NLMS update, thereby
learning a mapping from curvature‑modulated impedance‑aggregated neighbour
features to observed propagation speeds.  This single set of equations unifies
both topologies into one adaptive system.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent B – feature extraction (master vector)
# ----------------------------------------------------------------------
def extract_full_features(text: str) -> dict[str, float]:
    """Deterministic pseudo‑random feature dictionary from a string."""
    features: dict[str, float] = {}
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
        "rainmaker_pitch_formatting_ratio", "telemetry_agent_symmetry_ratio",
        "telemetry_protocol_discipline", "telemetry_manic_velocity",
    ]
    for k in keys:
        features[k] = rnd.random() * 10.0
    return features


def extract_master_vector(text: str) -> dict[str, float]:
    """Select a subset of the full feature set to act as the master vector."""
    f = extract_full_features(text)
    master_vector = {
        "visceral_ratio": f.get("operator_visceral_ratio", 0.0),
        "tech_ratio": f.get("operator_tech_ratio", 0.0),
        "legal_osint_ratio": f.get("operator_legal_osint_ratio", 0.0),
        "ledger_density": f.get("operator_ledger_density", 0.0),
        "recursion_score": f.get("operator_recursion_score", 0.0),
        "directive_ratio": f.get("operator_directive_ratio", 0.0),
        "target_density": f.get("operator_target_density", 0.0),
        "wrath_velocity": f.get("psyche_wrath_velocity", 0.0),
        "resource_exhaustion": f.get("resilience_resource_exhaustion_metric", 0.0),
        "swarm_density": f.get("resilience_swarm_orchestration_density", 0.0),
    }
    return master_vector


def master_vector_to_numpy(vec: dict[str, float]) -> np.ndarray:
    """Convert the master‑vector dict to a deterministic 1‑D NumPy array."""
    # Sorting keys guarantees reproducible ordering.
    keys = sorted(vec.keys())
    return np.array([vec[k] for k in keys], dtype=float)


# ----------------------------------------------------------------------
# Graph utilities – synthetic generation, impedance, curvature
# ----------------------------------------------------------------------
def generate_synthetic_graph(num_nodes: int, seed: int = 42) -> Tuple[Dict[int, Dict[int, float]], Dict[int, str]]:
    """
    Create an undirected graph with random impedances and a dummy text per node.
    Returns (adjacency, node_texts).
    adjacency[i][j] = impedance_ij  (float > 0)
    """
    rnd = random.Random(seed)
    adjacency: Dict[int, Dict[int, float]] = {i: {} for i in range(num_nodes)}
    node_texts: Dict[int, str] = {}

    # Assign a random short text to each node (used for feature extraction)
    for i in range(num_nodes):
        node_texts[i] = f"node_{i}_seed_{seed}"

    # Randomly connect nodes (Erdős‑Rényi style) with impedance in (0.1, 5.0)
    p_edge = 0.3
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if rnd.random() < p_edge:
                imp = rnd.uniform(0.1, 5.0)
                adjacency[i][j] = imp
                adjacency[j][i] = imp
    return adjacency, node_texts


def compute_ollivier_ricci_curvature(adjacency: Dict[int, Dict[int, float]]) -> Dict[Tuple[int, int], float]:
    """
    Approximate Ollivier‑Ricci curvature for each edge using a simple degree‑based formula:
        κ_ij = 2 / (1 + deg(i) + deg(j))
    The result is a dictionary keyed by (i, j) with i < j.
    """
    curvature: Dict[Tuple[int, int], float] = {}
    degrees = {i: len(neigh) for i, neigh in adjacency.items()}
    for i, neigh in adjacency.items():
        for j in neigh:
            if i < j:
                κ = 2.0 / (1.0 + degrees[i] + degrees[j])
                curvature[(i, j)] = κ
                curvature[(j, i)] = κ  # store both directions for convenience
    return curvature


def simulate_wavefront_velocities(adjacency: Dict[int, Dict[int, float]]) -> Dict[int, float]:
    """
    Produce a synthetic wavefront velocity `v_i` for each node.
    Stress is modelled as the sum of impedances of incident edges; then
        v_i = 1 / max(stress_i, 1)
    """
    velocities: Dict[int, float] = {}
    for i, neigh in adjacency.items():
        stress = sum(neigh.values())
        velocities[i] = 1.0 / max(stress, 1.0)
    return velocities


# ----------------------------------------------------------------------
# NLMS core (Parent A) – unchanged except for type hints
# ----------------------------------------------------------------------
def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Dot‑product prediction w·x."""
    return float(weights @ x)


def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, float]:
    """
    Perform a single Normalized LMS weight update.
    Returns (new_weights, error).
    """
    y = nlms_predict(weights, x)
    e = target - y
    norm_sq = float(x @ x) + eps
    delta = (mu * e / norm_sq) * x
    new_weights = weights + delta
    return new_weights, e


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def build_node_input_vector(
    node: int,
    adjacency: Dict[int, Dict[int, float]],
    curvature: Dict[Tuple[int, int], float],
    node_feature_matrix: Dict[int, np.ndarray],
) -> np.ndarray:
    """
    Compute the curvature‑weighted neighbourhood vector x_i for a given node.
    x_i = Σ_j (imp_ij * κ_ij) * φ_j
    """
    dim = next(iter(node_feature_matrix.values())).shape[0]
    x = np.zeros(dim, dtype=float)
    for neighbor, imp in adjacency[node].items():
        κ = curvature.get((node, neighbor), 0.0)
        weight = imp * κ
        x += weight * node_feature_matrix[neighbor]
    return x


def hybrid_nlms_iteration(
    adjacency: Dict[int, Dict[int, float]],
    node_texts: Dict[int, str],
    weights: np.ndarray,
    mu: float = 0.5,
) -> Tuple[np.ndarray, Dict[int, float]]:
    """
    One full hybrid iteration:
      1. Extract master vectors for all nodes (B).
      2. Compute curvature for all edges (B).
      3. Simulate wavefront velocities (A).
      4. For each node, build x_i, predict ŷ_i, and update the *global* weight vector.
    Returns the updated weight vector and a dictionary of per‑node prediction errors.
    """
    # 1. Master vectors → matrix
    node_feature_matrix: Dict[int, np.ndarray] = {
        i: master_vector_to_numpy(extract_master_vector(txt))
        for i, txt in node_texts.items()
    }

    # 2. Curvature
    curvature = compute_ollivier_ricci_curvature(adjacency)

    # 3. Target velocities
    targets = simulate_wavefront_velocities(adjacency)

    # 4. NLMS updates (global weight vector)
    errors: Dict[int, float] = {}
    w = weights.copy()
    for i in adjacency.keys():
        x_i = build_node_input_vector(i, adjacency, curvature, node_feature_matrix)
        if np.allclose(x_i, 0.0):
            # Isolated node – skip update
            continue
        w, e = nlms_update(w, x_i, targets[i], mu=mu)
        errors[i] = e
    return w, errors


def run_hybrid_demo(num_nodes: int = 12, epochs: int = 5, mu: float = 0.4) -> None:
    """
    Demonstration driver:
      * builds a synthetic graph,
      * initializes a zero weight vector,
      * runs several hybrid NLMS epochs,
      * prints the final weight vector and average error.
    """
    adjacency, node_texts = generate_synthetic_graph(num_nodes)
    # Determine feature dimensionality from a single master vector
    sample_vec = master_vector_to_numpy(extract_master_vector(next(iter(node_texts.values()))))
    dim = sample_vec.shape[0]
    weights = np.zeros(dim, dtype=float)

    for epoch in range(1, epochs + 1):
        weights, errors = hybrid_nlms_iteration(adjacency, node_texts, weights, mu=mu)
        avg_err = float(np.mean(list(errors.values()))) if errors else 0.0
        print(f"Epoch {epoch:02d} – avg error: {avg_err:.5f}")

    print("\nFinal weight vector:")
    for idx, val in enumerate(weights):
        print(f"  w[{idx}] = {val:.6f}")


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple smoke test that runs without external data.
    run_hybrid_demo(num_nodes=15, epochs=3, mu=0.3)