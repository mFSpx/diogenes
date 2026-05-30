# DARWIN HAMMER — match 2710, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_ternary_route_hybrid_hybrid_model__m13_s0.py (gen4)
# parent_b: hybrid_hybrid_krampus_brain_ttt_linear_m4_s1.py (gen2)
# born: 2026-05-29T23:43:53Z

"""Hybrid Algorithm combining:
- Parent A: TTT‑Linear update rule with split‑gain modulation and SSIM evaluation.
- Parent B: Feature extraction & graph‑theoretic Ollivier‑Ricci curvature framework.

Mathematical Bridge:
The split‑gain computed from the logistic‑loss gradient/Hessian (Parent A) is used as a
probabilistic pruning signal for edges of the feature‑graph constructed in Parent B.
After the adjacency matrix is updated, the Structural Similarity Index (SSIM) is
applied to the node‑feature matrix before and after the update, providing a unified
metric that links the graph‑theoretic structure to the TTT‑Linear performance measure.
"""

import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent B – Feature extraction utilities
# ----------------------------------------------------------------------
def extract_full_features(text: str) -> Dict[str, float]:
    """Deterministic pseudo‑random feature vector based on the input text."""
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
    return {k: rnd.random() * 10.0 for k in keys}


def extract_master_vector(text: str) -> Dict[str, float]:
    """Select a subset of features that will serve as the master representation."""
    if not text.strip():
        return {}
    f = extract_full_features(text)
    return {
        "visceral_ratio": f.get("operator_visceral_ratio", 0.0),
        "tech_ratio": f.get("operator_tech_ratio", 0.0),
        "legal_osint_ratio": f.get("operator_legal_osint_ratio", 0.0),
        "ledger_density": f.get("operator_ledger_density", 0.0),
        "recursion_score": f.get("operator_recursion_score", 0.0),
        "directive_ratio": f.get("operator_directive_ratio", 0.0),
        "target_density": f.get("operator_target_density", 0.0),
        "forensic_shield_ratio": f.get("psyche_forensic_shield_ratio", 0.0),
        "poetic_entropy": f.get("psyche_poetic_entropy", 0.0),
        "dissociative_index": f.get("psyche_dissociative_index", 0.0),
    }

# ----------------------------------------------------------------------
# Parent A – TTT‑Linear logistic utilities and SSIM
# ----------------------------------------------------------------------
def sigmoid(x: np.ndarray) -> np.ndarray:
    """Element‑wise sigmoid."""
    return 1.0 / (1.0 + np.exp(-x))


def logistic_grad_hess(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gradient and Hessian of binary logistic loss.
    y_true ∈ {0,1}, y_pred = sigmoid(logits)
    """
    grad = y_pred - y_true
    hess = y_pred * (1.0 - y_pred)
    return grad, hess


def split_gain(grad_sum: float, hess_sum: float, lam: float = 1.0, gamma: float = 0.0) -> float:
    """
    XGBoost‑style split gain:
        G^2 / (H + λ) - γ
    where G = Σgrad, H = Σhess.
    """
    if hess_sum + lam == 0.0:
        return 0.0
    return (grad_sum ** 2) / (hess_sum + lam) - gamma


def ssim_matrix(a: np.ndarray, b: np.ndarray, C1: float = 0.01 ** 2, C2: float = 0.03 ** 2) -> float:
    """
    Simplified SSIM over two matrices (flattened).
    Returns a scalar similarity in [0,1].
    """
    mu_a = a.mean()
    mu_b = b.mean()
    sigma_a = a.var()
    sigma_b = b.var()
    sigma_ab = ((a - mu_a) * (b - mu_b)).mean()
    numerator = (2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2)
    denominator = (mu_a ** 2 + mu_b ** 2 + C1) * (sigma_a + sigma_b + C2)
    return float(numerator / denominator) if denominator != 0 else 0.0

# ----------------------------------------------------------------------
# Hybrid core – graph update guided by split‑gain and SSIM evaluation
# ----------------------------------------------------------------------
def update_adjacency_with_gain(
    adj: np.ndarray,
    node_features: np.ndarray,
    y_true: np.ndarray,
    weight_vec: np.ndarray,
    lam: float = 1.0,
    gamma: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform one hybrid update:
    1. Compute linear logits = node_features @ weight_vec, then sigmoid predictions.
    2. Obtain per‑node gradient/hessian.
    3. For each edge (i,j) compute edge‑wise gain = split_gain(gi+gj, hi+hj).
    4. Prune edges whose gain falls below the median gain (probabilistic pruning).
    5. Return the new adjacency matrix and the updated node feature matrix
       (here we simply apply a linear residual update using the gain as a scaling factor).
    """
    # 1. Predictions
    logits = node_features @ weight_vec
    y_pred = sigmoid(logits)

    # 2. Gradient / Hessian per node
    grad, hess = logistic_grad_hess(y_true, y_pred)

    # 3. Edge‑wise gain matrix
    n = adj.shape[0]
    gain_matrix = np.zeros_like(adj, dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] != 0:
                g_sum = grad[i] + grad[j]
                h_sum = hess[i] + hess[j]
                gain = split_gain(g_sum, h_sum, lam, gamma)
                gain_matrix[i, j] = gain
                gain_matrix[j, i] = gain

    # 4. Prune edges below median gain (retain edges with higher informational value)
    median_gain = np.median(gain_matrix[gain_matrix > 0])
    pruned_adj = adj.copy()
    pruned_adj[gain_matrix < median_gain] = 0.0

    # 5. Feature update: a simple residual that pushes nodes toward the master vector direction
    #    using the edge gains as a diffusion coefficient.
    diffusion = np.zeros_like(node_features)
    for i in range(n):
        for j in range(n):
            if pruned_adj[i, j] > 0:
                diffusion[i] += gain_matrix[i, j] * (node_features[j] - node_features[i])
    updated_features = node_features + 0.01 * diffusion  # small step size

    return pruned_adj, updated_features


def hybrid_process(
    text: str,
    adj: np.ndarray,
    node_features: np.ndarray,
    weight_vec: np.ndarray,
    lam: float = 1.0,
    gamma: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    End‑to‑end hybrid operation:
    * Extract a master feature vector from `text`.
    * Append it to the node feature matrix (as an additional node) to allow the
      graph to incorporate the new semantic signal.
    * Run the adjacency update guided by split‑gain.
    * Compute SSIM between the feature matrix before and after the update.
    Returns the new adjacency, the updated feature matrix, and the SSIM score.
    """
    # Master vector extraction (Parent B)
    master_dict = extract_master_vector(text)
    master_vec = np.array([master_dict[k] for k in sorted(master_dict.keys())], dtype=float)

    # Align dimensions: if master vector length differs, pad/truncate to match node_features columns
    dim = node_features.shape[1]
    if master_vec.shape[0] < dim:
        master_vec = np.pad(master_vec, (0, dim - master_vec.shape[0]), constant_values=0.0)
    elif master_vec.shape[0] > dim:
        master_vec = master_vec[:dim]

    # Append as a new node
    extended_features = np.vstack([node_features, master_vec[None, :]])
    n_nodes = extended_features.shape[0]

    # Extend adjacency matrix with zero connections for the new node
    extended_adj = np.zeros((n_nodes, n_nodes), dtype=float)
    extended_adj[: adj.shape[0], : adj.shape[1]] = adj

    # Dummy binary targets for all nodes (for illustration we use random 0/1)
    y_true = np.random.randint(0, 2, size=n_nodes).astype(float)

    # Hybrid adjacency & feature update (core fusion)
    new_adj, new_features = update_adjacency_with_gain(
        extended_adj, extended_features, y_true, weight_vec, lam, gamma
    )

    # SSIM evaluation between pre‑ and post‑update feature matrices (Parent A)
    ssim_score = ssim_matrix(extended_features, new_features)

    return new_adj, new_features, ssim_score


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    # Create a small random graph (5 nodes) and random node features (5×8)
    num_nodes = 5
    feature_dim = 8
    adj_matrix = (np.random.rand(num_nodes, num_nodes) > 0.7).astype(float)
    np.fill_diagonal(adj_matrix, 0.0)  # no self‑loops
    adj_matrix = np.triu(adj_matrix) + np.triu(adj_matrix, 1).T  # symmetrize

    node_feats = np.random.rand(num_nodes, feature_dim)

    # Linear weight vector for TTT‑Linear component
    weight_vector = np.random.randn(feature_dim)

    sample_text = "The quick brown fox jumps over the lazy dog."

    new_adj, new_feats, similarity = hybrid_process(
        sample_text, adj_matrix, node_feats, weight_vector, lam=1.0, gamma=0.0
    )

    print("Original adjacency:\n", adj_matrix)
    print("Updated adjacency:\n", new_adj)
    print("SSIM between feature states:", similarity)
    # Verify shapes
    assert new_adj.shape[0] == new_feats.shape[0], "Node count mismatch after hybrid step"
    print("Hybrid step completed successfully.")