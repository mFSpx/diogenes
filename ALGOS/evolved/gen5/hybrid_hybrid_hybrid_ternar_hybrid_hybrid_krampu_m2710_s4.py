# DARWIN HAMMER — match 2710, survivor 4
# gen: 5
# parent_a: hybrid_hybrid_ternary_route_hybrid_hybrid_model__m13_s0.py (gen4)
# parent_b: hybrid_hybrid_krampus_brain_ttt_linear_m4_s1.py (gen2)
# born: 2026-05-29T23:43:53Z

import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

def extract_full_features(text: str) -> Dict[str, float]:
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

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def logistic_grad_hess(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    grad = y_pred - y_true
    hess = y_pred * (1.0 - y_pred)
    return grad, hess

def split_gain(grad_sum: float, hess_sum: float, lam: float = 1.0, gamma: float = 0.0) -> float:
    if hess_sum + lam == 0.0:
        return 0.0
    return (grad_sum ** 2) / (hess_sum + lam) - gamma

def ssim_matrix(a: np.ndarray, b: np.ndarray, C1: float = 0.01 ** 2, C2: float = 0.03 ** 2) -> float:
    mu_a = a.mean()
    mu_b = b.mean()
    sigma_a = a.var()
    sigma_b = b.var()
    sigma_ab = ((a - mu_a) * (b - mu_b)).mean()
    numerator = (2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2)
    denominator = (mu_a ** 2 + mu_b ** 2 + C1) * (sigma_a + sigma_b + C2)
    return float(numerator / denominator) if denominator != 0 else 0.0

def update_adjacency_with_gain(
    adj: np.ndarray,
    node_features: np.ndarray,
    y_true: np.ndarray,
    weight_vec: np.ndarray,
    lam: float = 1.0,
    gamma: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    logits = node_features @ weight_vec
    y_pred = sigmoid(logits)

    grad, hess = logistic_grad_hess(y_true, y_pred)

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

    median_gain = np.median(gain_matrix[gain_matrix > 0])
    pruned_adj = adj.copy()
    pruned_adj[gain_matrix < median_gain] = 0.0

    diffusion = np.zeros_like(node_features)
    for i in range(n):
        for j in range(n):
            if pruned_adj[i, j] > 0:
                diffusion[i] += gain_matrix[i, j] * (node_features[j] - node_features[i])
    updated_features = node_features + 0.01 * diffusion

    return pruned_adj, updated_features

def hybrid_process(
    text: str,
    adj: np.ndarray,
    node_features: np.ndarray,
    weight_vec: np.ndarray,
    lam: float = 1.0,
    gamma: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    master_dict = extract_master_vector(text)
    master_vec = np.array([master_dict[k] for k in sorted(master_dict.keys())], dtype=float)

    dim = node_features.shape[1]
    if master_vec.shape[0] < dim:
        master_vec = np.pad(master_vec, (0, dim - master_vec.shape[0]), constant_values=0.0)
    elif master_vec.shape[0] > dim:
        master_vec = master_vec[:dim]

    extended_features = np.vstack([node_features, master_vec[None, :]])
    n_nodes = extended_features.shape[0]

    pruned_adj, updated_features = update_adjacency_with_gain(
        np.pad(adj, ((0, 1), (0, 1))), 
        extended_features, 
        np.pad(y_true, (0, 1)), 
        weight_vec, 
        lam, 
        gamma
    )
    pruned_adj = pruned_adj[:-1, :-1]
    updated_features = updated_features[:-1]

    ssim_score = ssim_matrix(node_features, updated_features)

    return pruned_adj, updated_features, ssim_score