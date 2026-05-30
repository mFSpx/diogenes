# DARWIN HAMMER — match 3501, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_distri_m1510_s2.py (gen6)
# parent_b: hybrid_hybrid_krampus_brain_regret_engine_m384_s1.py (gen2)
# born: 2026-05-29T23:50:32Z

import math
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import re

# ----------------------------------------------------------------------
# Linguistic feature extraction (pronoun frequencies)
# ----------------------------------------------------------------------
FUNCTION_CATS: Dict[str, set[str]] = {
    "pronoun": {
        "i", "me", "my", "mine", "myself", "you", "your", "yours", "yourself",
        "he", "him", "his", "himself", "she", "her", "hers", "herself",
        "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
        "what", "which", "who", "whom", "this", "that", "these", "those",
        "am", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "shall", "should",
        "will", "would", "may", "might", "must", "can", "could", "ought",
        "i'm", "you're", "he's", "she's", "it's", "we're", "they're",
        "i've", "you've", "we've", "they've", "i'd", "you'd", "he'd",
        "she'd", "we'd", "they'd", "i'll", "you'll", "he'll", "she'll",
        "we'll", "they'll"
    }
}


def stylometry_features(text: str) -> Dict[str, int]:
    """Count pronoun occurrences in the supplied text."""
    words = re.findall(r"\b\w+'\w+|\b\w+\b", text.lower())
    counts = Counter(words)
    return {
        cat: sum(counts.get(tok, 0) for tok in tokens)
        for cat, tokens in FUNCTION_CATS.items()
    }


# ----------------------------------------------------------------------
# Graph construction from feature vectors
# ----------------------------------------------------------------------
def _vector_from_features(feat: Dict[str, int]) -> np.ndarray:
    """Convert a feature dict to a dense numpy vector (ordered by key)."""
    return np.array([feat[k] for k in sorted(feat.keys())], dtype=float)


def build_weighted_graph(features_list: List[Dict[str, int]]) -> Tuple[np.ndarray, List[float]]:
    """
    Build a symmetric similarity graph.
    Edge weight = cosine similarity between feature vectors.
    Returns adjacency matrix and a flat list of the non‑zero weights.
    """
    n = len(features_list)
    if n == 0:
        raise ValueError("features_list must contain at least one element")
    vectors = np.stack([_vector_from_features(f) for f in features_list])
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Avoid division by zero
    norms[norms == 0] = 1.0
    normalized = vectors / norms
    similarity = normalized @ normalized.T  # cosine similarity matrix
    np.fill_diagonal(similarity, 0.0)       # no self‑loops
    # Ensure non‑negative weights
    similarity = np.clip(similarity, 0.0, None)
    # Collect upper‑triangular non‑zero weights for later use
    weights = similarity[np.triu_indices(n, k=1)].tolist()
    return similarity, weights


# ----------------------------------------------------------------------
# Ollivier‑Ricci curvature approximation
# ----------------------------------------------------------------------
def _neighbor_distribution(adj: np.ndarray, i: int) -> np.ndarray:
    """Return the probability distribution over neighbors of node i."""
    row = adj[i]
    total = row.sum()
    if total == 0:
        # Isolated node – treat as Dirac at itself
        dist = np.zeros_like(row)
        dist[i] = 1.0
        return dist
    return row / total


def compute_ollivier_ricci_curvature(adj: np.ndarray) -> np.ndarray:
    """
    Approximate Ollivier‑Ricci curvature using the 1‑Wasserstein distance
    between neighbor distributions.
    curvature_{ij} = 1 - W1(μ_i, μ_j) / d(i, j)
    where d(i, j) = 1 / (adj[i, j] + ε) to keep distances finite.
    """
    n = adj.shape[0]
    curvature = np.zeros((n, n))
    epsilon = 1e-12
    for i in range(n):
        mu_i = _neighbor_distribution(adj, i)
        for j in range(i + 1, n):
            mu_j = _neighbor_distribution(adj, j)
            # 1‑Wasserstein distance on a complete graph reduces to L1 distance / 2
            w1 = 0.5 * np.abs(mu_i - mu_j).sum()
            # Effective graph distance (larger similarity → smaller distance)
            d_ij = 1.0 / (adj[i, j] + epsilon)
            curv = 1.0 - w1 / d_ij
            curvature[i, j] = curvature[j, i] = curv
    return curvature


# ----------------------------------------------------------------------
# Regret‑weighted strategy enriched by curvature
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathAction:
    id: int
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True)
class MathCounterfactual:
    action_id: int
    outcome_value: float
    probability: float = 1.0


def regret_weighted_strategy(
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual],
    curvature: np.ndarray,
) -> Dict[int, float]:
    """
    Compute regret‑weighted values.
    The raw expected value of each action is multiplied by a curvature‑adjusted factor:
        factor_i = 1 + mean_{j≠i} curvature_{ij}
    This rewards actions that reside in positively curved neighborhoods.
    """
    # Base expected values from counterfactuals
    base_vals: Dict[int, float] = {a.id: 0.0 for a in actions}
    for cf in counterfactuals:
        base_vals[cf.action_id] += cf.outcome_value * cf.probability

    # Curvature adjustment
    n = curvature.shape[0]
    adjusted: Dict[int, float] = {}
    for a in actions:
        # mean curvature to all other nodes (ignore self)
        mean_curv = (curvature[a.id].sum() - curvature[a.id, a.id]) / (n - 1) if n > 1 else 0.0
        factor = 1.0 + mean_curv
        adjusted[a.id] = base_vals[a.id] * factor
    return adjusted


# ----------------------------------------------------------------------
# End‑to‑end hybrid algorithm
# ----------------------------------------------------------------------
def extract_full_features(text: str) -> Dict[str, float]:
    """
    Deterministic pseudo‑random feature generator.
    The same text always yields the same vector.
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
    ]
    return {k: rnd.random() for k in keys}


def hybrid_algorithm(texts: List[str]) -> Tuple[np.ndarray, Dict[int, float]]:
    """
    Run the hybrid pipeline on a collection of texts.
    Returns:
        - Ollivier‑Ricci curvature matrix of the stylometry similarity graph.
        - Curvature‑adjusted expected values for each text (keyed by index).
    """
    # 1. Stylometric feature extraction
    feature_dicts = [stylometry_features(t) for t in texts]

    # 2. Build similarity graph
    adjacency, _ = build_weighted_graph(feature_dicts)

    # 3. Compute curvature
    curvature = compute_ollivier_ricci_curvature(adjacency)

    # 4. Build actions / counterfactuals from full features
    actions: List[MathAction] = []
    counterfactuals: List[MathCounterfactual] = []
    for idx, txt in enumerate(texts):
        full_feat = extract_full_features(txt)
        # Use the mean of the pseudo‑random vector as a scalar proxy for expected value
        ev = float(np.mean(list(full_feat.values())))
        actions.append(MathAction(id=idx, expected_value=ev))
        counterfactuals.append(MathCounterfactual(action_id=idx, outcome_value=ev))

    # 5. Regret‑weighted strategy enriched by curvature
    expected_values = regret_weighted_strategy(actions, counterfactuals, curvature)

    return curvature, expected_values


if __name__ == "__main__":
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "In a hole in the ground there lived a hobbit.",
        "To be, or not to be, that is the question."
    ]
    curv_mat, exp_vals = hybrid_algorithm(sample_texts)
    np.set_printoptions(precision=4, suppress=True)
    print("Ollivier‑Ricci curvature matrix:")
    print(curv_mat)
    print("\nCurvature‑adjusted expected values (by text index):")
    print(exp_vals)