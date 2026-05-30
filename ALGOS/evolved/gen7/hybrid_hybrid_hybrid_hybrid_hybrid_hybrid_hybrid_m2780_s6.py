# DARWIN HAMMER — match 2780, survivor 6
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1213_s2.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_decrea_m527_s5.py (gen5)
# born: 2026-05-29T23:46:00Z

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import numpy as np

# ----------------------------------------------------------------------
# Data structures (shared by both parents)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

@dataclass(frozen=True)
class MathAction:
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0

@dataclass(frozen=True)
class MathCounterfactual:
    action_id: str
    outcome_value: float
    probability: float = 1.0

# ----------------------------------------------------------------------
# Parent A – similarity, entropy and RLCT utilities
# ----------------------------------------------------------------------
def sphericity_index(length: float, width: float, height: float) -> float:
    """Return a dimensionless sphericity index."""
    vol = length * width * height
    surface = (length * width + width * height + height * length) * 2
    return (math.pi ** (1/3)) * (vol ** (2/3)) / surface

def morphology_vector(m: Morphology) -> np.ndarray:
    """Map a Morphology to a 4‑D feature vector."""
    sph = sphericity_index(m.length, m.width, m.height)
    return np.array([m.length, m.width, m.height, sph * m.mass])

def similarity_score(morph_a: Morphology, morph_b: Morphology) -> float:
    """SSIM‑like similarity based on normalized dot product."""
    a = morphology_vector(morph_a)
    b = morphology_vector(morph_b)
    num = np.dot(a, b)
    den = np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
    return num / den

def token_entropy(tokens: List[str]) -> float:
    """Normalized Shannon entropy of token frequencies."""
    if not tokens:
        return 0.0
    counts = np.array(list(dict((t, tokens.count(t)) for t in set(tokens)).values()), dtype=float)
    probs = counts / counts.sum()
    H = -np.sum(probs * np.log(probs + 1e-12))
    return H / np.log(len(probs))  # normalised to [0,1]

def real_log_canonical_threshold(morph: Morphology) -> float:
    """
    Toy RLCT: log of the determinant of the covariance matrix of the morphology
    feature vector with a tiny isotropic jitter.  Returns a positive scalar.
    """
    vec = morphology_vector(morph)
    cov = np.cov(vec.reshape(1, -1) + 1e-6 * np.random.randn(1, 4).T)
    # For a 1‑D covariance the determinant is just the variance
    var = np.maximum(cov, 1e-12).item()
    return max(math.log(var + 1e-12), 0.0)

def hybrid_similarity_entropy(
    morph_a: Morphology,
    morph_b: Morphology,
    log_tokens: List[str]
) -> Tuple[float, float, float]:
    """
    Compute similarity S, normalised entropy H and RLCT factor λ_rlct.
    Returns (S, H, λ_rlct) where λ_rlct = exp(‑RLCT) ∈ (0,1].
    """
    S = similarity_score(morph_a, morph_b)
    H = token_entropy(log_tokens)
    rlct = (real_log_canonical_threshold(morph_a) + real_log_canonical_threshold(morph_b)) / 2.0
    lambda_rlct = math.exp(-rlct)
    return S, H, lambda_rlct

# ----------------------------------------------------------------------
# Parent B – regret, Gini and edge‑weight utilities
# ----------------------------------------------------------------------
def gini_coefficient(values: np.ndarray) -> float:
    """Gini coefficient of a 1‑D non‑negative array."""
    if values.ndim != 1:
        raise ValueError("gini_coefficient expects a 1‑D array")
    sorted_vals = np.sort(values)
    n = values.size
    cumulative = np.cumsum(sorted_vals, dtype=float)
    if cumulative[-1] == 0:
        return 0.0
    return (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n

def compute_regret_gini(actions: List[MathAction]) -> Tuple[np.ndarray, float]:
    """
    Compute per‑action regret r_i = cost_i·risk_i,
    normalise to sum‑one, and return (r̂, G) where G is the Gini coefficient.
    """
    regrets = np.array([a.cost * a.risk for a in actions], dtype=float)
    total = regrets.sum()
    if total == 0:
        normalized = np.zeros_like(regrets)
    else:
        normalized = regrets / total
    G = gini_coefficient(normalized)
    return normalized, G

def euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    return float(np.linalg.norm(vec1 - vec2))

def compose_hybrid_edge_weights(
    actions: List[MathAction],
    morphologies: Dict[str, Morphology],
    marginal_probs: Dict[Tuple[str, str], float],
    log_tokens: List[str]
) -> Dict[Tuple[str, str], float]:
    """
    Build composite edge weights W_ij = d_ij·(1+r̂_i)·(1+r̂_j)·(1‑G)·M_ij·λ_rlct
    where λ_rlct is derived from the two endpoint morphologies.
    """
    # Regret & Gini
    r_norm, G = compute_regret_gini(actions)

    # Map action id -> index for regret lookup
    idx_map = {a.id: i for i, a in enumerate(actions)}

    # Pre‑compute RLCT factor per action pair using their morphologies
    rlct_factors: Dict[Tuple[str, str], float] = {}
    for (i, j) in marginal_probs.keys():
        m_i = morphologies[i]
        m_j = morphologies[j]
        _, _, lambda_rlct = hybrid_similarity_entropy(m_i, m_j, log_tokens)
        rlct_factors[(i, j)] = lambda_rlct

    weights: Dict[Tuple[str, str], float] = {}
    for (i, j), M_ij in marginal_probs.items():
        vec_i = morphology_vector(morphologies[i])
        vec_j = morphology_vector(morphologies[j])
        d_ij = euclidean_distance(vec_i, vec_j) + 1e-12
        r_i = r_norm[idx_map[i]]
        r_j = r_norm[idx_map[j]]
        λ = rlct_factors[(i, j)]
        w = d_ij * (1 + r_i) * (1 + r_j) * (1 - G) * M_ij * λ
        weights[(i, j)] = w
    return weights

def decreasing_rate(t: int, lam: float = 0.9, alpha: float = 0.05) -> float:
    """p(t) = λ·exp(−α·t) decreasing pruning probability."""
    return lam * math.exp(-alpha * t)

def hybrid_prune_and_rank(
    edge_weights: Dict[Tuple[str, str], float],
    actions: List[MathAction],
    max_iterations: int = 10
) -> List[Tuple[str, float]]:
    """
    Perform decreasing‑rate pruning on the edge weight graph and rank actions
    by a hybrid expected value:
        HV_i = expected_value_i · (1 – avg_retained_weight_i)
    Returns a list of (action_id, HV_i) sorted descending.
    """
    # Build adjacency list
    adj: Dict[str, List[float]] = {a.id: [] for a in actions}
    for (i, j), w in edge_weights.items():
        adj[i].append(w)
        adj[j].append(w)

    retained: Dict[Tuple[str, str], bool] = {}
    for _ in range(max_iterations):
        to_retain = {}
        for (i, j) in edge_weights:
            p_retain = decreasing_rate(len(retained.get((i, j), [])))
            to_retain[(i, j)] = random.random() < p_retain
        retained = to_retain

    action_hvs: Dict[str, float] = {}
    for a in actions:
        total_weight = 0.0
        count = 0
        for j, w in adj[a.id]:
            if retained.get((a.id, j), False):
                total_weight += w
                count += 1
        if count > 0:
            avg_weight = total_weight / count
            hv = a.expected_value * (1 - avg_weight)
            action_hvs[a.id] = hv

    return sorted(action_hvs.items(), key=lambda x: x[1], reverse=True)

def unified_hybrid_score(
    morph_a: Morphology,
    morph_b: Morphology,
    actions: List[MathAction],
    marginal_probs: Dict[Tuple[str, str], float],
    log_tokens: List[str]
) -> Dict[Tuple[str, str], float]:
    S, H, lambda_rlct = hybrid_similarity_entropy(morph_a, morph_b, log_tokens)
    r_norm, G = compute_regret_gini(actions)

    weights: Dict[Tuple[str, str], float] = {}
    for (i, j), M_ij in marginal_probs.items():
        vec_i = morphology_vector(morphologies[i])
        vec_j = morphology_vector(morphologies[j])
        d_ij = euclidean_distance(vec_i, vec_j) + 1e-12
        Ψ = S * (1 - H) * (1 - G) * lambda_rlct
        w = d_ij * Ψ * M_ij
        weights[(i, j)] = w
    return weights

# Minimal smoke test
if __name__ == "__main__":
    morph_a = Morphology(1.0, 2.0, 3.0, 4.0)
    morph_b = Morphology(1.1, 2.1, 3.1, 4.1)
    actions = [MathAction("a", 10.0), MathAction("b", 20.0)]
    marginal_probs = {("a", "b"): 0.5}
    log_tokens = ["token1", "token2"]
    weights = unified_hybrid_score(morph_a, morph_b, actions, marginal_probs, log_tokens)
    print(weights)