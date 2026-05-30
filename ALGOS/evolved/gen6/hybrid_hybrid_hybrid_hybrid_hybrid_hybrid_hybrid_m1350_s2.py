# DARWIN HAMMER — match 1350, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_bandit_router_m999_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_distri_hybrid_hybrid_xgboos_m642_s2.py (gen5)
# born: 2026-05-29T23:35:39Z

import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Set, Tuple, Hashable

import numpy as np

# ----------------------------------------------------------------------
# Parent‑A components (TERNAR‑TTT + Schoolfield)
# ----------------------------------------------------------------------
def variational_free_energy(mu: float, Wx: float) -> float:
    """Variational free energy term."""
    return (mu - Wx) ** 2


def c_to_k(celsius: float) -> float:
    """Convert Celsius to Kelvin."""
    return celsius + 273.15


class SchoolfieldParams:
    rho_25: float = 1.0
    delta_h_activation: float = 12_000.0
    t_low: float = 283.15
    t_high: float = 307.15
    delta_h_low: float = -45_000.0
    delta_h_high: float = 65_000.0
    r_cal: float = 1.987  # universal gas constant (cal·K⁻¹·mol⁻¹)


def developmental_rate(temp_k: float, params: SchoolfieldParams = SchoolfieldParams()) -> float:
    """Full Schoolfield rate as a function of absolute temperature."""
    if temp_k <= 0 or params.rho_25 < 0:
        raise ValueError("temperature must be Kelvin‑positive and rho_25 non‑negative")
    # Numerator for activation term
    num = params.rho_25 * (temp_k / 298.15) * math.exp(
        -params.delta_h_activation / params.r_cal * (1.0 / temp_k - 1.0 / 298.15)
    )
    # Low‑temperature inhibition term
    low = 1.0 + math.exp(
        params.delta_h_low / params.r_cal * (1.0 / params.t_low - 1.0 / temp_k)
    )
    # High‑temperature inhibition term
    high = 1.0 + math.exp(
        params.delta_h_high / params.r_cal * (1.0 / temp_k - 1.0 / params.t_high)
    )
    return num / (low * high)


def hybrid_vfe(mu: float, Wx: float, temp_c: float) -> float:
    """Temperature‑modulated VFE: 𝓔 = VFE·ρ(T)."""
    temp_k = c_to_k(temp_c)
    rho = developmental_rate(temp_k)
    return variational_free_energy(mu, Wx) * rho


# ----------------------------------------------------------------------
# Parent‑B components (Leader‑Tree Election + XGBoost Regret‑MinHash)
# ----------------------------------------------------------------------
def acceptance_probability(delta_e: float, temperature: float) -> float:
    """Metropolis acceptance probability."""
    if delta_e < 0:
        return 1.0
    return math.exp(-delta_e / temperature)


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    """Numerically stable sigmoid."""
    x_arr = np.asarray(x, dtype=float)
    pos = x_arr >= 0
    neg = ~pos
    out = np.empty_like(x_arr)
    out[pos] = 1.0 / (1.0 + np.exp(-x_arr[pos]))
    exp_x = np.exp(x_arr[neg])
    out[neg] = exp_x / (1.0 + exp_x)
    return float(out) if np.isscalar(x) else out


def tropical_max_plus(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Tropical (max‑plus) matrix product: (A⊗B)_ij = max_k (A_ik + B_kj)."""
    if A.shape[1] != B.shape[0]:
        raise ValueError("Incompatible shapes for tropical product")
    result = np.full((A.shape[0], B.shape[1]), -np.inf)
    for k in range(A.shape[1]):
        candidate = A[:, k, None] + B[k, None, :]
        result = np.maximum(result, candidate)
    return result


def binary_logistic_grad_hess(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Gradient and Hessian of binary logistic loss."""
    # Ensure predictions are probabilities
    p = sigmoid(y_pred)
    grad = p - y_true
    hess = p * (1.0 - p)
    return grad, hess


def minhash_similarity(sig_a: List[int], sig_b: List[int]) -> float:
    """Estimate Jaccard similarity from MinHash signatures."""
    if len(sig_a) != len(sig_b) or not sig_a:
        return 0.0
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / len(sig_a)


def shannon_entropy(probs: np.ndarray) -> float:
    """Shannon entropy of a probability distribution."""
    eps = 1e-12
    probs = np.clip(probs, eps, 1.0)
    return -float(np.sum(probs * np.log2(probs)))


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
Node = Hashable
Graph = Mapping[Node, Set[Node]]


def broadcast_leader_election(
    graph: Graph,
    node_energy: Dict[Node, float],
    temperature: float,
) -> Node:
    """
    Perform a leader election on ``graph`` using tropical max‑plus broadcast.

    Steps
    -----
    1. Build adjacency matrix A where A_ij = node_energy[i] if (i, j) is an edge, else -inf.
    2. Propagate a broadcast field B = A ⊗ I (max‑plus power‑1) which is just A.
       Re‑apply the product once more to model two‑hop influence: B = A ⊗ A.
    3. For each node compute Δ𝓔 = max(B_i) - node_energy[i].
    4. Accept the node with highest acceptance probability; ties broken randomly.
    Returns the elected leader node.
    """
    nodes = list(graph.keys())
    index = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    neg_inf = -np.inf
    A = np.full((n, n), neg_inf, dtype=float)

    for i, src in enumerate(nodes):
        for dst in graph[src]:
            j = index[dst]
            A[i, j] = node_energy.get(src, 0.0)

    # Two‑hop tropical propagation with normalization
    row_sums = np.sum(A, axis=1)
    A_normalized = A / np.maximum(row_sums[:, np.newaxis], 1e-12)
    B = tropical_max_plus(A_normalized, A)

    # Compute broadcast strength per node (max over incoming columns)
    broadcast_strength = np.max(B, axis=0)

    # Determine acceptance probabilities
    probs = []
    for i, node in enumerate(nodes):
        delta_e = broadcast_strength[i] - node_energy.get(node, 0.0)
        probs.append(acceptance_probability(delta_e, temperature))

    # Normalise to a distribution
    prob_arr = np.array(probs)
    if prob_arr.sum() == 0:
        prob_arr = np.ones_like(prob_arr) / len(prob_arr)
    else:
        prob_arr = prob_arr / prob_arr.sum()

    leader = random.choices(nodes, weights=prob_arr, k=1)[0]
    return leader


def regret_weighted_gradient(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sig_a: List[int],
    sig_b: List[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a gradient‑Hessian pair that blends XGBoost logistic gradients
    with a regret factor derived from MinHash similarity and entropy regularisation.

    Regret factor ρ = 1 - similarity(sig_a, sig_b).
    Regulariser   λ = Shannon entropy of the empirical distribution of sig_a.
    Final gradient = ρ * grad + λ * mean(grad)
    Final hessian = ρ * hess + λ * mean(hess)
    """
    grad, hess = binary_logistic_grad_hess(y_true, y_pred)

    # Regret factor from MinHash similarity
    sim = minhash_similarity(sig_a, sig_b)
    regret = 1.0 - sim

    # Entropy regulariser from signature distribution
    # Build empirical probability of each hash value in sig_a
    vals, counts = np.unique(sig_a, return_counts=True)
    probs = counts.astype(float) / counts.sum()
    entropy = shannon_entropy(probs)

    # Blend with gradient and Hessian clipping
    grad_clipped = np.clip(grad, -1e6, 1e6)
    hess_clipped = np.clip(hess, 1e-12, 1e6)
    grad_blend = regret * grad_clipped + entropy * np.mean(grad_clipped)
    hess_blend = regret * hess_clipped + entropy * np.mean(hess_clipped)
    return grad_blend, hess_blend