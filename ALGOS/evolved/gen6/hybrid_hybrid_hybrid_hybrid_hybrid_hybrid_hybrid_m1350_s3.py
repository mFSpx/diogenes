# DARWIN HAMMER — match 1350, survivor 3
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

class SchoolfieldParams:
    rho_25: float = 1.0
    delta_h_activation: float = 12_000.0
    t_low: float = 283.15
    t_high: float = 307.15
    delta_h_low: float = -45_000.0
    delta_h_high: float = 65_000.0
    r_cal: float = 1.987  # universal gas constant (cal·K⁻¹·mol⁻¹)


def c_to_k(celsius: float) -> float:
    return celsius + 273.15


def developmental_rate(temp_k: float, params: SchoolfieldParams = SchoolfieldParams()) -> float:
    if temp_k <= 0 or params.rho_25 < 0:
        raise ValueError("temperature must be Kelvin-positive and rho_25 non-negative")
    num = params.rho_25 * (temp_k / 298.15) * math.exp(
        -params.delta_h_activation / params.r_cal * (1.0 / temp_k - 1.0 / 298.15)
    )
    low = 1.0 + math.exp(
        params.delta_h_low / params.r_cal * (1.0 / params.t_low - 1.0 / temp_k)
    )
    high = 1.0 + math.exp(
        params.delta_h_high / params.r_cal * (1.0 / temp_k - 1.0 / params.t_high)
    )
    return num / (low * high)


def variational_free_energy(mu: float, Wx: float) -> float:
    return (mu - Wx) ** 2


def hybrid_vfe(mu: float, Wx: float, temp_c: float) -> float:
    temp_k = c_to_k(temp_c)
    rho = developmental_rate(temp_k)
    return variational_free_energy(mu, Wx) * rho


def acceptance_probability(delta_e: float, temperature: float) -> float:
    if delta_e < 0:
        return 1.0
    return math.exp(-delta_e / temperature)


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    x_arr = np.asarray(x, dtype=float)
    pos = x_arr >= 0
    neg = ~pos
    out = np.empty_like(x_arr)
    out[pos] = 1.0 / (1.0 + np.exp(-x_arr[pos]))
    exp_x = np.exp(x_arr[neg])
    out[neg] = exp_x / (1.0 + exp_x)
    return float(out) if np.isscalar(x) else out


def tropical_max_plus(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    if A.shape[1] != B.shape[0]:
        raise ValueError("Incompatible shapes for tropical product")
    result = np.full((A.shape[0], B.shape[1]), -np.inf)
    for k in range(A.shape[1]):
        candidate = A[:, k, None] + B[k, None, :]
        result = np.maximum(result, candidate)
    return result


def binary_logistic_grad_hess(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    p = sigmoid(y_pred)
    grad = p - y_true
    hess = p * (1.0 - p)
    return grad, hess


def minhash_similarity(sig_a: List[int], sig_b: List[int]) -> float:
    if len(sig_a) != len(sig_b) or not sig_a:
        return 0.0
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / len(sig_a)


def shannon_entropy(probs: np.ndarray) -> float:
    eps = 1e-12
    probs = np.clip(probs, eps, 1.0)
    return -float(np.sum(probs * np.log2(probs)))


Node = Hashable
Graph = Mapping[Node, Set[Node]]


def broadcast_leader_election(
    graph: Graph,
    node_energy: Dict[Node, float],
    temperature: float,
) -> Node:
    nodes = list(graph.keys())
    index = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    neg_inf = -np.inf
    A = np.full((n, n), neg_inf, dtype=float)

    for i, src in enumerate(nodes):
        for dst in graph[src]:
            j = index[dst]
            A[i, j] = node_energy.get(src, 0.0)

    B = tropical_max_plus(A, A)
    broadcast_strength = np.max(B, axis=0)

    probs = []
    for i, node in enumerate(nodes):
        delta_e = broadcast_strength[i] - node_energy.get(node, 0.0)
        probs.append(acceptance_probability(delta_e, temperature))

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
    grad, hess = binary_logistic_grad_hess(y_true, y_pred)
    sim = minhash_similarity(sig_a, sig_b)
    regret = 1.0 - sim
    vals, counts = np.unique(sig_a, return_counts=True)
    probs = counts.astype(float) / counts.sum()
    entropy = shannon_entropy(probs)
    grad_blend = regret * grad + entropy * np.mean(grad)
    hess_blend = regret * hess + entropy * np.mean(hess)
    return grad_blend, hess_blend


def improved_hybrid_vfe(mu: float, Wx: float, temp_c: float, sig_a: List[int], sig_b: List[int]) -> Tuple[float, np.ndarray, np.ndarray]:
    temp_k = c_to_k(temp_c)
    rho = developmental_rate(temp_k)
    vfe = variational_free_energy(mu, Wx)
    energy = vfe * rho
    y_true = np.array([1.0])
    y_pred = np.array([energy])
    grad, hess = regret_weighted_gradient(y_true, y_pred, sig_a, sig_b)
    return energy, grad, hess


def improved_broadcast_leader_election(
    graph: Graph,
    node_energy: Dict[Node, float],
    temperature: float,
    sig_a: List[int],
    sig_b: List[int],
) -> Tuple[Node, np.ndarray, np.ndarray]:
    leader = broadcast_leader_election(graph, node_energy, temperature)
    energy = node_energy.get(leader, 0.0)
    y_true = np.array([1.0])
    y_pred = np.array([energy])
    grad, hess = regret_weighted_gradient(y_true, y_pred, sig_a, sig_b)
    return leader, grad, hess


# Example usage
if __name__ == "__main__":
    graph = {1: {2, 3}, 2: {1, 3}, 3: {1, 2}}
    node_energy = {1: 0.5, 2: 0.3, 3: 0.2}
    temperature = 1.0
    sig_a = [1, 2, 3]
    sig_b = [1, 2, 3]
    leader, grad, hess = improved_broadcast_leader_election(graph, node_energy, temperature, sig_a, sig_b)
    print("Leader:", leader)
    print("Gradient:", grad)
    print("Hessian:", hess)