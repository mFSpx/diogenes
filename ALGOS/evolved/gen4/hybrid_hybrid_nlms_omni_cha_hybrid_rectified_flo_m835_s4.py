# DARWIN HAMMER — match 835, survivor 4
# gen: 4
# parent_a: hybrid_nlms_omni_chaotic_sprint_m59_s2.py (gen1)
# parent_b: hybrid_rectified_flow_hybrid_hybrid_hard_t_m184_s1.py (gen3)
# born: 2026-05-29T23:31:18Z

import numpy as np
import random
from collections import deque
from typing import Dict, List, Tuple

def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    return float(weights @ x)

def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, float]:
    pred = nlms_predict(weights, x)
    error = target - pred
    norm = float(np.dot(x, x) + eps)
    new_weights = weights + (mu * error / norm) * x
    return new_weights, error

def interpolant(x0: np.ndarray, x1: np.ndarray, t: np.ndarray) -> np.ndarray:
    t = np.reshape(t, (-1, 1))  
    return t * x1 + (1.0 - t) * x0

def flow_target(x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
    return x1 - x0

def _spline_piecewise_linear(x: np.ndarray, knots: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    return np.interp(x, knots, coeffs)

def kan_transform(x: np.ndarray, params: Dict[str, List[np.ndarray]]) -> np.ndarray:
    knots_list = params["knots"]
    coeffs_list = params["coeffs"]
    assert len(knots_list) == len(coeffs_list) == x.shape[0]

    transformed = np.empty_like(x, dtype=float)
    for idx, (kn, cf) in enumerate(zip(knots_list, coeffs_list)):
        transformed[idx] = _spline_piecewise_linear(
            np.array([x[idx]]), kn, cf
        )[0]
    return np.concatenate([transformed, np.array([1.0])])

def generate_synthetic_graph(
    num_nodes: int,
    avg_degree: int,
    seed: int | None = None,
) -> Tuple[Dict[int, List[Tuple[int, float]]], Dict[int, np.ndarray]]:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    adjacency: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(num_nodes)}
    features: Dict[int, np.ndarray] = {
        i: np.random.randn(2) for i in range(num_nodes)
    }

    prob = min(1.0, avg_degree / (num_nodes - 1))
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if random.random() < prob:
                impedance = random.uniform(0.1, 5.0)
                adjacency[i].append((j, impedance))
                adjacency[j].append((i, impedance))
    return adjacency, features

def impedance_weighted_input(
    node: int,
    adjacency: Dict[int, List[Tuple[int, float]]],
    features: Dict[int, np.ndarray],
) -> np.ndarray:
    neigh = adjacency.get(node, [])
    if not neigh:
        return np.zeros_like(next(iter(features.values())))

    acc = np.zeros_like(next(iter(features.values())))
    for nbr, imp in neigh:
        acc += imp * features[nbr]
    return acc

def hybrid_predict(
    weights: np.ndarray,
    node: int,
    adjacency: Dict[int, List[Tuple[int, float]]],
    features_t0: Dict[int, np.ndarray],
    features_t1: Dict[int, np.ndarray],
    t: float,
    kan_params: Dict[str, List[np.ndarray]],
) -> float:
    x_i_g = impedance_weighted_input(node, adjacency, features_t0)
    x0_i = features_t0[node]
    x1_i = features_t1[node]
    z_i_t = interpolant(x0_i, x1_i, np.array([t]))
    u_i_t = np.concatenate([x_i_g, z_i_t])
    psi_i_t = kan_transform(u_i_t, kan_params)
    return nlms_predict(weights, psi_i_t)

def hybrid_update(
    weights: np.ndarray,
    node: int,
    adjacency: Dict[int, List[Tuple[int, float]]],
    features_t0: Dict[int, np.ndarray],
    features_t1: Dict[int, np.ndarray],
    t: float,
    target: float,
    kan_params: Dict[str, List[np.ndarray]],
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, float]:
    x_i_g = impedance_weighted_input(node, adjacency, features_t0)
    x0_i = features_t0[node]
    x1_i = features_t1[node]
    z_i_t = interpolant(x0_i, x1_i, np.array([t]))
    u_i_t = np.concatenate([x_i_g, z_i_t])
    psi_i_t = kan_transform(u_i_t, kan_params)
    return nlms_update(weights, psi_i_t, target, mu, eps)

def main():
    num_nodes = 10
    avg_degree = 3
    adjacency, features_t0 = generate_synthetic_graph(num_nodes, avg_degree)
    features_t1 = {i: np.random.randn(2) for i in range(num_nodes)}
    kan_params = {
        "knots": [np.array([-1, 0, 1]) for _ in range(4)],
        "coeffs": [np.array([0, 1, 0]) for _ in range(4)],
    }
    weights = np.random.randn(4)
    t = 0.5
    target = 1.0
    new_weights, error = hybrid_update(weights, 0, adjacency, features_t0, features_t1, t, target, kan_params)
    print(new_weights)

if __name__ == "__main__":
    main()