# DARWIN HAMMER — match 1630, survivor 5
# gen: 5
# parent_a: hybrid_hybrid_decreasing_pr_hybrid_fisher_locali_m215_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_decisi_hybrid_hybrid_bandit_m14_s0.py (gen4)
# born: 2026-05-29T23:38:04Z

import math
import random
import sys
from pathlib import Path
from collections.abc import Hashable
from typing import Dict, Tuple, List, Iterable

import numpy as np

EPISTEMIC_FLAGS: Tuple[str, ...] = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")
_EPISTEMIC_CERTAINTY = {
    "FACT": 1.0,
    "PROBABLE": 0.8,
    "POSSIBLE": 0.6,
    "SURE_MAYBE": 0.4,
    "BULLSHIT": 0.2,
}

def length(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def prune_probability(t: float, lam: float = 1.0, alpha: float = 0.2) -> float:
    if t < 0 or lam < 0 or alpha < 0:
        raise ValueError('t, lam, and alpha must be non‑negative')
    return min(1.0, lam * math.exp(-alpha * t))

def prune_edges(edges: List[Hashable], t: float,
                lam: float = 1.0, alpha: float = 0.2,
                seed: int | str | None = None) -> List[Hashable]:
    rng = random.Random(seed)
    p = prune_probability(t, lam, alpha)
    return [e for e in edges if rng.random() >= p]

def haversine_distance(lat1: float, lon1: float,
                       lat2: float, lon2: float) -> float:
    R = 6371000.0  
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    Δφ = math.radians(lat2 - lat1)
    Δλ = math.radians(lon2 - lon1)
    a = math.sin(Δφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(Δλ / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))

def compute_resource_vector(
    nodes: Iterable[Hashable],
    positions: Dict[Hashable, Tuple[float, float]],
    reference: Tuple[float, float],
    signatures: Dict[Hashable, str],
    external_scores: Dict[Hashable, float],
    beta: float = 1.0,
) -> Dict[Hashable, np.ndarray]:
    sig_counts = {}
    for sig in signatures.values():
        sig_counts[sig] = sig_counts.get(sig, 0) + 1

    resource = {}
    for n in nodes:
        d = length(positions[n], reference)
        sigma = 1 if sig_counts.get(signatures.get(n, ""), 0) > 1 else 0
        p = beta * sigma
        s = float(external_scores.get(n, 0.0))

        resource[n] = np.array([d, p, s], dtype=float)

    return resource

def compute_composite_weights(
    edges: List[Tuple[Hashable, Hashable]],
    positions: Dict[Hashable, Tuple[float, float]],
    epistemic_flags: Dict[Tuple[Hashable, Hashable], str],
    weight_matrix: np.ndarray,
    resource_vec: Dict[Hashable, np.ndarray],
) -> Dict[Tuple[Hashable, Hashable], float]:
    all_vecs = np.stack(list(resource_vec.values()))
    norm = np.linalg.norm(all_vecs, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    normed = {k: v / norm[i] for i, (k, v) in enumerate(resource_vec.items())}

    comp_weights = {}
    for (u, v) in edges:
        ℓ = length(positions[u], positions[v])

        flag = epistemic_flags.get((u, v), "POSSIBLE")
        C_f = _EPISTEMIC_CERTAINTY.get(flag, 0.6)

        eu = normed[u]
        ev = normed[v]
        similarity = 1.0 + float(np.dot(eu, ev))  

        idx_u = int(u) if isinstance(u, (int, np.integer)) else hash(u) % weight_matrix.shape[0]
        idx_v = int(v) if isinstance(v, (int, np.integer)) else hash(v) % weight_matrix.shape[1]
        W_uv = float(weight_matrix[idx_u, idx_v])

        comp_weights[(u, v)] = ℓ * C_f * similarity * W_uv

    return comp_weights

def prune_edges_dynamic(
    edges: List[Tuple[Hashable, Hashable]],
    t: float,
    comp_weights: Dict[Tuple[Hashable, Hashable], float],
    base_lam: float = 1.0,
    base_alpha: float = 0.2,
    seed: int | str | None = None,
) -> List[Tuple[Hashable, Hashable]]:
    if not comp_weights:
        return edges.copy()

    mean_w = float(np.mean(list(comp_weights.values())))

    lam = max(0.1, min(5.0, base_lam * (1 + mean_w/10))) # prevent exploding values
    alpha = max(0.01, min(1.0, base_alpha * (1 + mean_w/10)))

    rng = random.Random(seed)
    p = prune_probability(t, lam, alpha)
    return [e for e in edges if rng.random() >= p]

class HybridFusion:
    def __init__(
        self,
        n_nodes: int,
        seed: int = 0,
        base_eta: float = 0.01,
        gamma: float = 0.9,
        epsilon: float = 1e-8,
    ):
        self.W = np.random.rand(n_nodes, n_nodes)
        self.S = 0.0
        self.eta = base_eta
        self.gamma = gamma
        self.epsilon = epsilon

    def update(self, rewards: float):
        self.S = self.gamma * self.S + rewards

    def get_effective_eta(self) -> float:
        return self.eta * (1 + self.S)

    def update_weight_matrix(self, comp_weights: Dict[Tuple[Hashable, Hashable], float]):
        for (u, v), w_uv in comp_weights.items():
            idx_u = int(u) if isinstance(u, (int, np.integer)) else hash(u) % self.W.shape[0]
            idx_v = int(v) if isinstance(v, (int, np.integer)) else hash(v) % self.W.shape[1]
            self.W[idx_u, idx_v] += self.get_effective_eta() * w_uv

def main():
    # Initialize HybridFusion
    n_nodes = 10
    hf = HybridFusion(n_nodes)

    # Example usage
    positions = {(i, j): (i * 1.0, j * 1.0) for i in range(n_nodes) for j in range(n_nodes)}
    reference = (0.0, 0.0)
    signatures = {(i, j): f"sig_{i}_{j}" for i in range(n_nodes) for j in range(n_nodes)}
    external_scores = {(i, j): 1.0 for i in range(n_nodes) for j in range(n_nodes)}
    resource_vec = compute_resource_vector(positions.keys(), positions, reference, signatures, external_scores)
    edges = [(i, j) for i in range(n_nodes) for j in range(n_nodes) if i != j]
    epistemic_flags = {(i, j): "FACT" for i in range(n_nodes) for j in range(n_nodes)}
    weight_matrix = np.random.rand(n_nodes, n_nodes)

    comp_weights = compute_composite_weights(edges, positions, epistemic_flags, weight_matrix, resource_vec)

    pruned_edges = prune_edges_dynamic(edges, 1.0, comp_weights)

    rewards = 1.0
    hf.update(rewards)
    hf.update_weight_matrix(comp_weights)

if __name__ == "__main__":
    main()