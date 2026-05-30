# DARWIN HAMMER — match 4665, survivor 3
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_minimum_cost__m539_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1887_s0.py (gen6)
# born: 2026-05-29T23:57:27Z

import math
import random
import sys
import pathlib
import numpy as np
from dataclasses import dataclass

@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str

    def lsm_vector(self) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(self.name)) % (2**32))
        return rng.random(3)


def reconstruction_risk_score(unique_quasi_identifiers: int, total_records: int) -> float:
    if total_records == 0:
        return 1.0
    return 1.0 / (1.0 + math.exp(-unique_quasi_identifiers / total_records))


def compute_lsm_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    dist = np.linalg.norm(v1 - v2)
    return 1.0 / (1.0 + dist)


def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    if not all(0.0 <= x <= 1.0 for x in (prior, likelihood, false_positive)):
        raise ValueError("probabilities must be in [0,1]")
    return likelihood * prior + false_positive * (1.0 - prior)


def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    if marginal <= 0.0:
        raise ValueError("marginal must be > 0")
    return (likelihood * prior) / marginal


class Sheaf:
    def __init__(self, node_dims: dict, edges: list):
        self.node_dims = node_dims
        self.edges = edges
        self._restrictions = {}
        self._sections = {}

    def set_restriction(self, edge: tuple, src_map: np.ndarray, dst_map: np.ndarray) -> None:
        u, v = edge
        if src_map.shape[1] != self.node_dims[u]:
            raise ValueError("src_map column dimension must match dim(node u)")
        if dst_map.shape[1] != self.node_dims[v]:
            raise ValueError("dst_map column dimension must match dim(node v)")
        if src_map.shape[0] != dst_map.shape[0]:
            raise ValueError("src_map and dst_map must have the same row dimension")
        self._restrictions[(u, v)] = (np.asarray(src_map, dtype=float),
                                      np.asarray(dst_map, dtype=float))

    def set_section(self, node, value: np.ndarray) -> None:
        if value.shape[0] != self.node_dims[node]:
            raise ValueError("section vector dimension mismatch")
        self._sections[node] = np.asarray(value, dtype=float)

    def get_section(self, node):
        return self._sections.get(node, np.zeros(self.node_dims[node]))

    def restriction_residual(self, edge: tuple) -> float:
        src_map, dst_map = self._restrictions[edge]
        u, v = edge
        su = self.get_section(u)
        sv = self.get_section(v)
        return np.linalg.norm(src_map @ su - dst_map @ sv)


class DenseAssociativeMemory:
    def __init__(self, dim: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(scale=0.1, size=(dim, dim))
        self.W = (self.W + self.W.T) / 2.0
        self.b = rng.normal(scale=0.1, size=dim)

    def energy(self, x: np.ndarray) -> float:
        return -0.5 * x @ self.W @ x + self.b @ x

    def retrieve(self, query: np.ndarray, steps: int = 20, lr: float = 0.1) -> np.ndarray:
        x = query.astype(float).copy()
        for _ in range(steps):
            grad = -self.W @ x + self.b
            x -= lr * grad
        return x


class RegretWeightedStrategy:
    def __init__(self, base_lr: float = 0.01):
        self.base_lr = base_lr

    def adjusted_lr(self, prev_loss: float, curr_loss: float) -> float:
        regret = prev_loss - curr_loss
        factor = 1.0 + max(regret, 0.0)
        return self.base_lr * factor


def hybrid_edge_weight(
    tier_u: ModelTier,
    tier_v: ModelTier,
    prior: float,
    likelihood: float,
    false_pos: float,
) -> float:
    lsm_u = tier_u.lsm_vector()
    lsm_v = tier_v.lsm_vector()
    bayes = bayes_marginal(prior, likelihood, false_pos)
    sim = compute_lsm_similarity(lsm_u, lsm_v)
    return bayes * sim


def hybrid_energy(
    sheaf: Sheaf,
    dam: DenseAssociativeMemory,
    edge_weights: dict,
    beta: float = 0.5,
) -> float:
    sheaf_term = 0.0
    for edge in sheaf.edges:
        u, v = edge
        w = hybrid_edge_weight(ModelTier(u, 0, ''), ModelTier(v, 0, ''), 0.5, 0.5, 0.1)
        sheaf_term += w * sheaf.restriction_residual(edge)
    dam_term = dam.energy(np.array([dam.retrieve(np.zeros(dam.W.shape[0]))]))
    return beta * sheaf_term + (1-beta) * dam_term


def train_hybrid_model(
    model_tiers: list,
    sheaf_edges: list,
    prior: float,
    likelihood: float,
    false_pos: float,
    beta: float = 0.5,
) -> tuple:
    node_dims = {tier.name: 3 for tier in model_tiers}
    sheaf = Sheaf(node_dims, sheaf_edges)
    dam = DenseAssociativeMemory(3)

    for edge in sheaf_edges:
        u, v = edge
        src_map = np.random.normal(scale=0.1, size=(3, 3))
        dst_map = np.random.normal(scale=0.1, size=(3, 3))
        sheaf.set_restriction(edge, src_map, dst_map)

    regret_strategy = RegretWeightedStrategy()
    prev_loss = float('inf')

    for _ in range(100):
        edge_weights = {}
        for edge in sheaf_edges:
            u, v = edge
            edge_weights[edge] = hybrid_edge_weight(model_tiers[0], model_tiers[1], prior, likelihood, false_pos)
        
        loss = hybrid_energy(sheaf, dam, edge_weights, beta)
        lr = regret_strategy.adjusted_lr(prev_loss, loss)
        prev_loss = loss

        # Update sheaf sections using gradient descent on the hybrid energy
        for node in node_dims:
            section = sheaf.get_section(node)
            grad = np.zeros(3)
            for edge in sheaf_edges:
                if edge[0] == node:
                    src_map, _ = sheaf._restrictions[edge]
                    grad += 2 * src_map.T @ (src_map @ section - src_map @ sheaf.get_section(edge[1]))
                elif edge[1] == node:
                    _, dst_map = sheaf._restrictions[edge]
                    grad += 2 * dst_map.T @ (dst_map @ sheaf.get_section(edge[0]) - dst_map @ section)
            sheaf.set_section(node, section - lr * grad)

    return sheaf, dam