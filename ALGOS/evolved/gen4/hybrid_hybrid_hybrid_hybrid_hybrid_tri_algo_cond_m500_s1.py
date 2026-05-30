# DARWIN HAMMER — match 500, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_hybrid_sketch_dense_associative_me_m32_s2.py (gen3)
# parent_b: hybrid_tri_algo_conduit_hybrid_cockpit_metri_m68_s0.py (gen2)
# born: 2026-05-29T23:29:33Z

import numpy as np
import math
import random
import sys
from dataclasses import dataclass

__all__ = [
    "Sheaf",
    "DenseAssociativeMemory",
    "signal_scores",
    "cockpit_honesty",
    "hybrid_energy",
    "hybrid_update_rule",
    "hybrid_retrieve",
]

def _byte_entropy(data: bytes, sample: int = 8192) -> float:
    if not data:
        return 0.0
    chunk = data[:sample]
    return shannon_entropy(list(chunk)) / 8.0  

def shannon_entropy(chunk):
    entropy = 0.0
    for x in set(chunk):
        p_x = chunk.count(x) / len(chunk)
        entropy += -p_x * math.log(p_x, 2)
    return entropy

def signal_scores(
    data: bytes,
    status_code: int | None = None,
    mime: str = "",
    keyword_hits: int = 0,
    structural_links: int = 0,
) -> tuple[float, float]:
    size = len(data)
    entropy = _byte_entropy(data)
    status_bonus = 0.18 if status_code and 200 <= status_code < 300 else -0.10
    mime_bonus = 0.12 if any(x in (mime or "").lower() for x in ("html", "json", "text", "xml")) else 0.02
    size_bonus = min(0.22, math.log1p(size) / 60.0)
    keyword_bonus = min(0.20, keyword_hits * 0.05)
    structure_bonus = min(0.16, structural_links * 0.01)
    signal = max(0.0, min(1.0,
        0.20 + status_bonus + mime_bonus + size_bonus + keyword_bonus + structure_bonus + 0.12 * entropy))
    noise = max(0.0, min(1.0,
        0.58 - 0.22 * entropy - keyword_bonus - structure_bonus + (0.12 if size < 64 else 0.0)))
    return signal, noise

def cockpit_honesty(signal_score: float) -> float:
    return signal_score

class Sheaf:
    def __init__(self, node_dims: dict, edges: list):
        self.node_dims = {k: int(v) for k, v in node_dims.items()}
        self.edges = list(edges)
        self._restrictions = {}          
        self._sections = {}              

    def set_restriction(self, edge: tuple, src_map: np.ndarray, dst_map: np.ndarray) -> None:
        u, v = edge
        src_map = np.asarray(src_map, dtype=float)
        dst_map = np.asarray(dst_map, dtype=float)

        if src_map.shape[1] != self.node_dims[u]:
            raise ValueError("src_map column dimension must match dim(node u)")
        if dst_map.shape[1] != self.node_dims[v]:
            raise ValueError("dst_map column dimension must match dim(node v)")
        if src_map.shape[0] != dst_map.shape[0]:
            raise ValueError("src_map and dst_map must have the same row dimension")

        self._restrictions[(u, v)] = (src_map, dst_map)

    def set_section(self, node, value: np.ndarray) -> None:
        value = np.asarray(value, dtype=float).reshape(-1)
        if value.shape[0] != self.node_dims[node]:
            raise ValueError(f"Section vector size for node {node} must be {self.node_dims[node]}")
        self._sections[node] = value

    def get_section(self, node):
        return self._sections.get(node, np.zeros(self.node_dims[node]))

    def all_section_vectors(self) -> list[np.ndarray]:
        return [self.get_section(node) for node in self.node_dims]

    def concatenated_section(self) -> np.ndarray:
        return np.concatenate(self.all_section_vectors())

    def consistency_penalty(self) -> float:
        total = 0.0
        for (u, v), (src_map, dst_map) in self._restrictions.items():
            xu = self.get_section(u)
            xv = self.get_section(v)
            diff = src_map @ xu - dst_map @ xv
            total += np.linalg.norm(diff) ** 2
        return total

class DenseAssociativeMemory:
    def __init__(self, dim: int, seed: int | None = None):
        rng = np.random.default_rng(seed)
        A = rng.normal(scale=0.1, size=(dim, dim))
        self.W = (A + A.T) / 2.0
        self.b = rng.normal(scale=0.1, size=dim)

    def energy(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        return -0.5 * x @ self.W @ x + self.b @ x

    def gradient(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return -self.W @ x + self.b

    def update(self, grad: np.ndarray, lr: float) -> None:
        self.W += lr * np.outer(grad, grad)  
        self.b += lr * grad

def hybrid_energy(sheaf: Sheaf, memory: DenseAssociativeMemory, signal_score: float) -> float:
    x = sheaf.concatenated_section()
    E_mem = memory.energy(x)
    E_sheaf = sheaf.consistency_penalty()
    return (1 - signal_score) * E_mem + signal_score * E_sheaf

def hybrid_update_rule(
    sheaf: Sheaf, 
    memory: DenseAssociativeMemory, 
    signal_score: float, 
    lr: float
) -> None:
    x = sheaf.concatenated_section()
    grad_mem = memory.gradient(x)
    grad_sheaf = np.zeros_like(x)
    for (u, v), (src_map, dst_map) in sheaf._restrictions.items():
        xu = sheaf.get_section(u)
        xv = sheaf.get_section(v)
        grad_u = 2 * (src_map.T @ (src_map @ xu - dst_map @ xv))
        grad_v = -2 * (dst_map.T @ (src_map @ xu - dst_map @ xv))
        grad_sheaf[u] += grad_u
        grad_sheaf[v] += grad_v

    grad_hybrid = (1 - signal_score) * grad_mem + signal_score * grad_sheaf
    memory.update(grad_hybrid, lr)

def hybrid_retrieve(sheaf: Sheaf, memory: DenseAssociativeMemory, signal_score: float, lr: float, steps: int) -> np.ndarray:
    for _ in range(steps):
        hybrid_update_rule(sheaf, memory, signal_score, lr)
    return sheaf.concatenated_section()