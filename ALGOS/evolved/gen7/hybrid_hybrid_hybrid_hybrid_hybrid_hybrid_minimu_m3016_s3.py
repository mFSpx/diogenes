# DARWIN HAMMER — match 3016, survivor 3
# gen: 7
# parent_a: hybrid_hybrid_hybrid_infota_hybrid_hybrid_hybrid_m1526_s1.py (gen6)
# parent_b: hybrid_hybrid_minimum_cost__hybrid_hybrid_bandit_m249_s0.py (gen3)
# born: 2026-05-29T23:47:19Z

import math
import numpy as np
import random
import sys
import pathlib
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib

@dataclass(frozen=True)
class VramSlotPlan:
    artifact_id: str
    artifact_kind: str
    action: str
    estimated_mb: int
    reason: str
    detail: dict[str, any]

    def as_dict(self) -> dict[str, any]:
        return asdict(self)

def entropy(probabilities: list[float], eps: float = 1e-12) -> float:
    total = sum(probabilities)
    if total <= 0:
        raise ValueError('positive probability mass required')
    return -sum((p/total) * math.log(max(p/total, eps)) for p in probabilities if p > 0)

def signature(tokens: list[str], k: int = 128) -> list[int]:
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError('k must be positive')
    if not toks:
        return [2**64 - 1] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]

def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, 'big') + token.encode('utf-8', errors='ignore')
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), 'big')

def similarity(sig_a: list[int], sig_b: list[int]) -> float:
    if len(sig_a) != len(sig_b):
        raise ValueError('signatures must have equal length')
    if not sig_a:
        raise ValueError('signatures must not be empty')
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)

def length(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def tree_metrics(
    nodes: dict[str, tuple[float, float]],
    edges: list[tuple[str, str]],
    root: str,
) -> tuple[dict[str, list[str]], dict[tuple[str, str], float], dict[str, float]]:
    adj = defaultdict(list)
    edge_len = {}
    node_dist = {}

    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        edge_len[(a, b)] = length(nodes[a], nodes[b])
        edge_len[(b, a)] = length(nodes[b], nodes[a])

    queue = [(root, 0)]
    visited = set()
    while queue:
        node, dist = queue.pop(0)
        if node not in visited:
            visited.add(node)
            node_dist[node] = dist
            for neighbor in adj[node]:
                if neighbor not in visited:
                    queue.append((neighbor, dist + 1))

    return dict(adj), edge_len, node_dist

def hybrid_similarity(tree_nodes: dict[str, tuple[float, float]], tree_edges: list[tuple[str, str]], root: str, sig_a: list[int], sig_b: list[int]) -> float:
    adj, edge_len, node_dist = tree_metrics(tree_nodes, tree_edges, root)
    minhash_similarity = similarity(sig_a, sig_b)
    modified_cost = {edge: length * (1 - minhash_similarity) for edge, length in edge_len.items()}
    hybrid_similarity = sum(modified_cost.values()) / len(modified_cost) if modified_cost else 0
    return hybrid_similarity

def bayesian_update(prior: dict[str, float], likelihood: dict[str, float]) -> dict[str, float]:
    posterior = {node: prior[node] * likelihood[node] for node in prior}
    posterior = {node: value / sum(posterior.values()) for node, value in posterior.items()}
    return posterior

def hybrid_vram_allocation(tree_nodes: dict[str, tuple[float, float]], tree_edges: list[tuple[str, str]], root: str, sig_a: list[int], sig_b: list[int]) -> VramSlotPlan:
    hybrid_similarity_value = hybrid_similarity(tree_nodes, tree_edges, root, sig_a, sig_b)
    prior = {node: 1 / len(tree_nodes) for node in tree_nodes}
    likelihood = {node: hybrid_similarity_value for node in tree_nodes}
    posterior = bayesian_update(prior, likelihood)
    allocated_node = max(posterior, key=posterior.get)
    allocated_amount = int(posterior[allocated_node] * 1024)
    return VramSlotPlan(allocated_node, "Hybrid", "Allocate", allocated_amount, "Hybrid similarity", {"similarity": hybrid_similarity_value})

def improved_hybrid_vram_allocation(tree_nodes: dict[str, tuple[float, float]], tree_edges: list[tuple[str, str]], root: str, sig_a: list[int], sig_b: list[int]) -> VramSlotPlan:
    hybrid_similarity_value = hybrid_similarity(tree_nodes, tree_edges, root, sig_a, sig_b)
    prior = {node: 1 / len(tree_nodes) for node in tree_nodes}
    likelihood = {node: hybrid_similarity_value + (1 - hybrid_similarity_value) * random.random() for node in tree_nodes}
    posterior = bayesian_update(prior, likelihood)
    allocated_node = max(posterior, key=posterior.get)
    allocated_amount = int(posterior[allocated_node] * 1024)
    return VramSlotPlan(allocated_node, "Improved Hybrid", "Allocate", allocated_amount, "Improved Hybrid similarity", {"similarity": hybrid_similarity_value})

if __name__ == "__main__":
    tree_nodes = {
        "A": (0, 0),
        "B": (1, 1),
        "C": (2, 2)
    }
    tree_edges = [("A", "B"), ("B", "C"), ("C", "A")]
    root = "A"
    sig_a = signature(["hello", "world"], 128)
    sig_b = signature(["hello", " foo"], 128)
    hybrid_similarity_value = hybrid_similarity(tree_nodes, tree_edges, root, sig_a, sig_b)
    hybrid_vram_plan = hybrid_vram_allocation(tree_nodes, tree_edges, root, sig_a, sig_b)
    improved_hybrid_vram_plan = improved_hybrid_vram_allocation(tree_nodes, tree_edges, root, sig_a, sig_b)
    print(hybrid_similarity_value)
    print(hybrid_vram_plan.as_dict())
    print(improved_hybrid_vram_plan.as_dict())