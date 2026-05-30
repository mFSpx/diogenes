# DARWIN HAMMER — match 4097, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_infotaxis_min_hybrid_hybrid_hybrid_m2692_s2.py (gen6)
# parent_b: hybrid_hybrid_nlms_hybrid_h_hybrid_hard_truth_ma_m1061_s2.py (gen5)
# born: 2026-05-29T23:53:42Z

import numpy as np
import math
import random
import sys
import pathlib
from collections import Counter
from dataclasses import dataclass

"""
This module fuses the hybrid_hybrid_infotaxis_min_hybrid_hybrid_hybrid_m2692_s2.py and 
hybrid_hybrid_nlms_hybrid_h_hybrid_hard_truth_ma_m1061_s2.py algorithms.
The mathematical bridge between these two algorithms lies in the concept of 
information entropy and pheromone decay, and the integration of high-dimensional 
text features onto a low-dimensional model space using a bilinear form and 
radial basis function (RBF) kernels. 
The labelled feature vectors from the second algorithm are used to calculate the 
signal value of the pheromone entries, which in turn are used to inform the 
infotaxis-based decision making process in the first algorithm.
"""

MAX_COMPONENT_TOKENS = 500

FUNCTION_CATS: dict[str, set[str]] = {
    "pronoun": set(
        "i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()
    ),
    "article": set("a an the".split()),
    "preposition": set(
        "about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()
    ),
    "auxiliary": set(
        "am are be been being can could did do does had has have is may might must shall should was were will would".split()
    ),
    "conjunction": set(
        "and but or nor so yet for".split()
    ),
    "adverb": set(
        "how very rather more".split()
    ),
}

@dataclass
class PheromoneEntry:
    uuid: str
    surface_key: str
    signal_kind: str
    signal_value: float
    half_life_seconds: int
    created_at: pathlib.Path
    last_decay: pathlib.Path

    def age_seconds(self) -> float:
        return np.random.uniform(0, 100)

    def decay_factor(self) -> float:
        if self.half_life_seconds <= 0:
            return 0.0
        return 0.5 ** (self.age_seconds() / self.half_life_seconds)

    def apply_decay(self) -> None:
        factor = self.decay_factor()
        self.signal_value *= factor
        self.last_decay = pathlib.Path.cwd()

MAX64 = (1 << 64) - 1

def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, "big") + token.encode("utf-8")
    return int.from_bytes(data, "big")

def gaussian(r: float, epsilon: float = 1.0) -> float:
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def compute_phash(values: list[float]) -> int:
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits

def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()

def similarity_matrix(features: dict[int, list[float]]) -> tuple[np.ndarray, list[int]]:
    nodes = list(features.keys())
    n = len(nodes)
    S = np.empty((n, n), dtype=np.float64)
    hashes = [compute_phash(list(features[n])) for n in nodes]

    for i in range(n):
        for j in range(i, n):
            d = hamming_distance(hashes[i], hashes[j])
            sim = 1.0 - d / 64.0
            S[i, j] = sim
            S[j, i] = sim
    return S, nodes

def rbf_kernel_matrix(features: dict[int, list[float]], epsilon: float = 1.0) -> tuple[np.ndarray, list[int]]:
    nodes = list(features.keys())
    n = len(nodes)
    K = np.empty((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(i, n):
            dist = euclidean(features[nodes[i]], features[nodes[j]])
            val = gaussian(dist, epsilon)
            K[i, j] = val
            K[j, i] = val
    return K, nodes

def fuse_features(features: dict[int, list[float]]) -> tuple[np.ndarray, list[int]]:
    S, nodes = similarity_matrix(features)
    K, _ = rbf_kernel_matrix(features)
    fused_matrix = np.multiply(S, K)
    return fused_matrix, nodes

def fuse_pheromones(fused_matrix: np.ndarray, pheromones: list[PheromoneEntry]) -> list[PheromoneEntry]:
    updated_pheromones = []
    for pheromone in pheromones:
        signal_value = pheromone.signal_value
        for i in range(len(fused_matrix)):
            signal_value += fused_matrix[i][i]
        updated_pheromone = PheromoneEntry(
            pheromone.uuid,
            pheromone.surface_key,
            pheromone.signal_kind,
            signal_value,
            pheromone.half_life_seconds,
            pheromone.created_at,
            pheromone.last_decay,
        )
        updated_pheromones.append(updated_pheromone)
    return updated_pheromones

def infotaxis_decision(features: dict[int, list[float]], pheromones: list[PheromoneEntry]) -> int:
    fused_matrix, nodes = fuse_features(features)
    updated_pheromones = fuse_pheromones(fused_matrix, pheromones)
    best_node = -1
    best_value = -1
    for i in range(len(nodes)):
        value = updated_pheromones[i].signal_value
        if value > best_value:
            best_value = value
            best_node = nodes[i]
    return best_node

if __name__ == "__main__":
    features = {0: [1.0, 2.0, 3.0], 1: [4.0, 5.0, 6.0]}
    pheromones = [
        PheromoneEntry("uuid1", "surface_key1", "signal_kind1", 1.0, 100, pathlib.Path("created_at1"), pathlib.Path("last_decay1")),
        PheromoneEntry("uuid2", "surface_key2", "signal_kind2", 2.0, 200, pathlib.Path("created_at2"), pathlib.Path("last_decay2")),
    ]
    best_node = infotaxis_decision(features, pheromones)
    print("Best node:", best_node)