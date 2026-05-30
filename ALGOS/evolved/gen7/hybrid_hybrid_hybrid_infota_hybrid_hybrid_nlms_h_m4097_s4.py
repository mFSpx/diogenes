# DARWIN HAMMER — match 4097, survivor 4
# gen: 7
# parent_a: hybrid_hybrid_infotaxis_min_hybrid_hybrid_hybrid_m2692_s2.py (gen6)
# parent_b: hybrid_hybrid_nlms_hybrid_h_hybrid_hard_truth_ma_m1061_s2.py (gen5)
# born: 2026-05-29T23:53:42Z

"""
This module fuses the hybrid_hybrid_infotaxis_min_hybrid_hybrid_hybrid_m2692_s2.py and 
hybrid_hybrid_nlms_hybrid_h_hybrid_hard_truth_ma_m1061_s2.py algorithms.
The mathematical bridge between these two algorithms lies in the concept of 
information entropy and Gaussian kernel similarity. The labelled feature vectors 
from the second algorithm are used to calculate the signal value of the pheromone 
entries, which in turn are used to inform the infotaxis-based decision making 
process in the first algorithm through a Gaussian-weighted similarity measure.
"""

import numpy as np
import math
import random
import sys
import pathlib
from collections import Counter
from dataclasses import dataclass

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
    "conjunction": set("and but or so yet for nor".split()),
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

def gaussian(r: float, epsilon: float = 1.0) -> float:
    return math.exp(-((epsilon * r) ** 2))

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
            dist = math.sqrt(sum((x - y) ** 2 for x, y in zip(features[nodes[i]], features[nodes[j]])))
            val = gaussian(dist, epsilon)
            K[i, j] = val
            K[j, i] = val
    return K, nodes

def hybrid_similarity(features: dict[int, list[float]], pheromone_entries: list[PheromoneEntry]) -> np.ndarray:
    K, nodes = rbf_kernel_matrix(features)
    S, _ = similarity_matrix(features)
    n = len(nodes)

    # Weight the similarity matrix by pheromone signals
    weighted_S = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i, n):
            signal = 0.0
            for entry in pheromone_entries:
                if entry.surface_key == nodes[i] + nodes[j]:
                    signal += entry.signal_value
            weighted_S[i, j] = S[i, j] * signal
            weighted_S[j, i] = weighted_S[i, j]
    return weighted_S

def infotaxis_decision(weighted_S: np.ndarray) -> int:
    n = len(weighted_S)
    probs = np.zeros(n)
    for i in range(n):
        probs[i] = np.sum(weighted_S[i, :])
    probs /= np.sum(probs)
    return np.random.choice(n, p=probs)

def generate_pheromone_entries() -> list[PheromoneEntry]:
    entries = []
    for i in range(10):
        entry = PheromoneEntry(
            uuid=str(i),
            surface_key=str(i),
            signal_kind="test",
            signal_value=random.random(),
            half_life_seconds=100,
            created_at=pathlib.Path.cwd(),
            last_decay=pathlib.Path.cwd(),
        )
        entries.append(entry)
    return entries

if __name__ == "__main__":
    features = {i: [random.random() for _ in range(10)] for i in range(10)}
    pheromone_entries = generate_pheromone_entries()
    weighted_S = hybrid_similarity(features, pheromone_entries)
    decision = infotaxis_decision(weighted_S)
    print(decision)