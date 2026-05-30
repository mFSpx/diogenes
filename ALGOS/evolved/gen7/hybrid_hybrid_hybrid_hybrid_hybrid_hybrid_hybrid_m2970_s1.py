# DARWIN HAMMER — match 2970, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_shap_a_hybrid_hybrid_hybrid_m2345_s1.py (gen6)
# parent_b: hybrid_hybrid_hybrid_decisi_hybrid_fractional_hd_m37_s1.py (gen3)
# born: 2026-05-29T23:47:00Z

import numpy as np
import random
import math
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple

Node = int
Graph = Dict[int, set]
Model = Dict[int, float]
MasterVector = np.ndarray
TextFeatures = Dict[str, float]
KrampusCoordinates = Tuple[float, float, float]

@dataclass
class PheromoneEntry:
    uuid: str
    surface_key: str
    signal_kind: str
    signal_value: float
    half_life_seconds: int
    created_at: str
    last_decay: str

    def __post_init__(self):
        self.uuid = str(np.random.randint(0, 1000000))
        self.created_at = ""
        self.last_decay = ""

    def age_seconds(self) -> float:
        return np.random.uniform(0, 100)

    def decay_factor(self) -> float:
        if self.half_life_seconds <= 0:
            return 0.0
        return 0.5 ** (self.age_seconds() / self.half_life_seconds)

    def apply_decay(self) -> None:
        factor = self.decay_factor()
        self.signal_value *= factor
        self.last_decay = ""

class PheromoneStore:
    _entries: Dict[str, PheromoneEntry] = {}

    @classmethod
    def add(cls, entry: PheromoneEntry) -> None:
        cls._entries[entry.uuid] = entry

    @classmethod
    def get_by_surface(cls, surface_key: str) -> List[PheromoneEntry]:
        return [e for e in cls._entries.values() if e.surface_key == surface_key]

def shannon_entropy(counts: Counter) -> float:
    total = sum(counts.values())
    entropy = 0.0
    for count in counts.values():
        prob = count / total
        entropy -= prob * math.log2(prob) if prob > 0 else 0
    return entropy

def hybrid_build_adj(master_vectors: List[MasterVector], threshold: float = 0.5) -> Graph:
    graph = {}
    for i, vector in enumerate(master_vectors):
        graph[i] = set()
        for j, other_vector in enumerate(master_vectors):
            if i != j and np.linalg.norm(vector - other_vector) < threshold:
                graph[i].add(j)
    return graph

def hybrid_decay_pheromones(pheros: List[PheromoneEntry]) -> List[PheromoneEntry]:
    for phero in pheros:
        phero.apply_decay()
    return pheros

def hybrid_shap_entropy(graph: Graph, master_vectors: List[MasterVector]) -> Dict[int, float]:
    shap_values = {}
    for node in graph:
        neighbors = graph[node]
        shap_value = 0.0
        for neighbor in neighbors:
            shap_value += np.linalg.norm(master_vectors[node] - master_vectors[neighbor])
        shap_values[node] = shap_value
    counts = Counter(shap_values.values())
    entropy = shannon_entropy(counts)
    return {node: entropy for node in graph}

def hybrid_entropy_shap(graph: Graph, master_vectors: List[MasterVector]) -> Dict[int, float]:
    shap_values = {}
    for node in graph:
        neighbors = graph[node]
        shap_value = 0.0
        for neighbor in neighbors:
            shap_value += np.linalg.norm(master_vectors[node] - master_vectors[neighbor])
        shap_values[node] = shap_value
    entropies = {}
    for node in graph:
        neighbors = graph[node]
        node_shap_values = [np.linalg.norm(master_vectors[node] - master_vectors[neighbor]) for neighbor in neighbors]
        node_counts = Counter(node_shap_values)
        node_entropy = shannon_entropy(node_counts)
        entropies[node] = node_entropy
    return entropies

if __name__ == "__main__":
    master_vectors = [np.random.rand(3) for _ in range(10)]
    graph = hybrid_build_adj(master_vectors)
    pheros = [PheromoneEntry("uuid", "surface_key", "signal_kind", 1.0, 100, "", "") for _ in range(10)]
    hybrid_decay_pheromones(pheros)
    shap_entropy = hybrid_shap_entropy(graph, master_vectors)
    hybrid_entropies = hybrid_entropy_shap(graph, master_vectors)
    print(shap_entropy)
    print(hybrid_entropies)