# DARWIN HAMMER — match 3195, survivor 0
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_model__m1238_s3.py (gen4)
# parent_b: hybrid_hybrid_hybrid_xgboos_hybrid_hybrid_hybrid_m1553_s2.py (gen6)
# born: 2026-05-29T23:48:36Z

import numpy as np
from dataclasses import dataclass
from datetime import datetime, timezone
from math import exp
from pathlib import Path
from typing import Any, Dict, Iterable, List

# Global constants & helpers
ROOT = Path(__file__).resolve().parents[2] if Path(__file__).exists() else Path.cwd()
DEFAULT_BUDGET_MB = 4096
DEFAULT_RESERVE_MB = 768

def now_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str

@dataclass
class Node:
    id: int
    ram_mb: int
    risk_score: float

def reconstruction_risk_score(unique_quasi_identifiers: int, total_records: int) -> float:
    if total_records <= 0:
        return 0.0
    return max(0.0, min(1.0, unique_quasi_identifiers / total_records))

class PheromoneEntry:
    """
    Simple container for a pheromone signal.  The original implementation
    used ``np.random.uuid1`` which does not exist; we replace it with the
    standard library ``uuid`` module and add a tiny amount of defensive
    programming.
    """
    __slots__ = (
        "uuid",
        "surface_key",
        "signal_kind",
        "signal_value",
        "half_life_seconds",
        "created_at",
        "last_decay",
    )

    def __init__(self, surface_key: str, signal_kind: str,
                 signal_value: float, half_life_seconds: int):
        self.uuid = str(uuid.uuid1())
        self.surface_key = surface_key
        self.signal_kind = signal_kind
        self.signal_value = float(signal_value)
        self.half_life_seconds = int(half_life_seconds)
        now = datetime.utcnow()
        self.created_at = now
        self.last_decay = now

    def age_seconds(self) -> float:
        """Age of the entry since the last decay event."""
        return (datetime.utcnow() - self.last_decay).total_seconds()

    def decay_factor(self) -> float:
        """Multiplicative decay factor based on half‑life."""
        if self.half_life_seconds <= 0:
            return 0.0
        return 0.5 ** (self.age_seconds() / self.half_life_seconds)

    def apply_decay(self) -> None:
        """Apply exponential decay to the stored signal value."""
        self.signal_value *= self.decay_factor()
        self.last_decay = datetime.utcnow()

# Mathematical bridge: Interpret the reconstruction risk score as a node attribute
# in the graph used for Ollivier-Ricci curvature computation. The risk score is used
# to weigh the node masses in the lazy random-walk distribution, allowing the curvature
# to reflect the privacy landscape of the artefacts.
def compute_ollivier_ricci_curvature(nodes: List[Node], 
                                    alpha: float = 0.5, 
                                    epsilon: float = 1.0) -> Dict[int, float]:
    num_nodes = len(nodes)
    adj_matrix = np.zeros((num_nodes, num_nodes))
    for i, node_i in enumerate(nodes):
        for j, node_j in enumerate(nodes):
            if i != j and node_i.ram_mb > 0 and node_j.ram_mb > 0:
                adj_matrix[i, j] = 1.0 / (node_i.ram_mb * node_j.ram_mb)
    
    # Weight node masses with reconstruction risk score
    masses = [node.risk_score * node.ram_mb for node in nodes]
    
    # Compute Ollivier-Ricci curvature
    curvatures = []
    for i in range(num_nodes):
        degree = np.sum(adj_matrix[i])
        if degree > 0:
            curvature = 1 - np.sum([masses[j] * adj_matrix[i, j] / degree for j in range(num_nodes)])
            curvatures.append(curvature)
        else:
            curvatures.append(0.0)
    
    return {i: curvatures[i] for i in range(num_nodes)}

def compute_pheromone_curvature(nodes: List[Node], pheromones: List[PheromoneEntry], 
                                alpha: float = 0.5, epsilon: float = 1.0) -> Dict[int, float]:
    num_nodes = len(nodes)
    adj_matrix = np.zeros((num_nodes, num_nodes))
    for i, node_i in enumerate(nodes):
        for j, node_j in enumerate(nodes):
            if i != j and node_i.ram_mb > 0 and node_j.ram_mb > 0:
                adj_matrix[i, j] = 1.0 / (node_i.ram_mb * node_j.ram_mb)
    
    # Weight node masses with reconstruction risk score
    masses = [node.risk_score * node.ram_mb for node in nodes]
    
    # Compute pheromone curvature
    curvatures = []
    for i in range(num_nodes):
        degree = np.sum(adj_matrix[i])
        if degree > 0:
            pheromone_signal = np.sum([pheromone.signal_value * adj_matrix[i, j] for j in range(num_nodes)])
            curvature = 1 - (masses[i] + alpha * pheromone_signal) / degree
            curvatures.append(curvature)
        else:
            curvatures.append(0.0)
    
    return {i: curvatures[i] for i in range(num_nodes)}

def hybrid_curvature(nodes: List[Node], pheromones: List[PheromoneEntry], 
                     alpha: float = 0.5, epsilon: float = 1.0) -> Dict[int, float]:
    # Compute Ollivier-Ricci curvature
    curvatures = compute_ollivier_ricci_curvature(nodes, alpha, epsilon)
    
    # Compute pheromone curvature
    pheromone_curvatures = compute_pheromone_curvature(nodes, pheromones, alpha, epsilon)
    
    # Combine curvatures
    hybrid_curvatures = {}
    for node_id in curvatures:
        hybrid_curvatures[node_id] = (curvatures[node_id] + pheromone_curvatures[node_id]) / 2
    
    return hybrid_curvatures

if __name__ == "__main__":
    # Smoke test: Create some nodes and pheromones, and compute the hybrid curvature
    nodes = [Node(1, 1024, 0.5), Node(2, 2048, 0.3), Node(3, 4096, 0.7)]
    pheromones = [PheromoneEntry("key1", "signal1", 0.2, 3600), PheromoneEntry("key2", "signal2", 0.8, 7200)]
    hybrid_curvatures = hybrid_curvature(nodes, pheromones)
    print(hybrid_curvatures)