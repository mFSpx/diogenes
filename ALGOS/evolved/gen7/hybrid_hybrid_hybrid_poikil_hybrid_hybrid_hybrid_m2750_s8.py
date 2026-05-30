# DARWIN HAMMER — match 2750, survivor 8
# gen: 7
# parent_a: hybrid_hybrid_poikilotherm__hybrid_regret_engine_m1595_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_ternar_m2473_s2.py (gen6)
# born: 2026-05-29T23:45:48Z

import math
import random
import sys
import pathlib
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Sequence, Iterable
import numpy as np

@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0
    delta_h_activation: float = 12_000.0      
    t_low: float = 283.15                     
    t_high: float = 307.15                    
    delta_h_low: float = -45_000.0            
    delta_h_high: float = 65_000.0            
    r_cal: float = 1.987                      

@dataclass(frozen=True)
class MathAction:
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0

@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "hybrid"

@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float

Point = Tuple[float, float]
Edge = Tuple[str, str]

def schoolfield_rate(temp_k: float, params: SchoolfieldParams) -> float:
    R = params.r_cal * 4.184  
    num = params.rho_25 * math.exp(
        -params.delta_h_activation / (R * 298.15)
    )
    denom = (1.0 +
             math.exp(params.delta_h_low / (R * params.t_low)) +
             math.exp(params.delta_h_high / (R * params.t_high)))
    temp_term = (1.0 +
                 math.exp(params.delta_h_low / (R * temp_k)) +
                 math.exp(params.delta_h_high / (R * temp_k)))
    return num * denom / temp_term

def gini_coefficient(values: Iterable[float]) -> float:
    xs = sorted(float(v) for v in values)
    if not xs or sum(xs) == 0:
        return 0.0
    n = len(xs)
    cumulative = sum((2 * i - n - 1) * x for i, x in enumerate(xs, 1))
    return cumulative / (n * sum(xs))

def euclidean_length(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def radial_basis_function(x: np.ndarray, centers: np.ndarray, epsilon: float = 1.0) -> np.ndarray:
    diff = x[:, None, :] - centers[None, :, :]
    sq_norm = np.sum(diff ** 2, axis=2)
    return np.exp(-epsilon * sq_norm)

class HybridPheromoneGraph:
    def __init__(self,
                 nodes: Dict[str, Point],
                 edges: List[Edge],
                 params: SchoolfieldParams):
        self.nodes = nodes
        self.edges = edges
        self.params = params
        self.pheromones: Dict[Edge, float] = {e: 1.0 for e in edges}
        self.half_life = 300.0
        self.lengths: Dict[Edge, float] = {e: euclidean_length(self.nodes[e[0]], self.nodes[e[1]]) for e in edges}

    def decay_factor(self, dt_seconds: float, gini: float) -> float:
        base = 0.5 ** (dt_seconds / self.half_life)
        return base ** (1.0 - gini)  

    def update_pheromones(self,
                          temperature_c: float,
                          actions: List[MathAction],
                          bandit_updates: List[BanditUpdate],
                          dt_seconds: float = 1.0) -> None:
        temp_k = temperature_c + 273.15
        phi_t = schoolfield_rate(temp_k, self.params)

        expected_vals = np.array([a.expected_value for a in actions])
        max_ev = expected_vals.max() if expected_vals.size else 0.0
        regrets = max_ev - expected_vals
        beta = 1.0 / max(1.0, temperature_c)
        propensities = np.exp(-beta * regrets)
        if propensities.sum() > 0:
            propensities /= propensities.sum()
        prop_map: Dict[str, float] = {a.id: p for a, p in zip(actions, propensities)}

        gini = gini_coefficient(propensities)

        decay = self.decay_factor(dt_seconds, gini)
        for e in self.pheromones:
            self.pheromones[e] *= decay

        for upd in bandit_updates:
            src = upd.context_id
            dst = upd.action_id
            edge = (src, dst) if (src, dst) in self.pheromones else (dst, src)
            if edge in self.pheromones:
                prop = prop_map.get(upd.action_id, 0.0)
                reinforcement = phi_t * prop * max(0.0, upd.reward)
                self.pheromones[edge] += reinforcement

    def edge_weights(self) -> Dict[Edge, float]:
        weights = {}
        for e in self.edges:
            weights[e] = self.lengths[e] / (1 + self.pheromones[e])
        return weights

# Improved with normalized lengths and pheromone levels
class NormalizedHybridPheromoneGraph(HybridPheromoneGraph):
    def __init__(self,
                 nodes: Dict[str, Point],
                 edges: List[Edge],
                 params: SchoolfieldParams):
        super().__init__(nodes, edges, params)
        self.min_length = min(self.lengths.values())
        self.max_length = max(self.lengths.values())
        self.min_pheromone = min(self.pheromones.values())
        self.max_pheromone = max(self.pheromones.values())

    def edge_weights(self) -> Dict[Edge, float]:
        weights = {}
        for e in self.edges:
            normalized_length = (self.lengths[e] - self.min_length) / (self.max_length - self.min_length)
            normalized_pheromone = (self.pheromones[e] - self.min_pheromone) / (self.max_pheromone - self.min_pheromone)
            weights[e] = normalized_length / (1 + normalized_pheromone)
        return weights

# Example usage
if __name__ == "__main__":
    nodes = {
        'A': (0.0, 0.0),
        'B': (1.0, 0.0),
        'C': (0.0, 1.0)
    }
    edges = [('A', 'B'), ('B', 'C'), ('C', 'A')]
    params = SchoolfieldParams()
    graph = NormalizedHybridPheromoneGraph(nodes, edges, params)
    actions = [MathAction('A', 1.0), MathAction('B', 2.0), MathAction('C', 3.0)]
    bandit_updates = [BanditUpdate('A', 'B', 1.0), BanditUpdate('B', 'C', 2.0), BanditUpdate('C', 'A', 3.0)]
    graph.update_pheromones(20.0, actions, bandit_updates)
    print(graph.edge_weights())