# DARWIN HAMMER — match 5266, survivor 5
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_sketch_m1264_s2.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_regret_engine_m8_s2.py (gen4)
# born: 2026-05-30T00:01:07Z

import sys
import math
import random
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Dict

# ----------------------------------------------------------------------
# Parent A core structures
# ----------------------------------------------------------------------
@dataclass
class Morphology:
    length: float
    width: float
    height: float
    mass: float

@dataclass
class StoreState:
    level: float = 0.0
    alpha: float = 1.0
    beta: float = 1.0
    dt: float = 1.0
    base: float = 1.0
    gain: float = 1.0
    limit: float = 10.0

    def update(self, inflow: List[float], outflow: List[float]) -> Tuple[float, float]:
        delta = self.alpha * sum(inflow) - self.beta * sum(outflow)
        self.level = max(0.0, self.level + self.dt * delta)
        return self.level, delta

# ----------------------------------------------------------------------
# Parent B core structures
# ----------------------------------------------------------------------
@dataclass
class Endpoint:
    failures: int
    failure_threshold: int
    righting_time_index: float

    @property
    def failure_rate(self) -> float:
        return self.failures / (self.failure_threshold + 1e-9)

@dataclass(frozen=True, slots=True)
class MathAction:
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0

@dataclass(frozen=True, slots=True)
class MathCounterfactual:
    action_id: str
    outcome_value: float
    probability: float = 1.0

# ----------------------------------------------------------------------
# Utility functions (Parent A)
# ----------------------------------------------------------------------
def tropical_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    n, m = A.shape[0], B.shape[1]
    result = np.full((n, m), np.inf)
    for i in range(n):
        for j in range(m):
            result[i, j] = np.min(A[i, :] + B[:, j])
    return result

def tropical_broadcast(adj: np.ndarray, steps: int = 3) -> np.ndarray:
    n = adj.shape[0]
    vec = np.zeros(n)  
    for _ in range(steps):
        vec = np.min(adj + vec[:, None], axis=0)
    return vec

def sphericity_index(morph: Morphology) -> float:
    volume = morph.length * morph.width * morph.height
    surface = 2 * (morph.length * morph.width +
                   morph.length * morph.height +
                   morph.width * morph.height)
    return volume / (surface + 1e-9)

def count_min_sketch(data: List[int],
                     width: int,
                     depth: int,
                     seed: int = 0) -> np.ndarray:
    rng = random.Random(seed)
    hashes = [lambda x, s=s: (hash((x, s)) % width) for s in range(depth)]
    sketch = np.zeros((depth, width), dtype=int)
    for item in data:
        for d, h in enumerate(hashes):
            idx = h(item)
            sketch[d, idx] += 1
    return sketch

def estimate_sketch_error(sketch: np.ndarray) -> float:
    total = sketch.sum()
    return (sketch.shape[1] - 1) / sketch.shape[1] * total

# ----------------------------------------------------------------------
# Utility functions (Parent B)
# ----------------------------------------------------------------------
def compute_health_scores(endpoints: List[Endpoint],
                          request_sequence: List[int]) -> List[float]:
    scores = []
    for req in request_sequence:
        ep = endpoints[req % len(endpoints)]
        health = (1.0 - ep.failure_rate) * ep.righting_time_index
        scores.append(health)
    return scores

def regret_engine(health_scores: List[float],
                  actions: List[MathAction]) -> Dict[str, float]:
    if not health_scores:
        avg_health = 0.0
    else:
        avg_health = sum(health_scores) / len(health_scores)
    regrets = {}
    for act in actions:
        regret = act.expected_value - act.cost * avg_health
        regrets[act.id] = regret
    return regrets

def hoeffding_bound(num_samples: int, delta: float = 0.05) -> float:
    return math.sqrt(math.log(2 / delta) / (2 * num_samples))

def hoeffding_split(candidates: List[float],
                    bound: float) -> List[int]:
    return [i for i, val in enumerate(candidates) if val > bound]

def simulated_annealing_accept(old: float,
                               new: float,
                               temperature: float) -> bool:
    if new < old:
        return True
    prob = math.exp(-(new - old) / max(temperature, 1e-9))
    return random.random() < prob

# ----------------------------------------------------------------------
# Hybrid pipeline (fusion of both parents)
# ----------------------------------------------------------------------
def hybrid_pipeline(adj_matrix: np.ndarray,
                    morphology: Morphology,
                    endpoints: List[Endpoint],
                    request_seq: List[int],
                    actions: List[MathAction],
                    sketch_depth: int = 4,
                    temperature: float = 1.0) -> Tuple[StoreState, np.ndarray, List[int]]:
    b = tropical_broadcast(adj_matrix, steps=3)  
    health = compute_health_scores(endpoints, request_seq)
    store = StoreState()
    outflow = (b * store.gain).tolist()
    store.update(inflow=health, outflow=outflow)

    sphericity = sphericity_index(morphology)
    mean_health = np.mean(health)
    modulated_width = max(1, int(sphericity * mean_health))
    data = request_seq
    sketch = count_min_sketch(data, modulated_width, sketch_depth)

    regrets = regret_engine(health, actions)
    bound = hoeffding_bound(len(health))
    leader_candidates = hoeffding_split(regrets.values(), bound)
    leader_actions = [list(regrets.keys())[i] for i in leader_candidates]

    current_leaders = leader_actions
    new_leaders = leader_actions
    if random.random() < 0.1: 
        new_leaders = random.sample(leader_actions, len(leader_actions))
    accept_prob = simulated_annealing_accept(len(current_leaders), len(new_leaders), temperature)
    if accept_prob:
        current_leaders = new_leaders

    return store, sketch, current_leaders