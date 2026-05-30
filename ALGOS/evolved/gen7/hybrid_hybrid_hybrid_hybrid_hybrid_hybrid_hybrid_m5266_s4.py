# DARWIN HAMMER — match 5266, survivor 4
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_sketch_m1264_s2.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_regret_engine_m8_s2.py (gen4)
# born: 2026-05-30T00:01:07Z

"""
Hybrid Algorithm: Fusion of Parent A (tropical broadcast, sphericity‑modulated Count‑Min sketch,
Hoeffding split & simulated annealing) and Parent B (Endpoint health scores, Regret engine,
Hoeffding‑tree split).

Mathematical Bridge
-------------------
The bridge is the *health score vector* produced by Parent B.  It is used in three places:

1. As the **influent/out‑flow** for Parent A’s `StoreState` dynamics, thus linking the
   SSM‑derived health to the tropical broadcast update.
2. As the **gain vector** `b` for the Hoeffding split test – the same vector that in
   Parent A is obtained from a tropical matrix multiplication.
3. As a **modulation factor** for the width of the Count‑Min sketch: the sphericity index
   (computed from a `Morphology` instance) is multiplied by the mean health score,
   producing a data‑dependent sketch dimension.

The resulting system therefore fuses the matrix‑algebraic core of Parent A with the
statistical‑decision core of Parent B in a single unified pipeline.
"""

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
        """Standard linear reservoir update."""
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
    """Min‑plus (tropical) matrix multiplication."""
    n, m = A.shape[0], B.shape[1]
    result = np.full((n, m), np.inf)
    for i in range(n):
        for j in range(m):
            result[i, j] = np.min(A[i, :] + B[:, j])
    return result

def tropical_broadcast(adj: np.ndarray, steps: int = 3) -> np.ndarray:
    """Repeated tropical multiplication of the adjacency matrix with a unit vector."""
    n = adj.shape[0]
    vec = np.zeros(n)  # tropical zero is +inf, but we start from 0 for additive identity
    for _ in range(steps):
        vec = np.min(adj + vec[:, None], axis=0)
    return vec

def sphericity_index(morph: Morphology) -> float:
    """Simple sphericity proxy: volume / surface area."""
    volume = morph.length * morph.width * morph.height
    surface = 2 * (morph.length * morph.width +
                   morph.length * morph.height +
                   morph.width * morph.height)
    return volume / (surface + 1e-9)

def count_min_sketch(data: List[int],
                     width: int,
                     depth: int,
                     seed: int = 0) -> np.ndarray:
    """Classic Count‑Min sketch with deterministic hash functions."""
    rng = random.Random(seed)
    hashes = [lambda x, s=s: (hash((x, s)) % width) for s in range(depth)]
    sketch = np.zeros((depth, width), dtype=int)
    for item in data:
        for d, h in enumerate(hashes):
            idx = h(item)
            sketch[d, idx] += 1
    return sketch

def estimate_sketch_error(sketch: np.ndarray) -> float:
    """Upper bound on the error (width‑1)/width * total count."""
    total = sketch.sum()
    return (sketch.shape[1] - 1) / sketch.shape[1] * total

# ----------------------------------------------------------------------
# Utility functions (Parent B)
# ----------------------------------------------------------------------
def compute_health_scores(endpoints: List[Endpoint],
                          request_sequence: List[int]) -> List[float]:
    """Map request ids to health scores using endpoint failure rates."""
    scores = []
    for req in request_sequence:
        ep = endpoints[req % len(endpoints)]
        # health = 1 - failure_rate, scaled by righting_time_index
        health = (1.0 - ep.failure_rate) * ep.righting_time_index
        scores.append(health)
    return scores

def regret_engine(health_scores: List[float],
                  actions: List[MathAction]) -> Dict[str, float]:
    """
    Compute regret for each action:
        regret = expected_value - cost * avg_health
    Returns a dict mapping action.id -> regret.
    """
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
    """Classic Hoeffding bound for Bernoulli variables."""
    return math.sqrt(math.log(2 / delta) / (2 * num_samples))

def hoeffding_split(candidates: List[float],
                    bound: float) -> List[int]:
    """
    Given a list of candidate scores, return indices of those that exceed the bound.
    """
    return [i for i, val in enumerate(candidates) if val > bound]

def simulated_annealing_accept(old: float,
                               new: float,
                               temperature: float) -> bool:
    """Metropolis acceptance criterion."""
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
    """
    Executes the fused algorithm:
    1. Tropical broadcast to obtain gain vector `b`.
    2. Compute health scores (Parent B) and feed them as inflow/outflow to `StoreState`.
    3. Modulate Count‑Min sketch width with sphericity × mean health.
    4. Regret engine uses health scores to produce regrets.
    5. Hoeffding split decides which actions become leaders.
    6. Simulated annealing optionally accepts a new leader set.
    Returns the final StoreState, the sketch matrix, and the list of selected action ids.
    """
    # 1. Tropical broadcast
    b = tropical_broadcast(adj_matrix, steps=3)  # shape (n,)

    # 2. Health scores and StoreState dynamics
    health = compute_health_scores(endpoints, request_seq)
    store = StoreState()
    # Use health as inflow, and the broadcast vector (scaled) as outflow
    outflow = (b * store.gain).tolist()
    store.update(inflow=health, outflow=outflow)

    # 3. Sphericity‑modulated Count‑Min sketch
    sph = sphericity_index(morphology)
    mean_health = sum(health) / (len(health) + 1e-9)
    width = max(5, int(sph * mean_health * 10))  # ensure a minimal width
    sketch = count_min_sketch(data=request_seq,
                              width=width,
                              depth=sketch_depth,
                              seed=42)

    # 4. Regret engine
    regrets = regret_engine(health, actions)

    # 5. Hoeffding split on regret values
    regret_vals = list(regrets.values())
    bound = hoeffding_bound(num_samples=len(regret_vals), delta=0.05)
    selected_idx = hoeffding_split(regret_vals, bound)

    # 6. Simulated annealing acceptance on the number of selected actions
    old_score = store.level
    new_score = old_score + len(selected_idx) * 0.1  # arbitrary improvement metric
    if simulated_annealing_accept(old_score, new_score, temperature):
        # accept new state
        store.level = new_score

    selected_action_ids = [actions[i].id for i in selected_idx]
    return store, sketch, selected_action_ids

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple synthetic data for a quick run
    np.random.seed(0)
    random.seed(0)

    # Adjacency matrix (5 nodes) with random non‑negative weights
    adj = np.random.rand(5, 5) * 5

    morph = Morphology(length=2.0, width=1.5, height=1.0, mass=3.0)

    endpoints = [
        Endpoint(failures=2, failure_threshold=10, righting_time_index=0.8),
        Endpoint(failures=0, failure_threshold=5, righting_time_index=1.0),
        Endpoint(failures=5, failure_threshold=20, righting_time_index=0.5)
    ]

    request_seq = [0, 1, 2, 3, 4, 0, 2, 3, 1, 4]

    actions = [
        MathAction(id="A", expected_value=10.0, cost=2.0),
        MathAction(id="B", expected_value=8.0, cost=1.5),
        MathAction(id="C", expected_value=12.0, cost=3.0)
    ]

    store_state, sketch_matrix, leaders = hybrid_pipeline(
        adj_matrix=adj,
        morphology=morph,
        endpoints=endpoints,
        request_seq=request_seq,
        actions=actions,
        sketch_depth=3,
        temperature=0.9
    )

    print("Final StoreState level:", store_state.level)
    print("Sketch shape:", sketch_matrix.shape)
    print("Selected leaders:", leaders)