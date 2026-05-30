# DARWIN HAMMER — match 1133, survivor 3
# gen: 4
# parent_a: hybrid_privacy_model_pool_m7_s2.py (gen1)
# parent_b: hybrid_hybrid_semantic_neig_hybrid_hybrid_geomet_m116_s1.py (gen3)
# born: 2026-05-29T23:33:02Z

import math
import numpy as np
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Iterable, List, Dict, Set, Tuple

@dataclass
class Model:
    ram_consumption: float
    tier: int

def reconstruction_risk_score(unique_quasi_identifiers: int, total_records: int) -> float:
    return 0.0 if total_records <= 0 else max(0.0, min(1.0, unique_quasi_identifiers / total_records))

def anonymize_for_indexing(record: dict[str, Any], redact_keys: Set[str] | None = None) -> dict[str, Any]:
    redact = redact_keys or {'email', 'phone', 'ssn', 'secret', 'token', 'password'}
    return {k: ('<redacted>' if k.lower() in redact else v) for k, v in record.items()}

def dp_aggregate(values: Iterable[float], epsilon: float = 1.0, sensitivity: float = 1.0) -> float:
    return sum(values)

def model_resource_matrix(models: List[Model], ram_ceiling: float, privacy_budget: float, 
                          alpha: float = 1.0) -> np.ndarray:
    A = np.zeros((len(models), 2))
    for i, model in enumerate(models):
        A[i, 0] = model.ram_consumption
        A[i, 1] = alpha * model.tier * reconstruction_risk_score(10, 100) 
    return A

def select_models_hybrid(models: List[Model], ram_ceiling: float, privacy_budget: float, 
                         alpha: float = 1.0) -> np.ndarray:
    A = model_resource_matrix(models, ram_ceiling, privacy_budget, alpha)
    x = np.zeros(len(models), dtype=int)
    for i in range(len(models)):
        if A[i, 0] <= ram_ceiling and A[i, 1] <= privacy_budget:
            x[i] = 1
    return x

def _cos(a, b):
    den = math.sqrt(sum(x*x for x in a)) * math.sqrt(sum(y*y for y in b))
    return 0.0 if den == 0 else sum(x*y for x, y in zip(a, b)) / den

def pheromone_probabilities(pheromones):
    total = sum(pheromones)
    return [p / total for p in pheromones]

def entropy(probabilities, eps=1e-12):
    total = sum(probabilities)
    if total <= 0:
        raise ValueError('positive probability mass required')
    return -sum((p / total) * math.log(max(p / total, eps)) for p in probabilities if p > 0)

def expected_entropy(p_hit, hit_state, miss_state):
    if not 0 <= p_hit <= 1:
        raise ValueError('p_hit must be in [0,1]')
    return p_hit * entropy(hit_state) + (1.0 - p_hit) * entropy(miss_state)

Point = tuple[float, float]

def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def nearest(point: Point, seeds: list[Point]) -> int:
    if not seeds:
        raise ValueError('seeds required')
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))

def geometric_privacy_risk(models: List[Model], points: List[Point], seeds: List[Point]) -> np.ndarray:
    A = model_resource_matrix(models, 100.0, 1.0)
    risks = np.zeros((len(points), len(models)))
    for i, point in enumerate(points):
        nearest_seed_idx = nearest(point, seeds)
        for j, model in enumerate(models):
            risks[i, j] = A[j, 1] * _cos(np.array(point), np.array(seeds[nearest_seed_idx]))
    return risks

def hybrid_select_models(models: List[Model], points: List[Point], seeds: List[Point], 
                         ram_ceiling: float, privacy_budget: float) -> np.ndarray:
    risks = geometric_privacy_risk(models, points, seeds)
    x = np.zeros(len(models), dtype=int)
    for i in range(len(models)):
        if np.mean(risks[:, i]) <= privacy_budget and models[i].ram_consumption <= ram_ceiling:
            x[i] = 1
    return x

def hybrid_privacy_semantic(models: List[Model], points: List[Point], seeds: List[Point], 
                            ram_ceiling: float, privacy_budget: float) -> Tuple[np.ndarray, np.ndarray]:
    x = hybrid_select_models(models, points, seeds, ram_ceiling, privacy_budget)
    risks = geometric_privacy_risk(models, points, seeds)
    return x, risks

def improved_hybrid_privacy_semantic(models: List[Model], points: List[Point], seeds: List[Point], 
                            ram_ceiling: float, privacy_budget: float, alpha: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    A = model_resource_matrix(models, ram_ceiling, privacy_budget, alpha)
    risks = geometric_privacy_risk(models, points, seeds)
    x = np.zeros(len(models), dtype=int)
    for i in range(len(models)):
        if np.mean(risks[:, i]) <= privacy_budget and A[i, 0] <= ram_ceiling:
            x[i] = 1
    return x, risks

if __name__ == "__main__":
    models = [Model(10.0, 2), Model(20.0, 3), Model(30.0, 1)]
    points = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]
    seeds = [(0.0, 0.0), (10.0, 10.0)]
    ram_ceiling = 50.0
    privacy_budget = 1.0
    x, risks = improved_hybrid_privacy_semantic(models, points, seeds, ram_ceiling, privacy_budget)
    print(x)
    print(risks)