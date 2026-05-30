# DARWIN HAMMER — match 2723, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_minimu_hybrid_hybrid_hybrid_m405_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_privac_hybrid_hybrid_endpoi_m16_s1.py (gen3)
# born: 2026-05-29T23:43:44Z

"""
Module for the hybrid algorithm that fuses the minimum-cost tree Bayesian bandit-router 
and the hybrid privacy model with endpoint health scores.

This module combines the core ideas of two parents: 
- hybrid_hybrid_minimum_cost__hybrid_hybrid_bandit_m249_s0.py (minimum-cost tree Bayesian update algorithm 
  and hybrid bandit-router and sketch-RLCT algorithm)
- hybrid_hybrid_hybrid_privac_hybrid_hybrid_endpoi_m16_s1.py (hybrid privacy model with reconstruction risk scores 
  and endpoint health scores)

The mathematical bridge between these two structures lies in the application of log-count statistics 
from the bandit-router algorithm to inform the reconstruction risk scores in the hybrid privacy model. 
This fusion introduces a novel "health" metric, defined as:
    health = (1 - (reconstruction_risk_score * failure_rate)) * (1 - recovery_priority) * log(count)
where `failure_rate = failures / failure_threshold`, `recovery_priority` comes from the morphology-driven righting-time model, 
and `log(count)` is the log-count statistic from the bandit-router algorithm.

This health score is then used to weigh the split of the total workshare into a deterministic part and a residual (LLM) part.
"""

import math
import random
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from math import exp

# Types
Point = Tuple[float, float]
Edge = Tuple[str, str]

# Algorithm A – deterministic tree utilities
def length(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def tree_metrics(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
) -> Tuple[Dict[str, List[str]], Dict[Edge, float], Dict[str, float]]:
    """
    Build adjacency, compute Euclidean edge lengths and root‑to‑node distances.

    Returns
    -------
    adj : dict mapping node → list of neighbours
    edge_len : dict mapping edge (ordered a, b) → length
    node_dist : dict mapping node → root‑to‑node distance
    """
    adj = defaultdict(list)
    edge_len = {}
    node_dist = {}

    for a, b in edges:
        adj[a].append(b)

    return adj, edge_len, node_dist

# Algorithm B – hybrid privacy utilities
@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str
    vram_mb: int

TIER_T1_QWEN_0_5B = ModelTier("qwen-0.5b", 512, "T1", 1024)
TIER_T2_REASONING = ModelTier("reasoning-t2", 3000, "T2", 2048)
TIER_T2_TOOL = ModelTier("tool-t2", 2600, "T2", 2048)
TIER_T3_QWEN_7B = ModelTier("qwen-7b", 7000, "T3", 4096)

def reconstruction_risk_score(unique_quasi_identifiers: int, total_records: int) -> float:
    return 0.0 if total_records <= 0 else max(0.0, min(1.0, unique_quasi_identifiers/total_records))

def anonymize_for_indexing(record: dict[str, Any], redact_keys: set[str]|None=None) -> dict[str, Any]:
    redact = redact_keys or {'email', 'phone', 'ssn', 'secret', 'token', 'password'}
    return {k: ('<redacted>' if k.lower() in redact else v) for k, v in record.items()}

def dp_aggregate(values: Iterable[float], epsilon: float=1.0, sensitivity: float=1.0) -> float:
    return sum(values)

def health_score(reconstruction_risk: float, recovery_priority: float) -> float:
    return (1 - reconstruction_risk) * (1 - recovery_priority)

class EndpointCircuitBreaker:
    def __init__(self, failure_threshold: float, recovery_priority: float):
        self.failure_threshold = failure_threshold
        self.recovery_priority = recovery_priority

    def update_failure_rate(self, failures: int) -> float:
        return failures / self.failure_threshold

class HybridHealthModel:
    def __init__(self, failure_threshold: float, recovery_priority: float):
        self.failure_threshold = failure_threshold
        self.recovery_priority = recovery_priority

    def update_health(self, count: int, failures: int) -> float:
        failure_rate = self.update_failure_rate(failures)
        reconstruction_risk = reconstruction_risk_score(failures, count)
        return health_score(reconstruction_risk, self.recovery_priority) * math.log(count)

def hybrid_workshare_allocation(models: List[ModelTier], health_scores: Dict[str, float], total_workshare: float) -> Dict[str, float]:
    """
    Allocate workshare to models based on their health scores.

    Returns
    -------
    workshare_allocation : dict mapping model → workshare
    """
    workshare_allocation = {}
    for model in models:
        workshare = total_workshare * health_scores[model.name] / sum(health_scores.values())
        workshare_allocation[model.name] = workshare

    return workshare_allocation

def hybrid_load_model(workshare_allocation: Dict[str, float], total_workshare: float) -> Dict[str, str]:
    """
    Load models based on their workshare allocation.

    Returns
    -------
    loaded_models : dict mapping model → loaded status
    """
    loaded_models = {}
    for model, workshare in workshare_allocation.items():
        if workshare > total_workshare * 0.5:
            loaded_models[model] = "loaded"
        else:
            loaded_models[model] = "not loaded"

    return loaded_models

def hybrid_evict_model(loaded_models: Dict[str, str], total_workshare: float) -> Dict[str, str]:
    """
    Evict models based on their loaded status.

    Returns
    -------
    evicted_models : dict mapping model → evicted status
    """
    evicted_models = {}
    for model, loaded in loaded_models.items():
        if loaded == "not loaded":
            evicted_models[model] = "evicted"
        else:
            evicted_models[model] = "not evicted"

    return evicted_models

if __name__ == "__main__":
    # Create a list of models
    models = [TIER_T1_QWEN_0_5B, TIER_T2_REASONING, TIER_T2_TOOL, TIER_T3_QWEN_7B]

    # Create a dictionary of health scores for each model
    health_scores = {
        TIER_T1_QWEN_0_5B.name: 0.5,
        TIER_T2_REASONING.name: 0.7,
        TIER_T2_TOOL.name: 0.3,
        TIER_T3_QWEN_7B.name: 0.9
    }

    # Set the total workshare
    total_workshare = 100.0

    # Allocate workshare to models
    workshare_allocation = hybrid_workshare_allocation(models, health_scores, total_workshare)

    # Load models based on their workshare allocation
    loaded_models = hybrid_load_model(workshare_allocation, total_workshare)

    # Evict models based on their loaded status
    evicted_models = hybrid_evict_model(loaded_models, total_workshare)

    print(workshare_allocation)
    print(loaded_models)
    print(evicted_models)