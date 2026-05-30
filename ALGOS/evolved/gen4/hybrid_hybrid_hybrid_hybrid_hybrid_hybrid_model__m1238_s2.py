# DARWIN HAMMER — match 1238, survivor 2
# gen: 4
# parent_a: hybrid_hybrid_hybrid_privac_hybrid_endpoint_circ_m28_s5.py (gen3)
# parent_b: hybrid_hybrid_model_vram_sc_hybrid_krampus_brain_m3_s2.py (gen2)
# born: 2026-05-29T23:34:44Z

"""
Hybrid Algorithm: Fusing Reconstruction Risk Score and Ollivier-Ricci Curvature

This module fuses two parent algorithms:

* **Parent A** – `hybrid_hybrid_hybrid_privac_hybrid_endpoint_circ_m28_s5.py` 
  provides `reconstruction_risk_score` that estimates the probability of record re-identification.
* **Parent B** – `hybrid_hybrid_model_vram_sc_hybrid_krampus_brain_m3_s2.py` 
  computes Ollivier-Ricci curvature on a graph.

The mathematical bridge between the two algorithms lies in interpreting the 
reconstruction risk score as a node attribute in the graph, influencing the 
lazy random-walk distribution for Ollivier-Ricci curvature computation. 
The curvature values reflect the record re-identification risk landscape.

The hybrid system integrates:

μ_i(v) = α·r_i·δ_{i=v} + (1-α)·r_i·(1/deg(i))·∑_{u∈N(i)} δ_{u=v}

where `r_i` is the normalised reconstruction risk score of node *i*. 
The curvature computed from these weighted measures is used to evaluate 
the risk associated with a set of records.
"""

import numpy as np
from dataclasses import dataclass
from datetime import datetime, timezone
from math import exp
from pathlib import Path
from typing import Any, Dict, Iterable, List

# ----------------------------------------------------------------------
# Shared data structures
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class Record:
    """Lightweight descriptor for a record."""
    quasi_identifiers: int
    total_records: int

# ----------------------------------------------------------------------
# Parent A – reconstruction risk score
# ----------------------------------------------------------------------

def reconstruction_risk_score(record: Record) -> float:
    """Probability that a record can be re-identified."""
    if record.total_records <= 0:
        return 0.0
    return max(0.0, min(1.0, record.quasi_identifiers / record.total_records))

# ----------------------------------------------------------------------
# Parent B – Ollivier-Ricci curvature
# ----------------------------------------------------------------------

def lazy_random_walk_distribution(alpha: float, 
                                 weights: List[float], 
                                 degree: int) -> List[float]:
    """Lazy random-walk distribution for Ollivier-Ricci curvature."""
    n = len(weights)
    distribution = [alpha * weights[i] if i == j else (1 - alpha) * weights[i] / degree for i in range(n) for j in range(n) if i == j]
    return [x / sum(distribution) for x in distribution]

def ollivier_ricci_curvature(alpha: float, 
                             weights: List[float], 
                             graph: Dict[int, List[int]]) -> List[float]:
    """Ollivier-Ricci curvature on a graph."""
    curvatures = []
    for node in graph:
        degree = len(graph[node])
        distribution = lazy_random_walk_distribution(alpha, weights, degree)
        curvature = 1 - (1 / degree) * sum([distribution[j] for j in graph[node]])
        curvatures.append(curvature)
    return curvatures

# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------

def hybrid_risk_curvature(alpha: float, 
                           records: List[Record], 
                           graph: Dict[int, List[int]]) -> List[float]:
    """Hybrid function integrating reconstruction risk score and Ollivier-Ricci curvature."""
    weights = [reconstruction_risk_score(record) for record in records]
    return ollivier_ricci_curvature(alpha, weights, graph)

def evaluate_risk_curvature(records: List[Record], 
                            graph: Dict[int, List[int]]) -> List[float]:
    """Evaluate risk curvature for a set of records."""
    alpha = 0.5
    return hybrid_risk_curvature(alpha, records, graph)

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------

if __name__ == "__main__":
    records = [Record(10, 100), Record(20, 100), Record(30, 100)]
    graph = {0: [1, 2], 1: [0, 2], 2: [0, 1]}
    curvatures = evaluate_risk_curvature(records, graph)
    print(curvatures)