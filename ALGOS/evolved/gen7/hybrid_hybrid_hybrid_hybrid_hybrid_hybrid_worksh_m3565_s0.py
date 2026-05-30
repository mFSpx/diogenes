# DARWIN HAMMER — match 3565, survivor 0
# gen: 7
# parent_a: hybrid_hybrid_hybrid_sheaf__hybrid_hybrid_hybrid_m2030_s3.py (gen6)
# parent_b: hybrid_hybrid_workshare_all_hybrid_krampus_brain_m23_s4.py (gen2)
# born: 2026-05-29T23:50:44Z

"""
Hybrid Sheaf-Cohomology & Regret-Weighted Workshare Allocator.

Parents:
- hybrid_hybrid_hybrid_sheaf__hybrid_hybrid_hybrid_m2030_s3.py
- hybrid_hybrid_workshare_all_hybrid_krampus_brain_m23_s4.py

Mathematical bridge:
The mathematical bridge between the two parents lies in the application of 
regret-weighted decision-making to the workshare allocation problem. By 
introducing a regret-weighted strategy to the workshare allocation, we can 
refine the deterministic allocation with a stochastic, feature-driven 
approach. The hybrid system treats the deterministic allocation as a 
baseline and refines the stochastic, LLM-driven share with a mathematically 
grounded feature weighting, achieving a single unified allocation routine.

The sheaf module builds a coboundary matrix Δ from restriction scalars on graph 
edges. The decision module supplies a *regret-weighted strategy* – a normalized 
scalar weight for each node – and a scalar epistemic-certainty factor. By 
scaling each edge’s restriction map with the average regret-weight of its 
incident nodes we obtain a *weighted sheaf*. The same exponential pruning 
probability `p(t)=λ·exp(-α·t)` is then applied to edges, yielding a pruned 
coboundary matrix whose null-space dimension reflects both topological 
connectivity and the underlying decision-theoretic information.

The workshare allocation is then performed using the weighted sheaf, where the 
regret-weighted strategy is used to distribute the workshare among the model 
groups.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Decision-theoretic data structures (from Parent A)
# ----------------------------------------------------------------------
EPISTEMIC_FLAGS = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")
_EPISTEMIC_FACTOR = {
    "FACT": 1.0,
    "PROBABLE": 0.8,
    "POSSIBLE": 0.5,
    "BULLSHIT": 0.2,
    "SURE_MAYBE": 0.6,
}

@dataclass(frozen=True)
class Action:
    """Action with cost, baseline probability, and epistemic certainty."""
    cost: float
    probability: float
    epistemic_certainty: str

    def epistemic_factor(self) -> float:
        """Map certainty flag to a scalar in [0,1]."""
        return _EPISTEMIC_FACTOR[self.epistemic_certainty]

# ----------------------------------------------------------------------
# Constants & helpers (shared)
# ----------------------------------------------------------------------
GROUPS = ("codex", "groq", "cohere", "local_models")

def _pct(value: float) -> float:
    """Round a float to six decimal places."""
    return round(float(value), 6)

def doomsday(year: int, month: int, day: int) -> int:
    """
    Return the Doomsday weekday index for a given Gregorian date.
    Monday → 0, …, Sunday → 6 (the original code used (weekday+1)%7).
    """
    return (date(year, month, day).weekday() + 1) % 7

# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------
def compute_regret_weights(actions: List[Action]) -> np.ndarray:
    """Turn a list of `Action`s into normalized regret-weighted node scalars."""
    regret_weights = np.array([action.cost * action.probability * action.epistemic_factor() for action in actions])
    return regret_weights / np.sum(regret_weights)

def build_hybrid_sheaf(regret_weights: np.ndarray, edges: List[Tuple[int, int]]) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Construct the coboundary matrix from those scalars, prune edges with `p(t)`, and return the matrix together with the retained edge list."""
    coboundary_matrix = np.zeros((len(regret_weights), len(regret_weights)))
    for edge in edges:
        coboundary_matrix[edge[0], edge[1]] = regret_weights[edge[0]] + regret_weights[edge[1]]
    # Apply exponential pruning probability
    pruning_probability = np.exp(-0.1 * np.sum(coboundary_matrix, axis=0))
    pruned_edges = [edge for edge in edges if random.random() > pruning_probability[edge[0]]]
    pruned_coboundary_matrix = np.zeros((len(regret_weights), len(regret_weights)))
    for edge in pruned_edges:
        pruned_coboundary_matrix[edge[0], edge[1]] = regret_weights[edge[0]] + regret_weights[edge[1]]
    return pruned_coboundary_matrix, pruned_edges

def allocate_workshare_hybrid(regret_weights: np.ndarray, groups: List[str]) -> Dict[str, float]:
    """Allocate workshare using the regret-weighted strategy."""
    workshare_allocations = {}
    for group in groups:
        workshare_allocations[group] = np.sum([regret_weights[i] for i, action in enumerate(actions) if action.epistemic_certainty == group])
    return workshare_allocations

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    actions = [Action(1.0, 0.5, "FACT"), Action(2.0, 0.3, "PROBABLE"), Action(3.0, 0.2, "POSSIBLE")]
    regret_weights = compute_regret_weights(actions)
    edges = [(0, 1), (1, 2), (2, 0)]
    coboundary_matrix, pruned_edges = build_hybrid_sheaf(regret_weights, edges)
    workshare_allocations = allocate_workshare_hybrid(regret_weights, GROUPS)
    print(workshare_allocations)