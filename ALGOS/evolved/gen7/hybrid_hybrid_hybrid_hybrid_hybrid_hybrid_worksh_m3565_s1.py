# DARWIN HAMMER — match 3565, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_sheaf__hybrid_hybrid_hybrid_m2030_s3.py (gen6)
# parent_b: hybrid_hybrid_workshare_all_hybrid_krampus_brain_m23_s4.py (gen2)
# born: 2026-05-29T23:50:44Z

"""
Hybrid Regret-Weighted Sheaf & Workshare Allocator

This module fuses the *hybrid_hybrid_hybrid_sheaf__hybrid_hybrid_hybrid_m2030_s3.py* 
and the *hybrid_hybrid_workshare_all_hybrid_krampus_brain_m23_s4.py*.

Mathematical bridge:
The mathematical interface between the two parents lies in their treatment of 
stochastic information. The regret-weighted sheaf module computes a weighted 
coboundary matrix Δ, where edge weights are determined by the average regret 
of incident nodes. Similarly, the workshare allocator module uses a feature 
vector to derive a probability distribution for allocating units among model 
groups. By integrating these two information streams, we create a hybrid 
system that allocates units based on both regret-weighted sheaf topology and 
feature-driven residuals.

The governing equations of the hybrid system are:

1. Compute regret-weighted node scalars `r_i` using the decision-theoretic 
   module from Parent A.
2. Construct a weighted coboundary matrix Δ using the regret-weighted 
   scalars and prune edges with a probability `p(t) = λ * exp(-α * t)`.
3. Compute the deterministic units `deterministic_units` using the 
   doomsday weekday factor.
4. Compute the feature-driven residual units `llm_units` using the 
   Krampus/Ollivier-Ricci feature vector.
5. Combine the deterministic and feature-driven units using the weighted 
   coboundary matrix Δ to produce a unified allocation.

"""

import math
import random
import sys
from datetime import date
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Decision‑theoretic data structures (from Parent A)
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

# ---------------------------------------------------------------------------
# Constants & helpers (shared)
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Hybrid Functions
# ---------------------------------------------------------------------------

def compute_regret_weights(actions: List[Action]) -> np.ndarray:
    """Compute regret-weighted node scalars."""
    costs = np.array([action.cost for action in actions])
    probabilities = np.array([action.probability for action in actions])
    epistemic_factors = np.array([action.epistemic_factor() for action in actions])
    regret_weights = costs * probabilities * epistemic_factors
    return regret_weights / np.sum(regret_weights)

def build_hybrid_sheaf(regret_weights: np.ndarray, 
                       num_nodes: int, 
                       num_edges: int, 
                       lambda_: float, 
                       alpha: float) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Construct the weighted coboundary matrix Δ and prune edges."""
    # Create a random graph with num_nodes and num_edges
    edges = [(i, j) for i in range(num_nodes) for j in range(i+1, num_nodes)]
    random.shuffle(edges)
    edges = edges[:num_edges]

    # Compute edge weights as average regret of incident nodes
    edge_weights = np.zeros(num_edges)
    for i, (node1, node2) in enumerate(edges):
        edge_weights[i] = (regret_weights[node1] + regret_weights[node2]) / 2

    # Prune edges with probability p(t) = λ * exp(-α * t)
    pruned_edges = []
    for i, edge in enumerate(edges):
        if random.random() > lambda_ * math.exp(-alpha * i):
            pruned_edges.append(edge)

    # Construct the weighted coboundary matrix Δ
    delta = np.zeros((num_nodes, num_nodes))
    for edge in pruned_edges:
        delta[edge[0], edge[1]] = edge_weights[edges.index(edge)]
        delta[edge[1], edge[0]] = edge_weights[edges.index(edge)]

    return delta, pruned_edges

def allocate_workshare_hybrid(delta: np.ndarray, 
                             deterministic_units: float, 
                             feature_vector: np.ndarray) -> Dict[str, float]:
    """Combine deterministic and feature-driven units using the weighted coboundary matrix Δ."""
    # Compute feature-driven residual units
    llm_units = np.dot(feature_vector, feature_vector) / np.sum(feature_vector)
    llm_units = {group: llm_units for group in GROUPS}

    # Combine deterministic and feature-driven units
    allocation = {}
    for group in GROUPS:
        allocation[group] = deterministic_units * delta[GROUPS.index(group), GROUPS.index(group)] + llm_units[group]

    return allocation

# ---------------------------------------------------------------------------
# Smoke Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    actions = [Action(1.0, 0.5, "FACT"), Action(2.0, 0.3, "PROBABLE"), Action(3.0, 0.2, "POSSIBLE")]
    regret_weights = compute_regret_weights(actions)
    delta, pruned_edges = build_hybrid_sheaf(regret_weights, 4, 6, 0.5, 0.1)
    deterministic_units = 10.0
    feature_vector = np.array([1.0, 2.0, 3.0, 4.0])
    allocation = allocate_workshare_hybrid(delta, deterministic_units, feature_vector)
    print(allocation)