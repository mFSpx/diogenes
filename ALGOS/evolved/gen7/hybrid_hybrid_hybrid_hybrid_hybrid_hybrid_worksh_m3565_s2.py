# DARWIN HAMMER — match 3565, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_sheaf__hybrid_hybrid_hybrid_m2030_s3.py (gen6)
# parent_b: hybrid_hybrid_workshare_all_hybrid_krampus_brain_m23_s4.py (gen2)
# born: 2026-05-29T23:50:44Z

"""
Hybrid Sheaf‑Cohomology & Regret‑Weighted Workshare Allocator.

This module fuses the *Hybrid Sheaf‑Cohomology & Regret‑Weighted Decision Engine*
(parent: `hybrid_hybrid_hybrid_sheaf__hybrid_hybrid_hybrid_m2030_s3.py`) and the
*Hybrid Workshare‑Feature Allocator* (parent: `hybrid_hybrid_workshare_all_hybrid_krampus_brain_m23_s4.py`).

The mathematical bridge between the two parents lies in their treatment of
stochastic information. The sheaf module constructs a weighted coboundary matrix
`Δ` from regret‑weighted node scalars and a pruning policy. The workshare
allocator uses a feature‑driven weight vector to distribute units among model
groups. By interpreting the feature weights as a probability distribution over
the nodes of the sheaf, we can use the regret‑weighted node scalars to modulate
the feature weights, effectively fusing the decision-theoretic and
feature-driven components.

The hybrid system achieves a unified allocation routine that accounts for both
topological connectivity and feature-driven information.

Imports:
- numpy
- standard library: math, random, sys, pathlib
"""

import math
import random
import sys
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
# Constants & helpers (shared with Parent B)
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

def compute_hybrid_weights(actions: List[Action], features: np.ndarray) -> np.ndarray:
    """
    Compute hybrid weights by modulating feature weights with regret‑weighted node scalars.

    Args:
    - actions (List[Action]): List of actions with cost, probability, and epistemic certainty.
    - features (np.ndarray): 24-dimensional feature vector.

    Returns:
    - hybrid_weights (np.ndarray): Modulated feature weights.
    """
    regret_weights = compute_regret_weights(actions)
    feature_weights = features / np.sum(features)  # normalize feature weights
    hybrid_weights = regret_weights * feature_weights
    return hybrid_weights / np.sum(hybrid_weights)  # normalize hybrid weights

def build_hybrid_sheaf(actions: List[Action], features: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Construct the weighted coboundary matrix Δ and prune edges using the hybrid weights.

    Args:
    - actions (List[Action]): List of actions with cost, probability, and epistemic certainty.
    - features (np.ndarray): 24-dimensional feature vector.

    Returns:
    - Δ (np.ndarray): Weighted coboundary matrix.
    - retained_edges (List[Tuple[int, int]]): List of retained edges.
    """
    hybrid_weights = compute_hybrid_weights(actions, features)
    Δ, retained_edges = build_sheaf(hybrid_weights)
    return Δ, retained_edges

def hybrid_nullspace_dimension(Δ: np.ndarray) -> int:
    """
    Compute the nullspace dimension of the weighted coboundary matrix Δ.

    Args:
    - Δ (np.ndarray): Weighted coboundary matrix.

    Returns:
    - dim ker(Δ) (int): Nullspace dimension.
    """
    U, s, Vh = np.linalg.svd(Δ, full_matrices=False)
    return np.sum(s < 1e-6)

def allocate_hybrid_workshare(actions: List[Action], features: np.ndarray) -> Dict[str, float]:
    """
    Allocate workshare units using the hybrid weights.

    Args:
    - actions (List[Action]): List of actions with cost, probability, and epistemic certainty.
    - features (np.ndarray): 24-dimensional feature vector.

    Returns:
    - workshare_allocation (Dict[str, float]): Dictionary with workshare allocation for each group.
    """
    hybrid_weights = compute_hybrid_weights(actions, features)
    deterministic_units = compute_deterministic_units()
    llm_units = 1 - deterministic_units
    workshare_allocation = {}
    for i, group in enumerate(GROUPS):
        workshare_allocation[group] = llm_units * hybrid_weights[i] + deterministic_units * (1 / len(GROUPS))
    return workshare_allocation

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def compute_regret_weights(actions: List[Action]) -> np.ndarray:
    """
    Compute regret‑weighted node scalars.

    Args:
    - actions (List[Action]): List of actions with cost, probability, and epistemic certainty.

    Returns:
    - regret_weights (np.ndarray): Regret‑weighted node scalars.
    """
    # implementation from Parent A
    costs = np.array([action.cost for action in actions])
    probabilities = np.array([action.probability for action in actions])
    epistemic_factors = np.array([action.epistemic_factor() for action in actions])
    regret_weights = costs * probabilities * epistemic_factors
    return regret_weights / np.sum(regret_weights)

def build_sheaf(weights: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Construct the weighted coboundary matrix Δ and prune edges.

    Args:
    - weights (np.ndarray): Weights for the sheaf.

    Returns:
    - Δ (np.ndarray): Weighted coboundary matrix.
    - retained_edges (List[Tuple[int, int]]): List of retained edges.
    """
    # implementation from Parent A
    num_nodes = len(weights)
    Δ = np.zeros((num_nodes, num_nodes))
    retained_edges = []
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if random.random() < 0.5:  # pruning probability
                Δ[i, j] = weights[i] * weights[j]
                retained_edges.append((i, j))
    return Δ, retained_edges

def compute_deterministic_units() -> float:
    """
    Compute deterministic units.

    Returns:
    - deterministic_units (float): Deterministic units.
    """
    # implementation from Parent B
    today = date.today()
    doomsday_index = doomsday(today.year, today.month, today.day)
    return (1 + doomsday_index / 7)

if __name__ == "__main__":
    actions = [Action(1.0, 0.5, "FACT"), Action(2.0, 0.3, "PROBABLE"), Action(3.0, 0.2, "POSSIBLE")]
    features = np.random.rand(24)
    Δ, retained_edges = build_hybrid_sheaf(actions, features)
    print("Nullspace dimension:", hybrid_nullspace_dimension(Δ))
    workshare_allocation = allocate_hybrid_workshare(actions, features)
    print("Workshare allocation:", workshare_allocation)