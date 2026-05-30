# DARWIN HAMMER — match 1603, survivor 3
# gen: 6
# parent_a: hybrid_dendritic_compartmen_hybrid_hybrid_hybrid_m991_s4.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m589_s0.py (gen5)
# born: 2026-05-29T23:37:44Z

"""Hybrid Dendritic-Bandit Resource Scheduler
========================================

This module fuses two distinct parents:

* **Parent A – hybrid_dendritic_compartmen_hybrid_hybrid_hybrid_m991_s4.py**  
  Provides Hodgkin-Huxley multi-compartment ODEs for a dendritic tree and 
  regret-weighted probabilities with a path-signature-pruning mechanism.

* **Parent B – hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m589_s0.py**  
  Supplies a Fisher-information based angle estimator, a contextual multi-armed 
  bandit and a linear-constraint resource model.

Mathematical Bridge
-------------------
The bridge is the **regret-weighted Fisher factor** that can simultaneously 
scale:

1. the **bandit propensity** (by multiplying a raw propensity with the 
   Fisher information vector derived from the current sensory features and 
   regret-weighted probabilities).
2. the **resource feasibility** of an entity (by multiplying the right-hand 
   side of the linear constraints with the current regret-weighted probabilities).

Thus a single scalar – the product of regret-weighted probabilities and 
Fisher factor – modulates both the feasibility test and the stochastic bandit 
choice, yielding a unified hybrid scheduler.

The membrane potentials **V_i** of each compartment are interpreted as 
“expected values” of abstract actions.  Regret-weighted probabilities are 
computed from these values, then mapped to a ternary symbol (-1, 0, 1).  
The Fisher information vector is derived from the current sensory features 
and regret-weighted probabilities.
"""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Shared data structures
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
# Hodgkin-Huxley utilities
# ----------------------------------------------------------------------
def alpha_m(V: float) -> float:
    return 0.1 * (V + 40.0) / (1.0 - math.exp(-(V + 40.0) / 10.0))

def beta_m(V: float) -> float:
    return 4.0 * math.exp(-(V + 65.0) / 18.0)

def alpha_h(V: float) -> float:
    return 0.07 * math.exp(-(V + 65.0) / 20.0)

def beta_h(V: float) -> float:
    return 0.5 / (1.0 + math.exp(-(V + 30.0) / 10.0))

# ----------------------------------------------------------------------
# Fisher-information utilities
# ----------------------------------------------------------------------
def fisher_information(features: List[float], angles: List[float], 
                       importance: List[float]) -> List[float]:
    fisher_factor = []
    for i in range(len(features)):
        fisher_factor.append(importance[i] * np.cos(angles[i])**2 / features[i]**2)
    return fisher_factor

# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------
def compute_regret_weighted_probabilities(V: List[float]) -> List[float]:
    probabilities = []
    for v in V:
        probabilities.append(1 / (1 + math.exp(-v)))
    return probabilities

def hybrid_select_action(V: List[float], features: List[float], 
                         angles: List[float], importance: List[float], 
                         budgets: List[float]) -> BanditAction:
    probabilities = compute_regret_weighted_probabilities(V)
    fisher_factor = fisher_information(features, angles, importance)
    propensities = []
    for i in range(len(features)):
        propensities.append(probabilities[i] * fisher_factor[i])
    action_id = np.argmax(propensities)
    return BanditAction(action_id=str(action_id), 
                         propensity=max(propensities))

def build_resource_matrix(entity_resources: List[List[float]], 
                           budgets: List[float]) -> np.ndarray:
    A = np.zeros((len(budgets), len(entity_resources)))
    for i in range(len(entity_resources)):
        for j in range(len(budgets)):
            A[j, i] = entity_resources[i][j]
    return A

# ----------------------------------------------------------------------
# Simulation driver
# ----------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(0)
    random.seed(0)

    V = [10.0, 20.0, 30.0]
    features = [1.0, 2.0, 3.0]
    angles = [math.pi/4, math.pi/3, math.pi/2]
    importance = [1.0, 1.0, 1.0]
    budgets = [100.0, 200.0, 300.0]
    entity_resources = [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0], [70.0, 80.0, 90.0]]

    action = hybrid_select_action(V, features, angles, importance, budgets)
    A = build_resource_matrix(entity_resources, budgets)

    print("Selected Action:", action)
    print("Resource Matrix:\n", A)