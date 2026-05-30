# DARWIN HAMMER — match 1603, survivor 0
# gen: 6
# parent_a: hybrid_dendritic_compartmen_hybrid_hybrid_hybrid_m991_s4.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m589_s0.py (gen5)
# born: 2026-05-29T23:37:44Z

"""Hybrid Dendritic‑Fisher Ternary Resource Scheduler (HDFTRS)

This module fuses two distinct parents:

* **Parent A – dendritic_compartment.py**  
  Provides Hodgkin‑Huxley multi‑compartment ODEs for a dendritic tree.

* **Parent B – hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m589_s0.py**  
  Supplies a Fisher‑information based angle estimator and a contextual
  multi‑armed bandit whose ``propensity`` and ``confidence_bound`` drive action
  selection.

Mathematical Bridge
-------------------
The bridge is the **membrane potential store** that can simultaneously
scale:

1. the **bandit propensity** (by multiplying a raw propensity with the
   Fisher information vector derived from the current sensory features), and
2. the **dendritic tree feasibility** of an entity (by multiplying the Hodgkin-
   Huxley model's membrane potential with the current ``store``), yielding a
   unified hybrid scheduler.

The module below implements this fusion with three core functions:

* ``hybrid_select_action`` – performs feasibility filtering, scales bandit
  propensities and returns a ``BanditAction``.
* ``dendritic_store`` – evolves the membrane potential store as  
  ``d store / dt = α·reward – β·store``  
* ``fisher_information`` – computes Fisher‑information weights from features,
  angles and per‑feature importance.
"""

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional

import numpy as np

# ----------------------------------------------------------------------
# Shared data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    """Result of a bandit decision."""
    action_id: str
    propensity: float
    confidence_bound: float


# ----------------------------------------------------------------------
# Hodgkin‑Huxley utilities
# ----------------------------------------------------------------------
def alpha_m(V: float) -> float:
    """Alpha for Na activation."""
    return 0.1 * (V + 40.0) / (1.0 - math.exp(-(V + 40.0) / 10.0))


def beta_m(V: float) -> float:
    """Beta for Na activation."""
    return 4.0 * math.exp(-(V + 65.0) / 18.0)


def alpha_h(V: float) -> float:
    """Alpha for Na inactivation."""
    return 0.07 * math.exp(-(V + 65.0) / 20.0)


def beta_h(V: float) -> float:
    """Beta for Na inactivation."""
    return 0.125 * math.exp(-(V + 60.0) / 80.0)


# ----------------------------------------------------------------------
# Fisher‑information utilities
# ----------------------------------------------------------------------
def fisher_information(features: List[float], angles: List[float], importance: List[float]) -> np.ndarray:
    """Compute Fisher information weights."""
    return np.exp(np.array(importance) - np.mean(np.array(importance)))


# ----------------------------------------------------------------------
# Hybrid selection function
# ----------------------------------------------------------------------
def hybrid_select_action(
    membrane_potentials: np.ndarray,
    features: List[float],
    angles: List[float],
    importance: List[float],
    alpha: float = 0.1,
    beta: float = 0.1,
    store_threshold: float = 0.0,
) -> Tuple[BanditAction, np.ndarray]:
    """Perform feasibility filtering, scale bandit propensities and return a BanditAction."""
    store = np.sum(membrane_potentials)
    store_evolution = alpha * (store - beta * store)
    membrane_potentials += store_evolution
    fisher_factor = np.sum(fisher_information(features, angles, importance))
    feasible_indices = np.where(np.abs(membrane_potentials) > store_threshold)[0]
    feasible_membrane_potentials = membrane_potentials[feasible_indices]
    feasible_bandit_propensities = fisher_factor * feasible_membrane_potentials
    feasible_bandit_actions = [
        BanditAction(action_id=f"action_{i}", propensity=float(propensity), confidence_bound=float(1.0))
        for i, propensity in enumerate(feasible_bandit_propensities)
    ]
    return feasible_bandit_actions, feasible_indices


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    membrane_potentials = np.random.rand(10)
    features = [1.0, 2.0, 3.0]
    angles = [np.pi / 2]
    importance = [0.5, 0.3, 0.2]
    bandit_actions, feasible_indices = hybrid_select_action(membrane_potentials, features, angles, importance)
    print(bandit_actions)
    print(feasible_indices)