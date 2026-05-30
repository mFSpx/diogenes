# DARWIN HAMMER — match 1635, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_hybrid_bandit_hybrid_voronoi_parti_m169_s0.py (gen3)
# parent_b: hybrid_hybrid_hybrid_endpoi_hybrid_hybrid_rbf_su_m726_s2.py (gen4)
# born: 2026-05-29T23:37:51Z

"""
Module for the hybrid algorithm that combines the bandit router and Voronoi partition 
from hybrid_hybrid_hybrid_bandit_hybrid_voronoi_parti_m169_s0.py with the 
morphology and surrogate modeling from hybrid_hybrid_hybrid_endpoi_hybrid_hybrid_rbf_su_m726_s2.py.
The mathematical bridge between these two structures is the use of Euclidean distance 
in both the Voronoi partition and the RBF surrogate model, which allows for a unified 
treatment of geometric and functional relationships.

Parents:
- hybrid_hybrid_hybrid_bandit_hybrid_voronoi_parti_m169_s0.py
- hybrid_hybrid_hybrid_endpoi_hybrid_hybrid_rbf_su_m726_s2.py
"""

import math
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np

# ----------------------------------------------------------------------
# Bandit core
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str

@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float

_POLICY: Dict[str, List[float]] = {}
_STORE: Dict[str, float] = {}          # virtual VRAM store per key
DEFAULT_BUDGET_MB = 1024 * 4  # Assuming 4GB as default budget

def reset_policy() -> None:
    """Clear all learned statistics and the virtual store."""
    _POLICY.clear()
    _STORE.clear()

def _reward(a: str) -> float:
    total, n = _POLICY.get(a, [0.0, 0.0])
    return total / n if n else 0.0

# ----------------------------------------------------------------------
# Voronoi helpers
# ----------------------------------------------------------------------
Point = Tuple[float, float]

def euclidean_distance(a: Point, b: Point) -> float:
    """Standard Euclidean distance in ℝ²."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

# ----------------------------------------------------------------------
# Morphology and surrogate modeling
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / max(length, width, height)

def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)

def rbf_surrogate(distance: float, epsilon: float = 1.0) -> float:
    return math.exp(-((epsilon * distance) ** 2))

# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------
def hybrid_action_selection(context: Point, actions: List[BanditAction]) -> BanditAction:
    distances = [euclidean_distance(context, (float(action.action_id.split('_')[0]), float(action.action_id.split('_')[1]))) for action in actions]
    surrogate_values = [rbf_surrogate(dist) for dist in distances]
    selected_action_index = np.argmax(surrogate_values)
    return actions[selected_action_index]

def hybrid_morphology_evaluation(morphology: Morphology, context: Point) -> float:
    distance = euclidean_distance((morphology.length, morphology.width), context)
    return rbf_surrogate(distance) * recovery_priority(morphology)

def hybrid_bandit_update(context: Point, action: BanditAction, reward: float) -> None:
    update = BanditUpdate(context_id=' '.join(map(str, context)), action_id=action.action_id, reward=reward, propensity=action.propensity)
    if update.context_id not in _POLICY:
        _POLICY[update.context_id] = [0.0, 0.0]
    _POLICY[update.context_id][0] += reward
    _POLICY[update.context_id][1] += 1

if __name__ == "__main__":
    context = (1.0, 2.0)
    actions = [BanditAction(action_id='1_2', propensity=0.5, expected_reward=10.0, confidence_bound=0.1, algorithm='epsilon_greedy')]
    selected_action = hybrid_action_selection(context, actions)
    print(selected_action.action_id)

    morphology = Morphology(length=10.0, width=5.0, height=2.0, mass=1.0)
    evaluation = hybrid_morphology_evaluation(morphology, context)
    print(evaluation)

    hybrid_bandit_update(context, actions[0], 10.0)
    print(_POLICY)