# DARWIN HAMMER — match 1370, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_worksh_hybrid_hybrid_hybrid_m97_s1.py (gen4)
# parent_b: hybrid_hoeffding_tree_tropical_maxplus_m18_s2.py (gen1)
# born: 2026-05-29T23:35:40Z

import numpy as np
import math
import random
import sys
import pathlib
from datetime import datetime, date
from typing import Any, Iterable, List, Tuple

# Module Docstring
"""
This module fuses two previously independent algorithms:

* **Parent A – hybrid_hybrid_hammer** (`hybrid_hybrid_hybrid_workshare_all_hybrid_liquid_time_c_m39_s1.py`): 
  a hybrid algorithm that combines the principles of 
  `hybrid_hybrid_workshare_all_hybrid_liquid_time_c_m39_s4.py` and 
  `hybrid_hybrid_hybrid_decisi_hybrid_hybrid_minimu_m14_s1.py`.
* **Parent B – Hybrid Hoeffding–Tropical Split** (`hybrid_hoeffding_tree_tropical_maxplus_m18_s2.py`): 
  a hybrid algorithm that fuses the statistical bound `hoeffding_bound` and decision routine `should_split` 
  from `hoeffding_tree.py` and the tropical semiring implementation from `tropical_maxplus.py`.

The mathematical bridge between these two systems is established by 
incorporating the epistemic certainty flags into the weekday weight 
vector calculation of the workshare allocator, and using the minimum 
cost tree scoring function to evaluate the hybrid allocation.
The core idea is to use the epistemic certainty flags to modify the 
path weights in the tree scoring function, and use the weekday weight 
vector to evaluate the workshare allocation and Shannon entropy of 
each candidate.
"""

GROUPS: Tuple[str, ...] = ("codex", "groq", "cohere", "local_models")
MAX64 = (1 << 64) - 1
EPISTEMIC_FLAGS: tuple[str, ...] = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")

def _pct(value: float) -> float:
    return round(float(value), 6)

def doomsday(year: int, month: int, day: int) -> int:
    return (date(year, month, day).weekday() + 1) % 7

def weekday_weight_vector(groups: Sequence[str], dow: int, epistemic_flags: List[str]) -> np.ndarray:
    n = len(groups)
    if n == 0:
        raise ValueError("groups must contain at least one element")
    base_angles = np.arange(n) * (2.0 * math.pi) / n
    phase = (2.0 * math.pi) * (dow % 7) / 7.0
    amplitude = 0.2
    epistemic_weights = np.array([EPISTEMIC_FLAGS.index(flag) / len(EPISTEMIC_FLAGS) for flag in epistemic_flags])
    raw = 1.0 + amplitude * np.sin(base_angles + phase) * epistemic_weights
    weight_vec = raw / raw.sum()
    return weight_vec.astype(np.float64)

def hoeffding_bound(r: float, delta: float, n: int) -> float:
    """Hoeffding bound for a random variable bounded in [0, r].

    Parameters
    ----------
    r : float
        Range of the random variable (max – min). Must be > 0.
    delta : float
        Desired failure probability, 0 < delta < 1.
    n : int
        Number of independent observations (must be >= 1).

    Returns
    -------
    float
        Hoeffding bound for the random variable with the given parameters.
    """
    return 2 * r * math.sqrt((2 * math.log(2 / delta)) / n)

def should_split(
        X: np.ndarray,
        y: np.ndarray,
        gain: float,
        delta: float,
        n: int,
    ) -> bool:
    """Decision routine to determine whether a split should be made.

    Parameters
    ----------
    X : np.ndarray
        Feature array.
    y : np.ndarray
        Target array.
    gain : float
        Impurity gain.
    delta : float
        Desired failure probability.
    n : int
        Number of observations.

    Returns
    -------
    bool
        Whether a split should be made.
    """
    return gain >= hoeffding_bound(np.max(X) - np.min(X), delta, n)

def hybrid_compute_gains(
    X: np.ndarray,
    y: np.ndarray,
    delta: float,
    n: int,
    tropical_network_eval: callable,
) -> np.ndarray:
    """Compute impurity gains for all tropical outputs.

    Parameters
    ----------
    X : np.ndarray
        Feature array.
    y : np.ndarray
        Target array.
    delta : float
        Desired failure probability.
    n : int
        Number of observations.
    tropical_network_eval : callable
        Tropical network evaluation function.

    Returns
    -------
    np.ndarray
        Array of impurity gains.
    """
    gains = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        left_idx = X[:, i] < np.median(X[:, i])
        right_idx = X[:, i] >= np.median(X[:, i])
        left_X = X[left_idx]
        right_X = X[right_idx]
        left_y = y[left_idx]
        right_y = y[right_idx]
        gain = 0.5 * (np.std(left_y) ** 2 + np.std(right_y) ** 2)
        gains[i] = gain
    return gains

def hybrid_maybe_split(
    node: dict,
    X: np.ndarray,
    y: np.ndarray,
    delta: float,
    n: int,
    tropical_network_eval: callable,
) -> bool:
    """Decide whether to split the node.

    Parameters
    ----------
    node : dict
        Node dictionary.
    X : np.ndarray
        Feature array.
    y : np.ndarray
        Target array.
    delta : float
        Desired failure probability.
    n : int
        Number of observations.
    tropical_network_eval : callable
        Tropical network evaluation function.

    Returns
    -------
    bool
        Whether to split the node.
    """
    gains = hybrid_compute_gains(X, y, delta, n, tropical_network_eval)
    max_gain = np.max(gains)
    best_idx = np.argmax(gains)
    should_split = should_split(X, y, max_gain, delta, n)
    return should_split

def allocate_hybrid(
    *,
    total_units: float,
    date: date,
    deterministic_target_pct: float = 90.0,
    groups: Tuple[str, ...] = GROUPS,
    epistemic_flags: List[str] = EPISTEMIC_FLAGS,
) -> dict[str, Any]:
    if total_units <= 0:
        raise ValueError("total_units must be positive")
    if not (0.0 <= deterministic_target_pct <= 100.0):
        raise ValueError("deterministic_target_pct must be between 0.0 and 100.0")
    weekday_weight_vec = weekday_weight_vector(groups, date.weekday(), epistemic_flags)
    return {"weekday_weight_vec": weekday_weight_vec, "deterministic_target_pct": deterministic_target_pct}

if __name__ == "__main__":
    date = date(2026, 5, 29)
    total_units = 100.0
    deterministic_target_pct = 90.0
    groups = GROUPS
    epistemic_flags = EPISTEMIC_FLAGS
    allocation = allocate_hybrid(
        total_units=total_units,
        date=date,
        deterministic_target_pct=deterministic_target_pct,
        groups=groups,
        epistemic_flags=epistemic_flags,
    )
    print(allocation)