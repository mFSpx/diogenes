# DARWIN HAMMER — match 1370, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_hybrid_worksh_hybrid_hybrid_hybrid_m97_s1.py (gen4)
# parent_b: hybrid_hoeffding_tree_tropical_maxplus_m18_s2.py (gen1)
# born: 2026-05-29T23:35:40Z

import numpy as np
import math
from datetime import date
from typing import Any, Iterable, List, Tuple

GROUPS: Tuple[str, ...] = ("codex", "groq", "cohere", "local_models")
MAX64 = (1 << 64) - 1
EPISTEMIC_FLAGS: tuple[str, ...] = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")

def _pct(value: float) -> float:
    return round(float(value), 6)

def doomsday(year: int, month: int, day: int) -> int:
    return (date(year, month, day).weekday() + 1) % 7

def weekday_weight_vector(groups: Iterable[str], dow: int, epistemic_flags: List[str]) -> np.ndarray:
    n = len(groups)
    if n == 0:
        raise ValueError("groups must contain at least one element")
    base_angles = np.arange(n) * (2.0 * math.pi) / n
    phase = (2.0 * math.pi) * (dow % 7) / 7.0
    amplitude = 0.2
    epistemic_weights = np.array([EPISTEMIC_FLAGS.index(flag) / len(EPISTEMIC_FLAGS) for flag in epistemic_flags])
    raw = 1.0 + amplitude * np.sin(base_angles + phase) * epistemic_weights[:, np.newaxis]
    weight_vec = raw / raw.sum(axis=0)
    return weight_vec.astype(np.float64)

def hoeffding_bound(r: float, delta: float, n: int) -> float:
    return 2 * r * math.sqrt((2 * math.log(2 / delta)) / n)

def should_split(
        X: np.ndarray,
        y: np.ndarray,
        gain: float,
        delta: float,
        n: int,
    ) -> bool:
    return gain >= hoeffding_bound(np.max(X) - np.min(X), delta, n)

def hybrid_compute_gains(
    X: np.ndarray,
    y: np.ndarray,
    delta: float,
    n: int,
) -> np.ndarray:
    gains = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        left_idx = X[:, i] < np.median(X[:, i])
        right_idx = X[:, i] >= np.median(X[:, i])
        left_X = X[left_idx]
        right_X = X[right_idx]
        left_y = y[left_idx]
        right_y = y[right_idx]
        left_std = np.std(left_y) if len(left_y) > 1 else 0
        right_std = np.std(right_y) if len(right_y) > 1 else 0
        gain = 0.5 * (left_std ** 2 + right_std ** 2)
        gains[i] = gain
    return gains

def hybrid_maybe_split(
    node: dict,
    X: np.ndarray,
    y: np.ndarray,
    delta: float,
    n: int,
) -> bool:
    gains = hybrid_compute_gains(X, y, delta, n)
    max_gain = np.max(gains)
    best_idx = np.argmax(gains)
    should_split = should_split(X[:, best_idx], y, max_gain, delta, n)
    return should_split

def allocate_hybrid(
    *,
    total_units: float,
    date: date,
    deterministic_target_pct: float = 90.0,
    groups: Iterable[str] = GROUPS,
    epistemic_flags: List[str] = EPISTEMIC_FLAGS,
) -> dict[str, Any]:
    if total_units <= 0:
        raise ValueError("total_units must be positive")
    if not (0.0 <= deterministic_target_pct <= 100.0):
        raise ValueError("deterministic_target_pct must be between 0.0 and 100.0")
    weekday_weight_vec = weekday_weight_vector(groups, date.weekday(), epistemic_flags)
    shannon_entropy = -np.sum(weekday_weight_vec * np.log2(weekday_weight_vec))
    return {
        "weekday_weight_vec": weekday_weight_vec,
        "deterministic_target_pct": deterministic_target_pct,
        "shannon_entropy": shannon_entropy,
    }

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