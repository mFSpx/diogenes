# DARWIN HAMMER — match 4273, survivor 3
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_nlms_o_m2125_s3.py (gen6)
# parent_b: hybrid_ternary_router_hybrid_hybrid_hybrid_m133_s1.py (gen4)
# born: 2026-05-29T23:54:34Z

"""Hybrid algorithm merging DARWIN HAMMER parent A and parent B.

Parent A provides a *weekday‑dependent weight vector* for a set of logical groups
(e.g., model providers) and a VRAM‑aware GPU selector.  
Parent B supplies a *Shapley value* machinery for attributing a global value to
individual features (here the groups) together with a simple circuit‑breaker
to guard against repeated selection failures.

The mathematical bridge is built by treating each *group* from parent A as a
feature in the Shapley game of parent B.  The global value function is defined
as the total free VRAM that can be harvested from GPUs belonging to a chosen
subset of groups.  Consequently:

* The weekday weight vector becomes a prior probability over groups.
* Exact Shapley values quantify each group’s marginal contribution to free VRAM.
* The circuit‑breaker monitors selection attempts that violate the memory budget.

The hybrid functions combine these concepts to rank groups, score GPUs and
perform a safe selection respecting both the temporal weighting and the
Shapley‑derived importance."""
import math
import random
import sys
import pathlib
import datetime as _dt
from typing import Any, Dict, List, Sequence, Tuple, Callable, FrozenSet

import numpy as np

# ----------------------------------------------------------------------
# Constants (mirroring parent A)
# ----------------------------------------------------------------------
GROUPS: Tuple[str, ...] = ("codex", "groq", "cohere", "local_models")
MAX64 = (1 << 64) - 1
DEFAULT_BUDGET_MB = 4096
DEFAULT_RESERVE_MB = 768

# ----------------------------------------------------------------------
# Parent‑A utilities
# ----------------------------------------------------------------------
def _pct(value: float) -> float:
    """Round a float to six decimal places."""
    return round(float(value), 6)


def doomsday(year: int, month: int, day: int) -> int:
    """
    Return weekday index where 0 = Sunday … 6 = Saturday.
    """
    return (int(_dt.date(year, month, day).weekday()) + 1) % 7


def weekday_weight_vector(groups: Sequence[str], dow: int) -> np.ndarray:
    """
    Normalised weight vector for *groups* based on weekday ``dow``.
    A sinusoidal rotation yields a row‑stochastic vector.
    """
    n = len(groups)
    if n == 0:
        raise ValueError("`groups` must contain at least one element.")
    base_angles = np.arange(n, dtype=np.float64) * (2.0 * math.pi) / n
    phase = (2.0 * math.pi) * (dow % 7) / 7.0
    amplitude = 0.2
    raw = 1.0 + amplitude * np.sin(base_angles + phase)
    weight_vec = raw / raw.sum()
    return weight_vec.astype(np.float64)


def vram_aware_gpu_selection(
    gpus: List[Dict[str, Any]],
    budget_mb: int,
    reserve_mb: int,
) -> List[Dict[str, Any]]:
    """
    Return GPUs whose *free* VRAM can satisfy ``budget_mb`` plus ``reserve_mb``.
    """
    required = budget_mb + reserve_mb
    return [gpu for gpu in gpus if gpu.get("memory.free", 0) >= required]

# ----------------------------------------------------------------------
# Parent‑B utilities
# ----------------------------------------------------------------------
def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / max(length, width, height)


def shapley_kernel_weight(subset_size: int, feature_count: int) -> float:
    """Weight factor used in the exact Shapley formula."""
    return (
        math.factorial(subset_size)
        * math.factorial(feature_count - subset_size - 1)
        / math.factorial(feature_count)
    )


def exact_shapley_value(
    value_fn: Callable[[FrozenSet[int]], float],
    feature_index: int,
    feature_count: int,
) -> float:
    """
    Compute the exact Shapley value for ``feature_index`` given a value function.
    """
    others = [i for i in range(feature_count) if i != feature_index]
    total = 0.0
    for k in range(len(others) + 1):
        for subset in combinations(others, k):
            s = frozenset(subset)
            marginal = value_fn(s | {feature_index}) - value_fn(s)
            total += shapley_kernel_weight(k, feature_count) * marginal
    return total


class EndpointCircuitBreaker:
    """Simple circuit‑breaker that opens after a configurable number of failures."""

    def __init__(self, failure_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False

    def record_success(self) -> None:
        self.failures = 0
        self.open = False

    def record_failure(self) -> None:
        self.failures += 1
        self.open = self.failures >= self.failure_threshold

    def allow(self) -> bool:
        return not self.open


# ----------------------------------------------------------------------
# Hybrid core
# ----------------------------------------------------------------------
def compute_group_weights(dow: int) -> np.ndarray:
    """
    Wrapper around :func:`weekday_weight_vector` that supplies the global ``GROUPS``.
    """
    return weekday_weight_vector(GROUPS, dow)


def _value_fn_factory(
    gpus: List[Dict[str, Any]],
    groups: Sequence[str],
) -> Callable[[FrozenSet[int]], float]:
    """
    Build a value function for Shapley calculation.

    The function receives a set of group *indices* and returns the total free VRAM
    (in MB) of all GPUs that belong to any of those groups.
    """
    # Build a mapping from group name to list of GPUs
    group_to_gpus: Dict[str, List[Dict[str, Any]]] = {g: [] for g in groups}
    for gpu in gpus:
        grp = gpu.get("group")
        if grp in group_to_gpus:
            group_to_gpus[grp].append(gpu)

    def value_fn(selected_indices: FrozenSet[int]) -> float:
        selected_groups = {groups[i] for i in selected_indices}
        total = 0.0
        for grp in selected_groups:
            for gpu in group_to_gpus.get(grp, []):
                total += float(gpu.get("memory.free", 0))
        return total

    return value_fn


def compute_shapley_scores(
    gpus: List[Dict[str, Any]],
    groups: Sequence[str] = GROUPS,
) -> np.ndarray:
    """
    Compute a Shapley value for each *group* where the coalition value is the
    aggregate free VRAM of GPUs belonging to the selected groups.
    """
    feature_count = len(groups)
    value_fn = _value_fn_factory(gpus, groups)

    shapley_vals = np.empty(feature_count, dtype=np.float64)
    for idx in range(feature_count):
        shapley_vals[idx] = exact_shapley_value(value_fn, idx, feature_count)
    # Normalise to a probability‑like vector
    total = shapley_vals.sum()
    if total == 0:
        return np.zeros_like(shapley_vals)
    return shapley_vals / total


def hybrid_gpu_selector(
    gpus: List[Dict[str, Any]],
    dow: int,
    budget_mb: int = DEFAULT_BUDGET_MB,
    reserve_mb: int = DEFAULT_RESERVE_MB,
    failure_threshold: int = 3,
) -> List[Dict[str, Any]]:
    """
    Perform a VRAM‑aware GPU selection that respects both the weekday‑driven
    group priors and the Shapley‑derived importance of each group.

    The algorithm proceeds as follows:
    1. Compute weekday weights (prior) for each group.
    2. Compute Shapley scores (marginal contribution) for each group.
    3. Blend the two vectors (geometric mean) to obtain a final ranking.
    4. Iterate over groups in descending order, attempting to pick GPUs that
       satisfy the memory requirement.  Failures are tracked by an
       :class:`EndpointCircuitBreaker`.
    """
    # 1. Prior from weekday
    prior = compute_group_weights(dow)  # shape (G,)

    # 2. Shapley importance
    shapley = compute_shapley_scores(gpus, GROUPS)  # shape (G,)

    # 3. Blend – geometric mean preserves proportionality while rewarding agreement
    blended = np.sqrt(prior * shapley)
    # Guard against zero entries
    blended = np.nan_to_num(blended, nan=0.0)

    # Order groups by blended score descending
    ordered_indices = np.argsort(-blended)

    selected: List[Dict[str, Any]] = []
    breaker = EndpointCircuitBreaker(failure_threshold)

    # Helper to check if current selection already meets budget
    def current_vram() -> int:
        return sum(int(g.get("memory.free", 0)) for g in selected)

    for idx in ordered_indices:
        if not breaker.allow():
            # Circuit open – stop trying further groups
            break

        group_name = GROUPS[idx]
        # Candidate GPUs belonging to this group and satisfying the per‑GPU budget
        candidates = [
            gpu
            for gpu in gpus
            if gpu.get("group") == group_name
            and gpu.get("memory.free", 0) >= (budget_mb + reserve_mb)
        ]

        if not candidates:
            breaker.record_failure()
            continue

        # Accept all candidates of the group (could be refined)
        selected.extend(candidates)
        breaker.record_success()

        if current_vram() >= (budget_mb + reserve_mb):
            # Requirement satisfied – early exit
            break

    return selected

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a synthetic GPU inventory
    rng = random.Random(42)
    synthetic_gpus = []
    for i in range(12):
        grp = GROUPS[i % len(GROUPS)]
        gpu = {
            "id": f"gpu_{i}",
            "group": grp,
            # free memory between 2 GB and 12 GB
            "memory.free": rng.randint(2000, 12000),
        }
        synthetic_gpus.append(gpu)

    dow = doomsday(2026, 5, 29)  # example date
    selected = hybrid_gpu_selector(
        synthetic_gpus,
        dow,
        budget_mb=DEFAULT_BUDGET_MB,
        reserve_mb=DEFAULT_RESERVE_MB,
        failure_threshold=2,
    )
    print("Selected GPUs:")
    for gpu in selected:
        print(f"  {gpu['id']} (group={gpu['group']}, free={gpu['memory.free']} MB)")

    # Show intermediate vectors for verification
    print("\nWeekday weight vector:", compute_group_weights(dow))
    print("Shapley scores:", compute_shapley_scores(synthetic_gpus))
    print("Blended ranking:", np.argsort(-np.sqrt(compute_group_weights(dow) * compute_shapley_scores(synthetic_gpus))))