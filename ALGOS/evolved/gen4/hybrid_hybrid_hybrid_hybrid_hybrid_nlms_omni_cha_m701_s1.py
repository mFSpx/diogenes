# DARWIN HAMMER — match 701, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_hybrid_worksh_hybrid_hybrid_model__m51_s2.py (gen3)
# parent_b: hybrid_nlms_omni_chaotic_sprint_m59_s1.py (gen1)
# born: 2026-05-29T23:30:28Z

"""Hybrid Workshare‑Liquid‑Time Scheduler with NLMS‑Adapted Allocation
===================================================================

This module fuses two parent algorithms:

* **Parent A** – *hybrid_workshare_liquid_time*: builds a weekday‑dependent
  resource‑allocation vector and schedules tasks according to GPU memory
  availability.

* **Parent B** – *nlms + omni‑chaotic sprint*: provides a Normalised Least‑Mean‑Squares
  (NLMS) weight‑adaptation rule that can be used to learn from an error signal.

**Mathematical bridge**  
Both parents operate on *vectors* that sum to a constant (the allocation vector
in A and the adaptive weight vector in B).  The bridge is therefore the NLMS
update applied **directly to the allocation vector**.  After a scheduling step
we compute an error between a *target* memory‑usage profile and the *actual*
usage produced by the current allocation.  The NLMS rule then adapts the
allocation vector so that future schedules better respect the target profile.
Thus the hybrid system continuously learns a weekday‑aware, memory‑aware
allocation strategy.

The core equations are:


# 1. Weekday weight base (Parent A)
w_base(d) = normalize( sin( 2π (d + i) / G ) + 1 ),   i = 0 … G‑1

# 2. Scheduling (Parent A)
usage = min( w * M_total , M_available )            # element‑wise

# 3. NLMS adaptation (Parent B)
e      = target – usage
norm_x = x·x + ε
w_new  = w + μ * e * x / norm_x


where `w` is the allocation vector, `M_total` the total memory demand,
`M_available` the GPU memory vector, `x` the input regressor (here we reuse the
allocation vector itself), `μ` the step‑size and `ε` a small stabiliser.

The functions below implement these steps and expose a single high‑level
routine `hybrid_scheduler` that can be called repeatedly to let the system
learn its optimal allocation.  A small smoke test demonstrates a few
iterations without error."""

import sys
import math
import random
from pathlib import Path
from datetime import datetime, timezone
from collections.abc import Sequence

import numpy as np

# ----------------------------------------------------------------------
# Constants (shared across the hybrid system)
# ----------------------------------------------------------------------
GROUPS = ("codex", "groq", "cohere", "local_models")
NUM_GROUPS = len(GROUPS)

MAX_GPU_MEMORY_MB = 1536                 # total GPU memory per group (simulated)
TARGET_USAGE_FRACTION = np.array([0.25, 0.30, 0.20, 0.25])  # desired share of total memory
TARGET_USAGE_FRACTION = TARGET_USAGE_FRACTION / TARGET_USAGE_FRACTION.sum()

NLMS_MU = 0.6          # learning rate for NLMS adaptation
NLMS_EPS = 1e-9       # stabiliser to avoid division by zero

# ----------------------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------------------
def _pct(value: float) -> float:
    """Round a float to six decimal places (helper for debugging)."""
    return round(float(value), 6)

def doomsday(year: int, month: int, day: int) -> int:
    """Return weekday index where 0 = Sunday … 6 = Saturday."""
    return (datetime(year, month, day).weekday() + 1) % 7

def weekday_weight_vector(groups: Sequence[str], dow: int) -> np.ndarray:
    """
    Produce a normalised weight vector for *groups* based on weekday ``dow``.
    The sinusoidal rotation yields a row‑stochastic vector (sum == 1).

    Parameters
    ----------
    groups: sequence of group identifiers
    dow:    integer weekday (0=Sunday … 6=Saturday)

    Returns
    -------
    np.ndarray of shape (len(groups),) with non‑negative entries summing to 1.
    """
    n = len(groups)
    if n == 0:
        raise ValueError("groups must contain at least one element")
    # Sinusoidal pattern shifted by weekday
    angles = 2 * math.pi * (dow + np.arange(n)) / n
    raw = np.sin(angles) + 1.0          # shift to make all entries non‑negative
    # Guard against pathological all‑zero (should never happen)
    if np.allclose(raw, 0):
        raw = np.ones_like(raw)
    weight_vec = raw / raw.sum()
    return weight_vec

# ----------------------------------------------------------------------
# Parent A – allocation and scheduling primitives
# ----------------------------------------------------------------------
def allocate_hybrid(dow: int) -> np.ndarray:
    """
    Allocate resources based on weekday weights.
    Returns a stochastic vector of length ``NUM_GROUPS``.
    """
    return weekday_weight_vector(GROUPS, dow)

def schedule_tasks(allocation: np.ndarray,
                   gpu_memory_mb: np.ndarray) -> np.ndarray:
    """
    Schedule tasks given an allocation vector and per‑group GPU memory limits.

    The allocation vector expresses the *desired* fraction of the total demand
    each group should receive.  The scheduler caps each group's usage by the
    available GPU memory for that group.

    Parameters
    ----------
    allocation   – stochastic vector (desired fractions)
    gpu_memory_mb – available memory per group (MB)

    Returns
    -------
    actual_usage – vector of memory actually used per group (MB)
    """
    # Assume a constant total demand equal to the sum of all available memory.
    total_demand = gpu_memory_mb.sum()
    desired_usage = allocation * total_demand
    # Real usage cannot exceed what the GPU can provide.
    actual_usage = np.minimum(desired_usage, gpu_memory_mb)
    return actual_usage

# ----------------------------------------------------------------------
# Parent B – NLMS adaptation core
# ----------------------------------------------------------------------
def nlms_update(weights: np.ndarray,
                x: np.ndarray,
                target: np.ndarray,
                mu: float = NLMS_MU,
                eps: float = NLMS_EPS) -> np.ndarray:
    """
    Perform a single NLMS weight update.

    The error is defined as ``target - predict(weights, x)`` where the
    prediction is the dot product ``weights·x``.  The same formulation is used
    to adapt the allocation vector.

    Parameters
    ----------
    weights – current weight (allocation) vector
    x       – regressor vector (here we reuse the allocation itself)
    target  – desired output vector (memory usage target)
    mu      – step size
    eps     – small constant to avoid division by zero

    Returns
    -------
    Updated weight vector.
    """
    # Prediction (scalar) – we treat each group independently, so we compute a
    # per‑component prediction by element‑wise multiplication.
    y = weights * x
    error = target - y
    norm_x = np.dot(x, x) + eps
    # NLMS update applied element‑wise
    new_weights = weights + mu * error * x / norm_x
    # Re‑normalise to keep the vector stochastic
    new_weights = np.maximum(new_weights, 0)          # enforce non‑negativity
    if new_weights.sum() == 0:
        # fallback to uniform distribution if everything collapsed
        new_weights = np.ones_like(new_weights) / len(new_weights)
    else:
        new_weights = new_weights / new_weights.sum()
    return new_weights

# ----------------------------------------------------------------------
# Hybrid operation
# ----------------------------------------------------------------------
def hybrid_scheduler(current_date: datetime,
                     gpu_memory_mb: np.ndarray,
                     allocation: np.ndarray | None = None) -> np.ndarray:
    """
    One iteration of the hybrid algorithm.

    1. Compute a weekday‑based base allocation (Parent A).
    2. If a previous ``allocation`` is supplied, blend it with the base using
       NLMS adaptation so that the allocation learns from the scheduling error.
    3. Schedule tasks with the adapted allocation.
    4. Return the *new* allocation vector for the next iteration.

    Parameters
    ----------
    current_date   – datetime used to extract the weekday.
    gpu_memory_mb  – vector of available GPU memory per group (MB).
    allocation     – optional previous allocation vector; if ``None`` the base
                     weekday allocation is used directly.

    Returns
    -------
    Updated allocation vector (stochastic, length ``NUM_GROUPS``).
    """
    dow = (current_date.weekday() + 1) % 7          # 0=Sunday … 6=Saturday
    base_alloc = allocate_hybrid(dow)

    if allocation is None:
        alloc = base_alloc
    else:
        # Use NLMS to adapt the previous allocation toward the base while also
        # accounting for the scheduling error.
        # Target usage is the desired fraction of total GPU memory.
        total_gpu = gpu_memory_mb.sum()
        target_usage = TARGET_USAGE_FRACTION * total_gpu

        # Perform a scheduling step with the *previous* allocation to obtain the
        # actual usage, then compute the error.
        actual_usage = schedule_tasks(allocation, gpu_memory_mb)

        # NLMS adaptation: treat allocation as weights, x as the base allocation,
        # and target as the desired usage expressed as a fraction of total demand.
        # We first map the usage error back to a fraction space.
        usage_error_frac = (target_usage - actual_usage) / total_gpu
        # Blend the base allocation with the error‑driven correction.
        alloc = nlms_update(allocation, base_alloc, allocation + usage_error_frac)

    # Final scheduling with the (potentially) updated allocation.
    _ = schedule_tasks(alloc, gpu_memory_mb)   # side‑effect free; could be logged
    return alloc

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simulate a week of scheduling to demonstrate learning.
    rng = np.random.default_rng(42)

    # Random per‑group GPU memory caps (within reasonable bounds)
    gpu_caps = rng.integers(300, 600, size=NUM_GROUPS).astype(float)

    alloc = None
    print("Day   Dow  Allocation (rounded)   Usage (MB)      Target (MB)")
    for day_offset in range(7):
        today = datetime.now(timezone.utc) + np.timedelta64(day_offset, 'D')
        # Run one hybrid iteration
        alloc = hybrid_scheduler(today, gpu_caps, allocation=alloc)
        usage = schedule_tasks(alloc, gpu_caps)
        total_gpu = gpu_caps.sum()
        target = TARGET_USAGE_FRACTION * total_gpu
        dow = (today.weekday() + 1) % 7
        print(f"{today.date()}  {dow}  {np.round(alloc, 3)}  {np.round(usage,1)}  {np.round(target,1)}")
    sys.exit(0)