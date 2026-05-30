# DARWIN HAMMER — match 5239, survivor 2
# gen: 4
# parent_a: hybrid_doomsday_calendar_hybrid_rlct_grokking_m396_s5.py (gen3)
# parent_b: hybrid_hybrid_hybrid_worksh_hybrid_hybrid_model__m51_s2.py (gen3)
# born: 2026-05-30T00:00:49Z

"""Hybrid Doomsday‑RLCT‑NLMS & Workshare‑Liquid‑Time Scheduler.

Parents
-------
* **Parent A** – ``hybrid_doomsday_calendar_hybrid_rlct_grokking_m396_s5.py``  
  Provides a deterministic weekday index and a NLMS adaptive filter whose
  learning‑rate is rescaled by a Real Log‑Canonical‑Threshold (RLCT) derived
  from recent errors.

* **Parent B** – ``hybrid_hybrid_hybrid_worksh_hybrid_hybrid_model__m51_s2.py``  
  Supplies a weekday‑dependent resource‑allocation vector for a set of model
  groups and a scheduler that respects GPU memory constraints.

Mathematical Bridge
-------------------
The bridge is the *weekday* (day‑of‑week) which appears in both parents:

* In **A** the weekday index is one‑hot encoded and concatenated to the NLMS
  input vector, allowing the filter to capture weekly periodicities.
* In **B** the same weekday drives a sinusoidal weight vector over the model
  groups, defining a stochastic allocation matrix.

The fused algorithm therefore builds a joint input vector  


x = [ encode_weekday(dow) ⊗ weight_vector(groups) ]


(the Kronecker product of the one‑hot weekday and the group weight vector).
This vector feeds the NLMS update whose effective learning‑rate  


μ_eff = μ₀ / (1 + λ·RLCT)


is modulated by the RLCT computed from the error history.  
After the weight update the allocation matrix is interpreted as a resource
distribution which is finally fed to a simple GPU‑memory‑aware scheduler.

The module implements the full pipeline in three public functions:
`hybrid_nlms_step`, `allocate_resources`, and `hybrid_workshare_nlms`.  A
smoke‑test at the bottom demonstrates a single update cycle."""

from __future__ import annotations

import math
import random
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence, Tuple, Deque, List

import numpy as np

# ----------------------------------------------------------------------
# Shared weekday utilities (Parent A)
# ----------------------------------------------------------------------
def weekday_index(year: int, month: int, day: int) -> int:
    """Return weekday as 0=Sunday … 6=Saturday."""
    # datetime.weekday(): 0=Monday … 6=Sunday → shift
    return (datetime(year, month, day).weekday() + 1) % 7


def encode_weekday(idx: int) -> np.ndarray:
    """One‑hot encode a weekday index into a length‑7 float vector."""
    vec = np.zeros(7, dtype=float)
    if 0 <= idx < 7:
        vec[idx] = 1.0
    else:
        raise ValueError(f"weekday index out of range: {idx}")
    return vec


# ----------------------------------------------------------------------
# RLCT helper (Parent A)
# ----------------------------------------------------------------------
def compute_rlct(error_history: Deque[float]) -> float:
    """
    Approximate the Real Log‑Canonical‑Threshold from a sliding window of errors.

    The implementation uses a simple variance‑based surrogate:
        RLCT ≈ log(1 + var(errors))

    This keeps the computation lightweight while preserving the monotonic
    relationship between error dispersion and the scaling factor used in the
    learning‑rate formula.
    """
    if not error_history:
        return 0.0
    arr = np.array(error_history, dtype=float)
    variance = arr.var()
    return math.log1p(variance)  # log(1+variance) is always non‑negative


# ----------------------------------------------------------------------
# NLMS step with RLCT‑adjusted learning‑rate (Parent A)
# ----------------------------------------------------------------------
def hybrid_nlms_step(
    w: np.ndarray,
    x: np.ndarray,
    d: float,
    mu0: float,
    rlct: float,
    lam: float = 1e-3,
) -> Tuple[np.ndarray, float]:
    """
    Perform one NLMS prediction‑update cycle.

    Parameters
    ----------
    w : np.ndarray
        Current filter coefficient vector (shape (N,)).
    x : np.ndarray
        Input vector (shape (N,)).
    d : float
        Desired scalar output.
    mu0 : float
        Base learning‑rate (0 < mu0 ≤ 2).
    rlct : float
        Real Log‑Canonical‑Threshold derived from recent errors.
    lam : float
        Damping factor for the RLCT term (default 1e‑3).

    Returns
    -------
    w_new : np.ndarray
        Updated coefficient vector.
    e : float
        Instantaneous error (d - y).
    """
    if w.shape != x.shape:
        raise ValueError("w and x must have the same shape")
    # Prediction
    y = float(np.dot(w, x))
    e = d - y

    # RLCT‑adjusted learning‑rate
    mu_eff = mu0 / (1.0 + lam * rlct)

    # Normalisation term (avoid division by zero)
    norm = float(np.dot(x, x)) + 1e-12

    # NLMS weight update
    w_new = w + (mu_eff / norm) * e * x
    return w_new, e


# ----------------------------------------------------------------------
# Weekday‑dependent group weight vector (Parent B)
# ----------------------------------------------------------------------
def weekday_weight_vector(groups: Sequence[str], dow: int) -> np.ndarray:
    """
    Produce a row‑stochastic weight vector for *groups* based on weekday ``dow``.

    A sinusoidal rotation yields smooth periodic variation:
        w_i = sin(2π (dow + i) / 7) + 1   (shifted to be non‑negative)
    The vector is then normalised to sum to 1.
    """
    n = len(groups)
    if n == 0:
        raise ValueError("groups sequence must contain at least one element")
    angles = np.arange(n, dtype=float) + dow
    raw = np.sin(2.0 * math.pi * angles / 7.0) + 1.0  # range [0,2]
    vec = raw / raw.sum()
    return vec


# ----------------------------------------------------------------------
# Resource allocation using the weekday‑group bridge
# ----------------------------------------------------------------------
def allocate_resources(
    dow: int,
    groups: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the joint allocation matrix and its flattened input vector.

    Returns
    -------
    allocation_matrix : np.ndarray, shape (7, G)
        Outer product of the one‑hot weekday vector and the group weight vector.
    x : np.ndarray, shape (7·G,)
        Flattened column‑major (C‑order) version of ``allocation_matrix``.
        This vector is suitable as the NLMS input.
    """
    one_hot = encode_weekday(dow)               # (7,)
    group_weights = weekday_weight_vector(groups, dow)  # (G,)
    allocation_matrix = np.outer(one_hot, group_weights)  # (7, G)
    x = allocation_matrix.ravel()  # flatten to 1‑D
    return allocation_matrix, x


# ----------------------------------------------------------------------
# Simple GPU‑memory‑aware task scheduler (Parent B)
# ----------------------------------------------------------------------
def schedule_tasks(
    allocation_matrix: np.ndarray,
    gpu_mem_bytes: int,
    mem_per_unit: float = 1e6,
) -> List[str]:
    """
    Schedule groups proportionally to their allocated weight while respecting
    a hard GPU memory budget.

    Parameters
    ----------
    allocation_matrix : np.ndarray, shape (7, G)
        Resource allocation produced by ``allocate_resources``.
    gpu_mem_bytes : int
        Available GPU memory in bytes.
    mem_per_unit : float, optional
        Memory cost (bytes) associated with a unit weight.  Default 1 MiB.

    Returns
    -------
    scheduled_groups : list of str
        Names of groups that can be executed within the memory budget.
    """
    # Sum over weekdays – only one weekday row is non‑zero, but we keep it generic.
    group_weights = allocation_matrix.sum(axis=0)  # shape (G,)
    # Compute required memory per group
    required_mem = group_weights * mem_per_unit
    # Greedy selection in descending weight order
    indices = np.argsort(-group_weights)  # descending
    scheduled = []
    used = 0.0
    for idx in indices:
        if used + required_mem[idx] <= gpu_mem_bytes:
            scheduled.append(idx)
            used += required_mem[idx]
    # Convert indices back to group names (caller must map)
    return scheduled


# ----------------------------------------------------------------------
# Integrated hybrid operation
# ----------------------------------------------------------------------
def hybrid_workshare_nlms(
    year: int,
    month: int,
    day: int,
    groups: Sequence[str],
    desired: float,
    w: np.ndarray,
    mu0: float,
    lam: float,
    error_hist: Deque[float],
    gpu_mem_bytes: int,
) -> Tuple[np.ndarray, float, List[int]]:
    """
    End‑to‑end hybrid step:

    1. Determine weekday and build the NLMS input vector via the allocation matrix.
    2. Compute RLCT from the error history.
    3. Perform the NLMS update with the RLCT‑adjusted learning‑rate.
    4. Append the new error to ``error_hist`` (maintaining a fixed size).
    5. Run the GPU‑memory scheduler on the allocation matrix.

    Returns
    -------
    w_new : np.ndarray
        Updated NLMS coefficients.
    e : float
        Current prediction error.
    scheduled_indices : list[int]
        Indices of groups that fit into the available GPU memory.
    """
    dow = weekday_index(year, month, day)

    # 1. Allocation matrix and NLMS input vector
    allocation_matrix, x = allocate_resources(dow, groups)

    # 2. RLCT from error history
    rlct = compute_rlct(error_hist)

    # 3. NLMS step
    w_new, e = hybrid_nlms_step(w, x, desired, mu0, rlct, lam)

    # 4. Update error history (keep last 50 errors)
    error_hist.append(e)
    if len(error_hist) > 50:
        error_hist.popleft()

    # 5. Scheduler (returns indices of groups that can run)
    scheduled_indices = schedule_tasks(allocation_matrix, gpu_mem_bytes)

    return w_new, e, scheduled_indices


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a modest set of model groups
    GROUPS = ("codex", "groq", "cohere", "local_models")
    G = len(GROUPS)

    # Initialise NLMS coefficients (size 7*G)
    w0 = np.zeros(7 * G, dtype=float)

    # Desired output – arbitrary scalar target for the demo
    desired_output = 0.5

    # Learning‑rate parameters
    mu0 = 0.8
    lam = 1e-3

    # Error history deque
    err_hist: Deque[float] = deque(maxlen=50)

    # Simulated GPU memory (e.g., 4 GiB)
    gpu_memory = 4 * (1 << 30)  # bytes

    # Run a single hybrid update for today's date
    today = datetime.now(timezone.utc)
    w1, err, scheduled = hybrid_workshare_nlms(
        year=today.year,
        month=today.month,
        day=today.day,
        groups=GROUPS,
        desired=desired_output,
        w=w0,
        mu0=mu0,
        lam=lam,
        error_hist=err_hist,
        gpu_mem_bytes=gpu_memory,
    )

    print("Updated NLMS weights (first 10 entries):", w1[:10])
    print("Current error:", err)
    print("Scheduled group indices:", scheduled)
    print("Scheduled group names:", [GROUPS[i] for i in scheduled])