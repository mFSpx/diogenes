# DARWIN HAMMER — match 2581, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_ternar_m1009_s1.py (gen4)
# parent_b: hybrid_hybrid_model_vram_sc_hybrid_hybrid_ternar_m868_s0.py (gen4)
# born: 2026-05-29T23:43:05Z

"""
Hybrid Memory‑Similarity Scheduler

This module fuses two distinct parent algorithms:

* **HybridSheaf + Bandit router** (parent A) – provides a sinusoidal,
  weekday‑dependent weight vector and an SSIM similarity metric.
* **GPU‑VRAM scheduler + Bayesian utilities** (parent B) – supplies a
  Bayesian marginal probability computation used to modulate resource
  allocation.

**Mathematical bridge**

Both parents treat resource distribution as a probability‑like weighting
problem.  Parent A yields a stochastic vector `w_dow` that depends on the
day‑of‑week, while parent B produces a Bayesian marginal `m_i` for each
task.  We combine them multiplicatively:


p_i = w_dow[i] * m_i


to obtain a *prior‑adjusted* allocation probability.  To incorporate the
similarity‑driven refinement of parent A, we compute the Structural
Similarity Index (SSIM) between each task’s performance profile and the
mean profile, yielding a similarity factor `s_i ∈ [0,1]`.  The final
allocation score is


a_i = p_i * (s_i + ε)


which is normalised to the available GPU memory.  This unified formulation
mathematically merges the sinusoidal weekday weighting, Bayesian
uncertainty handling, and SSIM‑based similarity adjustment into a single
resource‑allocation system.
"""

import math
import random
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Core building blocks (extracted from the parents)
# ----------------------------------------------------------------------


def weekday_weight_vector(groups: List[str], dow: int) -> np.ndarray:
    """
    Normalised sinusoidal weight vector for *groups* based on weekday ``dow``.
    The vector is row‑stochastic (sums to 1).

    Parameters
    ----------
    groups: List[str]
        Identifiers for the groups/tasks.
    dow: int
        Day of week where 0 = Sunday … 6 = Saturday.

    Returns
    -------
    np.ndarray
        Weight vector of shape (len(groups),).
    """
    n = len(groups)
    if n == 0:
        raise ValueError("groups must contain at least one element")
    base_angles = np.arange(n) * (2.0 * math.pi) / n
    phase = (2.0 * math.pi) * (dow % 7) / 7.0
    amplitude = 0.2
    raw = 1.0 + amplitude * np.sin(base_angles + phase)
    weight_vec = raw / raw.sum()
    return weight_vec.astype(np.float64)


def compute_ssim(
    x: List[float],
    y: List[float],
    dynamic_range: float = 1.0,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """
    Compute the Structural Similarity Index (SSIM) between two 1‑D signals.

    Parameters
    ----------
    x, y: List[float]
        Input sequences of equal length.
    dynamic_range: float, optional
        The dynamic range of the data (default 1.0).
    k1, k2: float, optional
        Stabilisation constants (default 0.01, 0.03).

    Returns
    -------
    float
        SSIM value in [-1, 1].
    """
    if len(x) != len(y):
        raise ValueError("samples must have equal length")
    if not x:
        raise ValueError("samples must not be empty")

    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    mx = x_arr.mean()
    my = y_arr.mean()
    vx = ((x_arr - mx) ** 2).mean()
    vy = ((y_arr - my) ** 2).mean()
    cov = ((x_arr - mx) * (y_arr - my)).mean()

    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2

    numerator = (2 * mx * my + c1) * (2 * cov + c2)
    denominator = (mx ** 2 + my ** 2 + c1) * (vx + vy + c2)

    return float(numerator / denominator)


def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """
    Compute the Bayesian marginal probability P(E) given prior, likelihood,
    and false‑positive rate.

    Parameters
    ----------
    prior: float
        Prior probability P(H) in [0, 1].
    likelihood: float
        Likelihood P(E|H) in [0, 1].
    false_positive: float
        False‑positive probability P(E|¬H) in [0, 1].

    Returns
    -------
    float
        Marginal probability P(E) in [0, 1].
    """
    if not (0.0 <= prior <= 1.0 and 0.0 <= likelihood <= 1.0 and 0.0 <= false_positive <= 1.0):
        raise ValueError("All probability arguments must be between 0 and 1")
    numerator = prior * likelihood
    denominator = numerator + (1.0 - prior) * false_positive
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------


def generate_task_profiles(num_tasks: int, length: int = 50) -> List[List[float]]:
    """
    Produce synthetic performance profiles for ``num_tasks``.  Each profile
    is a list of ``length`` floats drawn from a uniform distribution.
    """
    random.seed(0)  # deterministic for reproducibility
    return [[random.random() for _ in range(length)] for _ in range(num_tasks)]


def similarity_factors(profiles: List[List[float]]) -> np.ndarray:
    """
    Compute SSIM similarity of each task profile to the mean profile.
    Returns a vector of similarity scores in [0, 1].
    """
    if not profiles:
        raise ValueError("profiles must not be empty")
    # Mean profile (element‑wise)
    mean_profile = np.mean(np.array(profiles, dtype=np.float64), axis=0).tolist()
    sims = [compute_ssim(p, mean_profile) for p in profiles]
    # Clip to [0,1] because SSIM can be slightly negative for noisy data
    return np.clip(np.asarray(sims, dtype=np.float64), 0.0, 1.0)


def hybrid_memory_allocation(
    task_names: List[str],
    priors: List[float],
    likelihoods: List[float],
    false_positives: List[float],
    dow: int,
    profiles: List[List[float]],
    total_memory_mb: int = 4096,
    reserve_mb: int = 768,
    epsilon: float = 1e-6,
) -> List[int]:
    """
    Allocate GPU memory (in MB) to a set of tasks using the fused
    weekday‑weight, Bayesian marginal, and SSIM similarity scheme.

    Parameters
    ----------
    task_names: List[str]
        Identifiers for the tasks.
    priors, likelihoods, false_positives: List[float]
        Bayesian parameters per task (same length as ``task_names``).
    dow: int
        Day‑of‑week index (0 = Sunday … 6 = Saturday).
    profiles: List[List[float]]
        Performance profiles for each task (same length as ``task_names``).
    total_memory_mb: int, optional
        Total GPU memory budget.
    reserve_mb: int, optional
        Memory reserved for system overhead.
    epsilon: float, optional
        Small constant to avoid division by zero.

    Returns
    -------
    List[int]
        Memory allocation per task (in MB), summing to ``total_memory_mb - reserve_mb``.
    """
    if not (len(task_names) == len(priors) == len(likelihoods) == len(false_positives) == len(profiles)):
        raise ValueError("All per‑task input lists must have identical length")

    # 1. Weekday‑dependent stochastic base weights
    w_dow = weekday_weight_vector(task_names, dow)  # shape (n,)

    # 2. Bayesian marginal per task
    marginals = np.array(
        [bayes_marginal(p, l, fp) for p, l, fp in zip(priors, likelihoods, false_positives)],
        dtype=np.float64,
    )

    # 3. SSIM similarity factors
    sims = similarity_factors(profiles)  # shape (n,)

    # 4. Fuse the three components multiplicatively
    raw_scores = w_dow * marginals * (sims + epsilon)

    # 5. Normalise to the allocatable memory pool
    allocatable = max(total_memory_mb - reserve_mb, 0)
    if allocatable == 0:
        raise RuntimeError("No memory available for allocation after reserve")
    normalized = raw_scores / (raw_scores.sum() + epsilon)

    # 6. Convert to integer MB per task (largest‑remainder method to keep total exact)
    exact_mb = normalized * allocatable
    int_mb = np.floor(exact_mb).astype(int)
    remainder = int(allocatable - int_mb.sum())
    if remainder > 0:
        # Distribute the remaining MB to tasks with largest fractional parts
        fractional = exact_mb - int_mb
        idx_desc = np.argsort(-fractional)  # descending order
        int_mb[idx_desc[:remainder]] += 1

    return int_mb.tolist()


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Example configuration
    NUM_TASKS = 5
    TASK_NAMES = [f"task_{i}" for i in range(NUM_TASKS)]

    # Random but reproducible Bayesian parameters
    random.seed(42)
    priors = [random.uniform(0.2, 0.9) for _ in range(NUM_TASKS)]
    likelihoods = [random.uniform(0.5, 0.99) for _ in range(NUM_TASKS)]
    false_positives = [random.uniform(0.01, 0.3) for _ in range(NUM_TASKS)]

    # Current weekday (0 = Sunday … 6 = Saturday)
    from datetime import date

    today = date.today()
    dow_index = (today.weekday() + 1) % 7  # convert Python's 0=Mon to 0=Sun

    # Synthetic performance profiles
    profiles = generate_task_profiles(NUM_TASKS, length=100)

    # Perform allocation
    allocation = hybrid_memory_allocation(
        task_names=TASK_NAMES,
        priors=priors,
        likelihoods=likelihoods,
        false_positives=false_positives,
        dow=dow_index,
        profiles=profiles,
        total_memory_mb=4096,
        reserve_mb=768,
    )

    # Output results
    print("GPU Memory Allocation (MB) per task:")
    for name, mem in zip(TASK_NAMES, allocation):
        print(f"  {name}: {mem} MB")
    total_alloc = sum(allocation)
    print(f"Total allocated: {total_alloc} MB (expected {4096 - 768} MB)")