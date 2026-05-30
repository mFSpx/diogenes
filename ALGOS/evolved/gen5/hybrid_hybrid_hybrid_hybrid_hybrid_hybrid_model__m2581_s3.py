# DARWIN HAMMER — match 2581, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_ternar_m1009_s1.py (gen4)
# parent_b: hybrid_hybrid_model_vram_sc_hybrid_hybrid_ternar_m868_s0.py (gen4)
# born: 2026-05-29T23:43:05Z

import math
import random
import sys
from typing import List, Tuple
import numpy as np

def weekday_weight_vector(groups: List[str], dow: int) -> np.ndarray:
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
    if not (0.0 <= prior <= 1.0 and 0.0 <= likelihood <= 1.0 and 0.0 <= false_positive <= 1.0):
        raise ValueError("All probability arguments must be between 0 and 1")
    numerator = prior * likelihood
    denominator = numerator + (1.0 - prior) * false_positive
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def generate_task_profiles(num_tasks: int, length: int = 50) -> List[List[float]]:
    random.seed(0)  
    return [[random.random() for _ in range(length)] for _ in range(num_tasks)]


def similarity_factors(profiles: List[List[float]]) -> np.ndarray:
    if not profiles:
        raise ValueError("profiles must not be empty")
    mean_profile = np.mean(np.array(profiles, dtype=np.float64), axis=0).tolist()
    sims = [compute_ssim(p, mean_profile) for p in profiles]
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
    if not (len(task_names) == len(priors) == len(likelihoods) == len(false_positives) == len(profiles)):
        raise ValueError("All per‑task input lists must have identical length")

    w_dow = weekday_weight_vector(task_names, dow)  
    marginals = np.array([bayes_marginal(p, l, fp) for p, l, fp in zip(priors, likelihoods, false_positives)], dtype=np.float64)
    sims = similarity_factors(profiles)  

    raw_scores = w_dow * marginals * (sims + epsilon)
    allocatable = max(total_memory_mb - reserve_mb, 0)
    if allocatable == 0:
        raise RuntimeError("No memory available for allocation after reserve")
    normalized = raw_scores / (raw_scores.sum() + epsilon)

    exact_mb = normalized * allocatable
    int_mb = np.floor(exact_mb).astype(int)
    remainder = int(allocatable - int_mb.sum())
    if remainder > 0:
        sorted_indices = np.argsort(exact_mb - int_mb)
        for i in sorted_indices:
            if remainder > 0:
                int_mb[i] += 1
                remainder -= 1
            else:
                break

    return int_mb.tolist()


def improved_hybrid_memory_allocation(
    task_names: List[str],
    priors: List[float],
    likelihoods: List[float],
    false_positives: List[float],
    dow: int,
    profiles: List[List[float]],
    total_memory_mb: int = 4096,
    reserve_mb: int = 768,
    epsilon: float = 1e-6,
    alpha: float = 0.5,
) -> List[int]:
    if not (len(task_names) == len(priors) == len(likelihoods) == len(false_positives) == len(profiles)):
        raise ValueError("All per‑task input lists must have identical length")

    w_dow = weekday_weight_vector(task_names, dow)  
    marginals = np.array([bayes_marginal(p, l, fp) for p, l, fp in zip(priors, likelihoods, false_positives)], dtype=np.float64)
    sims = similarity_factors(profiles)  

    raw_scores = alpha * w_dow * marginals + (1 - alpha) * sims
    allocatable = max(total_memory_mb - reserve_mb, 0)
    if allocatable == 0:
        raise RuntimeError("No memory available for allocation after reserve")
    normalized = raw_scores / (raw_scores.sum() + epsilon)

    exact_mb = normalized * allocatable
    int_mb = np.floor(exact_mb).astype(int)
    remainder = int(allocatable - int_mb.sum())
    if remainder > 0:
        sorted_indices = np.argsort(exact_mb - int_mb)
        for i in sorted_indices:
            if remainder > 0:
                int_mb[i] += 1
                remainder -= 1
            else:
                break

    return int_mb.tolist()