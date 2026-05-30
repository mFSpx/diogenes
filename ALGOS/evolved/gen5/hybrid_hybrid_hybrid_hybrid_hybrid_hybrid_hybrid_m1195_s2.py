# DARWIN HAMMER — match 1195, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_distri_hybrid_bayes_claim_k_m166_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_endpoi_hybrid_ssim_hybrid_d_m1_s3.py (gen3)
# born: 2026-05-29T23:33:37Z

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, replace
from typing import List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# 1. Probabilistic acceptance (Parent A)
# ----------------------------------------------------------------------
def acceptance_probability(delta_energy: float, temperature: float) -> float:
    """Metropolis‑style acceptance probability.

    Args:
        delta_energy: Energy difference ΔE (positive → worse state).
        temperature: Simulated‑annealing temperature T > 0.

    Returns:
        Probability in [0,1].
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return math.exp(-delta_energy / temperature)


def hoeffding_bound(num_samples: int, epsilon: float, delta: float = 0.05) -> bool:
    """Hoeffding bound decision: returns True if the observed mean is
    statistically significant enough to trigger a split/leadership change.

    Args:
        num_samples: Number of independent observations.
        epsilon: Desired error bound.
        delta: Failure probability (default 0.05).

    Returns:
        Boolean decision.
    """
    if num_samples <= 0:
        return False
    bound = math.sqrt(math.log(2 / delta) / (2 * num_samples))
    return bound < epsilon


# ----------------------------------------------------------------------
# 2. Bayesian edge reliability (Parent A)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class EdgeBetaPrior:
    """Beta prior parameters for a Bernoulli edge reliability."""
    alpha: float = 1.0
    beta: float = 1.0


def bayesian_edge_update(prior: EdgeBetaPrior, successes: int, failures: int) -> Tuple[float, EdgeBetaPrior]:
    """Update edge reliability posterior with new evidence.

    Args:
        prior: Current Beta prior.
        successes: Number of observed successful transmissions.
        failures: Number of observed failures.

    Returns:
        posterior_mean: Expected reliability P(H|E) in [0,1].
        new_prior: Updated Beta parameters (conjugate posterior).
    """
    new_alpha = prior.alpha + successes
    new_beta = prior.beta + failures
    posterior_mean = new_alpha / (new_alpha + new_beta)
    return posterior_mean, EdgeBetaPrior(new_alpha, new_beta)


# ----------------------------------------------------------------------
# 3. Morphology & Recovery Priority (Parent B)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


def righting_time_index(m: Morphology, b: float = 1.0 / 3.0, k: float = 0.35, neck_lever: float = 1.0) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever


def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """Normalized priority in [0,1] based on righting time."""
    return max(0.0, min(1.0, righting_time_index(m) / max_index))


# ----------------------------------------------------------------------
# 4. SSIM endpoint circuit breaker (Parent B)
# ----------------------------------------------------------------------
def _mean_std(arr: np.ndarray) -> Tuple[float, float]:
    mu = float(np.mean(arr))
    sigma = float(np.std(arr, ddof=1))
    return mu, sigma


def ssim(x: List[float], y: List[float], dynamic_range: float = 255.0,
         k1: float = 0.01, k2: float = 0.03) -> float:
    """Structural Similarity Index Measure between two 1‑D signals."""
    if len(x) != len(y):
        raise ValueError("signals must have the same length")
    x_arr = np.array(x, dtype=float)
    y_arr = np.array(y, dtype=float)

    mu_x, sigma_x = _mean_std(x_arr)
    mu_y, sigma_y = _mean_std(y_arr)
    cov_xy = float(np.cov(x_arr, y_arr, ddof=1)[0, 1])

    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2

    numerator = (2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x ** 2 + sigma_y ** 2 + c2)

    if denominator == 0:
        return 0.0  # to prevent division by zero
    return float(numerator / denominator)


# ----------------------------------------------------------------------
# 5. Tropical max‑plus algebra utilities (Parent A)
# ----------------------------------------------------------------------
def t_add(a: float, b: float) -> float:
    """Tropical addition = max."""
    return max(a, b)


def t_mul(a: float, b: float) -> float:
    """Tropical multiplication = ordinary addition."""
    return a + b


def tropical_path_sum(matrix: np.ndarray, start: int, end: int) -> float:
    """Compute the tropical max‑plus path sum from start to end using
    a simple Floyd‑Warshall‑like relaxation (O(n³))."""
    n = matrix.shape[0]
    # Initialize tropical distance matrix: -inf for unreachable, 0 on diagonal
    dist = np.full((n, n), -np.inf)
    np.fill_diagonal(dist, 0.0)

    # Direct edges (tropical multiplication = addition)
    for i in range(n):
        for j in range(n):
            if not np.isinf(matrix[i, j]):  # assume np.inf marks no edge
                dist[i, j] = t_mul(0.0, matrix[i, j])  # = matrix[i,j]

    # Relaxation
    for k in range(n):
        for i in range(n):
            for j in range(n):
                via_k = t_mul(dist[i, k], dist[k, j])
                dist[i, j] = t_add(dist[i, j], via_k)

    return dist[start, end]


# ----------------------------------------------------------------------
# 6. Hybrid core: weighted tropical max‑plus evaluation
# ---------------------------------------
def compute_composite_weight(p_accept: float, posterior: float, ssim_value: float, recovery_priority: float) -> float:
    return p_accept * posterior * ssim_value * recovery_priority


def tropical_weighted_max_path(cost_matrix: np.ndarray, 
                              p_accepts: np.ndarray, 
                              posteriors: np.ndarray, 
                              ssim_values: np.ndarray, 
                              recovery_priorities: np.ndarray) -> float:
    n = cost_matrix.shape[0]
    weighted_cost_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            p_accept = p_accepts[i, j]
            posterior = posteriors[i, j]
            ssim_value = ssim_values[i, j]
            recovery_priority = recovery_priorities[i]

            composite_weight = compute_composite_weight(p_accept, posterior, ssim_value, recovery_priority)
            weighted_cost_matrix[i, j] = cost_matrix[i, j] * composite_weight

    return tropical_path_sum(weighted_cost_matrix, 0, n - 1)


def improved_hybrid_algorithm(delta_energies: np.ndarray, 
                             temperatures: np.ndarray, 
                             num_samples: np.ndarray, 
                             epsilon: float, 
                             delta: float, 
                             edge_priors: List[EdgeBetaPrior], 
                             successes: np.ndarray, 
                             failures: np.ndarray, 
                             morphologies: List[Morphology], 
                             max_index: float, 
                             signals_x: List[List[float]], 
                             signals_y: List[List[float]], 
                             dynamic_range: float, 
                             k1: float, 
                             k2: float) -> float:
    n = len(delta_energies)

    p_accepts = np.zeros((n, n))
    posteriors = np.zeros((n, n))
    ssim_values = np.zeros((n, n))
    recovery_priorities = np.zeros(n)

    for i in range(n):
        for j in range(n):
            p_accepts[i, j] = acceptance_probability(delta_energies[i], temperatures[i])
            if i == j:
                continue
            prior = edge_priors[i]
            posterior_mean, _ = bayesian_edge_update(prior, successes[i, j], failures[i, j])
            posteriors[i, j] = posterior_mean

        recovery_priorities[i] = recovery_priority(morphologies[i], max_index)

        if i < len(signals_x) and j < len(signals_y):
            ssim_values[i, j] = ssim(signals_x[i], signals_y[j], dynamic_range, k1, k2)

    cost_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                cost_matrix[i, j] = 0.0
            else:
                cost_matrix[i, j] = hoeffding_bound(num_samples[i], epsilon, delta)

    return tropical_weighted_max_path(cost_matrix, p_accepts, posteriors, ssim_values, recovery_priorities)