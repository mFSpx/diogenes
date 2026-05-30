# DARWIN HAMMER — match 3237, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_fisher_locali_hybrid_minimum_cost__m29_s2.py (gen2)
# parent_b: hybrid_hybrid_hybrid_caputo_hybrid_hybrid_cockpi_m2468_s0.py (gen4)
# born: 2026-05-29T23:48:37Z

"""Hybrid Fisher–Caputo–Trust Algorithm
Parents:
- hybrid_hybrid_fisher_locali_hybrid_minimum_cost__m29_s2.py (Gaussian beam, Fisher score, Bayesian marginal)
- hybrid_hybrid_hybrid_caputo_hybrid_hybrid_cockpi_m2468_s0.py (Caputo fractional derivative, Lanczos gamma approximation, trust‑weighted velocity)

Mathematical bridge:
Both parents manipulate probability‑like quantities (Fisher information and Bayesian priors) and weight them with
derivative‑based kernels (Gaussian derivative vs. Caputo fractional derivative).  The fusion treats the
Fisher scores of a temporal signal as a discrete time series, applies a Caputo fractional derivative to obtain
a fractional information flux, and then multiplies this flux by a trust‑weighted velocity field.  The resulting
scalar is used as a fractional‑weighted likelihood in a Bayesian update, closing the loop between the
probabilistic and fractional‑calculus components.
"""

import math
import random
import sys
import pathlib
from datetime import datetime
import numpy as np

# ----------------------------------------------------------------------
# Core pieces from Parent A
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian kernel centred at *center* with standard deviation *width*."""
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """
    Fisher information for a single Gaussian observation.
    Implements (dI/dθ)^2 / I where I is the Gaussian intensity.
    """
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def gaussian_filter(data: np.ndarray, sigma: float) -> np.ndarray:
    """Apply a 1‑D Gaussian smoothing filter using the Gaussian beam."""
    return np.array([gaussian_beam(x, 0.0, sigma) for x in data])


# ----------------------------------------------------------------------
# Core pieces from Parent B
# ----------------------------------------------------------------------
def lanczos_approximation(z: float) -> float:
    """Approximate Γ(z) using a Lanczos series (fallback for negative arguments)."""
    p = np.array([0.99999999999980993, 676.5203681218851, -1259.1392167224028,
                  771.32342877765313, -176.61502916214059, 12.507343278686905,
                  -0.13857109526572012, 9.9843695780195716e-6,
                  1.5056327351493116e-7])
    g = 7
    if z < 0.5:
        # Reflection formula
        return math.pi / (math.sin(math.pi * z) * lanczos_approximation(1 - z))
    z = z + g + 0.5
    x = p[0]
    for i in range(1, len(p)):
        x += p[i] / (z - i)
    t = z - 0.5
    return math.sqrt(2 * math.pi) * t ** (z - 0.5) * math.exp(-t) * x


def caputo_derivative(f: np.ndarray, t: np.ndarray, alpha: float) -> np.ndarray:
    """
    Discrete Caputo fractional derivative of order *alpha* for a scalar signal f(t).
    Uses the definition D^α f(t_n) ≈ Σ_{k=0}^{n-1} (f_{k+1}-f_k) (t_n - t_k)^{-α} / Γ(1-α).
    """
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0,1) for the simple implementation")
    dt = np.diff(t)
    df = np.diff(f)
    # Build matrix of (t_n - t_k)^{-α}
    n = len(t)
    kernel = np.zeros((n, n - 1))
    for i in range(n):
        for k in range(i):
            kernel[i, k] = (t[i] - t[k]) ** (-alpha)
    integral = (df * dt ** (-alpha)) / math.gamma(1 - alpha)  # element‑wise product
    # Cumulative sum weighted by kernel
    result = np.dot(kernel, integral)
    return result


def gamma_term(t: np.ndarray, alpha: float, sum_j_gamma: float) -> np.ndarray:
    """Vectorised gamma weighting term used in the fractional product."""
    vec = np.vectorize(lambda ti: math.gamma(lanczos_approximation(ti + alpha)))
    return vec(t) / sum_j_gamma


def trust_weighted_velocity(x0: np.ndarray, x1: np.ndarray, h: float) -> np.ndarray:
    """Linear trust‑weighted velocity field: v = h·(x1‑x0)."""
    return h * (x1 - x0)


# ----------------------------------------------------------------------
# Hybrid functions (the new unified topology)
# ----------------------------------------------------------------------
def hybrid_fisher_fractional_score(
    theta_series: np.ndarray,
    center: float,
    width: float,
    times: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """
    Compute Fisher scores for a temporal series *theta_series* and then apply a
    Caputo fractional derivative to obtain the fractional information flux.
    Returns an array of the same length as *times* (first element is zero).
    """
    if theta_series.shape != times.shape:
        raise ValueError("theta_series and times must have identical shapes")
    # Fisher scores point‑wise
    vec_fisher = np.vectorize(lambda th: fisher_score(th, center, width))
    scores = vec_fisher(theta_series)
    # Fractional derivative (order alpha)
    flux = caputo_derivative(scores, times, alpha)
    # Prepend zero to keep array length consistent
    return np.insert(flux, 0, 0.0)


def trust_weighted_fisher_energy(
    state_prev: np.ndarray,
    state_next: np.ndarray,
    h: float,
    fisher_val: float,
) -> float:
    """
    Combine the trust‑weighted velocity with a scalar Fisher information value.
    The resulting energy is v·fisher_val, mirroring the “trust‑weighted fractional
    weighted product” described in Parent B.
    """
    v = trust_weighted_velocity(state_prev, state_next, h)
    # Use the L2 norm of the velocity as a scalar magnitude
    v_norm = np.linalg.norm(v)
    return v_norm * fisher_val


def bayesian_fractional_update(
    prior: float,
    evidence: np.ndarray,
    times: np.ndarray,
    alpha: float,
    h: float,
    state_prev: np.ndarray,
    state_next: np.ndarray,
) -> float:
    """
    Perform a Bayesian update where the likelihood is weighted by a fractional
    product of the Caputo derivative of the evidence and a trust‑weighted
    Fisher energy term.

    Steps:
    1. Normalise *evidence* to obtain a likelihood vector.
    2. Compute the fractional derivative of the likelihood.
    3. Compute the gamma weighting term.
    4. Form the fractional weighted product.
    5. Multiply by the trust‑weighted Fisher energy (scalar) to obtain a
       fractional‑weighted likelihood.
    6. Apply the standard Bayesian update: posterior ∝ prior × likelihood.
    """
    if not (0 < prior <= 1):
        raise ValueError("prior must be a probability in (0,1]")
    if evidence.shape != times.shape:
        raise ValueError("evidence and times must share shape")
    # 1. Likelihood normalisation
    lik = evidence.astype(float)
    lik_sum = lik.sum()
    if lik_sum == 0:
        raise ValueError("evidence vector sums to zero")
    lik /= lik_sum

    # 2. Fractional derivative of the likelihood
    frac_der = caputo_derivative(lik, times, alpha)

    # 3. Gamma weighting term
    sum_j_gamma = np.sum([math.gamma(lanczos_approximation(ti + alpha)) for ti in times])
    gamma_w = gamma_term(times, alpha, sum_j_gamma)

    # 4. Fractional weighted product (vector)
    weighted_product = frac_der * gamma_w

    # 5. Trust‑weighted Fisher energy scalar
    # For the Fisher component we evaluate a single Fisher score at the mean theta
    mean_theta = np.mean(evidence)  # a proxy for a representative observation
    fisher_val = fisher_score(mean_theta, center=mean_theta, width=1.0)
    energy = trust_weighted_fisher_energy(state_prev, state_next, h, fisher_val)

    # Combine vector and scalar
    fractional_likelihood = weighted_product * energy
    # Normalise again to keep a proper probability distribution
    fl_sum = fractional_likelihood.sum()
    if fl_sum == 0:
        fractional_likelihood = np.full_like(fractional_likelihood, 1.0 / len(fractional_likelihood))
        fl_sum = 1.0
    fractional_likelihood /= fl_sum

    # 6. Bayesian update
    posterior_unnorm = prior * fractional_likelihood
    posterior = posterior_unnorm / posterior_unnorm.sum()
    # Return the mean posterior as a single representative probability
    return float(posterior.mean())


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic temporal data
    rng = np.random.default_rng(42)
    times = np.linspace(0, 10, 50)
    theta_series = rng.normal(loc=0.0, scale=1.0, size=times.shape)

    # Hybrid Fisher–fractional flux
    flux = hybrid_fisher_fractional_score(theta_series, center=0.0, width=1.0,
                                          times=times, alpha=0.3)
    print("Fractional Fisher flux (first 5):", flux[:5])

    # Trust‑weighted energy example
    state_prev = np.array([1.0, 2.0, 3.0])
    state_next = np.array([1.5, 2.5, 3.5])
    energy = trust_weighted_fisher_energy(state_prev, state_next, h=0.8,
                                          fisher_val=fisher_score(0.2, 0.0, 1.0))
    print("Trust‑weighted Fisher energy:", energy)

    # Bayesian fractional update
    evidence = np.abs(rng.normal(loc=0.0, scale=1.0, size=times.shape))
    posterior_mean = bayesian_fractional_update(
        prior=0.6,
        evidence=evidence,
        times=times,
        alpha=0.4,
        h=0.5,
        state_prev=state_prev,
        state_next=state_next,
    )
    print("Posterior mean after fractional Bayesian update:", posterior_mean)