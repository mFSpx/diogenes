# DARWIN HAMMER — match 2596, survivor 3
# gen: 4
# parent_a: hybrid_hybrid_distributed_l_hybrid_hoeffding_tre_m24_s6.py (gen2)
# parent_b: hybrid_hybrid_fisher_locali_hybrid_hybrid_decisi_m26_s6.py (gen3)
# born: 2026-05-29T23:43:02Z

"""Hybrid Algorithm integrating Probabilistic Annealing, Hoeffding‑Tropical Bounds,
Gaussian‑Fisher information and Semantic Regex features.

Parents:
- hybrid_hybrid_distributed_l_hybrid_hoeffding_tre_m24_s6 (probabilistic primitives,
  Hoeffding bound, tropical algebra)
- hybrid_hybrid_fisher_locali_hybrid_hybrid_decisi_m26_s6 (Gaussian beam,
  Fisher information, SSIM, regex‑based semantic weighting)

Mathematical bridge:
Both families manipulate quantities that are naturally expressed in the log‑domain.
The tropical semiring (max,+) is precisely the algebra of log‑probabilities:
tropical addition ↔ max corresponds to log‑sum‑exp approximation,
tropical multiplication ↔ + corresponds to multiplication of probabilities.
We therefore map:

* Gaussian intensity * Fisher information → tropical product (addition of logs)
* Hoeffding confidence intervals on SSIM differences → additive uncertainty term
* Acceptance probability (exp(‑ΔE/T)) → log‑probability ΔE/T, which is combined
  with broadcast probability via tropical addition (max) to obtain a unified
  acceptance score.

The code below implements three core hybrid operations that demonstrate this
fusion."""


import sys
import math
import random
import re
from pathlib import Path
import numpy as np

# ----------------------------------------------------------------------
# Parent A – probabilistic primitives & tropical algebra
# ----------------------------------------------------------------------
def broadcast_probability(total_phases: int, current_phase: int) -> float:
    if total_phases < 1 or current_phase < 1:
        raise ValueError('phases and phase must be positive')
    return min(1.0, 1.0 / (2 ** max(0, total_phases - current_phase)))


def acceptance_probability(delta_e: float, temperature: float) -> float:
    if delta_e < 0:
        return 1.0
    if temperature <= 0:
        return 0.0
    return math.exp(-delta_e / temperature)


def cooling_temperature(k: int, t0: float = 1.0, alpha: float = 0.95) -> float:
    if k < 0 or t0 < 0 or not (0 <= alpha <= 1):
        raise ValueError("invalid cooling parameters")
    return t0 * (alpha ** k)


def hoeffding_bound(r: float, delta: float, n: int) -> float:
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0 required")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))


def t_add(x, y):
    """Tropical addition = max."""
    return np.maximum(x, y)


def t_mul(x, y):
    """Tropical multiplication = + (addition of logs)."""
    return np.add(x, y)


def t_polyval(coeffs, x):
    """
    Tropical polynomial evaluation.
    coeffs[i] + i*x, then max over i.
    """
    coeffs = np.asarray(coeffs, dtype=float)
    x = np.asarray(x, dtype=float)
    exponents = np.arange(len(coeffs), dtype=float).reshape((-1,) + (1,) * x.ndim)
    terms = coeffs.reshape((-1,) + (1,) * x.ndim) + exponents * x
    return np.max(terms, axis=0)


# ----------------------------------------------------------------------
# Parent B – Gaussian/Fisher, SSIM and semantic regex
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian beam intensity."""
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for the Gaussian beam."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def ssim(x: np.ndarray, y: np.ndarray,
         dynamic_range: float = 255.0,
         k1: float = 0.01,
         k2: float = 0.03) -> float:
    """Structural Similarity Index Measure for 1‑D signals."""
    if x.shape != y.shape:
        raise ValueError('samples must have equal shape')
    if x.size == 0:
        raise ValueError('samples must not be empty')
    mx = np.mean(x)
    my = np.mean(y)
    vx = np.var(x)
    vy = np.var(y)
    cov = np.mean((x - mx) * (y - my))
    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2
    return ((2 * mx * my + c1) * (2 * cov + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))


# Regex patterns for semantic feature categories
EVIDENCE_RE = re.compile(r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b", re.I)
PLANNING_RE = re.compile(r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b", re.I)
DELAY_RE    = re.compile(r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|before i|first|after|review)\b", re.I)


def semantic_weight(text: str) -> float:
    """Assign a scalar weight based on the presence of semantic regex categories."""
    evidence = len(EVIDENCE_RE.findall(text))
    planning = len(PLANNING_RE.findall(text))
    delay = len(DELAY_RE.findall(text))
    # Positive evidence and planning increase weight, delays decrease it.
    return evidence * 0.6 + planning * 0.3 - delay * 0.4


# ----------------------------------------------------------------------
# Hybrid core functions (demonstrate the fused mathematics)
# ----------------------------------------------------------------------
def tropical_fisher(theta: float, center: float, width: float) -> float:
    """
    Combine Gaussian intensity and Fisher information in the tropical semiring.
    The log‑intensity and log‑Fisher are added (tropical multiplication) and
    then the maximum over the two representations is taken (tropical addition).
    """
    intensity = gaussian_beam(theta, center, width)
    fisher = fisher_score(theta, center, width)

    # Work in log‑domain to respect tropical algebra
    log_intensity = math.log(max(intensity, 1e-12))
    log_fisher = math.log(max(fisher, 1e-12))

    # Tropical multiplication = addition of logs
    combined = t_mul(log_intensity, log_fisher)   # = log_intensity + log_fisher

    # Tropical addition with the original logs (max)
    result = t_add(combined, np.array([log_intensity, log_fisher]))
    # Return to linear domain for downstream use
    return math.exp(float(result))


def hybrid_acceptance(delta_gain: float,
                      temperature_step: int,
                      total_phases: int,
                      current_phase: int,
                      text_context: str) -> float:
    """
    Compute a unified acceptance probability that merges:
    - Simulated‑annealing acceptance (exp(-ΔE/T))
    - Broadcast probability that decays with phase
    - Semantic weight from regex features that biases ΔE
    All quantities are combined using tropical addition (max) in the log‑domain.
    """
    # Adjust delta_gain with semantic bias (negative bias encourages acceptance)
    bias = -semantic_weight(text_context) * 0.05   # small scaling
    adjusted_delta = delta_gain + bias

    temperature = cooling_temperature(temperature_step)
    prob_anneal = acceptance_probability(adjusted_delta, temperature)

    prob_broadcast = broadcast_probability(total_phases, current_phase)

    # Convert to log‑probability, combine with tropical max, then back
    log_anneal = math.log(max(prob_anneal, 1e-12))
    log_broadcast = math.log(max(prob_broadcast, 1e-12))

    log_combined = t_add(log_anneal, log_broadcast)   # max in log‑space
    return math.exp(float(log_combined))


def hybrid_split_criterion(ssim_left: np.ndarray,
                           ssim_right: np.ndarray,
                           n_samples: int,
                           delta: float,
                           tropical_coeffs: list,
                           theta: float,
                           center: float,
                           width: float) -> bool:
    """
    Decide whether to split a node using a hybrid criterion:
    1. Compute the SSIM difference and bound it with Hoeffding.
    2. Evaluate a tropical polynomial on the Fisher‑derived gain.
    3. Accept split if the tropical gain exceeds the Hoeffding bound.
    """
    # 1. SSIM based confidence
    mu = ssim(ssim_left, ssim_right)
    bound = hoeffding_bound(r=1.0, delta=delta, n=n_samples)   # r=1 for normalized SSIM
    confident = mu - bound > 0.5   # arbitrary threshold for illustration

    # 2. Tropical gain from Fisher information
    gain = tropical_fisher(theta, center, width)

    # 3. Tropical polynomial evaluation (acts as a non‑linear scaling)
    scaled_gain = t_polyval(tropical_coeffs, gain)

    # Decision
    return confident and (scaled_gain > bound)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic data for testing
    theta_test = 0.75
    center_test = 0.5
    width_test = 0.2

    # Test tropical_fisher
    tf = tropical_fisher(theta_test, center_test, width_test)
    print(f"Tropical Fisher output: {tf:.6f}")

    # Test hybrid_acceptance
    accept = hybrid_acceptance(
        delta_gain=0.12,
        temperature_step=3,
        total_phases=10,
        current_phase=4,
        text_context="The plan was verified with supporting evidence."
    )
    print(f"Hybrid acceptance probability: {accept:.6f}")

    # Prepare SSIM arrays
    rng = np.random.default_rng(42)
    left = rng.random(100)
    right = left + rng.normal(0, 0.05, size=100)   # slightly perturbed

    split = hybrid_split_criterion(
        ssim_left=left,
        ssim_right=right,
        n_samples=100,
        delta=0.05,
        tropical_coeffs=[0.1, 0.3, 0.5],
        theta=theta_test,
        center=center_test,
        width=width_test
    )
    print(f"Hybrid split decision: {'split' if split else 'no split'}")