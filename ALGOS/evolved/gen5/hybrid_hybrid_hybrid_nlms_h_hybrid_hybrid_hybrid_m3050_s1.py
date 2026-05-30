# DARWIN HAMMER — match 3050, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_nlms_hybrid_h_hybrid_hybrid_ternar_m611_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_bayes__hybrid_hybrid_hybrid_m1978_s0.py (gen4)
# born: 2026-05-29T23:47:28Z

"""Hybrid Fusion of LMS Adaptive Filtering and Bayesian RBF Similarity
====================================================================

Parent algorithms:
- **hybrid_hybrid_nlms_hybrid_h_hybrid_hybrid_ternar_m611_s1.py** – provides a
  least‑mean‑square (LMS) weight update (`update_ltc`) together with a
  morphology description and an endpoint circuit‑breaker.
- **hybrid_hybrid_hybrid_bayes__hybrid_hybrid_hybrid_m1978_s0.py** – supplies a
  Bayesian posterior update where the likelihood is derived from a Radial Basis
  Function (RBF) similarity measure between a feature vector and a prototype.

**Mathematical bridge**

The RBF similarity `k(x, p)` is a scalar in `[0, 1]`.  We use it in two places:

1. **Bayesian likelihood** – the similarity directly becomes the likelihood
   for a hypothesised class (e.g. “match”).  The posterior therefore reflects
   perceptual similarity of the current observation to a stored prototype.
2. **Adaptive learning rate** – the posterior probability of the “match” class
   scales the LMS step size `μ`.  When the observation is highly similar to the
   prototype the filter adapts quickly; otherwise it adapts conservatively.

The circuit‑breaker halts updates after a configurable number of consecutive
large LMS errors, providing robustness.

The module below implements this fused system with three core functions:
`rbf_similarity`, `bayesian_update`, and `hybrid_lms_update`.  An orchestrating
`hybrid_operation` ties everything together."""

import numpy as np
import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

# ----------------------------------------------------------------------
# Morphology and utility functions (from Parent A)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length

def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)

class EndpointCircuitBreaker:
    """Simple failure‑count circuit breaker."""
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
# Feature extraction (light version of Parent B)
# ----------------------------------------------------------------------
def morphology_to_feature_vector(morph: Morphology) -> np.ndarray:
    """Convert morphology into a numeric feature vector."""
    return np.array([morph.length, morph.width, morph.height, morph.mass], dtype=float)

def extract_full_features(text: str) -> Dict[str, float]:
    """Deterministic pseudo‑random feature set based on a string."""
    features: Dict[str, float] = {}
    rnd = random.Random(hash(text) & 0xFFFFFFFFFFFFFFFF)
    keys = [
        "operator_visceral_ratio", "operator_tech_ratio",
        "operator_legal_osint_ratio", "psyche_forensic_shield_ratio",
        "psyche_poetic_entropy", "psyche_dissociative_index",
        "resilience_bureaucratic_weaponization_index",
        "resilience_resource_exhaustion_metric",
        "resilience_swarm_orchestration_density",
        "rainmaker_corporate_grit_tension", "rainmaker_countdown_density",
        "rainmaker_asset_structuring_weight", "telemetry_agent_symmetry"
    ]
    for key in keys:
        features[key] = rnd.random()
    return features

# ----------------------------------------------------------------------
# Core hybrid mathematics
# ----------------------------------------------------------------------
def rbf_similarity(x: np.ndarray, prototype: np.ndarray, sigma: float = 1.0) -> float:
    """
    Radial Basis Function similarity between two vectors.
    k(x, p) = exp(-||x-p||^2 / (2*sigma^2)),  range (0, 1].
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    diff = x - prototype
    sq_norm = float(np.dot(diff, diff))
    return math.exp(-sq_norm / (2.0 * sigma * sigma))

def bayesian_update(prior: Dict[str, float],
                    likelihood: Dict[str, float]) -> Tuple[Dict[str, float], float]:
    """
    Perform a single‑step Bayesian update.
    posterior_c = (likelihood_c * prior_c) / evidence
    Returns posterior dict and the evidence term.
    """
    unnorm = {c: likelihood[c] * prior.get(c, 0.0) for c in likelihood}
    evidence = sum(unnorm.values())
    if evidence == 0:
        # avoid division by zero – fall back to prior
        posterior = prior.copy()
    else:
        posterior = {c: unnorm[c] / evidence for c in unnorm}
    return posterior, evidence

def hybrid_lms_update(weights: List[float],
                     x: List[float],
                     target: float,
                     mu_base: float,
                     posterior_match: float,
                     eps: float = 1e-9) -> Tuple[List[float], float]:
    """
    LMS weight update whose step size is scaled by the posterior probability
    of the “match” hypothesis.
        μ_eff = μ_base * posterior_match
    Returns the new weight list and the instantaneous error.
    """
    if len(weights) != len(x):
        raise ValueError("weights and input must have equal length")
    mu_eff = mu_base * posterior_match
    # Guard against degenerate μ
    mu_eff = max(min(mu_eff, 1.9), 1e-6)

    y = sum(w * xi for w, xi in zip(weights, x))
    error = target - y
    power = sum(xi * xi for xi in x) + eps
    next_weights = [w + mu_eff * error * xi / power for w, xi in zip(weights, x)]
    return next_weights, error

# ----------------------------------------------------------------------
# High‑level hybrid operation
# ----------------------------------------------------------------------
def hybrid_operation(morphology: Morphology,
                     circuit_breaker: EndpointCircuitBreaker,
                     weights: List[float],
                     x: List[float],
                     target: float,
                     prototype_vec: np.ndarray,
                     prior: Dict[str, float],
                     mu_base: float = 0.5,
                     sigma: float = 1.0) -> Tuple[List[float], Dict[str, float], bool]:
    """
    Execute one hybrid step:
      1. Compute RBF similarity between current feature vector and prototype.
      2. Build a likelihood dict for two classes: 'match' and 'nonmatch'.
      3. Perform Bayesian update → posterior.
      4. If the circuit breaker permits, run LMS update with μ scaled by
         posterior['match']; otherwise keep weights unchanged.
      5. Record success/failure based on LMS error magnitude.
    Returns (updated_weights, posterior, circuit_open_flag).
    """
    if not circuit_breaker.allow():
        return weights, prior, True

    # Feature vector from morphology (could be extended with text features)
    feat_vec = morphology_to_feature_vector(morphology)

    # 1. Similarity
    sim = rbf_similarity(feat_vec, prototype_vec, sigma)

    # 2. Likelihood (simple mapping)
    likelihood = {
        "match": sim,
        "nonmatch": 1.0 - sim
    }

    # 3. Bayesian posterior
    posterior, _ = bayesian_update(prior, likelihood)

    # 4. LMS update conditioned on circuit breaker
    updated_weights, error = hybrid_lms_update(
        weights, x, target, mu_base, posterior.get("match", 0.0)
    )

    # 5. Record circuit‑breaker status
    tolerance = 0.01 * abs(target)  # relative tolerance
    if abs(error) <= tolerance:
        circuit_breaker.record_success()
    else:
        circuit_breaker.record_failure()

    return updated_weights, posterior, circuit_breaker.open

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a simple morphology instance
    morph = Morphology(length=2.0, width=1.5, height=1.0, mass=0.8)

    # Endpoint circuit breaker with default threshold
    cb = EndpointCircuitBreaker(failure_threshold=3)

    # Initial LMS weights (same dimension as input vector)
    init_weights = [0.1, -0.2, 0.05]

    # Input vector x and target output
    x_vec = [0.7, -1.2, 0.3]
    target_val = 0.4

    # Prototype vector for similarity (same size as morphology feature vector)
    prototype = np.array([2.0, 1.5, 1.0, 0.8])  # identical to morph -> sim≈1

    # Prior probabilities for the two hypotheses
    prior_probs = {"match": 0.6, "nonmatch": 0.4}

    # Run a few hybrid steps
    weights = init_weights
    for step in range(5):
        weights, posterior, broken = hybrid_operation(
            morphology=morph,
            circuit_breaker=cb,
            weights=weights,
            x=x_vec,
            target=target_val,
            prototype_vec=prototype,
            prior=prior_probs,
            mu_base=0.5,
            sigma=0.5
        )
        print(f"Step {step+1}:")
        print(f"  Weights   = {weights}")
        print(f"  Posterior = {posterior}")
        print(f"  CB open   = {broken}")
        # Update prior for next iteration
        prior_probs = posterior
        if broken:
            print("  Circuit breaker opened – stopping further updates.")
            break
    print("Smoke test completed without errors.")