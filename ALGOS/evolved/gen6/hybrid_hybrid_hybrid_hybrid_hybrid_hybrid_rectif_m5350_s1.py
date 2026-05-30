# DARWIN HAMMER — match 5350, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_nlms_o_m2306_s2.py (gen5)
# parent_b: hybrid_hybrid_rectified_flo_hybrid_hybrid_endpoi_m519_s2.py (gen4)
# born: 2026-05-30T00:01:20Z

"""Hybrid Algorithm Fusion of:
- Parent A: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_nlms_o_m2306_s2.py
- Parent B: hybrid_hybrid_rectified_flo_hybrid_hybrid_endpoi_m519_s2.py

Mathematical Bridge
-------------------
Parent A provides Bayesian probability updates, NLMS adaptive filtering and Euclidean
distances. Parent B supplies a categorical lexical‑semantic vector (LSM) derived from
text, a deterministic hash → 2‑D point mapping, and an endpoint health estimator.

The fusion treats the LSM vector as the NLMS input **x**. The NLMS prediction
produces a scalar *s* that is passed through a sigmoid to become a likelihood
\(L = \sigma(s)\). The prior for the Bayesian update is the health score **H**
computed by the `EndpointCircuitBreaker`. The marginal probability incorporates a
fixed false‑positive term. The resulting posterior probability **P** is then fed
back as the target for the next NLMS weight adaptation, closing the loop between
semantic representation, adaptive filtering, and probabilistic decision making.

Additionally, Euclidean distances between hash‑derived points are used as edge
costs, scaled by the complement of the health score, thereby unifying the graph
topology of Parent B with the probabilistic weighting of Parent A.

The module implements three core hybrid functions that demonstrate this
integration.
"""

import math
import random
import sys
import pathlib
import hashlib
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Utilities from Parent A
# ----------------------------------------------------------------------
Point = Tuple[float, float]


def length(a: Point, b: Point) -> float:
    """Euclidean distance between two 2‑D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """Marginal probability P(E) = L·π + FP·(1‑π)."""
    if not all(0.0 <= x <= 1.0 for x in (prior, likelihood, false_positive)):
        raise ValueError("All probability arguments must lie in [0, 1].")
    return likelihood * prior + false_positive * (1.0 - prior)


def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    """Bayesian posterior π' = π·L / P(E)."""
    if marginal <= 0.0:
        raise ValueError("Marginal probability must be > 0.")
    return prior * likelihood / marginal


def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Linear prediction w·x."""
    return float(weights @ x)


def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, float]:
    """
    Normalized Least‑Mean‑Squares weight update.

    Returns the new weight vector and the prediction error.
    """
    pred = nlms_predict(weights, x)
    error = target - pred
    norm = np.dot(x, x) + eps
    new_weights = weights + (mu / norm) * error * x
    return new_weights, error


# ----------------------------------------------------------------------
# Utilities from Parent B
# ----------------------------------------------------------------------
FUNCTION_CATS: Dict[str, set] = {
    "pronoun": set("i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()),
    "article": set("a an the".split()),
    "preposition": set("about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()),
    "auxiliary": set("am are be been being can could did do does had has have is may might must shall should was were will would".split()),
    "conjunction": set("and but or nor so yet because although while if when where whereas unless until".split()),
    "negation": set("no not never none neither cannot can't won't don't didn't isn't aren't was wasn't weren't".split()),
    "quantifier": set("all any both each few many more most much none several some such".split()),
    "adverb_common": set("very really just still already also even only then there here now often always sometimes".split()),
}


def words(text: str) -> List[str]:
    """Lower‑case tokenisation retaining alphabetic tokens only."""
    return [w for w in (text or "").lower().split() if w.isalpha()]


def lsm_vector(text: str) -> Dict[str, float]:
    """
    Lexical‑semantic (LSM) vector: for each functional category we compute the
    normalized frequency of words belonging to that category.
    """
    ws = words(text)
    total = max(1, len(ws))
    cnt = Counter(ws)
    return {
        cat: sum(cnt[w] for w in vocab) / total
        for cat, vocab in FUNCTION_CATS.items()
    }


def stable_hash(text: str) -> int:
    """Deterministic 256‑bit integer hash."""
    return int(hashlib.sha256(text.encode()).hexdigest(), 16)


def hash_to_point(text: str, scale: float = 1e-5) -> Point:
    """
    Map a string to a reproducible 2‑D point using its stable hash.
    The scale factor keeps coordinates in a numerically convenient range.
    """
    h = stable_hash(text)
    x = (h >> 128) & ((1 << 128) - 1)
    y = h & ((1 << 128) - 1)
    return (x * scale, y * scale)


@dataclass
class EndpointCircuitBreaker:
    """Simple failure counter that opens after a configurable threshold."""
    failure_threshold: int = 3
    failures: int = 0

    def register_failure(self) -> None:
        self.failures += 1

    def reset(self) -> None:
        self.failures = 0

    def compute_health(self, endpoint: str, breaker: str) -> float:
        """
        Health score ∈ [0, 1] decreasing with failures and with the length ratio
        between the breaker string and the endpoint string.
        """
        failures = self.failures
        threshold = self.failure_threshold
        failure_factor = max(0.0, 1.0 - failures / threshold)
        length_factor = 1.0 - len(breaker) / max(1, len(endpoint))
        health = failure_factor * length_factor
        return max(0.0, min(1.0, health))


# ----------------------------------------------------------------------
# Hybrid Core Functions
# ----------------------------------------------------------------------
def compute_hybrid_edge_weight(
    point_a: Point,
    point_b: Point,
    endpoint: str,
    breaker: str,
    circuit: EndpointCircuitBreaker,
) -> float:
    """
    Edge weight = Euclidean distance * (1 - health).

    The health term down‑weights edges connected to poorly performing endpoints.
    """
    dist = length(point_a, point_b)
    health = circuit.compute_health(endpoint, breaker)
    return dist * (1.0 - health)


def compute_semantic_posterior(
    text: str,
    weights: np.ndarray,
    endpoint: str,
    breaker: str,
    circuit: EndpointCircuitBreaker,
    false_positive: float = 0.05,
) -> float:
    """
    End‑to‑end hybrid inference:

    1. Convert *text* to an LSM feature vector x.
    2. Predict a scalar s = w·x with NLMS.
    3. Transform s → likelihood via sigmoid.
    4. Prior = health(endpoint, breaker).
    5. Bayesian posterior = update(prior, likelihood, marginal).

    Returns the posterior probability P ∈ [0, 1].
    """
    # 1. LSM vector → numpy array (ordered by FUNCTION_CATS keys)
    lsm = lsm_vector(text)
    x = np.array([lsm[cat] for cat in FUNCTION_CATS.keys()], dtype=float)

    # 2. NLMS prediction
    s = nlms_predict(weights, x)

    # 3. Sigmoid as likelihood
    likelihood = 1.0 / (1.0 + math.exp(-s))

    # 4. Prior from health score
    prior = circuit.compute_health(endpoint, breaker)

    # 5. Bayesian update
    marginal = bayes_marginal(prior, likelihood, false_positive)
    posterior = bayes_update(prior, likelihood, marginal)
    return posterior


def update_weights_from_feedback(
    weights: np.ndarray,
    text: str,
    target_posterior: float,
    mu: float = 0.5,
) -> Tuple[np.ndarray, float]:
    """
    Perform one NLMS adaptation step where the desired target is the
    *target_posterior* obtained from the Bayesian layer.

    Returns the updated weight vector and the prediction error.
    """
    lsm = lsm_vector(text)
    x = np.array([lsm[cat] for cat in FUNCTION_CATS.keys()], dtype=float)
    new_weights, error = nlms_update(weights, x, target_posterior, mu=mu)
    return new_weights, error


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Example strings
    endpoint_str = "api/v1/resource"
    breaker_str = "circuit"

    # Initialise circuit breaker
    circuit = EndpointCircuitBreaker(failure_threshold=5)

    # Derive deterministic points for two arbitrary identifiers
    p1 = hash_to_point("node_A")
    p2 = hash_to_point("node_B")

    # Compute edge weight
    edge_w = compute_hybrid_edge_weight(p1, p2, endpoint_str, breaker_str, circuit)
    print(f"Hybrid edge weight: {edge_w:.6f}")

    # Initialise NLMS weight vector (dimension equals number of categories)
    dim = len(FUNCTION_CATS)
    np.random.seed(0)
    w = np.random.randn(dim)

    # Sample text
    sample_text = "The quick brown fox jumps over the lazy dog while it runs quickly."

    # Perform hybrid inference
    posterior = compute_semantic_posterior(
        sample_text, w, endpoint_str, breaker_str, circuit
    )
    print(f"Posterior probability after inference: {posterior:.4f}")

    # Simulate feedback: suppose we desire a higher posterior (e.g., 0.8)
    desired = 0.8
    w, err = update_weights_from_feedback(w, sample_text, desired)
    print(f"Weight update error: {err:.6f}")

    # Re‑evaluate after weight adaptation
    posterior2 = compute_semantic_posterior(
        sample_text, w, endpoint_str, breaker_str, circuit
    )
    print(f"Posterior after weight update: {posterior2:.4f}")

    # Demonstrate health degradation after failures
    for _ in range(3):
        circuit.register_failure()
    degraded_health = circuit.compute_health(endpoint_str, breaker_str)
    print(f"Health after failures: {degraded_health:.4f}")

    # Re‑compute edge weight with degraded health
    edge_w2 = compute_hybrid_edge_weight(p1, p2, endpoint_str, breaker_str, circuit)
    print(f"Edge weight with degraded health: {edge_w2:.6f}")