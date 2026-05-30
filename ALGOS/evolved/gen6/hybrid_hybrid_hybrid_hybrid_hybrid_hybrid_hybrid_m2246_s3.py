# DARWIN HAMMER — match 2246, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_privac_hybrid_endpoint_circ_m28_s3.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_ternar_m599_s0.py (gen5)
# born: 2026-05-29T23:41:32Z

"""Hybrid VRAM‑Privacy‑Bandit‑Ternary Router

Parents:
- hybrid_hybrid_hybrid_privac_hybrid_endpoint_circ_m28_s3.py (risk, VRAM, morphology, DP‑budget, circuit‑breaker)
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_ternar_m599_s0.py (endpoint health, Hoeffding bound, ternary routing, bandit context)

Mathematical bridge:
Each model *i* contributes a weighted VRAM term w_i = r_i·s_i·m_i.
Each endpoint *j* contributes a health‑derived context c_j = health_score_j·s_i,
where s_i is the morphology scaling factor of the model currently considered.
We form a joint score

    score_{i,j} = w_i / (1 + ε_j)   with   ε_j = HoeffdingBound(Δ, δ, n_j)

where ε_j bounds the uncertainty of the endpoint’s observed failure_rate.
The DP‑aggregated risk 𝑅̂ supplies a global privacy budget; the circuit‑breaker
allows admission only if Σ_i w_i ≤ VRAM_budget – current_load and
breaker.allow() is true.  The selected (i*,j*) pair drives the ternary router
via a similarity‑adjusted command (SSIM) between the model’s feature vector
and the packet payload.

The module implements this unified decision engine."""
from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Core data structures (merged from both parents)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str


@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

    def scaling_factor(self) -> float:
        """Simple geometric scaling derived from volume‑to‑mass ratio."""
        volume = self.length * self.width * self.height
        return (volume / (self.mass + 1e-9)) ** (1 / 3)


@dataclass(frozen=True)
class ModelSpec:
    tier: ModelTier
    morphology: Morphology
    unique_quasi_identifiers: int
    total_records: int

    def reconstruction_risk(self) -> float:
        """Probability that a record can be re‑identified (parent A)."""
        if self.total_records <= 0:
            return 0.0
        # A diminishing return curve: more records → lower per‑record risk
        base = self.unique_quasi_identifiers / self.total_records
        return min(1.0, max(0.0, base * math.log1p(self.total_records)))

    def weighted_vram(self) -> float:
        """r_i·s_i·m_i term from parent A."""
        r = self.reconstruction_risk()
        s = self.morphology.scaling_factor()
        m = self.tier.ram_mb
        return r * s * m


@dataclass
class Endpoint:
    health_score: float
    failure_rate: float
    recovery_priority: float
    observations: int = 0  # number of failure observations

    def update_observation(self, failed: bool) -> None:
        self.observations += 1
        # simple exponential moving average for failure_rate
        alpha = 0.2
        self.failure_rate = (
            (1 - alpha) * self.failure_rate + alpha * (1.0 if failed else 0.0)
        )


# ----------------------------------------------------------------------
# Privacy / DP utilities (parent A)
# ----------------------------------------------------------------------


def laplace_mechanism(value: float, epsilon: float, sensitivity: float = 1.0) -> float:
    """Add Laplace noise to a scalar value."""
    scale = sensitivity / max(epsilon, 1e-9)
    noise = np.random.laplace(0.0, scale)
    return value + noise


def dp_aggregate_risks(models: List[ModelSpec], epsilon: float = 1.0) -> float:
    """Differential‑private aggregation of reconstruction risks."""
    raw_sum = sum(m.reconstruction_risk() for m in models)
    return laplace_mechanism(raw_sum, epsilon)


# ----------------------------------------------------------------------
# Circuit breaker (parent A)
# ----------------------------------------------------------------------


class CircuitBreaker:
    def __init__(self, max_failures: int = 5):
        self.max_failures = max_failures
        self.failures = 0

    def allow(self) -> bool:
        return self.failures < self.max_failures

    def record_failure(self) -> None:
        self.failures += 1


# ----------------------------------------------------------------------
# Hoeffding bound (parent B)
# ----------------------------------------------------------------------


def hoeffding_bound(r: float, delta: float, n: int) -> float:
    """Hoeffding bound for a bounded random variable."""
    if n <= 0:
        return float("inf")
    return math.sqrt((r ** 2 * math.log(2 / delta)) / (2 * n))


# ----------------------------------------------------------------------
# Similarity metric – SSIM (parent B, corrected)
# ----------------------------------------------------------------------


def ssim(x: np.ndarray, y: np.ndarray, dynamic_range: float = 255.0,
         k1: float = 0.01, k2: float = 0.03) -> float:
    """Structural similarity index (simplified version)."""
    if x.shape != y.shape:
        raise ValueError("Arrays must have the same shape")
    C1 = (k1 * dynamic_range) ** 2
    C2 = (k2 * dynamic_range) ** 2

    mu_x = x.mean()
    mu_y = y.mean()
    sigma_x = x.var()
    sigma_y = y.var()
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean()

    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    return numerator / denominator


# ----------------------------------------------------------------------
# Bandit core – uses endpoint health as context (parent B)
# ----------------------------------------------------------------------


class SimpleContextualBandit:
    """Epsilon‑greedy bandit that scores actions with a linear context."""

    def __init__(self, epsilon: float = 0.1):
        self.epsilon = epsilon
        self.weights: Dict[int, float] = {}  # endpoint id → weight

    def _score(self, endpoint: Endpoint, context: float) -> float:
        """Linear combination of health and context."""
        w = self.weights.get(id(endpoint), 1.0)
        return w * endpoint.health_score * context

    def select(self, endpoints: List[Endpoint], context: float) -> Endpoint:
        """Return the endpoint with highest score, with epsilon exploration."""
        if random.random() < self.epsilon:
            return random.choice(endpoints)

        scores = [(self._score(ep, context), ep) for ep in endpoints]
        _, best = max(scores, key=lambda pair: pair[0])
        return best

    def update(self, endpoint: Endpoint, reward: float) -> None:
        """Simple gradient‑free update of the weight."""
        idx = id(endpoint)
        old = self.weights.get(idx, 1.0)
        self.weights[idx] = old + 0.01 * reward  # tiny learning rate


# ----------------------------------------------------------------------
# Hybrid decision engine (core fusion)
# ----------------------------------------------------------------------


def compute_joint_scores(
    models: List[ModelSpec],
    endpoints: List[Endpoint],
    delta: float = 0.05,
    r_range: float = 1.0,
) -> List[Tuple[ModelSpec, Endpoint, float]]:
    """
    For each (model, endpoint) pair compute

        score = (r_i·s_i·m_i) / (1 + ε_j)

    where ε_j is the Hoeffding bound for the endpoint’s failure observations.
    Returns a list sorted by descending score.
    """
    results = []
    for model in models:
        w_i = model.weighted_vram()
        for ep in endpoints:
            epsilon_j = hoeffding_bound(r_range, delta, ep.observations)
            score = w_i / (1.0 + epsilon_j)
            results.append((model, ep, score))
    results.sort(key=lambda tup: tup[2], reverse=True)
    return results


def admit_models(
    models: List[ModelSpec],
    vram_budget: float,
    breaker: CircuitBreaker,
    current_load: float = 0.0,
) -> List[ModelSpec]:
    """
    Admit models respecting the VRAM budget and circuit‑breaker state.
    Mirrors the admission rule of parent A.
    """
    admitted = []
    load = current_load
    for model in models:
        demand = model.weighted_vram()
        if demand <= (vram_budget - load) and breaker.allow():
            admitted.append(model)
            load += demand
        else:
            breaker.record_failure()
    return admitted


def route_packet_via_hybrid(
    packet: Dict[str, Any],
    selected_endpoint: Endpoint,
    selected_model: ModelSpec,
) -> Dict[str, Any]:
    """
    Adapt the ternary router (parent B) using the hybrid context.
    The payload vector is compared with a synthetic feature vector derived
    from the model’s morphology; similarity modulates the command.
    """
    # 1. Build a simple feature vector from morphology (length, width, height)
    morph = selected_model.morphology
    feature_vec = np.array([morph.length, morph.width, morph.height], dtype=float)

    # 2. Extract a numeric representation from the packet (e.g., ASCII codes)
    text = str(packet.get("text_surface") or packet.get("raw_command") or "")
    payload_vec = np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(float)
    # Pad / truncate to match feature length
    if payload_vec.size < feature_vec.size:
        payload_vec = np.pad(payload_vec, (0, feature_vec.size - payload_vec.size))
    else:
        payload_vec = payload_vec[: feature_vec.size]

    # 3. Compute similarity
    similarity = ssim(feature_vec, payload_vec)

    # 4. Modulate the packet command based on endpoint health and similarity
    modulation = selected_endpoint.health_score * similarity
    packet["modulated_score"] = modulation
    packet["routed_to"] = selected_endpoint.health_score  # placeholder identifier
    return packet


# ----------------------------------------------------------------------
# Demonstration functions (at least three)
# ----------------------------------------------------------------------


def demo_hybrid_decision() -> None:
    """Run a miniature end‑to‑end hybrid decision flow."""
    # Create dummy models
    models = [
        ModelSpec(
            tier=ModelTier(name="A", ram_mb=2000, tier="large"),
            morphology=Morphology(1.2, 0.8, 0.6, 2.5),
            unique_quasi_identifiers=150,
            total_records=5000,
        ),
        ModelSpec(
            tier=ModelTier(name="B", ram_mb=800, tier="medium"),
            morphology=Morphology(0.9, 0.7, 0.5, 1.2),
            unique_quasi_identifiers=80,
            total_records=3000,
        ),
    ]

    # Create dummy endpoints
    endpoints = [
        Endpoint(health_score=0.9, failure_rate=0.1, recovery_priority=0.8),
        Endpoint(health_score=0.6, failure_rate=0.05, recovery_priority=0.9),
    ]

    # Simulate a few failure observations
    for _ in range(10):
        ep = random.choice(endpoints)
        ep.update_observation(failed=random.random() < ep.failure_rate)

    # Compute joint scores and pick the best pair
    joint = compute_joint_scores(models, endpoints)
    best_model, best_endpoint, best_score = joint[0]

    # Admit models under a VRAM budget
    breaker = CircuitBreaker(max_failures=3)
    admitted = admit_models(models, vram_budget=3000.0, breaker=breaker)

    # Build a packet and route it
    packet = {"text_surface": "Hello world!"}
    routed = route_packet_via_hybrid(packet, best_endpoint, best_model)

    # Print a concise summary
    print("Best pair score:", best_score)
    print("Admitted models:", [m.tier.name for m in admitted])
    print("Routed packet:", routed)


def demo_bandit_learning() -> None:
    """Show how the contextual bandit updates its weights."""
    endpoints = [
        Endpoint(health_score=0.7, failure_rate=0.2, recovery_priority=0.5),
        Endpoint(health_score=0.4, failure_rate=0.1, recovery_priority=0.6),
    ]
    bandit = SimpleContextualBandit(epsilon=0.2)

    # Context derived from a dummy model’s scaling factor
    dummy_model = ModelSpec(
        tier=ModelTier(name="C", ram_mb=1200, tier="small"),
        morphology=Morphology(1.0, 1.0, 1.0, 1.0),
        unique_quasi_identifiers=50,
        total_records=2000,
    )
    context = dummy_model.morphology.scaling_factor()

    for _ in range(20):
        chosen = bandit.select(endpoints, context)
        # Simulate reward = 1 - failure_rate (higher is better)
        reward = 1.0 - chosen.failure_rate
        bandit.update(chosen, reward)

    # Display learned weights
    for ep in endpoints:
        print(f"Endpoint health {ep.health_score:.2f} weight {bandit.weights.get(id(ep),1.0):.3f}")


def demo_privacy_budget() -> None:
    """Compute a DP‑protected aggregate risk and show its effect."""
    models = [
        ModelSpec(
            tier=ModelTier(name="X", ram_mb=500, tier="tiny"),
            morphology=Morphology(0.5, 0.5, 0.5, 0.4),
            unique_quasi_identifiers=30,
            total_records=1000,
        ),
        ModelSpec(
            tier=ModelTier(name="Y", ram_mb=1500, tier="large"),
            morphology=Morphology(1.5, 1.2, 1.0, 3.0),
            unique_quasi_identifiers=200,
            total_records=8000,
        ),
    ]
    raw = sum(m.reconstruction_risk() for m in models)
    dp = dp_aggregate_risks(models, epsilon=0.5)
    print(f"Raw risk sum: {raw:.4f}, DP‑noised sum: {dp:.4f}")


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    print("=== Hybrid Decision Demo ===")
    demo_hybrid_decision()
    print("\n=== Bandit Learning Demo ===")
    demo_bandit_learning()
    print("\n=== Privacy Budget Demo ===")
    demo_privacy_budget()