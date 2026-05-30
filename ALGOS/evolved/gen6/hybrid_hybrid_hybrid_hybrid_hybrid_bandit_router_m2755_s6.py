# DARWIN HAMMER — match 2755, survivor 6
# gen: 6
# parent_a: hybrid_hybrid_hybrid_endpoi_hybrid_hybrid_hybrid_m976_s6.py (gen5)
# parent_b: hybrid_bandit_router_hybrid_hybrid_hybrid_m2649_s4.py (gen5)
# born: 2026-05-29T23:45:38Z

"""Hybrid Morphology‑Bandit Router

This module fuses two distinct parent algorithms:

* **Parent A** – provides a physical *Morphology* description and an
  *EndpointCircuitBreaker* that tracks failures. Its core scalar is the
  *Fisher score* `f(morph)` computed from the morphology vector.
* **Parent B** – implements a contextual bandit with an empirical reward
  estimator `r̂_a = total_reward_a / count_a` and a non‑parametric RBF
  surrogate `ŷ_a(x)` that predicts reward from a context vector `x`.

The mathematical bridge is the **hybrid reward estimator**


r̃_a(x, m) = α·r̂_a   +   (1−α)·ŷ_a(x)   +   β·f(m)


where  

* `α ∈ [0,1]` is derived from the circuit‑breaker state (`α = 1 − failures /
  failure_threshold`),  
* `β` is a small scaling constant that injects the morphology‑based Fisher
  score, and  
* `x` is the stored context vector for the current decision.

The estimator is used by the bandit selection logic, thus marrying the
hardware‑aware morphology model with the learning‑driven bandit router."""

import math
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple, List, Dict, Sequence, Callable, Any

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Morphology & Circuit‑Breaker primitives
# ----------------------------------------------------------------------


def now_z() -> str:
    """Current UTC timestamp in ISO‑8601 Zulu format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class Morphology:
    """Geometric description of a physical (or logical) entity."""

    length: float
    width: float
    height: float
    mass: float

    def __post_init__(self) -> None:
        for name, value in (
            ("length", self.length),
            ("width", self.width),
            ("height", self.height),
            ("mass", self.mass),
        ):
            if not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"{name} must be a positive number")

    def as_vector(self) -> np.ndarray:
        """Return the morphology as a 1‑D numpy array."""
        return np.array([self.length, self.width, self.height, self.mass], dtype=float)


class EndpointCircuitBreaker:
    """Simple failure counter that opens after a configurable threshold."""

    def __init__(self, failure_threshold: int = 3):
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False
        self.last_event_at = now_z()

    def record_success(self) -> None:
        """Reset failures and close the breaker."""
        self.failures = 0
        self.open = False
        self.last_event_at = now_z()

    def record_failure(self) -> None:
        """Increment failures and open the breaker if threshold is hit."""
        self.failures += 1
        self.last_event_at = now_z()
        if self.failures >= self.failure_threshold:
            self.open = True

    def status(self) -> Tuple[bool, int]:
        """Return (open, failures)."""
        return self.open, self.failures


def fisher_score(morph: Morphology) -> float:
    """
    Compute a scalar Fisher‑like score from a morphology.
    The formulation uses the Mahalanobis‑type distance of the morphology
    vector to its mean (here the vector of ones) weighted by the inverse
    of the vector itself – a cheap proxy that yields a positive scalar.
    """
    v = morph.as_vector()
    # Prevent division by zero
    inv = np.where(v != 0, 1.0 / v, 0.0)
    mean = np.ones_like(v)
    diff = v - mean
    score = float(np.dot(diff * inv, diff))
    return score


# ----------------------------------------------------------------------
# Parent B – Contextual Bandit with RBF surrogate
# ----------------------------------------------------------------------


Vector = Sequence[float]


@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str


@dataclass(frozen=True)
class BanditUpdate:
    context_id: str          # identifier linking to a stored context vector
    action_id: str
    reward: float
    propensity: float


# Global stores (mutable)
_POLICY: Dict[str, List[float]] = {}          # action_id → [total_reward, count]
_CONTEXT_STORE: Dict[str, Vector] = {}        # context_id → vector representation
_SURROGATE: Dict[str, "RBFSurrogate"] = {}    # action_id → surrogate model


def reset_hybrid() -> None:
    """Clear all learned statistics, stored contexts and surrogate models."""
    _POLICY.clear()
    _CONTEXT_STORE.clear()
    _SURROGATE.clear()


class RBFSurrogate:
    """
    Simple Radial‑Basis‑Function surrogate model.
    Stores a list of centers `c_i` and associated weights `w_i`.
    Prediction: ŷ(x) = Σ_i w_i * exp(-γ * ||x - c_i||²)
    """

    def __init__(self, gamma: float = 0.5):
        self.gamma = float(gamma)
        self.centers: List[np.ndarray] = []
        self.weights: List[float] = []

    def update(self, x: Vector, y: float) -> None:
        """Append a new training sample (x, y)."""
        self.centers.append(np.asarray(x, dtype=float))
        self.weights.append(float(y))

    def predict(self, x: Vector) -> float:
        """Return the RBF‑weighted prediction for vector x."""
        if not self.centers:
            return 0.0
        x_arr = np.asarray(x, dtype=float)
        diffs = np.stack([c - x_arr for c in self.centers])
        dists_sq = np.einsum('ij,ij->i', diffs, diffs)
        kernels = np.exp(-self.gamma * dists_sq)
        pred = float(np.dot(self.weights, kernels) / np.sum(kernels))
        return pred


# ----------------------------------------------------------------------
# Hybrid core – mathematical bridge
# ----------------------------------------------------------------------


_BETA = 0.01  # scaling for the Fisher term; small so it perturbs but does not dominate


def hybrid_reward_estimate(
    action_id: str,
    context_id: str,
    morph: Morphology,
    breaker: EndpointCircuitBreaker,
) -> float:
    """
    Compute the hybrid reward estimate

        r̃ = α·r̂ + (1−α)·ŷ + β·f(morph)

    where
        α = 1 - failures / failure_threshold   (clamped to [0,1])
        r̂ = empirical mean reward for the action (0 if unseen)
        ŷ = RBF surrogate prediction for the stored context vector
        f(morph) = Fisher score of the morphology
    """
    # ---- α from circuit‑breaker ----
    failures = breaker.failures
    threshold = breaker.failure_threshold
    alpha = max(0.0, min(1.0, 1.0 - failures / threshold))

    # ---- Empirical reward r̂ ----
    total_reward, count = _POLICY.get(action_id, [0.0, 0.0])
    empirical = total_reward / count if count > 0 else 0.0

    # ---- Surrogate prediction ŷ ----
    surrogate = _SURROGATE.get(action_id)
    ctx_vec = _CONTEXT_STORE.get(context_id, [])
    surrogate_pred = surrogate.predict(ctx_vec) if surrogate and ctx_vec else 0.0

    # ---- Fisher term β·f(morph) ----
    fisher = fisher_score(morph)

    hybrid = alpha * empirical + (1.0 - alpha) * surrogate_pred + _BETA * fisher
    return hybrid


def select_action(
    context_id: str,
    morph: Morphology,
    breaker: EndpointCircuitBreaker,
    candidate_actions: List[str],
    exploration_eps: float = 0.1,
) -> str:
    """
    Choose an action using an ε‑greedy policy over the hybrid reward estimate.
    With probability `exploration_eps` a random action is returned;
    otherwise the action with the highest `r̃` + confidence bound is selected.
    """
    if not candidate_actions:
        raise ValueError("candidate_actions must contain at least one action_id")

    # Exploration step
    if random.random() < exploration_eps:
        return random.choice(candidate_actions)

    # Exploitation step
    best_action = None
    best_score = -math.inf
    for aid in candidate_actions:
        # Confidence bound similar to UCB1: sqrt(2 * log(N) / n)
        total, cnt = _POLICY.get(aid, [0.0, 0.0])
        n = cnt if cnt > 0 else 1.0
        N = sum(c[1] for c in _POLICY.values()) + 1.0
        confidence = math.sqrt(2.0 * math.log(N) / n)

        hybrid = hybrid_reward_estimate(aid, context_id, morph, breaker)
        score = hybrid + confidence
        if score > best_score:
            best_score = score
            best_action = aid

    assert best_action is not None
    return best_action


def update_hybrid(
    action_id: str,
    context_id: str,
    reward: float,
    propensity: float,
    morph: Morphology,
    breaker: EndpointCircuitBreaker,
) -> None:
    """
    Update all hybrid components after observing a reward:

    * Empirical bandit statistics (`_POLICY`)
    * Context storage (`_CONTEXT_STORE`) – creates a fresh entry if missing
    * RBF surrogate model (`_SURROGATE`)
    * Circuit‑breaker status – a negative reward counts as a failure
    """
    # ---- Update empirical stats ----
    total, cnt = _POLICY.get(action_id, [0.0, 0.0])
    _POLICY[action_id] = [total + reward, cnt + 1.0]

    # ---- Store (or update) context vector ----
    if context_id not in _CONTEXT_STORE:
        # For demo purposes we synthesize a context vector from morphology
        # plus a small random perturbation.
        base_vec = morph.as_vector()
        noise = np.random.normal(scale=0.01, size=base_vec.shape)
        _CONTEXT_STORE[context_id] = (base_vec + noise).tolist()

    # ---- Update surrogate model ----
    surrogate = _SURROGATE.setdefault(action_id, RBFSurrogate())
    ctx_vec = _CONTEXT_STORE[context_id]
    surrogate.update(ctx_vec, reward)

    # ---- Circuit‑breaker feedback ----
    if reward < 0:
        breaker.record_failure()
    else:
        breaker.record_success()


# ----------------------------------------------------------------------
# Demonstration helpers (three public functions)
# ----------------------------------------------------------------------


def register_action(action_id: str) -> None:
    """Create an entry for a new action with zeroed statistics."""
    if action_id not in _POLICY:
        _POLICY[action_id] = [0.0, 0.0]


def get_action_statistics(action_id: str) -> Tuple[float, float]:
    """Return (average_reward, count) for the given action."""
    total, cnt = _POLICY.get(action_id, [0.0, 0.0])
    avg = total / cnt if cnt > 0 else 0.0
    return avg, cnt


def list_registered_actions() -> List[str]:
    """Return a list of all known action identifiers."""
    return list(_POLICY.keys())


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Initialise components
    breaker = EndpointCircuitBreaker(failure_threshold=5)

    # Define two actions
    for aid in ("A", "B"):
        register_action(aid)

    # Create two distinct morphologies
    morph_a = Morphology(length=1.2, width=0.5, height=0.3, mass=2.0)
    morph_b = Morphology(length=0.8, width=0.4, height=0.6, mass=1.5)

    # Simulate a few interaction rounds
    for step in range(10):
        ctx_id = f"ctx_{step}"
        # Randomly pick a morphology for this round
        morph = morph_a if random.random() < 0.5 else morph_b

        # Choose action via hybrid policy
        chosen = select_action(
            context_id=ctx_id,
            morph=morph,
            breaker=breaker,
            candidate_actions=list_registered_actions(),
            exploration_eps=0.2,
        )

        # Simulate a stochastic reward (higher for action A)
        reward = random.gauss(1.0 if chosen == "A" else 0.2, 0.5)
        # Propensity is placeholder (1.0 for deterministic choice)
        propensity = 1.0

        # Update all hybrid structures
        update_hybrid(
            action_id=chosen,
            context_id=ctx_id,
            reward=reward,
            propensity=propensity,
            morph=morph,
            breaker=breaker,
        )

        # Optional debug output
        avg, cnt = get_action_statistics(chosen)
        print(
            f"Step {step:02d} | ctx={ctx_id} | morph={chosen} | reward={reward:.3f} "
            f"| avg={avg:.3f} (n={int(cnt)}) | breaker_open={breaker.open}"
        )

    print("\nFinal action statistics:")
    for aid in list_registered_actions():
        avg, cnt = get_action_statistics(aid)
        print(f"  Action {aid}: avg_reward={avg:.3f}, count={int(cnt)}")
    print(f"Circuit breaker status: open={breaker.open}, failures={breaker.failures}")