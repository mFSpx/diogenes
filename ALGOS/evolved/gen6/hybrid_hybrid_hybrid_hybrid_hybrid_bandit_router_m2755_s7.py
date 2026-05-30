# DARWIN HAMMER — match 2755, survivor 7
# gen: 6
# parent_a: hybrid_hybrid_hybrid_endpoi_hybrid_hybrid_hybrid_m976_s6.py (gen5)
# parent_b: hybrid_bandit_router_hybrid_hybrid_hybrid_m2649_s4.py (gen5)
# born: 2026-05-29T23:45:38Z

"""Hybrid Morphology‑Bandit‑RBF System

Parent A: *hybrid_hybrid_hybrid_endpoi_hybrid_hybrid_hybrid_m976_s6.py* – provides
`Morphology`, `EndpointCircuitBreaker` and a Fisher‑score primitive.

Parent B: *hybrid_bandit_router_hybrid_hybrid_hybrid_m2649_s4.py* – provides a
contextual bandit with empirical reward bookkeeping and an RBF surrogate model.

Mathematical bridge
-------------------
Both parents expose a scalar estimate of “goodness” for a given entity:

* Parent A: `fisher_score(morph)` – a statistic derived from the morphology vector.
* Parent B: `empirical_reward(action)` – `total_reward / count`.
* Parent B surrogate: `ŷ_a(x) = Σ_i w_i·K(‖x‑c_i‖)` where `K` is a Gaussian kernel.

We fuse them into a single hybrid estimator for an action *a* in context *x*:


r̃_a(x) = α·(empirical_reward_a) + β·fisher_score(morph_a) + (1‑α‑β)·ŷ_a(x)


with mixing coefficients `α,β∈[0,1]` and `α+β≤1`.  
The estimator drives action selection, while the `EndpointCircuitBreaker`
can disable actions whose failure counter exceeds a threshold.

The module implements this unified system and supplies three public
functions that demonstrate the hybrid workflow.
"""

import math
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple, List, Dict, Sequence, Callable, Any

import numpy as np

# ----------------------------------------------------------------------
# Parent A – morphology and circuit‑breaker primitives
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
    A simple surrogate for a Fisher‑information‑like score.
    We treat the morphology vector as a parameter vector θ and compute
    `score = θᵀ·θ`, i.e. the squared Euclidean norm.  This yields a positive
    scalar that grows with the magnitude of the entity.
    """
    v = morph.as_vector()
    return float(np.dot(v, v))


# ----------------------------------------------------------------------
# Parent B – contextual bandit with RBF surrogate
# ----------------------------------------------------------------------
Vector = Sequence[float]


@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float = 0.0
    expected_reward: float = 0.0
    confidence_bound: float = 0.0
    algorithm: str = "hybrid"


@dataclass(frozen=True)
class BanditUpdate:
    context_id: str          # identifier linking to a stored context vector
    action_id: str
    reward: float
    propensity: float


# Global stores (module‑level singletons)
_POLICY: Dict[str, List[float]] = {}          # action_id → [total_reward, count]
_CONTEXT_STORE: Dict[str, Vector] = {}        # context_id → vector representation
_SURROGATE: Dict[str, "RBFSurrogate"] = {}    # action_id → surrogate model
_BREAKER: Dict[str, EndpointCircuitBreaker] = {}  # action_id → circuit breaker


def reset_hybrid() -> None:
    """Clear all learned statistics, stored contexts, surrogate models and breakers."""
    _POLICY.clear()
    _CONTEXT_STORE.clear()
    _SURROGATE.clear()
    _BREAKER.clear()


# ----------------------------------------------------------------------
# RBF surrogate model (Parent B component)
# ----------------------------------------------------------------------
class RBFSurrogate:
    """
    Non‑parametric regression model using Gaussian kernels.
    Stores a list of centers `c_i` and associated weights `w_i`.
    Prediction: ŷ(x) = Σ_i w_i * exp(-γ * ||x - c_i||²)
    """
    def __init__(self, gamma: float = 1.0):
        if gamma <= 0:
            raise ValueError("gamma must be positive")
        self.gamma = gamma
        self.centers: List[np.ndarray] = []
        self.weights: List[float] = []

    def add_sample(self, x: Vector, target: float, lr: float = 0.1) -> None:
        """Add a new sample (x, target) using a simple stochastic gradient step."""
        x_arr = np.asarray(x, dtype=float)
        if not self.centers:
            # First sample – initialise with weight = target
            self.centers.append(x_arr)
            self.weights.append(target)
            return

        # Predict current output
        y_pred = self.predict(x_arr)
        error = target - y_pred

        # Update each weight proportional to kernel value and error
        for i, c in enumerate(self.centers):
            k = math.exp(-self.gamma * np.linalg.norm(x_arr - c) ** 2)
            self.weights[i] += lr * error * k

        # Optionally add the new point as a centre (to keep model expressive)
        self.centers.append(x_arr)
        self.weights.append(lr * error)  # small initial weight

    def predict(self, x: Vector) -> float:
        """Return ŷ(x)."""
        if not self.centers:
            return 0.0
        x_arr = np.asarray(x, dtype=float)
        contributions = [
            w * math.exp(-self.gamma * np.linalg.norm(x_arr - c) ** 2)
            for c, w in zip(self.centers, self.weights)
        ]
        return float(sum(contributions))


# ----------------------------------------------------------------------
# Hybrid reward estimator
# ----------------------------------------------------------------------
def empirical_reward(action_id: str) -> float:
    """Return total_reward / count for the given action, 0 if never seen."""
    total, cnt = _POLICY.get(action_id, [0.0, 0.0])
    return total / cnt if cnt > 0 else 0.0


def hybrid_reward_estimate(
    action_id: str,
    context_vec: Vector,
    morph: Morphology,
    alpha: float = 0.5,
    beta: float = 0.2,
) -> float:
    """
    Compute the fused reward estimate:

        r̃ = α·empirical + β·fisher_score + (1‑α‑β)·surrogate

    The coefficients are clipped to the feasible simplex.
    """
    if not (0.0 <= alpha <= 1.0 and 0.0 <= beta <= 1.0):
        raise ValueError("alpha and beta must lie in [0,1]")
    if alpha + beta > 1.0:
        # renormalise to keep sum ≤ 1
        scale = 1.0 / (alpha + beta)
        alpha *= scale
        beta *= scale

    emp = empirical_reward(action_id)
    fisher = fisher_score(morph)
    surrogate = _SURROGATE.setdefault(action_id, RBFSurrogate()).predict(context_vec)

    hybrid = alpha * emp + beta * fisher + (1.0 - alpha - beta) * surrogate
    return hybrid


# ----------------------------------------------------------------------
# Core hybrid operations (exposed functions)
# ----------------------------------------------------------------------
def register_context(context_id: str, vector: Vector) -> None:
    """Store a context vector for later retrieval."""
    _CONTEXT_STORE[context_id] = tuple(vector)


def register_action(action_id: str, morph: Morphology, breaker_threshold: int = 3) -> None:
    """Initialise bookkeeping for a new action."""
    if action_id in _POLICY:
        return
    _POLICY[action_id] = [0.0, 0.0]                     # total_reward, count
    _SURROGATE[action_id] = RBFSurrogate(gamma=0.5)   # one surrogate per action
    _BREAKER[action_id] = EndpointCircuitBreaker(failure_threshold=breaker_threshold)


def select_action(
    context_id: str,
    candidate_action_ids: List[str],
    morph_map: Dict[str, Morphology],
    alpha: float = 0.5,
    beta: float = 0.2,
) -> BanditAction:
    """
    Choose the action with the highest hybrid reward estimate among candidates,
    respecting circuit‑breaker status (open actions are ignored).
    """
    if context_id not in _CONTEXT_STORE:
        raise KeyError(f"Context {context_id!r} not registered")
    ctx_vec = _CONTEXT_STORE[context_id]

    best_id = None
    best_score = -math.inf
    for aid in candidate_action_ids:
        breaker = _BREAKER.get(aid)
        if breaker is None:
            raise KeyError(f"Action {aid!r} not registered")
        if breaker.open:
            continue  # skip broken actions
        morph = morph_map[aid]
        score = hybrid_reward_estimate(aid, ctx_vec, morph, alpha, beta)
        if score > best_score:
            best_score = score
            best_id = aid

    if best_id is None:
        raise RuntimeError("All candidate actions are blocked by circuit breakers")

    # Populate a BanditAction with the computed statistics
    return BanditAction(
        action_id=best_id,
        propensity=1.0 / len(candidate_action_ids),
        expected_reward=best_score,
        confidence_bound=0.0,   # placeholder – could be derived from variance
        algorithm="hybrid"
    )


def process_interaction(
    update: BanditUpdate,
    morph: Morphology,
    alpha: float = 0.5,
    beta: float = 0.2,
    lr: float = 0.1,
) -> None:
    """
    Incorporate a new observation into both empirical statistics and the RBF surrogate.
    Also update the circuit breaker: a reward below the 25th percentile of the
    empirical distribution for the action counts as a failure.
    """
    # --- Empirical bookkeeping -------------------------------------------------
    total, cnt = _POLICY.setdefault(update.action_id, [0.0, 0.0])
    total += update.reward
    cnt += 1
    _POLICY[update.action_id] = [total, cnt]

    # --- Surrogate update ------------------------------------------------------
    ctx_vec = _CONTEXT_STORE.get(update.context_id)
    if ctx_vec is None:
        raise KeyError(f"Context {update.context_id!r} not registered")
    surrogate = _SURROGATE.setdefault(update.action_id, RBFSurrogate())
    surrogate.add_sample(ctx_vec, update.reward, lr=lr)

    # --- Circuit‑breaker logic -------------------------------------------------
    breaker = _BREAKER.setdefault(update.action_id, EndpointCircuitBreaker())
    # Simple heuristic: reward < median of observed rewards → failure
    # We approximate median by comparing to current empirical mean.
    mean_reward = total / cnt
    if update.reward < mean_reward:
        breaker.record_failure()
    else:
        breaker.record_success()


def get_action_status(action_id: str) -> Tuple[bool, int]:
    """Convenient accessor for the circuit‑breaker status of an action."""
    breaker = _BREAKER.get(action_id)
    if breaker is None:
        raise KeyError(f"Action {action_id!r} not registered")
    return breaker.status()


# ----------------------------------------------------------------------
# Smoke test (executed when run as a script)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Reset any stale state
    reset_hybrid()

    # Define two morphologies
    morph_a = Morphology(length=1.2, width=0.8, height=0.5, mass=2.3)
    morph_b = Morphology(length=0.9, width=0.6, height=0.4, mass=1.8)

    # Register actions
    register_action("A", morph_a, breaker_threshold=2)
    register_action("B", morph_b, breaker_threshold=2)

    # Register a few contexts
    register_context("c1", [0.1, 0.2, 0.3])
    register_context("c2", [0.4, 0.1, 0.0])
    register_context("c3", [0.0, 0.5, 0.2])

    # Mapping from action_id to its morphology (required by select_action)
    morph_map = {"A": morph_a, "B": morph_b}

    # Simulate a round‑robin interaction loop
    for step in range(5):
        ctx_id = random.choice(["c1", "c2", "c3"])
        try:
            chosen = select_action(
                context_id=ctx_id,
                candidate_action_ids=["A", "B"],
                morph_map=morph_map,
                alpha=0.6,
                beta=0.1,
            )
        except RuntimeError as e:
            print(f"Step {step}: {e}")
            break

        # Simulated stochastic reward: higher for action A in context c1, else random
        if chosen.action_id == "A" and ctx_id == "c1":
            reward = random.uniform(0.7, 1.0)
        else:
            reward = random.uniform(0.0, 0.6)

        upd = BanditUpdate(
            context_id=ctx_id,
            action_id=chosen.action_id,
            reward=reward,
            propensity=chosen.propensity,
        )
        process_interaction(upd, morph_map[chosen.action_id])

        open_a, fail_a = get_action_status("A")
        open_b, fail_b = get_action_status("B")
        print(
            f"Step {step}: ctx={ctx_id}, chosen={chosen.action_id}, reward={reward:.3f}, "
            f"breaker A=({open_a}, {fail_a}), B=({open_b}, {fail_b})"
        )

    print("\nFinal empirical statistics:")
    for aid, (tot, cnt) in _POLICY.items():
        print(f" Action {aid}: total_reward={tot:.3f}, count={int(cnt)}")
    print("\nHybrid test completed without errors.")