# DARWIN HAMMER — match 403, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_bandit_router_hybrid_model_vram_sc_m35_s1.py (gen2)
# parent_b: hybrid_hybrid_endpoint_circ_hybrid_hybrid_fisher_m268_s1.py (gen4)
# born: 2026-05-29T23:28:47Z

"""Hybrid Bandit‑Circuit‑Fisher Algorithm
Integrates:
- Parent A: bandit router with propensity scores, confidence bounds, and linear‑type TTT core.
- Parent B: endpoint circuit‑breaker whose failure threshold is modulated by Fisher‑score based
  hygiene metrics.

Mathematical bridge:
1. For each candidate action a we obtain a propensity p_a from the bandit router.
2. A Fisher information scalar f_a is derived from the action’s feature vector x_a
   (f_a = x_aᵀ F x_a, where F is approximated by a Gaussian kernel on the feature space).
3. The TTT‑Linear prediction is taken as the element‑wise product p_a · f_a.
4. The confidence bound of the bandit (UCB style) is scaled by the same Fisher factor,
   yielding a fused score s_a = p_a·f_a + cb_a·f_a.
5. The circuit‑breaker adjusts its failure threshold τ by τ′ = τ · (1 + α·f̄) where f̄ is the
   mean Fisher score of the current action set; actions are pruned when the breaker is open.

The three core functions below showcase this fusion:
- `fisher_score` computes the scalar Fisher information for a feature vector.
- `adjust_failure_threshold` mutates the circuit‑breaker using the mean Fisher score.
- `hybrid_select_action` selects an action using the fused score while respecting the breaker.
"""

import math
import random
import sys
import pathlib
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Bandit router state (parent A)
# ----------------------------------------------------------------------
_POLICY: Dict[str, List[float]] = {}
_STORE: Dict[str, float] = {}

def reset_policy() -> None:
    _POLICY.clear()
    _STORE.clear()

def update_policy(updates: List["BanditUpdate"]) -> None:
    for u in updates:
        s = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        s[0] += float(u.reward)
        s[1] += 1.0

def _reward(a: str) -> float:
    total, n = _POLICY.get(a, [0.0, 0.0])
    return total / n if n else 0.0

def _count(a: str) -> float:
    return _POLICY.get(a, [0.0, 0.0])[1]

# ----------------------------------------------------------------------
# Data structures (parent A)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str

@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float

# ----------------------------------------------------------------------
# Circuit breaker (parent B)
# ----------------------------------------------------------------------
def now_z() -> str:
    """Current UTC timestamp in ISO‑8601 Zulu format."""
    import datetime, zoneinfo
    return datetime.datetime.now(zoneinfo.ZoneInfo("UTC")).isoformat().replace("+00:00", "Z")

class EndpointCircuitBreaker:
    """Simple failure counter that opens after a configurable (adaptive) threshold."""

    def __init__(self, failure_threshold: int = 3):
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
        self.base_threshold = failure_threshold
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False
        self.last_event_at = ""

    def record_success(self) -> None:
        self.failures = 0
        self.open = False
        self.last_event_at = now_z()

    def record_failure(self) -> None:
        self.failures += 1
        self.open = self.failures >= self.failure_threshold
        self.last_event_at = now_z()

    def allow(self) -> bool:
        """True if the circuit is closed (i.e. endpoint is usable)."""
        return not self.open

    def adapt_threshold(self, scaling_factor: float) -> None:
        """Adjust the failure threshold according to a scaling factor."""
        new_thr = max(1, int(round(self.base_threshold * scaling_factor)))
        self.failure_threshold = new_thr

    def as_dict(self) -> dict:
        return {
            "base_threshold": self.base_threshold,
            "failure_threshold": self.failure_threshold,
            "failures": self.failures,
            "open": self.open,
            "last_event_at": self.last_event_at,
        }

# ----------------------------------------------------------------------
# Fisher‑score utilities (parent B)
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian kernel used to approximate a Fisher information matrix."""
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(features: np.ndarray, sigma: float = 1.0) -> float:
    """
    Approximate Fisher information for a feature vector using a Gaussian kernel.
    f = xᵀ K x where K_ij = exp(-||x_i‑x_j||²/(2σ²)).  For a single vector we use the
    diagonal approximation K_ii = 1, yielding f = ||x||² scaled by sigma.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    norm_sq = float(np.dot(features, features))
    return norm_sq / (sigma * sigma)

def adjust_failure_threshold(breaker: EndpointCircuitBreaker, fisher_vals: List[float], alpha: float = 0.2) -> None:
    """
    Modulate the breaker’s failure threshold with the mean Fisher score.
    τ′ = τ · (1 + α·f̄)
    """
    if not fisher_vals:
        return
    mean_fisher = sum(fisher_vals) / len(fisher_vals)
    scaling = 1.0 + alpha * mean_fisher
    breaker.adapt_threshold(scaling)

# ----------------------------------------------------------------------
# Hybrid selection logic
# ----------------------------------------------------------------------
def hybrid_select_action(
    context: Dict[str, Dict[str, np.ndarray]],
    actions: List[str],
    breaker: EndpointCircuitBreaker,
    algorithm: str = "linucb",
    epsilon: float = 0.1,
    seed: int | str | None = 7,
) -> BanditAction:
    """
    Select an action by fusing bandit propensity, Fisher information,
    and circuit‑breaker state.

    Parameters
    ----------
    context : dict
        Mapping ``action_id -> {"features": np.ndarray}``.
    actions : list[str]
        Candidate action identifiers.
    breaker : EndpointCircuitBreaker
        Shared breaker whose state may prune actions.
    algorithm : str
        Bandit algorithm name (only 'linucb' and 'epsilon_greedy' are supported).
    epsilon : float
        Exploration probability for epsilon‑greedy.
    seed : int | str | None
        Random seed for reproducibility.

    Returns
    -------
    BanditAction
        The chosen action together with its fused statistics.
    """
    if not actions:
        raise ValueError("actions required")
    rng = random.Random(seed)

    # 1️⃣ Prune actions if the circuit is open
    if not breaker.allow():
        raise RuntimeError("Circuit breaker is open; no actions permitted")

    # 2️⃣ Compute raw propensities (simple UCB‑style estimate)
    total_counts = sum(_count(a) for a in actions) + 1.0
    propensities = {}
    confidence_bounds = {}
    for a in actions:
        # naive propensity = empirical mean reward
        mu = _reward(a)
        n = _count(a)
        propensities[a] = mu
        # classic LinUCB confidence term
        cb = math.sqrt(2 * math.log(total_counts) / (n + 1e-6))
        confidence_bounds[a] = cb

    # 3️⃣ Fisher scores per action (requires feature vectors)
    fisher_vals = {}
    for a in actions:
        feats = context.get(a, {}).get("features")
        if feats is None:
            raise ValueError(f"Missing feature vector for action '{a}'")
        fisher_vals[a] = fisher_score(feats)

    # 4️⃣ Adjust breaker threshold before selection (bridge step)
    adjust_failure_threshold(breaker, list(fisher_vals.values()))

    # 5️⃣ Fuse scores: s_a = (p_a + cb_a) * f_a
    fused_scores = {}
    for a in actions:
        fused = (propensities[a] + confidence_bounds[a]) * fisher_vals[a]
        fused_scores[a] = fused

    # 6️⃣ Exploration vs exploitation
    if algorithm == "epsilon_greedy" and rng.random() < epsilon:
        chosen_id = rng.choice(actions)
    else:
        chosen_id = max(actions, key=lambda a: fused_scores[a])

    # 7️⃣ Package result
    chosen = BanditAction(
        action_id=chosen_id,
        propensity=propensities[chosen_id],
        expected_reward=_reward(chosen_id),
        confidence_bound=confidence_bounds[chosen_id],
        algorithm=algorithm,
    )
    return chosen

def hybrid_update(
    updates: List[BanditUpdate],
    breaker: EndpointCircuitBreaker,
    reward_threshold: float = 0.1,
) -> None:
    """
    Update bandit statistics and feed back into the circuit breaker.
    A failure is recorded when the observed reward falls below
    (expected reward – reward_threshold).
    """
    update_policy(updates)
    for u in updates:
        expected = _reward(u.action_id)
        if u.reward < expected - reward_threshold:
            breaker.record_failure()
        else:
            breaker.record_success()

def simulate_hybrid_step(
    context_features: Dict[str, np.ndarray],
    actions: List[str],
    breaker: EndpointCircuitBreaker,
    rng_seed: int = 42,
) -> Tuple[BanditAction, EndpointCircuitBreaker]:
    """
    One simulation step:
    1. Build the context dict expected by `hybrid_select_action`.
    2. Choose an action.
    3. Mock a stochastic reward (Gaussian around true propensity).
    4. Update the hybrid system.
    Returns the selected action and the (potentially) updated breaker.
    """
    # Build nested context structure
    context = {a: {"features": context_features[a]} for a in actions}

    chosen = hybrid_select_action(
        context=context,
        actions=actions,
        breaker=breaker,
        algorithm="linucb",
        seed=rng_seed,
    )

    # Simulated reward: true propensity (empirical mean) plus noise
    true_mu = _reward(chosen.action_id)
    noise = np.random.default_rng(rng_seed).normal(0, 0.05)
    reward = max(0.0, true_mu + noise)

    update = BanditUpdate(
        context_id="sim",
        action_id=chosen.action_id,
        reward=reward,
        propensity=chosen.propensity,
    )
    hybrid_update([update], breaker)

    return chosen, breaker

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Initialise a tiny action set with synthetic features
    actions = ["a1", "a2", "a3"]
    np.random.seed(0)
    features = {a: np.random.randn(4) for a in actions}

    # Prime the policy with a few random updates so counts are non‑zero
    reset_policy()
    dummy_updates = [
        BanditUpdate(context_id="init", action_id=a, reward=float(i), propensity=0.5)
        for i, a in enumerate(actions, start=1)
    ]
    update_policy(dummy_updates)

    # Instantiate circuit breaker
    breaker = EndpointCircuitBreaker(failure_threshold=3)

    # Run a few hybrid steps
    for step in range(5):
        act, breaker = simulate_hybrid_step(features, actions, breaker, rng_seed=step)
        print(f"Step {step+1}: chose {act.action_id}, propensity={act.propensity:.3f}, "
              f"expected={act.expected_reward:.3f}, cb={act.confidence_bound:.3f}")
        print("Breaker state:", breaker.as_dict())
    sys.exit(0)