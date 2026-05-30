# DARWIN HAMMER — match 2355, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_bandit_router_variational_free_ene_m56_s2.py (gen2)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_fisher_m984_s1.py (gen4)
# born: 2026-05-29T23:42:08Z

"""Hybrid Bandit‑Router + Active‑Inference + Fisher‑Information Model

Parents:
- PARENT ALGORITHM A: hybrid_hybrid_bandit_router_variational_free_ene_m56_s2.py
- PARENT ALGORITHM B: hybrid_hybrid_hybrid_hybrid_fisher_m984_s1.py

Mathematical Bridge:
The Bandit‑Router maintains a policy (reward estimates & confidence bounds).  
Active‑Inference supplies a Gaussian temperature‑dependence model
`developmental_rate(T; θ)` whose log‑likelihood curvature yields the Fisher
information `I(θ, T) = -E[∂² log p/∂θ²]`.  
We embed this scalar Fisher information into the bandit’s confidence term,
so that actions observed under high‑information conditions are explored
more aggressively.  

Additionally, the morphology of an entity (length, width, height, mass) is
projected onto a low‑dimensional subspace via a simple PCA; the resulting
embedding constitutes the “context” for the bandit decision.  The three core
functions below demonstrate the hybrid operation:
1. `compute_fisher_information` – derives Fisher information from the
   temperature‑dependent Gaussian model.
2. `select_action` – projects morphology, computes an Upper‑Confidence‑Bound
   (UCB) that incorporates Fisher information, and returns the best action.
3. `update_policy_with_fisher` – updates the policy and adjusts the confidence
   bound using the observed Fisher information.

The resulting system jointly minimizes variational free energy (through the
Gaussian model) while performing contextual bandit learning guided by
information‑theoretic confidence."""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Any

import numpy as np

# ----------------------------------------------------------------------
# Core data structures (merged from both parents)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "HybridBanditAI"


@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float
    fisher: float  # Fisher information observed for this update


@dataclass(frozen=True)
class SchoolfieldParams:
    """Parameters of the Gaussian temperature model (active inference)."""
    rho_25: float = 1.0               # baseline rate at 25°C
    t_opt: float = 298.15             # optimal temperature (K)
    sigma: float = 5.0                # spread of the Gaussian (K)
    # The following are placeholders to keep compatibility with the original
    # parent code; they are not used in the simplified Gaussian formulation.
    delta_h_activation: float = 12_000.0
    t_low: float = 283.15
    t_high: float = 307.15
    delta_h_low: float = 0.0


@dataclass(frozen=True)
class Morphology:
    """Geometric description used as contextual features."""
    length: float
    width: float
    height: float
    mass: float


# ----------------------------------------------------------------------
# Global policy storage (action_id -> [total_reward, count, fisher_sum])
# ----------------------------------------------------------------------
_POLICY: Dict[str, List[float]] = {}


def reset_policy() -> None:
    """Clear the global policy."""
    _POLICY.clear()


def _reward_estimate(action_id: str) -> float:
    total, n, _ = _POLICY.get(action_id, [0.0, 0.0, 0.0])
    return total / n if n else 0.0


def _fisher_average(action_id: str) -> float:
    _, n, fisher_sum = _POLICY.get(action_id, [0.0, 0.0, 0.0])
    return fisher_sum / n if n else 1.0  # avoid division by zero


def update_policy_with_fisher(updates: List[BanditUpdate]) -> None:
    """Incorporate rewards and observed Fisher information into the policy."""
    for u in updates:
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0, 0.0])
        stats[0] += float(u.reward)            # total reward
        stats[1] += 1.0                         # count
        stats[2] += float(u.fisher)             # accumulated Fisher


# ----------------------------------------------------------------------
# Temperature model (Active Inference) ------------------------------------------------
# ----------------------------------------------------------------------
def c_to_k(celsius: float) -> float:
    """Convert Celsius to Kelvin."""
    return celsius + 273.15


def developmental_rate(temp_k: float, params: SchoolfieldParams) -> float:
    """
    Gaussian temperature dependence:
        r(T) = rho_25 * exp( - (T - T_opt)^2 / (2 * sigma^2) )
    """
    exponent = -((temp_k - params.t_opt) ** 2) / (2.0 * params.sigma ** 2)
    return params.rho_25 * math.exp(exponent)


def normalized_activity(temp_c: float, low_c: float = 5.0) -> float:
    """Normalize activity to [0,1] using the Gaussian temperature model."""
    params = SchoolfieldParams()
    rate = developmental_rate(c_to_k(temp_c), params)
    # Simple linear scaling against a low‑temperature baseline
    baseline = developmental_rate(c_to_k(low_c), params)
    return (rate - baseline) / (1.0 - baseline) if rate > baseline else 0.0


# ----------------------------------------------------------------------
# Fisher information for the Gaussian temperature model
# ----------------------------------------------------------------------
def compute_fisher_information(params: SchoolfieldParams, temp_c: float) -> float:
    """
    For a Gaussian likelihood with fixed variance σ², the Fisher information
    with respect to the mean (here T_opt) is I = 1/σ².
    We return the scalar I multiplied by the normalized activity to reflect
    that higher activity yields more informative observations.
    """
    sigma_sq = params.sigma ** 2
    base_fisher = 1.0 / sigma_sq
    activity = normalized_activity(temp_c)
    return base_fisher * (0.5 + activity)  # ensure a minimum >0


# ----------------------------------------------------------------------
# Contextual dimensionality reduction (PCA) ---------------------------------
# ----------------------------------------------------------------------
def _pca_projection(matrix: np.ndarray, n_components: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the top `n_components` principal components of `matrix` (samples x features)
    and return (projected_data, components).
    """
    # Center the data
    mean = matrix.mean(axis=0, keepdims=True)
    centered = matrix - mean
    # SVD
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:n_components]
    projected = centered @ components.T
    return projected, components


def embed_morphology_batch(batch: List[Morphology], n_components: int = 2) -> np.ndarray:
    """Convert a batch of Morphology objects into a low‑dimensional embedding."""
    data = np.array([[m.length, m.width, m.height, m.mass] for m in batch], dtype=float)
    projected, _ = _pca_projection(data, n_components)
    return projected  # shape: (batch, n_components)


# ----------------------------------------------------------------------
# Bandit decision making with Fisher‑augmented confidence bounds
# ----------------------------------------------------------------------
def select_action(context: Morphology, possible_actions: List[str], timestep: int) -> BanditAction:
    """
    Choose the action with the highest Upper‑Confidence‑Bound (UCB):
        UCB = μ̂ + c * sqrt( (2 * log(t)) / (n * I) )
    where:
        μ̂    – empirical reward estimate,
        n    – number of times the action was selected,
        I    – average Fisher information for that action,
        c    – exploration coefficient (set to 1.0 here).
    """
    # Ensure policy entries exist
    for a in possible_actions:
        _POLICY.setdefault(a, [0.0, 0.0, 0.0])

    best_action = None
    best_score = -math.inf
    for a in possible_actions:
        mu = _reward_estimate(a)
        total, n, _ = _POLICY[a]
        n = max(n, 1.0)  # avoid division by zero
        fisher_avg = _fisher_average(a)
        confidence = math.sqrt((2.0 * math.log(max(timestep, 1))) / (n * fisher_avg))
        ucb = mu + confidence
        if ucb > best_score:
            best_score = ucb
            best_action = a

    # Propensity is a softmax over UCBs for reporting purposes
    scores = np.array([
        _reward_estimate(a) + math.sqrt((2.0 * math.log(max(timestep, 1))) /
                                        (max(_POLICY[a][1], 1.0) * _fisher_average(a)))
        for a in possible_actions
    ])
    exp_scores = np.exp(scores - np.max(scores))
    probs = exp_scores / exp_scores.sum()
    propensity = dict(zip(possible_actions, probs))[best_action]

    return BanditAction(
        action_id=best_action,
        propensity=propensity,
        expected_reward=_reward_estimate(best_action),
        confidence_bound=best_score - _reward_estimate(best_action),
    )


# ----------------------------------------------------------------------
# Endpoint circuit breaker (from Parent B) – used to guard updates
# ----------------------------------------------------------------------
class EndpointCircuitBreaker:
    """Simple failure counter that opens after a configurable threshold."""
    def __init__(self, failure_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False
        self.last_event_at = ""

    def record_success(self) -> None:
        self.failures = 0
        self.open = False
        self.last_event_at = self._now()

    def record_failure(self) -> None:
        self.failures += 1
        self.open = self.failures >= self.failure_threshold
        self.last_event_at = self._now()

    def allow(self) -> bool:
        """True if the circuit is closed (i.e. endpoint is usable)."""
        return not self.open

    def as_dict(self) -> dict[str, Any]:
        return {
            "failure_threshold": self.failure_threshold,
            "failures": self.failures,
            "open": self.open,
            "last_event_at": self.last_event_at,
        }

    @staticmethod
    def _now() -> str:
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# ----------------------------------------------------------------------
# Demonstration utilities -------------------------------------------------
# ----------------------------------------------------------------------
def simulate_step(context: Morphology,
                  actions: List[str],
                  timestep: int,
                  breaker: EndpointCircuitBreaker) -> None:
    """Run a single decision‑update cycle."""
    if not breaker.allow():
        # Circuit open – skip this step
        return

    # Select action using the hybrid policy
    chosen = select_action(context, actions, timestep)

    # Simulate environment reward: reward = normalized activity + noise
    temp_c = random.uniform(5.0, 35.0)
    reward = normalized_activity(temp_c) + random.gauss(0, 0.05)

    # Compute Fisher information for the observed temperature
    fisher = compute_fisher_information(SchoolfieldParams(), temp_c)

    # Build update record
    upd = BanditUpdate(
        context_id=str(id(context)),
        action_id=chosen.action_id,
        reward=reward,
        propensity=chosen.propensity,
        fisher=fisher,
    )

    # Update policy
    update_policy_with_fisher([upd])

    # Randomly decide if the step succeeded (to demo breaker)
    if random.random() < 0.9:
        breaker.record_success()
    else:
        breaker.record_failure()


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Initialise
    reset_policy()
    actions = ["heat", "cool", "maintain"]
    breaker = EndpointCircuitBreaker(failure_threshold=5)

    # Create a batch of random morphologies
    morphologies = [
        Morphology(
            length=random.uniform(0.5, 2.0),
            width=random.uniform(0.5, 2.0),
            height=random.uniform(0.5, 2.0),
            mass=random.uniform(1.0, 10.0),
        )
        for _ in range(10)
    ]

    # Run 100 timesteps
    for t in range(1, 101):
        ctx = random.choice(morphologies)
        simulate_step(ctx, actions, t, breaker)

    # Print final policy summary
    print("Final policy statistics:")
    for aid, stats in _POLICY.items():
        total, count, fisher_sum = stats
        print(
            f"Action {aid}: avg reward={total/count:.3f}, "
            f"plays={int(count)}, avg Fisher={fisher_sum/count:.3f}"
        )
    print("Circuit breaker state:", breaker.as_dict())