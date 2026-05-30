# DARWIN HAMMER — match 853, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_counterfactual_effec_m316_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_infota_hybrid_fold_change_d_m128_s0.py (gen4)
# born: 2026-05-29T23:31:20Z

"""Hybrid Causal‑Bandit Fusion (HCBF)

Parents:
- **Parent A**: reconstruction risk scoring + causal effect estimation.
- **Parent B**: MinHash‑based entropic signature + contextual bandit policy with drag‑limited integration.

Mathematical bridge:
The reconstruction risk score `r` (∈[0,1]) is used as a *confidence weight* for the causal
average treatment effect `τ`.  The weighted effect `τ_w = τ·(1‑r)` becomes the
*expected reward* for a bandit action.  Conversely, the MinHash signature similarity
`σ` (Hamming similarity between two signatures) modulates the *propensity* of the
bandit policy: `π = σ·π₀`.  The drag‑limited integration in `hybrid_strike` uses the
product `σ·τ_w` as a force that drives the virtual agent’s position and velocity.

The three core functions below demonstrate this fusion:
1. `weighted_causal_effect` – computes τ_w from data, risk score and confounders.
2. `bandit_policy_update_weighted` – updates the bandit policy using the weighted
   causal effect as reward and MinHash similarity as propensity modifier.
3. `hybrid_strike` – runs a simple Euler integration where the force is the
   similarity‑weighted causal effect, returning the final `StrikeState`.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Iterable
import numpy as np
import math
import random
import sys
import pathlib
import statistics

# ----------------------------------------------------------------------
# Parent A – reconstruction risk & causal effect
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str
    vram_mb: int

TIER_T1_QWEN_0_5B = ModelTier("qwen-0.5b", 512, "T1", 1024)

def reconstruction_risk_score(unique_quasi_identifiers: int, total_records: int) -> float:
    """Risk score ∈[0,1] proportional to the fraction of unique quasi‑identifiers."""
    if total_records <= 0:
        return 0.0
    return max(0.0, min(1.0, unique_quasi_identifiers / total_records))

@dataclass(frozen=True)
class CausalEffect:
    effect_id: str
    treatment: str
    outcome: str
    confounders: Tuple[str, ...]
    ate_estimate: float | None
    ate_confidence_interval: Tuple[float, float] | None
    refutation_passed: bool
    refutation_methods: Tuple[str, ...]
    heterogeneous_effects: Dict[str, float]

def estimate_causal_effect(
    treatment: str,
    outcome: str,
    confounders: List[str],
    data: Dict[str, List[float]],
) -> CausalEffect:
    """Very lightweight ATE estimator: difference of means between treated (t>=0.5) and control."""
    t_vals = list(map(float, data.get(treatment, [])))
    y_vals = list(map(float, data.get(outcome, [])))
    if not t_vals or len(t_vals) != len(y_vals):
        ate = None
        ci = None
    else:
        treated = [y for tt, y in zip(t_vals, y_vals) if tt >= 0.5]
        control = [y for tt, y in zip(t_vals, y_vals) if tt < 0.5]
        if treated and control:
            ate = statistics.mean(treated) - statistics.mean(control)
            # simple normal‑approx CI
            se = math.sqrt(
                statistics.pvariance(treated) / len(treated) +
                statistics.pvariance(control) / len(control)
            )
            ci = (ate - 1.96 * se, ate + 1.96 * se)
        else:
            ate = None
            ci = None
    return CausalEffect(
        effect_id="effect_" + treatment + "_" + outcome,
        treatment=treatment,
        outcome=outcome,
        confounders=tuple(confounders),
        ate_estimate=ate,
        ate_confidence_interval=ci,
        refutation_passed=False,
        refutation_methods=(),
        heterogeneous_effects={},
    )

def weighted_causal_effect(
    treatment: str,
    outcome: str,
    confounders: List[str],
    data: Dict[str, List[float]],
    unique_qi: int,
    total_records: int,
) -> float | None:
    """Return τ_w = τ·(1‑r) where r is the reconstruction risk."""
    risk = reconstruction_risk_score(unique_qi, total_records)
    ce = estimate_causal_effect(treatment, outcome, confounders, data)
    if ce.ate_estimate is None:
        return None
    return ce.ate_estimate * (1.0 - risk)

# ----------------------------------------------------------------------
# Parent B – MinHash signature & bandit policy
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

@dataclass
class StrikeState:
    position: float = 0.0
    velocity: float = 0.0

_POLICY: Dict[str, List[float]] = {}          # action_id → [cumulative_reward, count]

def reset_policy() -> None:
    """Clear the global bandit policy."""
    _POLICY.clear()

def _reward(action: str) -> float:
    total, n = _POLICY.get(action, [0.0, 0.0])
    return total / n if n else 0.0

def _count(action: str) -> int:
    return int(_POLICY.get(action, [0.0, 0.0])[1])

def entropic_minhash(probs: List[float], n_hashes: int = 64) -> List[int]:
    """Create a MinHash signature from a probability distribution."""
    # Normalise to a proper distribution
    prob_arr = np.array(probs, dtype=float)
    if prob_arr.sum() == 0:
        prob_arr = np.ones_like(prob_arr) / prob_arr.size
    else:
        prob_arr = prob_arr / prob_arr.sum()
    # Simple hash: for each hash function pick the smallest index after random permutation
    signature = []
    rng = np.random.default_rng()
    for _ in range(n_hashes):
        perm = rng.permutation(len(prob_arr))
        # first index where cumulative prob exceeds a uniform draw
        u = rng.random()
        cum = np.cumsum(prob_arr[perm])
        idx = int(np.searchsorted(cum, u))
        signature.append(int(perm[idx]))
    return signature

def hamming_similarity(sig1: List[int], sig2: List[int]) -> float:
    """Return the fraction of equal components between two signatures."""
    if len(sig1) != len(sig2) or not sig1:
        return 0.0
    equal = sum(1 for a, b in zip(sig1, sig2) if a == b)
    return equal / len(sig1)

def bandit_policy_update_weighted(
    context_id: str,
    action_id: str,
    reward: float,
    base_propensity: float,
    similarity: float,
) -> BanditAction:
    """
    Update the global bandit policy.
    The effective propensity is π = base_propensity·(1+similarity).
    The reward is added to the cumulative reward for the action.
    Returns a snapshot of the updated BanditAction.
    """
    effective_propensity = base_propensity * (1.0 + similarity)
    total, count = _POLICY.get(action_id, [0.0, 0.0])
    total += reward
    count += 1
    _POLICY[action_id] = [total, count]
    exp_reward = total / count if count else 0.0
    # Upper confidence bound (UCB) using a simple sqrt term
    conf_bound = exp_reward + math.sqrt(2 * math.log(max(1, count)) / count) if count else float('inf')
    return BanditAction(
        action_id=action_id,
        propensity=effective_propensity,
        expected_reward=exp_reward,
        confidence_bound=conf_bound,
        algorithm="HCBF",
    )

def hybrid_strike(
    data: Dict[str, List[float]],
    treatment: str,
    outcome: str,
    confounders: List[str],
    unique_qi: int,
    total_records: int,
    base_propensity: float = 0.1,
    steps: int = 20,
    dt: float = 0.05,
) -> StrikeState:
    """
    Perform a drag‑limited integration where the force at each step is:
        F = σ·τ_w
    with σ = similarity between consecutive MinHash signatures,
    τ_w = weighted causal effect.
    The integration updates position and velocity using simple Euler steps.
    """
    # Initial causal effect (static across steps for simplicity)
    tau_w = weighted_causal_effect(treatment, outcome, confounders, data, unique_qi, total_records)
    if tau_w is None:
        tau_w = 0.0

    # Build an initial MinHash signature from the treatment distribution
    signature_prev = entropic_minhash(data.get(treatment, []))

    state = StrikeState()
    for step in range(steps):
        # New signature from the outcome distribution (acts as “next context”)
        signature_curr = entropic_minhash(data.get(outcome, []))
        sigma = hamming_similarity(signature_prev, signature_curr)

        # Bandit update using the weighted effect as reward
        action_id = f"step_{step}"
        bandit_action = bandit_policy_update_weighted(
            context_id=f"ctx_{step}",
            action_id=action_id,
            reward=tau_w,
            base_propensity=base_propensity,
            similarity=sigma,
        )

        # Force = similarity * weighted effect
        force = sigma * tau_w

        # Simple drag term proportional to velocity (linear damping)
        drag = -0.1 * state.velocity

        # Acceleration = (force + drag) (mass = 1)
        accel = force + drag

        # Euler integration
        state.velocity += accel * dt
        state.position += state.velocity * dt

        # Prepare for next iteration
        signature_prev = signature_curr

    return state

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic dataset
    np.random.seed(42)
    n = 200
    data = {
        "treatment": (np.random.rand(n) > 0.5).astype(int).tolist(),
        "outcome": (np.random.rand(n) * 10).tolist(),
        "age": (np.random.randint(20, 70, size=n)).tolist(),
        "income": (np.random.randint(30000, 120000, size=n)).tolist(),
    }

    # Parameters
    treatment_col = "treatment"
    outcome_col = "outcome"
    confounders = ["age", "income"]
    unique_qi = 150
    total_records = n

    # Run hybrid strike
    final_state = hybrid_strike(
        data=data,
        treatment=treatment_col,
        outcome=outcome_col,
        confounders=confounders,
        unique_qi=unique_qi,
        total_records=total_records,
    )

    print("Final StrikeState:", asdict(final_state))
    print("Policy snapshot (first 3 actions):")
    for i, (aid, vals) in enumerate(_POLICY.items()):
        if i >= 3:
            break
        print(f"  {aid}: reward={vals[0]:.3f}, count={vals[1]}")
    sys.exit(0)