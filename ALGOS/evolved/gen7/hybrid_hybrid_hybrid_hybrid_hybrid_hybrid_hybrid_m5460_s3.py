# DARWIN HAMMER — match 5460, survivor 3
# gen: 7
# parent_a: hybrid_hybrid_hybrid_bandit_hybrid_capybara_opti_m41_s0.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_regret_m2199_s0.py (gen6)
# born: 2026-05-30T00:02:00Z

"""Hybrid Bandit‑Regret‑Entropic Algorithm (HBREA)

Parents:
- hybrid_hybrid_hybrid_bandit_hybrid_capybara_opti_m41_s0.py (Bandit‑Capybara core)
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_regret_m2199_s0.py (Regret‑Entropic core)

Mathematical Bridge
-------------------
The Bandit core produces, for a given context, a *propensity* `p` and a
*confidence bound* `c`.  In the Regret‑Entropic core the Weighted Average
Treatment Effect (WATE) is derived from a reconstruction‑risk score `r`:


WATE = (1 - r) * N / (N + 1)          (N = total_records)


We fuse the two by using `p·c·WATE` as a *scalar confidence* that multiplies
the instantaneous regret


Δ = (Ē_R - r_actual)                     # base regret
Regret = Δ * p * c * ate * WATE          # ate = ATE estimate from causal effect


The resulting Regret term drives the update of the Bandit policy,
while the causal ATE modulates the learning rate, closing the loop between
exploration (Bandit) and causal‑aware exploitation (Regret‑Entropic)."""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Callable, Any

import numpy as np

# ----------------------------------------------------------------------
# Shared global stores
# ----------------------------------------------------------------------
_POLICY: Dict[str, List[float]] = {}          # action -> [total_reward, count]
_CAUSAL_STORE: Dict[str, "CausalEffect"] = {}  # effect_id -> CausalEffect


def reset_hybrid() -> None:
    """Reset all learned statistics and stored causal effects."""
    _POLICY.clear()
    _CAUSAL_STORE.clear()


# ----------------------------------------------------------------------
# Bandit core (adapted from Parent A)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float          # exploration probability (0..1)
    expected_reward: float     # current estimate of reward
    confidence_bound: float    # statistical confidence (0..1)
    algorithm: str


def _reward_estimate(action: str) -> float:
    total, n = _POLICY.get(action, [0.0, 0.0])
    return total / n if n else 0.0


def select_action(
    context: Dict[str, float],
    actions: List[str],
    algorithm: str = "linucb",
    epsilon: float = 0.1,
    seed: int | str | None = 7,
) -> BanditAction:
    """
    Very light‑weight placeholder for a contextual bandit selector.
    Returns a random action together with synthetic propensity,
    expected reward and confidence bound.
    """
    if not actions:
        raise ValueError("No actions provided")
    rng = random.Random(seed)
    chosen = rng.choice(actions)
    # synthetic statistics
    propensity = rng.random()
    expected = _reward_estimate(chosen)
    confidence = min(1.0, 1.0 / (1.0 + rng.random()))  # higher when few pulls
    return BanditAction(chosen, propensity, expected, confidence, algorithm)


def update_bandit_policy(action: str, reward: float) -> None:
    """Incremental update of the simple reward average stored in _POLICY."""
    total, n = _POLICY.get(action, [0.0, 0.0])
    _POLICY[action] = [total + reward, n + 1]


# ----------------------------------------------------------------------
# Regret‑Entropic core (adapted from Parent B)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str
    vram_mb: int


def reconstruction_risk_score(unique_quasi_identifiers: int, total_records: int) -> float:
    """Normalized risk score in [0,1] based on identifier uniqueness."""
    if total_records <= 0:
        return 0.0
    return max(0.0, min(1.0, unique_quasi_identifiers / total_records))


@dataclass(frozen=True)
class CausalEffect:
    effect_id: str
    treatment: str
    outcome: str
    confounders: Tuple[str, ...]
    ate_estimate: float | None               # Average Treatment Effect
    ate_confidence_interval: Tuple[float, float] | None
    refutation_passed: bool
    refutation_methods: Tuple[str, ...]


def weighted_average_treatment_effect(risk: float, total_records: int) -> float:
    """
    Compute WATE as a smooth function of risk and dataset size.
    The formula mirrors the bridge described in the module docstring.
    """
    return (1.0 - risk) * total_records / (total_records + 1.0)


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def compute_regret(
    bandit_action: BanditAction,
    actual_reward: float,
    causal_effect: CausalEffect | None,
    wate: float,
) -> float:
    """
    Core hybrid regret computation.

    Δ = (Ē_R - r_actual)                       # base regret
    Regret = Δ * p * c * ate * WATE
    """
    base_regret = bandit_action.expected_reward - actual_reward
    ate = causal_effect.ate_estimate if causal_effect and causal_effect.ate_estimate is not None else 1.0
    regret = base_regret * bandit_action.propensity * bandit_action.confidence_bound * ate * wate
    return regret


def hybrid_step(
    context: Dict[str, float],
    actions: List[str],
    causal_effects: List[CausalEffect],
    unique_quasi_identifiers: int,
    total_records: int,
    reward_function: Callable[[str, Dict[str, float]], float],
) -> Tuple[BanditAction, float]:
    """
    Execute one hybrid iteration:
      1. Bandit selects an action.
      2. WATE is derived from reconstruction risk.
      3. The causal effect matching the chosen action (if any) is fetched.
      4. Regret is computed and the bandit policy is updated.
    Returns the selected action and the computed regret value.
    """
    # 1. Bandit decision
    ba = select_action(context, actions)

    # 2. Risk → WATE
    risk = reconstruction_risk_score(unique_quasi_identifiers, total_records)
    wate = weighted_average_treatment_effect(risk, total_records)

    # 3. Locate causal effect (match by treatment == action_id)
    causal = next((c for c in causal_effects if c.treatment == ba.action_id), None)

    # 4. Observe reward
    reward = reward_function(ba.action_id, context)

    # 5. Regret calculation
    regret = compute_regret(ba, reward, causal, wate)

    # 6. Policy update using regret as a learning‑rate modifier
    #    (simple SGD‑like update on the average reward)
    lr = max(0.0, 1.0 - regret)  # keep learning rate in [0,1]
    adjusted_reward = reward * lr + ba.expected_reward * (1.0 - lr)
    update_bandit_policy(ba.action_id, adjusted_reward)

    return ba, regret


def batch_hybrid_run(
    contexts: List[Dict[str, float]],
    actions: List[str],
    causal_effects: List[CausalEffect],
    unique_quasi_identifiers: int,
    total_records: int,
    reward_function: Callable[[str, Dict[str, float]], float],
) -> List[Tuple[BanditAction, float]]:
    """
    Run the hybrid algorithm over a batch of contexts.
    Returns a list of (BanditAction, regret) tuples.
    """
    results = []
    for ctx in contexts:
        ba, rg = hybrid_step(
            ctx,
            actions,
            causal_effects,
            unique_quasi_identifiers,
            total_records,
            reward_function,
        )
        results.append((ba, rg))
    return results


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # deterministic seed for reproducibility
    random.seed(42)

    # Example actions
    actions = ["alpha", "beta", "gamma"]

    # Dummy causal effects (ATEs are arbitrary)
    causal_effects = [
        CausalEffect(
            effect_id="e1",
            treatment="alpha",
            outcome="outcome",
            confounders=(),
            ate_estimate=0.8,
            ate_confidence_interval=(0.6, 1.0),
            refutation_passed=True,
            refutation_methods=("method1",),
        ),
        CausalEffect(
            effect_id="e2",
            treatment="beta",
            outcome="outcome",
            confounders=(),
            ate_estimate=1.2,
            ate_confidence_interval=(1.0, 1.4),
            refutation_passed=True,
            refutation_methods=("method2",),
        ),
    ]

    # Simple stochastic reward function
    def reward_fn(action_id: str, ctx: Dict[str, float]) -> float:
        base = {"alpha": 1.0, "beta": 0.5, "gamma": 0.2}[action_id]
        noise = random.gauss(0, 0.1)
        return max(0.0, base + noise)

    # Single‑step demo
    ctx = {"feature1": 0.3, "feature2": 0.7}
    ba, rg = hybrid_step(
        context=ctx,
        actions=actions,
        causal_effects=causal_effects,
        unique_quasi_identifiers=42,
        total_records=1000,
        reward_function=reward_fn,
    )
    print("Selected:", ba)
    print("Regret:", rg)

    # Batch demo
    batch_ctxs = [ {"f": i/10.0} for i in range(5) ]
    batch_results = batch_hybrid_run(
        contexts=batch_ctxs,
        actions=actions,
        causal_effects=causal_effects,
        unique_quasi_identifiers=42,
        total_records=1000,
        reward_function=reward_fn,
    )
    print("\nBatch results:")
    for i, (act, reg) in enumerate(batch_results):
        print(f"  [{i}] action={act.action_id}, regret={reg:.4f}")

    # Show final policy estimates
    print("\nFinal policy estimates:")
    for a in actions:
        est = _reward_estimate(a)
        total, n = _POLICY.get(a, (0.0, 0))
        print(f"  {a}: avg_reward={est:.3f} (n={int(n)})")