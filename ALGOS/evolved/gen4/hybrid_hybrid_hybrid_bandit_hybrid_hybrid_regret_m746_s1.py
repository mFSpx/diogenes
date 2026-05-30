# DARWIN HAMMER — match 746, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_bandit_router_hybrid_model_vram_sc_m35_s3.py (gen2)
# parent_b: hybrid_hybrid_regret_engine_hybrid_ternary_lens__m3_s5.py (gen3)
# born: 2026-05-29T23:30:44Z

"""
Hybrid Regret-Weighted Ternary-Decision Analyzer (RW-TD-H) with Bandit Core

This module fuses:
* **Hybrid Bandit Router** (Parent A) – a regret-agnostic bandit exploration algorithm
* **Hybrid Regret-Weighted Ternary-Decision Analyzer** (Parent B) – a hybrid regret-and-hygiene analyzer

Mathematical bridge:
The hybrid algorithm maps the regret-weighted probabilities from Parent A onto the ternary alphabet by sign-quantisation,
concatenates the resulting symbolic sequence with the ternary vector from Parent B, and evaluates the Shannon entropy of the combined
empirical distribution.
"""

import math
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

# Shared data structures
@dataclass(frozen=True)
class HybridAction:
    action_id: str
    expected_reward: float
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0
    ternary_symbol: int = 0

@dataclass(frozen=True)
class HybridCounterfactual:
    action_id: str
    reward: float
    outcome_value: float
    probability: float = 1.0
    ternary_symbol: int = 0

# ----------------------------------------------------------------------
# Bandit core (from Parent A)
# ----------------------------------------------------------------------
def _reward(a: str) -> float:
    total, n = _POLICY.get(a, [0.0, 0.0])
    return total / n if n else 0.0

def select_action(
    context: Dict[str, float],
    actions: List[str],
    algorithm: str = "linucb",
    epsilon: float = 0.1,
    seed: int | str | None = 7,
) -> HybridAction:
    rng = random.Random(seed)
    if algorithm == "epsilon_greedy" and rng.random() < epsilon:
        chosen = rng.choice(actions)
    elif algorithm == "thompson":
        chosen = max(
            actions,
            key=lambda a: rng.betavariate(
                1 + max(0, _reward(a)),
                1 + max(0, 1 - _reward(a)),
            ),
        )
    else:  # linucb-style surrogate
        scale = math.sqrt(
            sum(float(v) * float(v) for v in context.values())
        ) if context else 1.0
        chosen = max(
            actions,
            key=lambda a: _reward(a)
            + 0.1 * scale / math.sqrt(1 + _POLICY.get(a, [0, 0])[1]),
        )

    expected_reward = _reward(chosen)
    propensity = 1.0 / len(actions)
    confidence = 1.0 / math.sqrt(1 + _POLICY.get(chosen, [0, 0])[1])
    ternary_symbol = int(np.sign(expected_reward))  # sign-quantisation
    return HybridAction(
        action_id=chosen,
        expected_reward=expected_reward,
        expected_value=0.0,  # not used in this hybrid
        cost=0.0,
        risk=0.0,
        ternary_symbol=ternary_symbol,
    )

# ----------------------------------------------------------------------
# Hybrid Regret-Weighted Ternary-Decision Analyzer (from Parent B)
# ----------------------------------------------------------------------
def shannon_entropy(x: Iterable[float]) -> float:
    """Calculate the Shannon entropy of a discrete probability distribution."""
    p = np.array(x) / sum(x)
    return -np.sum(p * np.log2(p))

def fuse_hybrid_action(
    p: HybridAction,
    t: HybridAction
) -> HybridAction:
    """Fuse the regret-weighted probability distribution with the ternary vector."""
    return HybridAction(
        action_id=p.action_id,
        expected_reward=p.expected_reward,
        expected_value=0.0,  # not used in this hybrid
        cost=p.cost,
        risk=p.risk,
        ternary_symbol=t.ternary_symbol,
    )

def hybrid_update(
    context: Dict[str, float],
    action: HybridAction,
    outcome: HybridCounterfactual,
) -> HybridAction:
    """Update the regret-weighted probability distribution and the ternary vector."""
    # Update the regret-weighted probability distribution
    _POLICY[action.action_id][0] += outcome.reward
    _POLICY[action.action_id][1] += 1

    # Update the ternary vector
    t = HybridAction(
        action_id=action.action_id,
        expected_reward=0.0,  # not used in this hybrid
        expected_value=0.0,  # not used in this hybrid
        cost=0.0,
        risk=0.0,
        ternary_symbol=int(np.sign(outcome.reward)),
    )

    # Fuse the updated regret-weighted probability distribution with the updated ternary vector
    return fuse_hybrid_action(p=action, t=t)

# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------
def hybrid_select_action(
    context: Dict[str, float],
    actions: List[str],
    algorithm: str = "linucb",
    epsilon: float = 0.1,
    seed: int | str | None = 7,
) -> HybridAction:
    return select_action(
        context=context,
        actions=actions,
        algorithm=algorithm,
        epsilon=epsilon,
        seed=seed,
    )

def hybrid_update_with_entropy(
    context: Dict[str, float],
    action: HybridAction,
    outcome: HybridCounterfactual,
) -> float:
    """Return the Shannon entropy of the fused empirical distribution."""
    p = np.array([1.0, 1.0])  # placeholder for regret-weighted probability distribution
    t = np.array([action.ternary_symbol])  # placeholder for ternary vector
    return shannon_entropy(np.concatenate((p, t)))

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    from random import randint
    _POLICY = {str(i): [0.0, 0.0] for i in range(10)}  # placeholder for learned statistics
    context = {f'feature_{i}': randint(0, 100) for i in range(5)}
    actions = [str(i) for i in range(10)]
    algorithm = "linucb"
    epsilon = 0.1
    seed = 7
    action = hybrid_select_action(
        context=context,
        actions=actions,
        algorithm=algorithm,
        epsilon=epsilon,
        seed=seed,
    )
    outcome = HybridCounterfactual(
        action_id=action.action_id,
        reward=randint(0, 100),
        outcome_value=randint(0, 100),
    )
    entropy = hybrid_update_with_entropy(
        context=context,
        action=action,
        outcome=outcome,
    )
    print(entropy)