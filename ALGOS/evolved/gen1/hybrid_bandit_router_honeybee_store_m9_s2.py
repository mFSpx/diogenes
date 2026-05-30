# DARWIN HAMMER — match 9, survivor 2
# gen: 1
# parent_a: bandit_router.py (gen0)
# parent_b: honeybee_store.py (gen0)
# born: 2026-05-29T23:16:48Z

"""Hybrid Bandit-Store Algorithm

This module fuses two independent prototypes:

* **bandit_router.py** – a lightweight contextual bandit router offering
  epsilon‑greedy, Thompson sampling and a LinUCB‑style upper‑confidence bound.
* **honeybee_store.py** – a simple store dynamics primitive where a scalar
  “store” evolves according to inflow/outflow (Δ = α·∑in – β·∑out) and a
  “dance duration” function maps that Δ to a bounded control signal.

### Mathematical Bridge
The bridge is built on the observation that both algorithms maintain a scalar
state that can be interpreted as a *resource level*:

* In the bandit, the per‑action confidence term `c_a = 1/√(1+N_a)` shrinks as the
  number of pulls `N_a` grows.
* In the store, the scalar `S` evolves with Δ and influences downstream behaviour
  through `dance_duration`.

We let the current store value `S` modulate the confidence term of the bandit,
creating a coupled system:


c_a(S) = (1 + S/(S+1)) / √(1 + N_a)
U_a = μ_a + η·scale·c_a(S)


where `μ_a` is the empirical mean reward, `η` a tunable exploration factor,
and `scale` the norm of the context vector. After an action is taken, its
reward is fed back as *inflow* to the store, while a fixed *cost* can be treated
as outflow. This closes a feedback loop: the store shapes exploration, and the
resulting actions drive store dynamics.

The implementation below provides three core functions demonstrating this
hybrid operation, together with a small smoke test.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Shared data structures (derived from bandit_router.py)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class BanditAction:
    """Result of an action selection."""
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str


@dataclass(frozen=True)
class BanditUpdate:
    """Single observation used to update the policy."""
    context_id: str
    action_id: str
    reward: float
    propensity: float


# ----------------------------------------------------------------------
# Global policy store (action → [cumulative reward, count])
# ----------------------------------------------------------------------
_POLICY: Dict[str, List[float]] = {}


def reset_policy() -> None:
    """Erase all learned statistics."""
    _POLICY.clear()


def _reward(action: str) -> float:
    """Empirical mean reward for *action* (0 if never observed)."""
    total, n = _POLICY.get(action, [0.0, 0.0])
    return total / n if n else 0.0


def _count(action: str) -> float:
    """Number of times *action* has been selected."""
    return _POLICY.get(action, [0.0, 0.0])[1]


def update_policy(updates: List[BanditUpdate]) -> None:
    """Incorporate a batch of observations into the global policy."""
    for u in updates:
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        stats[0] += float(u.reward)
        stats[1] += 1.0


# ----------------------------------------------------------------------
# Store dynamics (derived from honeybee_store.py)
# ----------------------------------------------------------------------


def update_store(
    store: float,
    inflow: List[float],
    outflow: List[float],
    alpha: float = 1.0,
    beta: float = 1.0,
    dt: float = 1.0,
) -> Tuple[float, float]:
    """
    Apply the honeybee store equation.

    Δ = α·Σ(inflow) – β·Σ(outflow)
    S' = max(0, S + dt·Δ)

    Returns the new store value and the raw Δ.
    """
    delta = alpha * sum(inflow) - beta * sum(outflow)
    new_store = max(0.0, store + dt * delta)
    return new_store, delta


def dance_duration(
    delta_store: float,
    base: float = 1.0,
    gain: float = 1.0,
    limit: float = 10.0,
) -> float:
    """
    Map a store change Δ to a bounded control signal.
    """
    return max(0.0, min(limit, base + gain * delta_store))


# ----------------------------------------------------------------------
# Hybrid core: action selection that depends on the current store
# ----------------------------------------------------------------------


def hybrid_select_action(
    context: Dict[str, float],
    actions: List[str],
    store: float,
    algorithm: str = "linucb",
    epsilon: float = 0.1,
    eta: float = 0.1,
    seed: int | str | None = 7,
) -> BanditAction:
    """
    Choose an action using a bandit rule whose confidence term is scaled by the
    current store value.

    Parameters
    ----------
    context : dict
        Feature vector for the current decision (used only for its Euclidean norm).
    actions : list[str]
        Candidate action identifiers.
    store : float
        Current store level (non‑negative). Higher store → more aggressive
        exploration via an inflated confidence bound.
    algorithm : {'linucb', 'epsilon_greedy', 'thompson'}
        Which underlying bandit policy to employ.
    epsilon : float
        Exploration probability for epsilon‑greedy.
    eta : float
        Base scaling factor for the confidence term (mirrors the 0.1 constant
        in the original LinUCB stub).
    seed : int|str|None
        Random seed for reproducibility.

    Returns
    -------
    BanditAction
        The selected action together with diagnostic quantities.
    """
    if not actions:
        raise ValueError("actions required")
    rng = random.Random(seed)

    # Helper: store‑dependent confidence multiplier
    store_factor = 1.0 + store / (store + 1.0)  # lies in (1,2)

    if algorithm == "epsilon_greedy" and rng.random() < epsilon:
        chosen = rng.choice(actions)
    elif algorithm == "thompson":
        # Thompson sampling with Beta posterior, modulated by store_factor
        def sample(a: str) -> float:
            r = _reward(a)
            n = _count(a)
            # Pseudo‑counts for Beta; add store_factor as a weak prior
            a_param = 1.0 + max(0.0, r) * store_factor
            b_param = 1.0 + max(0.0, 1.0 - r) * store_factor
            return rng.betavariate(a_param, b_param)

        chosen = max(actions, key=sample)
    else:  # default: LinUCB‑style upper confidence bound
        # Context norm (scale) – identical to original implementation
        scale = math.sqrt(sum(float(v) * float(v) for v in context.values())) if context else 1.0

        def ucb_score(a: str) -> float:
            mean = _reward(a)
            cnt = _count(a)
            conf = store_factor / math.sqrt(1.0 + cnt)
            return mean + eta * scale * conf

        chosen = max(actions, key=ucb_score)

    # Diagnostic quantities
    prop = 1.0 / len(actions)
    exp_reward = _reward(chosen)
    conf_bound = store_factor / math.sqrt(1.0 + _count(chosen))

    return BanditAction(
        action_id=chosen,
        propensity=prop,
        expected_reward=exp_reward,
        confidence_bound=conf_bound,
        algorithm=algorithm,
    )


# ----------------------------------------------------------------------
# Hybrid step: select → observe reward → update policy & store
# ----------------------------------------------------------------------


def hybrid_step(
    context: Dict[str, float],
    actions: List[str],
    store: float,
    true_reward_fn,
    algorithm: str = "linucb",
    epsilon: float = 0.1,
    eta: float = 0.1,
    alpha: float = 1.0,
    beta: float = 1.0,
    dt: float = 1.0,
    seed: int | str | None = 7,
) -> Tuple[BanditAction, float, float]:
    """
    Execute one hybrid iteration.

    *Select* an action, *sample* its stochastic reward via ``true_reward_fn``,
    *update* the bandit statistics, and *propagate* the reward as inflow to the
    store.

    Returns
    -------
    (action, reward, new_store)
    """
    # 1. Action selection (store‑aware)
    action = hybrid_select_action(
        context,
        actions,
        store,
        algorithm=algorithm,
        epsilon=epsilon,
        eta=eta,
        seed=seed,
    )

    # 2. Observe stochastic reward (external environment)
    reward = float(true_reward_fn(action.action_id, context))

    # 3. Update bandit policy
    update_policy(
        [
            BanditUpdate(
                context_id="".join(str(k) for k in sorted(context.keys())),
                action_id=action.action_id,
                reward=reward,
                propensity=action.propensity,
            )
        ]
    )

    # 4. Feed reward as inflow; assume a constant unit cost as outflow
    new_store, delta = update_store(
        store,
        inflow=[reward],
        outflow=[1.0],  # fixed cost per decision
        alpha=alpha,
        beta=beta,
        dt=dt,
    )

    # 5. (Optional) expose the dance duration as a side‑channel metric
    _ = dance_duration(delta, base=1.0, gain=0.5, limit=10.0)

    return action, reward, new_store


# ----------------------------------------------------------------------
# Convenience wrapper for batch processing
# ----------------------------------------------------------------------


def hybrid_run(
    contexts: List[Dict[str, float]],
    actions: List[str],
    init_store: float = 5.0,
    true_reward_fn=None,
    algorithm: str = "linucb",
    epsilon: float = 0.1,
    eta: float = 0.1,
    alpha: float = 1.0,
    beta: float = 1.0,
    dt: float = 1.0,
    seed: int | str | None = 7,
) -> Tuple[List[BanditAction], List[float], float]:
    """
    Run the hybrid system over a sequence of contexts.

    Parameters
    ----------
    contexts : list[dict]
        Sequence of feature dictionaries.
    actions : list[str]
        Global action set (unchanged across steps).
    init_store : float
        Starting store level.
    true_reward_fn : callable(action_id, context) → float
        Function that returns a stochastic reward. If ``None`` a simple
        synthetic Bernoulli model is used.
    """
    if true_reward_fn is None:

        def default_reward(a_id: str, ctx: Dict[str, float]) -> float:
            # Synthetic reward: probability = sigmoid of (bias + context norm)
            bias = {"A": 0.2, "B": 0.5, "C": -0.1}.get(a_id, 0.0)
            norm = math.sqrt(sum(v * v for v in ctx.values())) if ctx else 0.0
            p = 1.0 / (1.0 + math.exp(-(bias + 0.1 * norm)))
            return 1.0 if random.random() < p else 0.0

        true_reward_fn = default_reward

    store = float(init_store)
    chosen_actions: List[BanditAction] = []
    rewards: List[float] = []

    rng = random.Random(seed)

    for ctx in contexts:
        act, rew, store = hybrid_step(
            ctx,
            actions,
            store,
            true_reward_fn,
            algorithm=algorithm,
            epsilon=epsilon,
            eta=eta,
            alpha=alpha,
            beta=beta,
            dt=dt,
            seed=rng.randint(0, 2 ** 31 - 1),
        )
        chosen_actions.append(act)
        rewards.append(rew)

    return chosen_actions, rewards, store


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Minimal sanity check: run a short horizon with synthetic data.
    actions_list = ["A", "B", "C"]
    test_contexts = [
        {"x1": random.uniform(-1, 1), "x2": random.uniform(-1, 1)} for _ in range(10)
    ]

    # Reset any lingering state from previous imports
    reset_policy()

    acts, rew, final_store = hybrid_run(
        test_contexts,
        actions_list,
        init_store=3.0,
        algorithm="linucb",
        epsilon=0.2,
        eta=0.05,
        alpha=0.8,
        beta=0.3,
        dt=0.5,
        seed=42,
    )

    print("Selected actions:", [a.action_id for a in acts])
    print("Rewards:", rew)
    print("Final store level:", final_store)

    # Ensure the policy dictionary has been populated
    assert len(_POLICY) > 0, "Policy should contain at least one entry"
    sys.exit(0)