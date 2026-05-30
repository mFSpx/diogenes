# DARWIN HAMMER — match 56, survivor 2
# gen: 2
# parent_a: hybrid_ternary_router_ssim_m1_s2.py (gen1)
# parent_b: hybrid_bandit_router_honeybee_store_m9_s3.py (gen1)
# born: 2026-05-29T23:24:01Z

"""Hybrid SSIM‑Bandit Router
================================
Parent A: ``hybrid_ternary_router_ssim_m1_s2.py`` – provides a NumPy
implementation of the Structural Similarity Index (SSIM) and a routing
decision based on a numeric payload.

Parent B: ``hybrid_bandit_router_honeybee_store_m9_s3.py`` – implements a
contextual multi‑armed bandit with a mutable “store” that influences the
exploration/exploitation trade‑off.

Mathematical Bridge
-------------------
Both parents expose scalar metrics that modulate a decision:
* **SSIM** yields a similarity score *s* ∈ [−1, 1] derived from means,
  variances and covariance of two vectors.
* **Bandit** computes a *store factor*  

  f(store) = 1 + store/(store + 1)  

  which scales confidence bounds and expected rewards.

The fusion treats the SSIM score as an additional *contextual scaling*
that multiplicatively adjusts the store factor:


f̂ = f(store) × (½ + ½·s)          # map s∈[−1,1] → scaling∈[0,1]


Thus the similarity of the packet payload directly influences the
exploration intensity and the confidence bound used by the bandit
algorithm.  The resulting hybrid decision is a single scalar‑driven
action that can also be interpreted as a routing channel.

The module below implements this fused system, exposing three core
functions that demonstrate the combined operation:
`compute_ssim`, `hybrid_select_action`, and `hybrid_step`.  A small
smoke‑test runs the pipeline end‑to‑end."""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# SSIM implementation (Parent A)
# ----------------------------------------------------------------------
def compute_ssim(
    x: List[float] | Tuple[float, ...],
    y: List[float] | Tuple[float, ...],
    dynamic_range: float = 1.0,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """Structural Similarity Index between two equal‑length numeric sequences."""
    if len(x) != len(y):
        raise ValueError("samples must have equal length")
    if not x:
        raise ValueError("samples must not be empty")

    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    mx = x_arr.mean()
    my = y_arr.mean()
    vx = ((x_arr - mx) ** 2).mean()
    vy = ((y_arr - my) ** 2).mean()
    cov = ((x_arr - mx) * (y_arr - my)).mean()

    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2

    numerator = (2 * mx * my + c1) * (2 * cov + c2)
    denominator = (mx * mx + my * my + c1) * (vx + vy + c2)
    return float(numerator / denominator)


# ----------------------------------------------------------------------
# Payload‑to‑prototype similarity (shared by both parents)
# ----------------------------------------------------------------------
PROTOTYPE_VECTOR = np.array([0.2, 0.5, 0.3, 0.7, 0.1], dtype=np.float64)


def payload_similarity(packet: Dict[str, Any]) -> float:
    """Return SSIM similarity between packet payload and a fixed prototype.

    Missing or malformed payloads yield a similarity of 0.0.
    """
    payload = packet.get("payload")
    if not isinstance(payload, (list, tuple)):
        return 0.0
    try:
        vec = np.asarray(payload, dtype=np.float64)
        if vec.size < PROTOTYPE_VECTOR.size:
            vec = np.pad(vec, (0, PROTOTYPE_VECTOR.size - vec.size))
        elif vec.size > PROTOTYPE_VECTOR.size:
            vec = vec[: PROTOTYPE_VECTOR.size]
        return compute_ssim(vec, PROTOTYPE_VECTOR)
    except Exception:
        return 0.0


# ----------------------------------------------------------------------
# Bandit data structures (Parent B)
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


_POLICY: Dict[str, List[float]] = {}  # action_id → [cumulative_reward, count]


def reset_policy() -> None:
    _POLICY.clear()


def _reward(action: str) -> float:
    total, n = _POLICY.get(action, [0.0, 0.0])
    return total / n if n else 0.0


def _count(action: str) -> float:
    return _POLICY.get(action, [0.0, 0.0])[1]


def update_policy(updates: List[BanditUpdate]) -> None:
    for u in updates:
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        stats[0] += float(u.reward)
        stats[1] += 1.0


def update_store(
    store: float,
    inflow: List[float],
    outflow: List[float],
    alpha: float = 1.0,
    beta: float = 1.0,
    dt: float = 1.0,
) -> Tuple[float, float]:
    """Simple leaky‑bucket dynamics."""
    delta = alpha * sum(inflow) - beta * sum(outflow)
    new_store = max(0.0, store + dt * delta)
    return new_store, delta


# ----------------------------------------------------------------------
# Hybrid decision making – the mathematical fusion
# ----------------------------------------------------------------------
def hybrid_select_action(
    context: Dict[str, float],
    actions: List[str],
    store: float,
    packet: Dict[str, Any],
    algorithm: str = "linucb",
    epsilon: float = 0.1,
    eta: float = 0.1,
    seed: int | str | None = 7,
) -> BanditAction:
    """Select an action using a bandit algorithm whose confidence is scaled by SSIM.

    The SSIM similarity *s* is mapped to a factor in [0, 1] via (0.5 + 0.5·s)
    and multiplied with the classic store factor f(store) = 1 + store/(store+1).
    The resulting fused factor ``store_factor_hat`` modulates both the
    exploration term and the confidence bound.
    """
    if not actions:
        raise ValueError("actions required")
    rng = random.Random(seed)

    # ---- similarity‑driven scaling ------------------------------------
    sim = payload_similarity(packet)  # typical range [0,1] for well‑formed data
    sim_factor = 0.5 + 0.5 * sim      # maps to [0,1]
    base_store_factor = 1.0 + store / (store + 1.0)
    store_factor_hat = base_store_factor * sim_factor

    # ---- algorithmic branch -----------------------------------------
    if algorithm == "epsilon_greedy" and rng.random() < epsilon:
        chosen = rng.choice(actions)
    elif algorithm == "thompson":
        # Thompson sampling with beta parameters biased by the fused factor
        def sample(a: str) -> float:
            r = _reward(a)
            n = _count(a)
            a_param = 1.0 + max(0.0, r) * store_factor_hat
            b_param = 1.0 + max(0.0, 1.0 - r) * store_factor_hat
            return rng.betavariate(a_param, b_param)

        chosen = max(actions, key=sample)
    else:
        # LinUCB‑style upper confidence bound
        scale = np.linalg.norm(list(context.values()))
        def ucb_score(a: str) -> float:
            mean = _reward(a)
            cnt = _count(a)
            conf = store_factor_hat / math.sqrt(1.0 + cnt)
            return mean + eta * scale * conf

        chosen = max(actions, key=ucb_score)

    prop = 1.0 / len(actions)
    exp_reward = _reward(chosen)
    conf_bound = store_factor_hat / math.sqrt(1.0 + _count(chosen))

    return BanditAction(
        action_id=chosen,
        propensity=prop,
        expected_reward=exp_reward,
        confidence_bound=conf_bound,
        algorithm=algorithm,
    )


def hybrid_step(
    context: Dict[str, float],
    actions: List[str],
    store: float,
    packet: Dict[str, Any],
    true_reward_fn,
    algorithm: str = "linucb",
    epsilon: float = 0.1,
    eta: float = 0.1,
    alpha: float = 1.0,
    beta: float = 1.0,
    dt: float = 1.0,
    seed: int | str | None = 7,
    cost: float = 0.0,
) -> Tuple[BanditAction, float, float]:
    """One interaction step that fuses SSIM similarity into the bandit loop.

    Returns the chosen action, the (similarity‑scaled) reward, and the updated
    store value.
    """
    action = hybrid_select_action(context, actions, store, packet, algorithm, epsilon, eta, seed)

    # The true environment reward is modulated by payload similarity to keep
    # the whole pipeline mathematically coupled.
    base_reward = true_reward_fn(action.action_id)
    sim = payload_similarity(packet)
    reward = base_reward * (0.5 + 0.5 * sim)  # same mapping used in the selector

    update_policy([BanditUpdate("", action.action_id, reward, 1.0)])
    new_store, _ = update_store(store, [reward], [cost], alpha, beta, dt)
    return action, reward, new_store


def route_packet_hybrid(
    packet: Dict[str, Any],
    context: Dict[str, float],
    actions: List[str],
    store: float,
    true_reward_fn,
    algorithm: str = "linucb",
) -> Dict[str, Any]:
    """High‑level routing API that returns a routing dictionary enriched with
    bandit decision data and the SSIM similarity metric.
    """
    action, reward, new_store = hybrid_step(
        context,
        actions,
        store,
        packet,
        true_reward_fn,
        algorithm=algorithm,
    )
    similarity = payload_similarity(packet)

    route = {
        "payload_similarity": similarity,
        "chosen_action": action.action_id,
        "action_propensity": action.propensity,
        "expected_reward": action.expected_reward,
        "confidence_bound": action.confidence_bound,
        "algorithm": action.algorithm,
        "reward_observed": reward,
        "store_before": store,
        "store_after": new_store,
    }
    return route


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # deterministic seed for reproducibility
    random.seed(42)
    np.random.seed(0)

    # Dummy packet with a numeric payload
    test_packet = {"payload": [0.25, 0.45, 0.35, 0.65, 0.15]}

    # Context for the bandit (could be derived from packet metadata)
    test_context = {"feature1": 0.8, "feature2": 1.2}

    # Available actions
    test_actions = ["engine_ternary", "engine_binary", "engine_gpu"]

    # Initial store value
    test_store = 1.0

    # Simple stochastic reward function
    def true_reward(action_id: str) -> float:
        # Reward is higher for the ternary engine to illustrate influence
        base = {"engine_ternary": 0.9, "engine_binary": 0.4, "engine_gpu": 0.6}
        return base.get(action_id, 0.0) + random.random() * 0.1

    # Run the hybrid routing once
    result = route_packet_hybrid(
        packet=test_packet,
        context=test_context,
        actions=test_actions,
        store=test_store,
        true_reward_fn=true_reward,
        algorithm="linucb",
    )

    print("Hybrid routing result:")
    for k, v in result.items():
        print(f"  {k}: {v}")

    sys.exit(0)