# DARWIN HAMMER — match 41, survivor 3
# gen: 3
# parent_a: hybrid_hybrid_bandit_router_hybrid_model_vram_sc_m35_s2.py (gen2)
# parent_b: hybrid_capybara_optimizatio_tri_algo_conduit_m55_s2.py (gen1)
# born: 2026-05-29T23:25:33Z

"""Hybrid Bandit‑Capybara Scheduler‑Optimizer

This module fuses the two parent algorithms:

* **Parent A** – a contextual multi‑armed bandit router that emits a
  `BanditAction` with a *propensity* (interpreted as an inflow) and a
  *confidence_bound* (interpreted as an outflow).  It also drives a
  linear “TTT” weight‑matrix update whose learning rate is scaled by the
  propensity.

* **Parent B** – a capybara‑style continuous optimisation routine that
  uses an exponential evasion schedule `δ(t)` and a Hoeffding‑derived
  epsilon `ε` to compute a hybrid evasion magnitude `δ_h = δ(t)*(1+ε)`.

**Mathematical bridge**

The bridge is built on the *store equation* of the honeybee primitive
and the Hoeffding‑epsilon of the tri‑algo conduit:


Δstore = α·propensity – β·confidence_bound
storeₜ₊₁ = max(0, storeₜ + Δstore·dt)                (1)

η      = η₀·(1 + propensity)                         (2)

ε      = sqrt( (log(2/δ_hoeffding)) / (2·n) )        (Hoeffding bound)
δ_h    = δ(t)·(1 + ε)                                 (3)

grad   = ∂L/∂W  (loss derived from current positions)
Wₜ₊₁   = Wₜ – η·grad                                 (4)

posᵢₜ₊₁ = clamp( posᵢₜ + δ_h·r·direction , lo , hi ) (5)


Equations (1)–(5) are evaluated sequentially for every bandit decision,
so a single `BanditAction` simultaneously drives the virtual‑VRAM store,
the learning‑rate‑scaled matrix update, and the evasion‑driven position
perturbation.  The result is a unified hybrid system that couples
resource‑allocation dynamics with continuous optimisation dynamics.

"""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Tuple, Sequence

import numpy as np

# ----------------------------------------------------------------------
# Shared constants
# ----------------------------------------------------------------------
ALPHA = 0.6          # store inflow coefficient
BETA = 0.4           # store outflow coefficient
DT = 1.0             # time step for store dynamics
ETA0 = 0.01          # base learning rate for matrix updates
DELTA_MAX = 1.0      # max evasion magnitude
ALPHA_EVASION = 3.0  # decay rate for evasion schedule
HOEFFDING_DELTA = 0.05  # confidence level for Hoeffding bound
CLAMP_LO = -5.0
CLAMP_HI = 5.0

# ----------------------------------------------------------------------
# Bandit core (Parent A)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float          # interpreted as inflow
    expected_reward: float
    confidence_bound: float    # interpreted as outflow
    algorithm: str


@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float


_POLICY: Dict[str, List[float]] = {}   # action_id -> [total_reward, count]
_STORE: Dict[str, float] = {}          # virtual VRAM store per key


def reset_policy() -> None:
    """Clear learned statistics and the virtual store."""
    _POLICY.clear()
    _STORE.clear()


def _reward_estimate(action_id: str) -> float:
    total, n = _POLICY.get(action_id, [0.0, 0.0])
    return total / n if n else 0.0


def _confidence_bound(action_id: str) -> float:
    """Hoeffding‑type confidence bound for the given action."""
    _, n = _POLICY.get(action_id, [0.0, 0.0])
    if n == 0:
        return float("inf")
    return math.sqrt(math.log(2.0 / HOEFFDING_DELTA) / (2.0 * n))


def select_action(
    context: Dict[str, float],
    actions: List[str],
    algorithm: str = "linucb",
    epsilon: float = 0.1,
    seed: int | str | None = 7,
) -> BanditAction:
    """Select an action using a simple LinUCB‑like rule."""
    if seed is not None:
        random.seed(seed)

    if not actions:
        raise ValueError("action list cannot be empty")

    # For each action compute: μ + ε·conf where μ is estimated reward
    # and conf is the Hoeffding bound.
    scores: List[Tuple[str, float, float]] = []
    for a in actions:
        mu = _reward_estimate(a)
        conf = _confidence_bound(a)
        score = mu + epsilon * conf
        scores.append((a, score, conf))

    # Choose the action with the highest score
    best_action, best_score, best_conf = max(scores, key=lambda x: x[1])

    # Propensity is a softmax‑like mapping of the score to (0,1)
    exp_vals = [math.exp(s[1]) for s in scores]
    total_exp = sum(exp_vals)
    propensity = math.exp(best_score) / total_exp if total_exp > 0 else 0.0

    return BanditAction(
        action_id=best_action,
        propensity=propensity,
        expected_reward=_reward_estimate(best_action),
        confidence_bound=best_conf,
        algorithm=algorithm,
    )


def update_policy(update: BanditUpdate) -> None:
    """Incorporate observed reward into the policy statistics."""
    total, n = _POLICY.get(update.action_id, [0.0, 0.0])
    _POLICY[update.action_id] = [total + update.reward, n + 1]


# ----------------------------------------------------------------------
# Capybara optimisation primitives (Parent B)
# ----------------------------------------------------------------------
Vector = Sequence[float]


def evasion_delta(t: int, t_max: int, delta_max: float = DELTA_MAX, alpha: float = ALPHA_EVASION) -> float:
    """Exponential decay schedule for evasion magnitude."""
    if t < 0 or t_max <= 0 or delta_max < 0 or alpha < 0:
        raise ValueError("invalid evasion schedule parameters")
    return delta_max * math.exp(-alpha * min(t, t_max) / t_max)


def clamp(x: Vector, lower: float, upper: float) -> list[float]:
    """Clamp each component of a vector to [lower, upper]."""
    return [min(upper, max(lower, xi)) for xi in x]


def _shannon_entropy(data: Sequence[int]) -> float:
    """Return Shannon entropy (bits) for a sequence of byte values."""
    if not data:
        return 0.0
    counts = np.bincount(np.array(data, dtype=np.uint8), minlength=256)
    probs = counts[counts > 0] / len(data)
    return -np.sum(probs * np.log2(probs))


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def _update_store(action: BanditAction, key: str = "global") -> float:
    """Update the virtual VRAM store using Eq. (1) and return the new store."""
    inflow = action.propensity
    outflow = action.confidence_bound
    delta_store = ALPHA * inflow - BETA * outflow
    prev = _STORE.get(key, 0.0)
    new_store = max(0.0, prev + delta_store * DT)
    _STORE[key] = new_store
    return new_store


def _gradient_from_positions(W: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """
    Simple synthetic loss: L = 0.5 * ||W·pos - g||²,
    where g is the global best (mean of positions).
    Returns the gradient ∂L/∂W.
    """
    if positions.size == 0:
        return np.zeros_like(W)

    g_best = positions.mean(axis=0)               # shape (d,)
    preds = positions @ W.T                        # shape (n, d)
    error = preds - g_best                         # shape (n, d)
    grad = (error.T @ positions) / positions.shape[0]  # shape (d, d)
    return grad


def hybrid_step(
    W: np.ndarray,
    positions: np.ndarray,
    t: int,
    t_max: int,
    context: Dict[str, float],
    actions: List[str],
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Perform one hybrid iteration:

    1. Bandit selects an action.
    2. Store is updated (Eq. 1).
    3. Learning rate η is computed (Eq. 2) and the weight matrix W is updated (Eq. 4).
    4. Hoeffding epsilon ε is derived from the selected action’s confidence bound.
    5. Hybrid evasion magnitude δ_h is computed (Eq. 3).
    6. Each position vector is perturbed (Eq. 5) and clamped.

    Returns updated (W, positions, store_value).
    """
    # 1. Bandit decision
    ba = select_action(context, actions)

    # 2. Store dynamics
    store_val = _update_store(ba)

    # 3. Matrix learning
    eta = ETA0 * (1.0 + ba.propensity)
    grad = _gradient_from_positions(W, positions)
    W_new = W - eta * grad

    # 4. Hoeffding epsilon (derived from confidence bound)
    # The confidence bound itself is already a Hoeffding‑type term,
    # we turn it into a scaled epsilon for the evasion schedule.
    epsilon_h = ba.confidence_bound / (1.0 + store_val)  # simple scaling

    # 5. Hybrid evasion magnitude
    delta_t = evasion_delta(t, t_max)
    delta_h = delta_t * (1.0 + epsilon_h)

    # 6. Position perturbation
    n, dim = positions.shape
    random_dirs = np.random.randn(n, dim)
    random_dirs /= np.linalg.norm(random_dirs, axis=1, keepdims=True)  # unit vectors
    perturb = delta_h * random_dirs
    new_positions = positions + perturb
    # Clamp to admissible region
    new_positions = np.array([clamp(p, CLAMP_LO, CLAMP_HI) for p in new_positions])

    # Optional: simulate a reward and feed back to the bandit
    simulated_reward = -np.mean(np.linalg.norm(new_positions, axis=1))  # encourage proximity to origin
    update_policy(BanditUpdate(
        context_id="default",
        action_id=ba.action_id,
        reward=simulated_reward,
        propensity=ba.propensity,
    ))

    return W_new, new_positions, store_val


def run_hybrid(
    dim: int = 4,
    n_agents: int = 10,
    steps: int = 20,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Initialise a random weight matrix and a population of agents,
    then execute `steps` hybrid iterations.

    Returns the final (W, positions, store).
    """
    # Initialise
    W = np.random.randn(dim, dim)
    positions = np.random.uniform(CLAMP_LO, CLAMP_HI, size=(n_agents, dim))
    actions = [f"act_{i}" for i in range(5)]
    context = {"dummy": 0.0}

    for t in range(steps):
        W, positions, store = hybrid_step(
            W=W,
            positions=positions,
            t=t,
            t_max=steps,
            context=context,
            actions=actions,
        )
    return W, positions, store


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple sanity run – should complete without exception.
    final_W, final_pos, final_store = run_hybrid(dim=3, n_agents=6, steps=15)
    print("Final weight matrix (W):")
    print(final_W)
    print("\nFinal positions (first 3 agents):")
    print(final_pos[:3])
    print("\nFinal virtual store value:", final_store)
    # Demonstrate entropy calculation on recent rewards
    recent_rewards = [update.reward for update in [
        BanditUpdate("default", a, random.random(), 0.1) for a in ["act_0", "act_1", "act_2"]
    ]]
    entropy = _shannon_entropy([int(r * 255) for r in recent_rewards])
    print("\nShannon entropy of synthetic rewards:", entropy)