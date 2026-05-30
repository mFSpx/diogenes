# DARWIN HAMMER — match 746, survivor 2
# gen: 4
# parent_a: hybrid_hybrid_bandit_router_hybrid_model_vram_sc_m35_s3.py (gen2)
# parent_b: hybrid_hybrid_regret_engine_hybrid_ternary_lens__m3_s5.py (gen3)
# born: 2026-05-29T23:30:44Z

"""Hybrid Bandit‑Regret‑Ternary Engine

This module fuses the core of two parent algorithms:

* **Parent A – Bandit Router with Virtual‑VRAM store** – provides a contextual
  multi‑armed bandit that learns expected rewards, tracks propensities and
  maintains a virtual VRAM budget per action.

* **Parent B – Regret‑Weighted MinHash & Ternary‑Decision Hygiene Analyzer** –
  supplies a regret‑weighted probability vector over actions and a deterministic
  ternary representation of an arbitrary payload.

**Mathematical bridge**

Both parents ultimately emit *discrete distributions*:

1. The bandit yields a propensity‑based distribution `π_a = 1/|A|` for the
   selected action and a learned reward estimate `r̂_a`.
2. The regret engine produces a regret‑adjusted probability vector
   `p_i = exp(−(R_max−r̂_i))/∑_j exp(−(R_max−r̂_j))`.

We map the regret‑adjusted probabilities to the ternary alphabet by a
sign‑quantisation `s_i = sign(p_i − 1/|A|) ∈ {−1,0,+1}` and concatenate this
symbolic sequence with the deterministic ternary payload vector `t`.  The
empirical distribution of the combined symbols `{−1,0,+1}` is then evaluated
with Shannon entropy.  This entropy simultaneously reflects (a) the bandit’s
confidence (through regret‑weighted probabilities) and (b) the payload‑level
“hygiene” (through the ternary vector).  Finally the entropy is used to
scale the virtual VRAM budget allocated to the chosen action.

The implementation below provides three public functions that realise the
hybrid operation and a smoke‑test when executed as a script.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Shared structures – Bandit core (Parent A)
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


_POLICY: Dict[str, List[float]] = {}          # [total_reward, count]
_STORE: Dict[str, float] = {}                 # virtual VRAM store per action
DEFAULT_BUDGET_MB = 1024 * 4                  # 4 GiB default budget


def reset_policy() -> None:
    """Clear all learned statistics and the virtual store."""
    _POLICY.clear()
    _STORE.clear()


def _reward(action: str) -> float:
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
    Choose an action using an epsilon‑greedy / Thompson / LinUCB surrogate.
    Returns a BanditAction descriptor containing the chosen action, its
    propensity, the current expected reward estimate and a simple confidence
    bound.
    """
    if not actions:
        raise ValueError("actions required")
    rng = random.Random(seed)

    # --- Exploration / exploitation decision --------------------------------
    if algorithm == "epsilon_greedy" and rng.random() < epsilon:
        chosen = rng.choice(actions)
    elif algorithm == "thompson":
        # Beta‑Bernoulli posterior (pseudo‑counts derived from reward)
        chosen = max(
            actions,
            key=lambda a: rng.betavariate(
                1 + max(0.0, _reward(a)),
                1 + max(0.0, 1 - _reward(a)),
            ),
        )
    else:  # default: LinUCB‑style surrogate
        scale = math.sqrt(
            sum(float(v) * float(v) for v in context.values())
        ) if context else 1.0
        chosen = max(
            actions,
            key=lambda a: _reward(a) + 0.1 * scale / math.sqrt(1 + _POLICY.get(a, [0, 0])[1]),
        )

    propensity = 1.0 / len(actions)
    confidence = 1.0 / math.sqrt(1 + _POLICY.get(chosen, [0, 0])[1])
    return BanditAction(
        action_id=chosen,
        propensity=propensity,
        expected_reward=_reward(chosen),
        confidence_bound=confidence,
        algorithm=algorithm,
    )


def update_policy(update: BanditUpdate) -> None:
    """
    Incorporate a new observation into the bandit statistics and adjust the
    virtual VRAM store for the associated action.
    """
    total, cnt = _POLICY.get(update.action_id, [0.0, 0.0])
    _POLICY[update.action_id] = [total + update.reward, cnt + 1]

    # Simple VRAM bookkeeping: reward proportional credit
    credit = update.reward * update.propensity
    _STORE[update.action_id] = _STORE.get(update.action_id, 0.0) + credit


# ----------------------------------------------------------------------
# Regret‑Weighted utilities – Parent B (partial)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathAction:
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


def regret_weighted_probabilities(actions: List[MathAction]) -> np.ndarray:
    """
    Compute a regret‑adjusted probability vector over the supplied actions.
    Regret for an action i is defined as R_i = max_j E_j − E_i.
    The weight w_i = exp(−R_i) yields a soft‑max distribution.
    """
    if not actions:
        raise ValueError("actions required")
    expected = np.array([a.expected_value for a in actions], dtype=float)
    max_exp = np.max(expected)
    regrets = max_exp - expected
    weights = np.exp(-regrets)
    probs = weights / np.sum(weights)
    return probs


# ----------------------------------------------------------------------
# Deterministic ternary payload encoder – Parent B (partial)
# ----------------------------------------------------------------------
def ternary_vector(payload: List[str], dim: int) -> np.ndarray:
    """
    Deterministic mapping from a list of strings to a ternary vector of length `dim`.
    For each position i the SHA‑256 hash of the i‑th payload element (or of the
    string "pad" if the payload is shorter) is taken, reduced modulo 3 and mapped
    to {−1,0,+1}.
    """
    vec = np.empty(dim, dtype=int)
    for i in range(dim):
        token = payload[i] if i < len(payload) else "pad"
        h = hashlib.sha256(token.encode("utf-8")).digest()
        mod = h[i % len(h)] % 3  # simple deterministic reduction
        vec[i] = {0: -1, 1: 0, 2: 1}[mod]
    return vec


# ----------------------------------------------------------------------
# Hybrid entropy & VRAM allocation
# ----------------------------------------------------------------------
def hybrid_entropy(
    regret_probs: np.ndarray,
    ternary_vec: np.ndarray,
) -> float:
    """
    Fuse the regret‑weighted probability vector with the ternary payload.
    1. Sign‑quantise the probability vector:
         s_i = sign(p_i − 1/|A|)  ∈ {−1,0,+1}
    2. Concatenate the symbolic sequence `s` with the payload ternary vector `t`.
    3. Compute the empirical Shannon entropy of the combined multiset over
       the three symbols.
    """
    if regret_probs.ndim != 1:
        raise ValueError("regret_probs must be 1‑D")
    n_actions = regret_probs.shape[0]
    uniform = 1.0 / n_actions
    sign_quant = np.sign(regret_probs - uniform).astype(int)  # -1,0,1

    combined = np.concatenate([sign_quant, ternary_vec])
    # Empirical frequencies
    counts = np.array([np.sum(combined == s) for s in (-1, 0, 1)], dtype=float)
    probs = counts / counts.sum()
    # Guard against log2(0)
    nonzero = probs > 0
    entropy = -np.sum(probs[nonzero] * np.log2(probs[nonzero]))
    return float(entropy)


def allocate_vram(action_id: str, entropy: float) -> float:
    """
    Scale the virtual VRAM budget for `action_id` according to the entropy.
    The maximal entropy for a ternary alphabet is log2(3); we linearly map
    [0, log2(3)] → [0, DEFAULT_BUDGET_MB].
    """
    max_entropy = math.log2(3)  # ≈1.585
    factor = min(max(entropy / max_entropy, 0.0), 1.0)
    allocation = DEFAULT_BUDGET_MB * factor
    _STORE[action_id] = _STORE.get(action_id, 0.0) + allocation
    return allocation


# ----------------------------------------------------------------------
# Public hybrid operation
# ----------------------------------------------------------------------
def hybrid_step(
    context: Dict[str, float],
    action_ids: List[str],
    payload: List[str],
    algorithm: str = "linucb",
    seed: int | str | None = 7,
) -> Tuple[BanditAction, float, float]:
    """
    Perform a full hybrid iteration:
      1. Bandit selects an action.
      2. Regret‑weighted probabilities are built for *all* actions
         (using dummy expected values derived from current bandit estimates).
      3. A ternary vector is generated from `payload`.
      4. Entropy is computed and used to allocate VRAM to the chosen action.
    Returns the BanditAction descriptor, the entropy value and the VRAM
    allocation (in MB).
    """
    # 1. Bandit selection
    bandit_act = select_action(context, action_ids, algorithm=algorithm, seed=seed)

    # 2. Build MathAction list from current bandit reward estimates
    math_actions = [
        MathAction(id=a, expected_value=_reward(a))
        for a in action_ids
    ]
    probs = regret_weighted_probabilities(math_actions)

    # 3. Ternary payload encoding
    ternary_vec = ternary_vector(payload, dim=8)  # fixed small dimension

    # 4. Entropy & VRAM allocation
    ent = hybrid_entropy(probs, ternary_vec)
    vram = allocate_vram(bandit_act.action_id, ent)

    return bandit_act, ent, vram


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple reproducible scenario
    random.seed(42)
    reset_policy()

    ctx = {"feature1": 0.7, "feature2": -0.2}
    actions = ["A", "B", "C", "D"]
    payload = ["alpha", "beta", "gamma"]

    # Run a few hybrid steps
    for step in range(3):
        ba, entropy, vram_mb = hybrid_step(ctx, actions, payload, algorithm="linucb", seed=step)
        # Simulated reward (random for demo)
        reward = random.random()
        upd = BanditUpdate(
            context_id=f"ctx{step}",
            action_id=ba.action_id,
            reward=reward,
            propensity=ba.propensity,
        )
        update_policy(upd)
        print(
            f"Step {step}: action={ba.action_id}, reward={reward:.3f}, "
            f"entropy={entropy:.3f}, vram_alloc={vram_mb:.1f} MB"
        )

    # Final policy snapshot
    print("\nFinal policy statistics:")
    for aid, stats in _POLICY.items():
        print(f"  {aid}: total_reward={stats[0]:.3f}, count={int(stats[1])}")

    print("\nVirtual VRAM store:")
    for aid, used in _STORE.items():
        print(f"  {aid}: {used:.1f} MB")