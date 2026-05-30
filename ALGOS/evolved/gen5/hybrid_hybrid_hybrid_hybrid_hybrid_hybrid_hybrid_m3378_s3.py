# DARWIN HAMMER — match 3378, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_hybrid_bandit_hybrid_hybrid_hybrid_m742_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_bandit_hybrid_fisher_locali_m1077_s0.py (gen3)
# born: 2026-05-29T23:49:38Z

"""Hybrid Bandit‑Sketch‑Fisher algorithm.

Parents:
- hybrid_hybrid_hybrid_bandit_hybrid_hybrid_hybrid_m742_s0.py (Bandit‑Sketch‑Workshare)
- hybrid_hybrid_hybrid_bandit_hybrid_fisher_locali_m1077_s0.py (Bandit‑Fisher)

Mathematical bridge:
The propensity scores produced by the bandit router are interpreted as a
probability distribution over actions.  Those probabilities weight the
variational free‑energy (VFE) computation that measures similarity between an
action’s group‑one‑hot representation and the weekday‑dependent weight vector.
Simultaneously the Fisher information I(p)=1/(p(1‑p)) derived from the same
propensity scores defines a data‑dependent learning‑rate factor that scales the
policy’s reward accumulation.  Thus the bandit, sketch (VFE), and Fisher
components are coupled through the shared propensity variable.

The hybrid reward for an action a on day d is

    R_hybrid(a) = (1 – recon_risk(a, N)) *
                  VFE( one_hot(a), w_d ) *
                  f(I(prop_a), cb_a)

where
- recon_risk is a placeholder reconstruction‑risk score,
- w_d is the weekday weight vector,
- VFE is implemented as a negative KL‑divergence,
- I(prop_a) is the Fisher information of the propensity,
- cb_a is the confidence bound from the bandit and f(·) is a simple scaling
  (here multiplication).

The policy update multiplies the observed reward by the Fisher‑scaled learning
rate, ensuring that high‑confidence (low‑variance) propensities drive faster
adaptation.

The module provides three core functions:
    • weekday_weight_vector – weekday‑dependent group weights (Parent A)
    • hybrid_select_action   – bandit action selection with propensity output (Parent B)
    • hybrid_update_policy   – policy update using Fisher‑scaled learning (fusion)
"""

import math
import random
import sys
from pathlib import Path
from datetime import date
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Global structures (shared policy store)
# ----------------------------------------------------------------------
_POLICY: Dict[str, List[float]] = {}   # action_id -> [total_reward, count]
_STORE: Dict[str, float] = {}         # placeholder for any auxiliary storage


# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
GROUPS = ("codex", "groq", "cohere", "local_models")
ALPHA_UCB = 0.1          # exploration coefficient for LinUCB‑style score
BASE_LR = 0.05           # base learning rate before Fisher scaling


# ----------------------------------------------------------------------
# Helper utilities (Parent A)
# ----------------------------------------------------------------------
def _pct(value: float) -> float:
    """Round a float to six decimal places."""
    return round(float(value), 6)


def doomsday(year: int, month: int, day: int) -> int:
    """
    Return weekday index where 0 = Sunday … 6 = Saturday.
    """
    # Python's weekday(): Monday=0 … Sunday=6 → shift to Sunday=0
    return (date(year, month, day).weekday() + 1) % 7


def weekday_weight_vector(groups: Tuple[str, ...], dow: int) -> np.ndarray:
    """
    Produce a normalized weight vector for *groups* based on the weekday ``dow``.
    The construction follows a circular embedding where each group receives a
    sinusoidal weight that depends on the day of week.
    """
    n = len(groups)
    angles = np.arange(n) * (2.0 * math.pi) / n
    phase = (2.0 * math.pi) * dow / 7.0
    raw = np.cos(angles - phase) + 1.0          # shift to non‑negative
    return raw / raw.sum()                      # L1‑normalize


def reconstruction_risk_score(action_id: str, total_records: int) -> float:
    """
    Placeholder reconstruction‑risk estimator.
    Returns a value in [0,1] where larger values indicate higher risk.
    """
    # Simple heuristic: longer identifiers are assumed riskier
    length_factor = min(len(action_id) / 20.0, 1.0)
    size_factor = min(total_records / 1_000_000.0, 1.0)
    return _pct(0.5 * length_factor + 0.5 * size_factor)


def action_one_hot(action_id: str) -> np.ndarray:
    """
    Map an action identifier to a one‑hot vector over GROUPS.
    The mapping is deterministic via a hash modulo the number of groups.
    """
    idx = hash(action_id) % len(GROUPS)
    vec = np.zeros(len(GROUPS))
    vec[idx] = 1.0
    return vec


def variational_free_energy(action_vec: np.ndarray, weight_vec: np.ndarray) -> float:
    """
    Simple VFE proxy: negative KL‑divergence between the action distribution
    (treated as a categorical) and the weekday weight distribution.
    """
    # Avoid log(0) by clipping
    eps = 1e-12
    p = np.clip(action_vec, eps, 1.0)
    q = np.clip(weight_vec, eps, 1.0)
    kl = np.sum(p * np.log(p / q))
    return -_pct(kl)   # negative KL → higher similarity yields larger (less negative) value


# ----------------------------------------------------------------------
# Fisher‑related utilities (Parent B)
# ----------------------------------------------------------------------
def fisher_information(propensity: float) -> float:
    """
    Fisher information for a Bernoulli trial with success probability ``propensity``.
    I(p) = 1 / (p (1-p)).
    """
    eps = 1e-12
    p = max(min(propensity, 1.0 - eps), eps)
    return 1.0 / (p * (1.0 - p))


# ----------------------------------------------------------------------
# Data classes (Parent B)
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
# Core hybrid functions
# ----------------------------------------------------------------------
def hybrid_select_action(
    context: Dict[str, float],
    actions: List[str],
    algorithm: str = "linucb",
    epsilon: float = 0.1,
    seed: int | str | None = 7,
) -> BanditAction:
    """
    Select an action using a LinUCB / Thompson / epsilon‑greedy router.
    Returns a BanditAction that also carries the propensity (probability) of the
    chosen action, which will be used downstream as the bridge to the VFE and
    Fisher components.
    """
    if not actions:
        raise ValueError("actions required")
    rng = random.Random(seed)

    # Compute a score for each action
    scores = {}
    for a in actions:
        total, n = _POLICY.get(a, [0.0, 0.0])
        avg_reward = total / n if n > 0 else 0.0
        if algorithm == "epsilon_greedy":
            # pure exploitation score
            scores[a] = avg_reward
        elif algorithm == "thompson":
            # draw from Beta posterior
            alpha = 1.0 + max(0.0, total)
            beta = 1.0 + max(0.0, n - total)
            scores[a] = rng.betavariate(alpha, beta)
        else:  # LinUCB‑style
            scale = math.sqrt(
                sum(float(v) ** 2 for v in context.values())
            ) if context else 1.0
            confidence = ALPHA_UCB * scale / math.sqrt(1.0 + n)
            scores[a] = avg_reward + confidence

    # epsilon‑greedy exploration wrapper
    if algorithm == "epsilon_greedy" and rng.random() < epsilon:
        chosen = rng.choice(actions)
    else:
        # softmax over scores to obtain propensities
        max_score = max(scores.values())
        exp_vals = {a: math.exp(scores[a] - max_score) for a in actions}
        Z = sum(exp_vals.values())
        propensities = {a: exp_vals[a] / Z for a in actions}
        # pick the action with highest score (deterministic tie‑break)
        chosen = max(actions, key=lambda a: scores[a])

    # final propensity for the chosen action
    if algorithm == "epsilon_greedy":
        propensity = 1.0 / len(actions) if rng.random() < epsilon else 1.0
    else:
        propensity = propensities[chosen]

    total, n = _POLICY.get(chosen, [0.0, 0.0])
    exp_reward = total / n if n else 0.0
    confidence = 1.0 / math.sqrt(1.0 + n)

    return BanditAction(
        action_id=chosen,
        propensity=propensity,
        expected_reward=exp_reward,
        confidence_bound=confidence,
        algorithm=algorithm,
    )


def hybrid_reward(
    action_id: str,
    total_records: int,
    dow: int,
) -> float:
    """
    Compute the fused reward:
        (1 - recon_risk) * VFE(action_one_hot, weekday_weights)
    """
    recon = reconstruction_risk_score(action_id, total_records)
    w_vec = weekday_weight_vector(GROUPS, dow)
    a_vec = action_one_hot(action_id)
    vfe = variational_free_energy(a_vec, w_vec)
    return (1.0 - recon) * vfe


def hybrid_update_policy(updates: List[BanditUpdate]) -> None:
    """
    Update the global policy store using Fisher‑scaled learning rates.
    For each update the reward contribution is multiplied by
    lr = BASE_LR * sqrt(I(propensity)), where I is the Fisher information.
    """
    for u in updates:
        fisher = fisher_information(u.propensity)
        lr = BASE_LR * math.sqrt(fisher)

        # Retrieve current aggregates
        agg = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        total, count = agg

        # Apply Fisher‑scaled incremental update
        total += lr * u.reward
        count += lr
        _POLICY[u.action_id] = [total, count]


def hybrid_step(
    context: Dict[str, float],
    actions: List[str],
    total_records: int,
    today: Tuple[int, int, int],
    algorithm: str = "linucb",
    epsilon: float = 0.1,
    seed: int | str | None = 7,
) -> Tuple[BanditAction, float]:
    """
    Perform a single hybrid iteration:
        1. Select an action (propensity produced)
        2. Compute hybrid reward using VFE & reconstruction risk
        3. Update policy with Fisher‑scaled learning
    Returns the BanditAction and the computed reward.
    """
    year, month, day = today
    dow = doomsday(year, month, day)

    # 1. Action selection
    ba = hybrid_select_action(context, actions, algorithm, epsilon, seed)

    # 2. Reward computation
    rew = hybrid_reward(ba.action_id, total_records, dow)

    # 3. Policy update
    upd = BanditUpdate(
        context_id="ctx_" + "_".join(f"{k}{v}" for k, v in context.items()),
        action_id=ba.action_id,
        reward=rew,
        propensity=ba.propensity,
    )
    hybrid_update_policy([upd])

    return ba, rew


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # deterministic seed for reproducibility
    random.seed(42)

    # Dummy context: feature vector of a user/session
    dummy_context = {"feat_a": 0.3, "feat_b": 0.7, "feat_c": 0.1}

    # Action pool
    action_pool = ["search_codex", "generate_groq", "summarize_cohere", "run_local_models"]

    # Simulated total number of records in the privacy store
    total_records_sim = 850_000

    # Today's date
    today_tuple = (2026, 5, 29)

    # Run a few hybrid steps
    for i in range(5):
        ba, r = hybrid_step(
            context=dummy_context,
            actions=action_pool,
            total_records=total_records_sim,
            today=today_tuple,
            algorithm="linucb",
            epsilon=0.15,
            seed=i,
        )
        print(
            f"Step {i+1}: chosen={ba.action_id}, propensity={_pct(ba.propensity)}, "
            f"reward={_pct(r)}, total_reward={_pct(_POLICY[ba.action_id][0])}"
        )