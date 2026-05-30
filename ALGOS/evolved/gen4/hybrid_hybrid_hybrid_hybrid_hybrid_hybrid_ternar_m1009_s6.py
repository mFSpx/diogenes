# DARWIN HAMMER — match 1009, survivor 6
# gen: 4
# parent_a: hybrid_hybrid_hybrid_worksh_hybrid_sheaf_cohomol_m179_s1.py (gen3)
# parent_b: hybrid_hybrid_ternary_route_hybrid_bandit_router_m56_s1.py (gen2)
# born: 2026-05-29T23:32:17Z

"""Hybrid Sheaf‑Bandit Router
================================

This module fuses the *weekday‑rotated weight vector* from
``hybrid_hybrid_hybrid_worksh_hybrid_sheaf_cohomol_m179_s1.py`` with the
*SSIM‑driven multi‑armed bandit* from
``hybrid_hybrid_ternary_route_hybrid_bandit_router_m56_s1.py``.

Mathematical bridge
-------------------
Both parents rely on a linear weighting step:

* **Parent A** builds a row‑stochastic vector `w(dow)` by a sinusoidal
  rotation that depends on the day‑of‑week.  It can be seen as a linear
  map `ℝⁿ → Δⁿ` (the probability simplex).

* **Parent B** maintains a bandit policy `π(a)` (propensity) for each
  action `a` and updates an expected reward `r̂(a)` using a similarity
  score (SSIM) between packet payloads.

The fusion treats the weekday weight vector as a *pre‑conditioner* for the
bandit propensities: the effective propensity of an action `a` on a given
day is `π_eff(a) = π(a)·w_a(dow)`.  The SSIM value between the current packet
and a reference payload is used as the instantaneous reward that updates the
bandit statistics, mirroring the restriction maps of a cellular sheaf.

The resulting system therefore performs:


r̂_t(a) ← r̂_{t‑1}(a) + α·SSIM(payload_t, ref_a)·w_a(dow)
π_eff(a) ← (π(a) + r̂_t(a))·w_a(dow)


where `α` is a learning rate.  The three public functions below expose this
hybrid behaviour.
"""

import math
import random
import sys
import pathlib
import datetime as dt
from typing import List, Tuple, Sequence, Dict, Any

import numpy as np

# ----------------------------------------------------------------------
# Parent A – weekday weight vector (sinusoidal rotation)
# ----------------------------------------------------------------------


def weekday_weight_vector(groups: Sequence[str], dow: int) -> np.ndarray:
    """
    Return a row‑stochastic weight vector for *groups* on the given weekday.

    Parameters
    ----------
    groups: sequence of group identifiers
    dow: integer weekday where 0 = Sunday … 6 = Saturday

    Returns
    -------
    np.ndarray of shape (len(groups),) with dtype float64
    """
    n = len(groups)
    if n == 0:
        raise ValueError("groups must contain at least one element")
    base_angles = np.arange(n) * (2.0 * math.pi) / n
    phase = (2.0 * math.pi) * (dow % 7) / 7.0
    amplitude = 0.2
    raw = 1.0 + amplitude * np.sin(base_angles + phase)
    weight_vec = raw / raw.sum()
    return weight_vec.astype(np.float64)


# ----------------------------------------------------------------------
# Parent B – SSIM similarity metric
# ----------------------------------------------------------------------


def compute_ssim(
    x: List[float],
    y: List[float],
    dynamic_range: float = 1.0,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """
    Compute the Structural Similarity (SSIM) index between two 1‑D signals.

    Parameters
    ----------
    x, y : list‑like of equal length
    dynamic_range : the range of the data (default 1.0)
    k1, k2 : stability constants

    Returns
    -------
    SSIM value in [0, 1]
    """
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
# Hybrid policy storage (mirrors Parent B's _POLICY)
# ----------------------------------------------------------------------


_POLICY: Dict[str, List[float]] = {}  # action_id -> [cumulative_reward, count]


def reset_policy() -> None:
    """Clear all learned statistics."""
    _POLICY.clear()


def _reward(action: str) -> float:
    """Mean reward observed for *action* (0 if never taken)."""
    total, n = _POLICY.get(action, [0.0, 0.0])
    return total / n if n else 0.0


def _count(action: str) -> float:
    """Number of times *action* has been selected."""
    return _POLICY.get(action, [0.0, 0.0])[1]


def update_policy(updates: List[Dict[str, float]]) -> None:
    """
    Apply a batch of reward updates.

    Each dict must contain:
        - ``action_id`` (str)
        - ``reward``   (float)
    """
    for u in updates:
        stats = _POLICY.setdefault(u["action_id"], [0.0, 0.0])
        stats[0] += float(u["reward"])
        stats[1] += 1.0


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------


def weighted_ssim_score(
    packet: Dict[str, Any],
    reference_payload: List[float],
    groups: Sequence[str],
    dow: int,
) -> float:
    """
    Compute a SSIM similarity between the packet payload and a reference,
    then weight the result by the weekday‑dependent group weight.

    The packet is assumed to contain a ``payload`` key with a list/tuple of
    floats.  The first group in *groups* is taken as the action identifier for
    this score (the mapping can be customised by the caller).

    Returns
    -------
    float – the weighted similarity in [0, 1]
    """
    payload = packet.get("payload")
    if not isinstance(payload, (list, tuple)):
        raise ValueError("packet must contain a numeric 'payload' sequence")
    ssim = compute_ssim(list(payload), reference_payload)
    weights = weekday_weight_vector(groups, dow)
    # Use the first group's weight as a simple proxy; more elaborate schemes
    # could map each group to a distinct action.
    return ssim * float(weights[0])


def update_hybrid_policy(
    action_id: str,
    packet: Dict[str, Any],
    reference_payload: List[float],
    groups: Sequence[str],
    dow: int,
    learning_rate: float = 0.5,
) -> None:
    """
    Incorporate a new packet observation into the bandit policy.

    The instantaneous reward is the weighted SSIM score; it is blended with
    the existing mean reward using an exponential moving average controlled by
    ``learning_rate`` (0 < α ≤ 1).

    Parameters
    ----------
    action_id : identifier of the bandit arm (must match a key in _POLICY)
    packet    : incoming packet dict with a ``payload`` field
    reference_payload : baseline payload used for SSIM comparison
    groups    : list of group names (used for weekday weighting)
    dow       : day‑of‑week integer (0 = Sunday … 6 = Saturday)
    learning_rate : EMA coefficient
    """
    raw_reward = weighted_ssim_score(packet, reference_payload, groups, dow)
    # Blend with previous mean reward
    prev_mean = _reward(action_id)
    blended = learning_rate * raw_reward + (1 - learning_rate) * prev_mean
    update_policy([{"action_id": action_id, "reward": blended}])


def select_hybrid_action(
    action_ids: Sequence[str],
    groups: Sequence[str],
    dow: int,
    exploration_prob: float = 0.1,
) -> str:
    """
    Choose an action using the weekday‑scaled propensities.

    The effective propensity for each action is:
        π_eff(a) = (mean_reward(a) + ε) * w_a(dow)

    where ε is a small constant to avoid zero probabilities.
    With probability ``exploration_prob`` a random action is returned.

    Returns
    -------
    str – the selected ``action_id``
    """
    if not action_ids:
        raise ValueError("at least one action must be provided")
    if random.random() < exploration_prob:
        return random.choice(list(action_ids))

    weights = weekday_weight_vector(groups, dow)
    # Align weights with actions (assume same ordering)
    if len(weights) != len(action_ids):
        raise ValueError("number of groups must match number of actions")
    epsilon = 1e-6
    scores = np.array(
        [_reward(a) + epsilon for a in action_ids], dtype=np.float64
    ) * weights
    # Normalise to obtain a probability distribution
    probs = scores / scores.sum()
    chosen = np.random.choice(action_ids, p=probs)
    return str(chosen)


# ----------------------------------------------------------------------
# Demonstration / smoke test
# ----------------------------------------------------------------------


def _demo() -> None:
    """Run a tiny end‑to‑end demonstration of the hybrid system."""
    reset_policy()
    groups = ("codex", "groq", "cohere", "local_models")
    actions = groups  # one action per group for simplicity
    reference = [0.2, 0.5, 0.3, 0.7, 0.1]  # fixed reference payload

    # Simulate a stream of 20 packets with random payloads
    for i in range(20):
        dow = dt.datetime.utcnow().weekday()  # 0 = Monday … 6 = Sunday
        # Convert to 0=Sunday…6=Saturday as used by our weight function
        dow = (dow + 1) % 7
        payload = [random.random() for _ in range(len(reference))]
        packet = {"payload": payload}
        # Randomly pick an action to update (in a real system this would be
        # the action taken for the packet)
        act = random.choice(actions)
        update_hybrid_policy(
            action_id=act,
            packet=packet,
            reference_payload=reference,
            groups=groups,
            dow=dow,
            learning_rate=0.4,
        )
        # Occasionally select a new action based on the learned policy
        if i % 5 == 0:
            chosen = select_hybrid_action(actions, groups, dow, exploration_prob=0.05)
            print(f"Step {i:02d} – chosen action: {chosen}")

    # Final policy dump
    print("\nFinal policy statistics:")
    for a in actions:
        print(
            f"  {a}: mean_reward={_reward(a):.4f}, count={int(_count(a))}"
        )


if __name__ == "__main__":
    _demo()