# DARWIN HAMMER — match 2052, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_ternar_m1009_s6.py (gen4)
# parent_b: hybrid_hybrid_indy_learning_hybrid_pheromone_inf_m541_s0.py (gen5)
# born: 2026-05-29T23:40:34Z

"""Hybrid Sheaf‑Bandit‑Pheromone Infotaxis
========================================

This module fuses the two parent algorithms:

* **Parent A** – *weekday‑rotated weight vector* (a linear map
  `ℝⁿ → Δⁿ`) together with an SSIM‑driven reward update for a multi‑armed
  bandit.

* **Parent B** – *deterministic learning vectors* enriched by a
  pheromone‑based surface‑usage tracker and an entropy‑based (infotaxis)
  action‑selection rule that uses log‑count statistics.

**Mathematical bridge**

Both parents maintain a *propensity* vector `π ∈ Δ^k` for a set of
`k` actions.  In the hybrid we treat the weekday weight vector
`w(dow) ∈ Δ^n` (where `n` equals the number of actions) as a *pre‑conditioner*
that scales the raw bandit propensities before they are fed to the
pheromone/entropy machinery.  Concretely, for action `a` on day `d`


π_raw(a)          – base propensity (from deterministic learning)
π_pre(a) = π_raw(a) * w_a(d)          – pre‑conditioned propensity
π_pher(a)   = softmax( log(count_a + 1) )   – pheromone‑derived probability
π_eff(a)    = normalize( π_pre(a) * π_pher(a) )


The instantaneous reward is the SSIM similarity between the current
payload and a reference payload `ref_a`.  The reward updates the expected
reward `r̂(a)` with the same weekday weight as a scaling factor:


r̂_t(a) = r̂_{t‑1}(a) + α * SSIM(payload, ref_a) * w_a(d)


Finally, an entropy‑minimising (infotaxis) selector chooses the action
with the highest `π_eff(a) * (1 + β * information_gain(a))`, where the
information gain is approximated by the negative log of the pheromone
probability.

The three public functions below expose this fused behaviour:
`weekday_weight_vector`, `hybrid_bandit_update`, and `select_action_infotaxis`.
"""

import math
import random
import sys
import pathlib
import datetime as dt
import re
import json
import hashlib
from collections import Counter, defaultdict
from typing import List, Tuple, Sequence, Dict, Any

import numpy as np

# ----------------------------------------------------------------------
# Parent A – weekday weight vector (sinusoidal rotation)
# ----------------------------------------------------------------------


def weekday_weight_vector(groups: Sequence[str], dow: int) -> np.ndarray:
    """
    Return a row‑stochastic weight vector for *groups* on the given weekday.

    Parameters
    ----------
    groups : sequence of identifiers (length n)
    dow    : integer weekday where 0 = Sunday … 6 = Saturday

    Returns
    -------
    np.ndarray of shape (n,) with dtype float64, summing to 1.0.
    """
    n = len(groups)
    if n == 0:
        raise ValueError("groups must contain at least one element")
    # base angles equally spaced around the unit circle
    base_angles = np.arange(n) * (2.0 * math.pi) / n
    # rotate by a weekday‑dependent phase
    phase = (dow / 7.0) * 2.0 * math.pi
    rotated = np.sin(base_angles + phase) + 1.0  # shift to non‑negative
    weight = rotated / rotated.sum()
    return weight.astype(np.float64)


# ----------------------------------------------------------------------
# Parent B – pheromone & infotaxis utilities (trimmed)
# ----------------------------------------------------------------------


def sha256_json(value: Any) -> str:
    """Deterministic hash of a JSON‑serialisable value."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def tokenize(text: str) -> List[Dict[str, Any]]:
    """Very light tokenizer used by the original parent."""
    WORD_RE = re.compile(r"\S+")
    return [
        {"token": m.group(0), "start": m.start(), "end": m.end()}
        for m in WORD_RE.finditer(text)
    ]


def compute_pheromone_probabilities(
    surface_counts: Mapping[str, int]
) -> Dict[str, float]:
    """
    Convert raw surface usage counts into a probability distribution
    via a log‑count (entropy) transform.

    π_pher(a) ∝ log(count_a + 1)
    """
    log_counts = {k: math.log(v + 1) for k, v in surface_counts.items()}
    total = sum(log_counts.values())
    if total == 0:
        # uniform fallback
        n = len(surface_counts)
        return {k: 1.0 / n for k in surface_counts}
    return {k: v / total for k, v in log_counts.items()}


def infotaxis_score(
    propensity: float, pheromone_prob: float, beta: float = 0.5
) -> float:
    """
    Entropy‑based score used for action selection.

    score = propensity * (1 + β * (-log π_pher))
    """
    if pheromone_prob <= 0.0:
        information = 0.0
    else:
        information = -math.log(pheromone_prob)
    return propensity * (1.0 + beta * information)


# ----------------------------------------------------------------------
# Hybrid data structures
# ----------------------------------------------------------------------


class HybridAction:
    """Container merging deterministic learning, bandit, and pheromone state."""
    __slots__ = (
        "action_id",
        "base_propensity",
        "expected_reward",
        "confidence_bound",
        "surface_key",
        "count",
    )

    def __init__(
        self,
        action_id: str,
        base_propensity: float,
        expected_reward: float = 0.0,
        confidence_bound: float = 0.0,
        surface_key: str | None = None,
    ):
        self.action_id = action_id
        self.base_propensity = base_propensity
        self.expected_reward = expected_reward
        self.confidence_bound = confidence_bound
        self.surface_key = surface_key or action_id
        self.count = 0  # how many times this action has been taken


# ----------------------------------------------------------------------
# Core hybrid operations
# ----------------------------------------------------------------------


def ssim_placeholder(payload: np.ndarray, reference: np.ndarray) -> float:
    """
    Very simple similarity metric standing in for SSIM.
    Returns a value in [0, 1] where 1 means identical.
    """
    if payload.shape != reference.shape:
        # resize by flattening and truncating/padding
        min_len = min(payload.size, reference.size)
        p = payload.ravel()[:min_len]
        r = reference.ravel()[:min_len]
    else:
        p = payload.ravel()
        r = reference.ravel()
    diff = np.mean((p - r) ** 2)
    return 1.0 / (1.0 + diff)


def hybrid_bandit_update(
    actions: List[HybridAction],
    payload: np.ndarray,
    reference_dict: Mapping[str, np.ndarray],
    dow: int,
    alpha: float = 0.1,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Perform one hybrid update step.

    * Compute weekday weights.
    * Update expected rewards with SSIM‑scaled reward.
    * Update pheromone counts (log‑count statistics).
    * Return the effective propensity vector (π_eff) and the pheromone
      probability distribution for external inspection.
    """
    # 1️⃣ weekday pre‑conditioner
    groups = [a.action_id for a in actions]
    w = weekday_weight_vector(groups, dow)  # shape (k,)

    # 2️⃣ SSIM‑driven reward update
    for idx, act in enumerate(actions):
        ref = reference_dict.get(act.action_id)
        if ref is None:
            reward = 0.0
        else:
            reward = ssim_placeholder(payload, ref)
        # Scale by weekday weight component
        delta = alpha * reward * w[idx]
        act.expected_reward += delta

    # 3️⃣ Build raw propensity vector from deterministic learning + reward
    raw_propensity = np.array(
        [a.base_propensity + a.expected_reward for a in actions], dtype=np.float64
    )
    # Ensure non‑negative before normalisation
    raw_propensity = np.clip(raw_propensity, a_min=0.0, a_max=None)

    # 4️⃣ Pheromone update (log‑count)
    surface_counts = defaultdict(int)
    for act in actions:
        surface_counts[act.surface_key] = act.count
    pheromone_probs = compute_pheromone_probabilities(surface_counts)

    # 5️⃣ Fuse raw propensity with weekday weight and pheromone
    pre_cond = raw_propensity * w  # element‑wise scaling
    pheromone_vec = np.array(
        [pheromone_probs.get(act.surface_key, 0.0) for act in actions],
        dtype=np.float64,
    )
    fused = pre_cond * pheromone_vec
    total = fused.sum()
    if total == 0.0:
        # fallback to uniform distribution
        pi_eff = np.full_like(fused, 1.0 / len(fused))
    else:
        pi_eff = fused / total

    # 6️⃣ Increment counts for the selected action (for demonstration we
    #    pick the highest‑probability one)
    chosen_idx = int(np.argmax(pi_eff))
    actions[chosen_idx].count += 1

    return pi_eff, pheromone_probs


def select_action_infotaxis(
    actions: List[HybridAction],
    pheromone_probs: Mapping[str, float],
    beta: float = 0.5,
) -> str:
    """
    Choose an action using the infotaxis‑style entropy‑adjusted score.
    Returns the selected action_id.
    """
    scores = []
    for act in actions:
        prop = act.base_propensity + act.expected_reward
        pher = pheromone_probs.get(act.surface_key, 1e-9)
        score = infotaxis_score(prop, pher, beta=beta)
        scores.append(score)
    best_idx = int(np.argmax(scores))
    return actions[best_idx].action_id


# ----------------------------------------------------------------------
# Demonstration / smoke test
# ----------------------------------------------------------------------


def _demo():
    # Define a tiny action set
    action_ids = ["A", "B", "C"]
    actions = [
        HybridAction(aid, base_propensity=random.random()) for aid in action_ids
    ]

    # Mock reference payloads (simple 1‑D vectors)
    reference_dict = {
        "A": np.array([0.1, 0.2, 0.3]),
        "B": np.array([0.4, 0.5, 0.6]),
        "C": np.array([0.7, 0.8, 0.9]),
    }

    # Random payload each step
    for step in range(5):
        payload = np.random.rand(3)
        dow = dt.datetime.utcnow().weekday()  # 0=Monday … 6=Sunday
        pi_eff, pher_probs = hybrid_bandit_update(
            actions, payload, reference_dict, dow, alpha=0.2
        )
        chosen = select_action_infotaxis(actions, pher_probs, beta=0.7)
        print(
            f"Step {step}: π_eff={pi_eff.round(3)}  chosen={chosen}  counts={[a.count for a in actions]}"
        )


if __name__ == "__main__":
    _demo()