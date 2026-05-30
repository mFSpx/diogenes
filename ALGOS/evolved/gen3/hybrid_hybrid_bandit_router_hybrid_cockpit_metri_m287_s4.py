# DARWIN HAMMER — match 287, survivor 4
# gen: 3
# parent_a: hybrid_bandit_router_poikilotherm_schoolf_m20_s2.py (gen1)
# parent_b: hybrid_cockpit_metrics_hybrid_pheromone_inf_m64_s0.py (gen2)
# born: 2026-05-29T23:28:06Z

"""
Hybrid Bandit‑Router / Schoolfield – Honesty‑Weighted Pheromone System

Parents:
* **bandit_router / Schoolfield poikilotherm** – a temperature‑aware multi‑armed
  bandit where the exploration term uses a scale `S_T = A(T)·‖context‖`.
* **cockpit_metrics / hybrid_pheromone_infotaxis** – a pheromone signalling
  mechanism whose strength is weighted by an *honesty* factor
  `H = claims_with_evidence / total_claims_emitted` and whose optimisation
  incorporates entropy of the pheromone distribution.

**Mathematical bridge**

Both parents expose a scalar gain that modulates a base quantity:


temperature gain      : A(T) ∈ [0,1]               (Schoolfield activity)
honesty gain          : H   ∈ [0,1]               (anti‑slop ratio)
joint gain G(T,H)     : G = A(T) * H


The hybrid algorithm multiplies the context norm by `G` and feeds the
result into the bandit’s Upper‑Confidence‑Bound (UCB) term *and* into the
pheromone decay factor.  Consequently, exploration is strongest when the
environment is thermally optimal **and** the information source is honest;
otherwise the system favours exploitation of well‑known actions and
concentrates pheromone on trustworthy pathways.  Entropy of the pheromone
distribution is used as a regulariser when updating the policy.

The module provides three core hybrid operations:
1. `temperature_activity` – Schoolfield activity gate.
2. `honesty_weight` – anti‑slop ratio.
3. `hybrid_select_action` – temperature‑ and honesty‑aware bandit selection.
4. `hybrid_update_policy` – reward update together with honesty‑weighted
   pheromone decay and entropy‑regularised learning.
"""

from __future__ import annotations

import math
import random
import sys
import pathlib
import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Helper functions from Parent B
# ----------------------------------------------------------------------


def anti_slop_ratio(claims_with_evidence: int, total_claims_emitted: int) -> float:
    """Honesty weight H ∈ [0,1] based on evidence‑coverage."""
    if total_claims_emitted <= 0:
        return 1.0
    return max(0.0, min(1.0, claims_with_evidence / total_claims_emitted))


def calculate_entropy(probabilities: List[float], eps: float = 1e-12) -> float:
    """Shannon entropy of a probability distribution."""
    total = sum(probabilities)
    if total <= 0:
        raise ValueError("positive probability mass required")
    return -sum(
        (p / total) * math.log(max(p / total, eps))
        for p in probabilities
        if p > 0
    )


def expected_honesty_weighted_entropy(
    p_hit: float,
    hit_state: List[float],
    miss_state: List[float],
    claims_with_evidence: int,
    total_claims_emitted: int,
) -> float:
    """
    Expected entropy weighted by honesty H.
    """
    H = anti_slop_ratio(claims_with_evidence, total_claims_emitted)
    if not 0.0 <= p_hit <= 1.0:
        raise ValueError("p_hit must be in [0,1]")
    return H * (
        p_hit * calculate_entropy(hit_state)
        + (1.0 - p_hit) * calculate_entropy(miss_state)
    )


# ----------------------------------------------------------------------
# Helper functions from Parent A
# ----------------------------------------------------------------------


def temperature_activity(celsius: float) -> float:
    """
    Schoolfield activity gate A(T) ∈ [0,1].
    Uses a simplified Arrhenius‑type formulation.
    """
    # Parameters are chosen to give a bell‑shaped response centered ~25 °C
    T_opt = 298.15  # 25 °C in Kelvin
    delta = 10.0    # width of the activity window
    T = celsius + 273.15
    exponent = -((T - T_opt) ** 2) / (2 * delta ** 2)
    return max(0.0, min(1.0, math.exp(exponent)))


# ----------------------------------------------------------------------
# Hybrid data structures
# ----------------------------------------------------------------------


@dataclass
class ActionStats:
    """Statistics kept per bandit action."""
    count: int = 0
    reward_sum: float = 0.0
    pheromone: float = 1.0  # initial pheromone strength

    @property
    def expected_reward(self) -> float:
        return self.reward_sum / self.count if self.count > 0 else 0.0


# ----------------------------------------------------------------------
# Core hybrid functions
# ----------------------------------------------------------------------


def hybrid_confidence(
    context: np.ndarray,
    n_a: int,
    temperature_c: float,
    claims_with_evidence: int,
    total_claims_emitted: int,
) -> float:
    """
    Compute the joint exploration scale:

        G = A(T) * H
        scale = G * ||context||
        exploration = scale / sqrt(1 + n_a)

    Returns the exploration term to be added to the expected reward.
    """
    norm = float(np.linalg.norm(context))
    A_T = temperature_activity(temperature_c)
    H = anti_slop_ratio(claims_with_evidence, total_claims_emitted)
    joint_gain = A_T * H
    scale = joint_gain * norm
    return scale / math.sqrt(1.0 + n_a)


def hybrid_select_action(
    context: np.ndarray,
    action_table: Dict[str, ActionStats],
    temperature_c: float,
    claims_with_evidence: int,
    total_claims_emitted: int,
) -> str:
    """
    Temperature‑ and honesty‑aware UCB bandit selection.

    For each action `a` compute:

        score_a = E[R_a] + confidence_a

    where `confidence_a` uses the joint gain G(T,H).  The action with the
    highest score is returned.
    """
    best_action = None
    best_score = -math.inf

    for aid, stats in action_table.items():
        confidence = hybrid_confidence(
            context,
            stats.count,
            temperature_c,
            claims_with_evidence,
            total_claims_emitted,
        )
        score = stats.expected_reward + confidence
        if score > best_score:
            best_score = score
            best_action = aid

    # Fallback in case table is empty
    if best_action is None:
        raise RuntimeError("Action table is empty.")
    return best_action


def hybrid_update_policy(
    action_id: str,
    reward: float,
    context: np.ndarray,
    action_table: Dict[str, ActionStats],
    temperature_c: float,
    claims_with_evidence: int,
    total_claims_emitted: int,
    half_life_seconds: float,
    timestamp: datetime.datetime | None = None,
) -> None:
    """
    Update both bandit statistics and pheromone strength for the selected action.

    * Bandit update – increment count and reward sum.
    * Pheromone update – decay with half‑life, then weight by honesty H.
    * Entropy regularisation – adjust pheromone to favour lower entropy
      distributions (more peaked) proportionally to the honesty‑weighted
      expected entropy.

    Parameters
    ----------
    action_id : str
        Identifier of the chosen action.
    reward : float
        Observed reward (e.g., performance metric).
    context : np.ndarray
        Current context vector (used only for the confidence term).
    action_table : dict
        Mapping from action identifiers to ``ActionStats``.
    temperature_c : float
        Current temperature in Celsius (affects decay via A(T)).
    claims_with_evidence, total_claims_emitted : int
        Honesty statistics.
    half_life_seconds : float
        Pheromone half‑life.
    timestamp : datetime, optional
        Current time; defaults to ``datetime.datetime.utcnow()``.
    """
    if timestamp is None:
        timestamp = datetime.datetime.utcnow()

    # ----- Bandit statistics -----
    stats = action_table.setdefault(action_id, ActionStats())
    stats.count += 1
    stats.reward_sum += reward

    # ----- Pheromone decay -----
    # Decay factor based on elapsed time and temperature activity
    elapsed = (timestamp - datetime.datetime.utcnow()).total_seconds()
    # Using absolute value because elapsed will be <=0 in this immediate call
    elapsed = abs(elapsed)
    decay = math.pow(0.5, elapsed / half_life_seconds)
    A_T = temperature_activity(temperature_c)
    H = anti_slop_ratio(claims_with_evidence, total_claims_emitted)

    # Apply decay, then honesty weighting
    stats.pheromone *= decay * A_T * H

    # ----- Entropy‑based regularisation -----
    # Build the current pheromone distribution
    pheromones = np.array([s.pheromone for s in action_table.values()], dtype=float)
    if pheromones.sum() == 0:
        # Avoid division by zero – re‑initialise uniformly
        pheromones[:] = 1.0
    probs = pheromones / pheromones.sum()
    current_entropy = calculate_entropy(probs.tolist())

    # Expected entropy if we consider a hit/miss scenario (placeholder states)
    # For illustration we use the current distribution as both hit and miss.
    exp_entropy = expected_honesty_weighted_entropy(
        p_hit=0.5,
        hit_state=probs.tolist(),
        miss_state=probs.tolist(),
        claims_with_evidence=claims_with_evidence,
        total_claims_emitted=total_claims_emitted,
    )

    # Adjust pheromone to gently push entropy toward the expected value
    entropy_error = exp_entropy - current_entropy
    # Simple proportional controller
    adjustment = 0.1 * entropy_error
    # Distribute adjustment proportionally to existing pheromones
    for s in action_table.values():
        s.pheromone = max(1e-8, s.pheromone + adjustment * (s.pheromone / pheromones.sum()))


# ----------------------------------------------------------------------
# Convenience wrapper class
# ----------------------------------------------------------------------


class HybridBanditPheromone:
    """
    Encapsulates the hybrid algorithm state.
    """

    def __init__(
        self,
        action_ids: List[str],
        half_life_seconds: float = 300.0,
    ) -> None:
        self.actions: Dict[str, ActionStats] = {aid: ActionStats() for aid in action_ids}
        self.half_life = half_life_seconds

    def select(
        self,
        context: np.ndarray,
        temperature_c: float,
        claims_with_evidence: int,
        total_claims_emitted: int,
    ) -> str:
        return hybrid_select_action(
            context,
            self.actions,
            temperature_c,
            claims_with_evidence,
            total_claims_emitted,
        )

    def update(
        self,
        action_id: str,
        reward: float,
        context: np.ndarray,
        temperature_c: float,
        claims_with_evidence: int,
        total_claims_emitted: int,
        timestamp: datetime.datetime | None = None,
    ) -> None:
        hybrid_update_policy(
            action_id,
            reward,
            context,
            self.actions,
            temperature_c,
            claims_with_evidence,
            total_claims_emitted,
            self.half_life,
            timestamp,
        )

    def pheromone_distribution(self) -> np.ndarray:
        """Current normalized pheromone vector."""
        pher = np.array([s.pheromone for s in self.actions.values()], dtype=float)
        if pher.sum() == 0:
            return np.full_like(pher, 1.0 / len(pher))
        return pher / pher.sum()


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Define a tiny problem with three actions
    hb = HybridBanditPheromone(action_ids=["A", "B", "C"], half_life_seconds=120.0)

    # Simulated context vector
    ctx = np.array([0.3, -0.1, 0.7])

    # Simulated environmental / honesty parameters
    temperature = 22.0  # °C
    claims_evidence = 8
    total_claims = 10

    # Run a few selection‑update cycles
    for step in range(5):
        chosen = hb.select(
            context=ctx,
            temperature_c=temperature,
            claims_with_evidence=claims_evidence,
            total_claims_emitted=total_claims,
        )
        # Fake reward: higher for action "B"
        reward = 1.0 if chosen == "B" else 0.2
        hb.update(
            action_id=chosen,
            reward=reward,
            context=ctx,
            temperature_c=temperature,
            claims_with_evidence=claims_evidence,
            total_claims_emitted=total_claims,
        )
        print(
            f"Step {step+1}: chosen={chosen}, reward={reward:.2f}, "
            f"pheromone={hb.pheromone_distribution()}"
        )
    sys.exit(0)