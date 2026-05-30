# DARWIN HAMMER — match 3854, survivor 3
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_nlms_h_m2129_s1.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1481_s2.py (gen6)
# born: 2026-05-29T23:52:04Z

"""Hybrid Fusion of Bandit‑Router NLMS (Parent A) and Certainty‑Weighted Geometric‑RBF (Parent B).

Mathematical Bridge
-------------------
*Parent A* supplies a bandit policy 𝑟̂ₐ (expected reward) and a NLMS step‑size that
depends on the confidence of the chosen action.  
*Parent B* supplies a certainty flag (confidence in basis points) and a geometric
product that rotates the raw context vector 𝑥 into a feature space where an
RBF kernel is evaluated.

The fusion is performed by

1. Rotating the context with a GA‑rotor 𝑅 (represented as an orthogonal matrix) →  
   𝑥̃ = 𝑅 𝑥.
2. Computing an RBF similarity vector 𝑠ₐ = exp(−‖𝑥̃‑pₐ‖² / (2σ²)) for each stored
   prototype pₐ.
3. Weighting the bandit expected reward with the certainty flag confidence
   𝑐 = confidence_bps / 10000 → 𝑟̃ₐ = 𝑟̂ₐ · 𝑐.
4. Forming a precision‑modulated score  
   scoreₐ = 𝑟̃ₐ · 𝑠ₐ .
5. Selecting the action with maximal score and performing an NLMS weight update
   with a step‑size that also incorporates the certainty‑weighted reward:

   μₐ = ε · (1 + 𝑟̃ₐ) / (‖𝑥̃‖² + δ)

Thus the epistemic certainty from Parent B modulates both the bandit decision
and the NLMS adaptation, while the geometric‑RBF machinery from Parent B provides
the similarity measure that drives the bandit’s routing.

The module implements this unified system with three core functions:
`select_action`, `nlms_update`, and `update_policy`."""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Data structures (combined from both parents)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class BanditAction:
    """Action description used by the bandit router."""
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "HybridBandit"


@dataclass(frozen=True)
class CertaintyFlag:
    """Epistemic certainty attached to an action."""
    label: str
    confidence_bps: int  # 0 .. 10000  (basis points)
    authority_class: str
    rationale: str
    evidence_refs: Tuple[str, ...] = ()
    generated_at: str = field(default_factory=lambda: "")

    def __post_init__(self) -> None:
        allowed = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")
        if self.label not in allowed:
            raise ValueError(f"unknown epistemic flag: {self.label!r}")
        if not (0 <= int(self.confidence_bps) <= 10000):
            raise ValueError("confidence_bps must be 0..10000")
        if not self.generated_at:
            object.__setattr__(self, "generated_at", _now_z())


def _now_z() -> str:
    """Current UTC time in ISO‑8601 Z‑suffix format."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# ----------------------------------------------------------------------
# Global mutable stores (simplified for demonstration)
# ----------------------------------------------------------------------
_POLICY: Dict[str, Tuple[float, int]] = {}          # action_id → (cumulative_reward, count)
_WEIGHTS: Dict[str, np.ndarray] = {}               # action_id → weight vector for NLMS
_PROTOTYPES: Dict[str, np.ndarray] = {}            # action_id → prototype vector (RBF centre)


# ----------------------------------------------------------------------
# Core mathematical primitives
# ----------------------------------------------------------------------


def _geometric_rotor(dim: int) -> np.ndarray:
    """Generate a random orthogonal matrix (GA‑rotor) of size dim×dim."""
    # Use QR decomposition of a random Gaussian matrix to obtain an orthogonal matrix.
    q, _ = np.linalg.qr(np.random.randn(dim, dim))
    return q


def _rotate_context(x: np.ndarray, rotor: np.ndarray) -> np.ndarray:
    """Apply the geometric product (here a rotation) to the context vector."""
    return rotor @ x


def _rbf_similarity(x: np.ndarray, prototypes: Dict[str, np.ndarray], sigma: float) -> Dict[str, float]:
    """
    Compute RBF similarity between rotated context x and each prototype.

    sₐ = exp( -||x - pₐ||² / (2σ²) )
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    denom = 2 * sigma * sigma
    sims = {}
    for aid, p in prototypes.items():
        dist2 = np.linalg.norm(x - p) ** 2
        sims[aid] = math.exp(-dist2 / denom)
    return sims


def _certainty_weighted_reward(action: BanditAction, flag: CertaintyFlag) -> float:
    """
    Modulate the bandit expected reward by epistemic confidence.

        r̃ₐ = r̂ₐ * (confidence_bps / 10000)
    """
    confidence = flag.confidence_bps / 10000.0
    return action.expected_reward * confidence


def select_action(
    context: np.ndarray,
    rotor: np.ndarray,
    sigma: float,
    bandit_actions: List[BanditAction],
    certainty_flags: Dict[str, CertaintyFlag],
    prototypes: Dict[str, np.ndarray],
) -> Tuple[BanditAction, CertaintyFlag]:
    """
    Fuse the two parent topologies to pick an action.

    Steps:
    1. Rotate the context (Parent B geometric product).
    2. Compute RBF similarity to each prototype (Parent B RBF).
    3. Weight bandit expected reward by certainty flag (Parent B certainty).
    4. Form scoreₐ = r̃ₐ * sₐ (Parent A precision‑modulated score).
    5. Return the action with maximal score.
    """
    x_rot = _rotate_context(context, rotor)
    sim = _rbf_similarity(x_rot, prototypes, sigma)

    best_score = -math.inf
    best_action = None
    best_flag = None

    for act in bandit_actions:
        flag = certainty_flags.get(act.action_id)
        if flag is None:
            continue  # skip actions without a certainty flag
        r_tilde = _certainty_weighted_reward(act, flag)
        s = sim.get(act.action_id, 0.0)
        score = r_tilde * s
        if score > best_score:
            best_score = score
            best_action = act
            best_flag = flag

    if best_action is None or best_flag is None:
        raise RuntimeError("No viable action found during selection.")
    return best_action, best_flag


def nlms_update(
    action_id: str,
    x: np.ndarray,
    desired: float,
    bandit_action: BanditAction,
    certainty_flag: CertaintyFlag,
    epsilon: float = 0.1,
    delta: float = 1e-6,
) -> None:
    """
    Perform an NLMS weight update for the chosen action.

    Step size μₐ = ε·(1 + r̃ₐ) / (||x̃||² + δ)

    The weight vector is stored in the global _WEIGHTS dict.
    """
    # Ensure weight vector exists
    w = _WEIGHTS.get(action_id)
    if w is None:
        w = np.zeros_like(x)
        _WEIGHTS[action_id] = w

    # Rotate context to stay consistent with selection step
    rotor = _geometric_rotor(len(x))
    x_rot = _rotate_context(x, rotor)

    # Confidence‑weighted reward
    r_tilde = _certainty_weighted_reward(bandit_action, certainty_flag)

    mu = epsilon * (1.0 + r_tilde) / (np.linalg.norm(x_rot) ** 2 + delta)

    # NLMS update: w ← w + μ * e * x / (||x||² + δ)
    y = np.dot(w, x_rot)
    error = desired - y
    w += mu * error * x_rot / (np.linalg.norm(x_rot) ** 2 + delta)

    # Store back (numpy arrays are mutable, but explicit assignment for clarity)
    _WEIGHTS[action_id] = w


def update_policy(action_id: str, reward: float, propensity: float) -> None:
    """
    Incrementally update the bandit policy statistics for an action.
    Maintains cumulative reward and selection count to recompute expected_reward.
    """
    cum_reward, count = _POLICY.get(action_id, (0.0, 0))
    cum_reward += reward
    count += 1
    _POLICY[action_id] = (cum_reward, count)


def expected_reward(action_id: str) -> float:
    """Return the current expected reward estimate for an action."""
    cum_reward, count = _POLICY.get(action_id, (0.0, 0))
    return cum_reward / count if count > 0 else 0.0


# ----------------------------------------------------------------------
# Demo / Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    # Dimensionality of context vectors
    D = 5

    # Create a random rotor (orthogonal matrix)
    rotor = _geometric_rotor(D)

    # Generate synthetic prototypes for three actions
    action_ids = ["A", "B", "C"]
    for aid in action_ids:
        _PROTOTYPES[aid] = np.random.randn(D)

    # Initialise bandit actions (propensity and confidence bound are placeholders)
    bandit_actions = []
    for aid in action_ids:
        exp_r = expected_reward(aid)  # initially zero
        bandit_actions.append(
            BanditAction(
                action_id=aid,
                propensity=1.0 / len(action_ids),
                expected_reward=exp_r,
                confidence_bound=0.0,
            )
        )

    # Attach certainty flags (varying confidence)
    certainty_flags = {
        "A": CertaintyFlag(label="FACT", confidence_bps=9000, authority_class="A", rationale="high confidence"),
        "B": CertaintyFlag(label="PROBABLE", confidence_bps=6000, authority_class="B", rationale="moderate confidence"),
        "C": CertaintyFlag(label="POSSIBLE", confidence_bps=3000, authority_class="C", rationale="low confidence"),
    }

    # Simulate a single interaction
    context = np.random.randn(D)

    # Select action using the hybrid mechanism
    chosen_action, chosen_flag = select_action(
        context=context,
        rotor=rotor,
        sigma=1.0,
        bandit_actions=bandit_actions,
        certainty_flags=certainty_flags,
        prototypes=_PROTOTYPES,
    )
    print(f"Chosen action: {chosen_action.action_id} with certainty {chosen_flag.label}")

    # Simulated environment response (desired output)
    desired_output = float(np.dot(_PROTOTYPES[chosen_action.action_id], context) + np.random.randn() * 0.1)

    # Perform NLMS update
    nlms_update(
        action_id=chosen_action.action_id,
        x=context,
        desired=desired_output,
        bandit_action=chosen_action,
        certainty_flag=chosen_flag,
        epsilon=0.05,
    )

    # Mock reward (e.g., negative absolute error)
    reward = -abs(desired_output - np.dot(_WEIGHTS[chosen_action.action_id], context))
    update_policy(chosen_action.action_id, reward, chosen_action.propensity)

    # Refresh expected rewards in bandit actions for next round
    for act in bandit_actions:
        act_exp = expected_reward(act.action_id)
        # dataclasses are frozen; create a new instance to reflect updated reward
        updated = BanditAction(
            action_id=act.action_id,
            propensity=act.propensity,
            expected_reward=act_exp,
            confidence_bound=act.confidence_bound,
            algorithm=act.algorithm,
        )
        # replace in list (simple approach)
        idx = bandit_actions.index(act)
        bandit_actions[idx] = updated

    print("Smoke test completed without errors.")