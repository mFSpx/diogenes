# DARWIN HAMMER — match 2390, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hdc_se_m1184_s0.py (gen5)
# parent_b: hybrid_hybrid_cockpit_metri_hard_truth_math_m27_s4.py (gen2)
# born: 2026-05-29T23:42:08Z

"""Hybrid Bandit‑RBF‑HDC & Cockpit‑Trust Style Transport
================================================================

This module fuses the two parent algorithms:

* **Parent A** – a contextual multi‑armed bandit that uses an RBF surrogate
  and hyperdimensional computing (HDC) with sparse winner‑take‑all (WTA).
* **Parent B** – a cockpit‑honesty metric ``h∈[0,1]`` that scales a
  constant‑velocity flow between two high‑dimensional linguistic style
  vectors ``v₀`` and ``v₁``.

**Mathematical bridge**

The scalar trust factor ``h`` from Parent B is used as a *mixing weight* that
blends the bandit’s expected reward with the HDC similarity when computing a
priority value.  The same ``h`` also scales the velocity field that transports
the style vector (Parent B).  Consequently a single priority drives both the
decision‑making of the bandit/HDC pool **and** the step size of an Euler
integrator that moves a current style vector toward a trust‑weighted target.

Formally, for a given action ``a`` with expected reward ``r̂`` and a
hyperdimensional similarity ``s∈[−1,1]`` we define


π(a;h) = h·r̂ + (1−h)·s                                          (1)


For style vectors ``v₀, v₁∈ℝᴰ`` the trust‑scaled flow is


v_target = v₀ + h·(v₁ − v₀)                                      (2)


An Euler step with step size ``α = clip(π,0,1)`` updates the current style
``v_cur`` as


v_next = v_cur + α·(v_target − v_cur)                            (3)


The functions below implement this unified system, together with lightweight
HDC encoders needed by the bandit side.  All operations rely only on NumPy and
the Python standard library.  The ``__main__`` block provides a smoke test."""

import math
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Sequence, Callable

import numpy as np

# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
Vector = Sequence[float]
BipolarVector = np.ndarray  # values in {−1, +1}
Dim = int

# ----------------------------------------------------------------------
# Parent A – Bandit / HDC core (trimmed to essentials)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    """Immutable description of a single bandit arm."""
    action_id: str
    propensity: float
    expected_reward: float      # RBF surrogate prediction for [context, action_one_hot]
    confidence_bound: float
    algorithm: str = "HybridBanditRBFHDC"


def rbf_surrogate(x: np.ndarray, centers: np.ndarray, gamma: float = 1.0) -> float:
    """
    Very small RBF surrogate used as a stand‑in for the parent’s full model.
    Returns the weighted average of centre rewards.
    """
    diffs = centers[:, :-1] - x  # last column of centres holds reward
    dists = np.linalg.norm(diffs, axis=1)
    weights = np.exp(-gamma * dists ** 2)
    if weights.sum() == 0:
        return 0.0
    rewards = centers[:, -1]
    return float(np.dot(weights, rewards) / weights.sum())


def encode_to_hv(vec: np.ndarray, dim: Dim = 10000, seed: int = 0) -> BipolarVector:
    """
    Random‑projection encoder turning a real‑valued vector into a bipolar
    hypervector (values ∈{−1,+1}).
    """
    rng = np.random.default_rng(seed)
    proj = rng.standard_normal((vec.shape[0], dim))
    hv = np.sign(vec @ proj)
    hv[hv == 0] = 1
    return hv.astype(np.int8)


def hyperdim_similarity(hv1: BipolarVector, hv2: BipolarVector) -> float:
    """
    Normalised dot product similarity ∈[−1,1] between two bipolar hypervectors.
    """
    assert hv1.shape == hv2.shape, "Hypervectors must have the same dimensionality"
    return float(np.dot(hv1, hv2) / hv1.size)


def hybrid_priority(action: BanditAction, similarity: float, trust: float) -> float:
    """
    Fuse bandit expected reward and HDC similarity using the trust factor.
    Implements equation (1).
    """
    return trust * action.expected_reward + (1.0 - trust) * similarity


# ----------------------------------------------------------------------
# Parent B – Cockpit trust & style transport core
# ----------------------------------------------------------------------
def anti_slop_ratio(claims_with_evidence: int, total_claims_emitted: int) -> float:
    """Ratio of supported claims, clamped to [0, 1]."""
    if total_claims_emitted <= 0:
        return 1.0
    return max(0.0, min(1.0, claims_with_evidence / total_claims_emitted))


def cockpit_honesty(displayed_ok: int, unknown_displayed_as_ok: int) -> float:
    """Fraction of displayed items that are known‑good, clamped to [0, 1]."""
    total = displayed_ok + unknown_displayed_as_ok
    if total == 0:
        return 1.0
    return max(0.0, min(1.0, displayed_ok / total))


def compute_trust_factor(
    claims_with_evidence: int,
    total_claims_emitted: int,
    displayed_ok: int,
    unknown_displayed_as_ok: int,
) -> float:
    """
    Combine the two cockpit metrics into a single trust scalar h∈[0,1].
    Simple average is used – more sophisticated fusions are possible.
    """
    slop = anti_slop_ratio(claims_with_evidence, total_claims_emitted)
    honesty = cockpit_honesty(displayed_ok, unknown_displayed_as_ok)
    return (slop + honesty) / 2.0


def hybrid_style_target(v0: np.ndarray, v1: np.ndarray, trust: float) -> np.ndarray:
    """
    Compute the trust‑weighted style target.
    Implements equation (2).
    """
    assert v0.shape == v1.shape, "Style vectors must share dimensionality"
    return v0 + trust * (v1 - v0)


def hybrid_style_loss(prediction: np.ndarray, target: np.ndarray) -> float:
    """Mean‑squared error between a prediction and the trust‑scaled target."""
    return float(np.mean((prediction - target) ** 2))


def hybrid_euler_step(
    current: np.ndarray,
    target: np.ndarray,
    step_factor: float,
) -> np.ndarray:
    """
    One Euler integration step toward ``target`` with step size ``step_factor``.
    Implements equation (3). ``step_factor`` is clipped to [0,1] to keep the
    update stable.
    """
    alpha = max(0.0, min(1.0, step_factor))
    return current + alpha * (target - current)


# ----------------------------------------------------------------------
# Unified hybrid operations
# ----------------------------------------------------------------------
def hybrid_decision_cycle(
    context_vec: np.ndarray,
    action_space: List[BanditAction],
    hdc_centers: np.ndarray,
    style_v0: np.ndarray,
    style_v1: np.ndarray,
    trust: float,
) -> Tuple[BanditAction, np.ndarray]:
    """
    Perform a single decision cycle:

    1. Encode the context into a hypervector.
    2. Compute similarity of each action’s hypervector (here we reuse the
       action ID hashed into a vector) with the context HV.
    3. Fuse similarity and expected reward into a priority via ``hybrid_priority``.
    4. Select the action with maximal priority.
    5. Compute the trust‑scaled style target and take an Euler step whose size
       is the normalised priority of the chosen action.

    Returns the selected ``BanditAction`` and the updated style vector.
    """
    # 1. Context HV
    ctx_hv = encode_to_hv(context_vec, dim=hdc_centers.shape[1], seed=42)

    # 2. Similarities (use HDC centre hypervectors as prototypes)
    similarities = []
    for act in action_space:
        # simple deterministic pseudo‑vector for the action based on its ID
        rng = np.random.default_rng(hash(act.action_id) % (2**32))
        act_vec = rng.random(context_vec.shape[0])
        act_hv = encode_to_hv(act_vec, dim=hdc_centers.shape[1], seed=7)
        sim = hyperdim_similarity(ctx_hv, act_hv)
        similarities.append(sim)

    # 3. Priorities
    priorities = [
        hybrid_priority(act, sim, trust) for act, sim in zip(action_space, similarities)
    ]

    # 4. Choose best action
    best_idx = int(np.argmax(priorities))
    chosen_action = action_space[best_idx]
    best_priority = priorities[best_idx]

    # 5. Style transport
    v_target = hybrid_style_target(style_v0, style_v1, trust)
    updated_style = hybrid_euler_step(style_v0, v_target, best_priority)

    return chosen_action, updated_style


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Deterministic random seed for reproducibility
    rng = np.random.default_rng(123)

    # Synthetic context vector (e.g. 8‑dimensional)
    context = rng.random(8)

    # Dummy RBF centre matrix: each row = [features..., reward]
    centres = rng.random((20, 9))  # last column is reward
    centres[:, -1] = rng.random(20) * 5.0  # rewards in [0,5]

    # Build a small action set with RBF‑predicted expected rewards
    actions: List[BanditAction] = []
    for i in range(5):
        act_id = f"arm_{i}"
        # One‑hot action encoding concatenated to context for RBF query
        action_one_hot = np.zeros(5)
        action_one_hot[i] = 1.0
        query = np.concatenate([context, action_one_hot])
        exp_reward = rbf_surrogate(query, centres, gamma=2.0)
        actions.append(
            BanditAction(
                action_id=act_id,
                propensity=1.0 / 5,
                expected_reward=exp_reward,
                confidence_bound=0.1,
            )
        )

    # HDC prototype matrix (centres for similarity) – using random bipolar vectors
    hdc_dim = 4096
    hdc_prototypes = rng.integers(0, 2, size=(len(actions), hdc_dim)).astype(np.int8) * 2 - 1

    # Style vectors (high‑dimensional, e.g. D=300)
    style_dim = 300
    style0 = rng.random(style_dim)
    style1 = rng.random(style_dim)

    # Cockpit metrics → trust factor
    h = compute_trust_factor(
        claims_with_evidence=42,
        total_claims_emitted=50,
        displayed_ok=80,
        unknown_displayed_as_ok=20,
    )
    print(f"Trust factor h = {h:.3f}")

    # Run a single hybrid decision cycle
    chosen, new_style = hybrid_decision_cycle(
        context_vec=context,
        action_space=actions,
        hdc_centers=hdc_prototypes,
        style_v0=style0,
        style_v1=style1,
        trust=h,
    )
    print(f"Chosen action: {chosen.action_id}")
    print(f"Updated style norm (L2): {np.linalg.norm(new_style):.3f}")

    # Verify loss computation against the target
    target_style = hybrid_style_target(style0, style1, h)
    loss = hybrid_style_loss(new_style, target_style)
    print(f"Style MSE loss after one step: {loss:.6f}")