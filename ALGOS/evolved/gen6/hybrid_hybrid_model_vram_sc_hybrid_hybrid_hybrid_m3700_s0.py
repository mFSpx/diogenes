# DARWIN HAMMER — match 3700, survivor 0
# gen: 6
# parent_a: hybrid_model_vram_scheduler_ttt_linear_m11_s0.py (gen1)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m630_s0.py (gen5)
# born: 2026-05-29T23:51:16Z

"""
Hybrid VRAM‑NLMS‑Bandit Scheduler

This module fuses:

* **Parent A** – the linear test‑time‑training (ttt_linear) weight‑matrix dynamics
  used as a proxy for GPU‑VRAM usage evolution.
* **Parent B** – the Normalised Least‑Mean‑Squares (NLMS) predictor together with a
  diffusion‑driven store and a multi‑armed bandit that selects pre‑emption actions.

**Mathematical bridge**

The ttt_linear matrix **W ∈ ℝ^{n×n}** is interpreted as a linear map from an
artifact‑feature vector **f** to an estimated VRAM consumption vector **u = W·f**.
The NLMS predictor supplies a scalar prediction **p = wᵀ·x** of the *total* VRAM
required for a new artifact, where **w** are the NLMS weights and **x** are the same
features used by the linear map.

The prediction error **e = target – p** drives two coupled updates:

1. **Diffusive store** – a scalar state **S** obeys a first‑order diffusion
   equation `dS/dt = -α·S + β·e`.  This mirrors the “sheaf cohomology diffusion
   forcing” of Parent B and provides a smooth signal that modulates the stochastic
   bandit propensities.

2. **Bandit propensities** – each action **a** has a propensity **πₐ** that is
   multiplicatively scaled by `exp(γ·S)`.  High store values (large recent errors)
   increase exploration, while low values reinforce exploitation.

Finally, the selected bandit action determines a corrective update to the VRAM
matrix **W**, closing the loop between resource planning, prediction, and
action selection.

The three core functions below showcase this hybrid dynamics:
`estimate_vram`, `nlms_step`, and `bandit_select`.
"""

from __future__ import annotations
import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Dict, Any

import numpy as np

# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class VramSlotPlan:
    artifact_id: str
    artifact_kind: str
    action: str
    estimated_mb: int
    reason: str
    detail: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BanditAction:
    action_id: str
    base_propensity: float
    expected_reward: float = 0.0
    confidence_bound: float = 0.0
    algorithm: str = "HybridScheduler"


@dataclass
class StoreState:
    level: float = 0.0          # S(t) in the diffusion equation
    alpha: float = 0.5          # decay coefficient
    beta: float = 1.0           # error‑to‑store gain
    dt: float = 1.0             # integration step size
    gamma: float = 0.1          # bandit scaling factor


# ----------------------------------------------------------------------
# Core mathematical components
# ----------------------------------------------------------------------
def estimate_vram(features: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Linear VRAM estimator from Parent A.
    Computes u = W · f, where:
    - features : (n,) column vector describing an artifact (size, type encoding …)
    - W        : (n, n) linear map learned during test‑time training.
    Returns a vector of estimated per‑slot VRAM consumption.
    """
    if features.ndim != 1:
        raise ValueError("features must be a 1‑D vector")
    if W.shape[0] != W.shape[1] or W.shape[1] != features.shape[0]:
        raise ValueError("W must be square and compatible with feature dimension")
    return W @ features


def nlms_step(
    w: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, float]:
    """
    One NLMS adaptation step (Parent B).
    - w      : current NLMS weight vector.
    - x      : input feature vector.
    - target : true total VRAM consumption for the artifact.
    Returns the updated weight vector and the instantaneous prediction error.
    The step size is normalised by ||x||² + eps.
    """
    if x.ndim != 1 or w.shape != x.shape:
        raise ValueError("w and x must be 1‑D vectors of the same shape")
    pred = float(w @ x)
    error = target - pred
    norm_sq = float(x @ x) + eps
    step = (mu / norm_sq) * error
    w_new = w + step * x
    return w_new, error


def update_store(store: StoreState, error: float) -> None:
    """
    Diffusive store dynamics driven by the NLMS error.
    Implements a forward Euler step of:
        dS/dt = -α·S + β·error
    The store level is clipped to a reasonable range to avoid numerical blow‑up.
    """
    dS = -store.alpha * store.level + store.beta * error
    store.level += store.dt * dS
    # Clip for stability
    store.level = max(min(store.level, 1e6), -1e6)


def bandit_select(
    actions: List[BanditAction],
    store: StoreState,
    rng: random.Random | None = None,
) -> BanditAction:
    """
    Stochastic action selection (Parent B) whose propensities are modulated by the
    current store level.  The modulation follows:
        πₐ' = base_propensity * exp(γ·S)
    The action is drawn proportionally to πₐ'.
    """
    rng = rng or random.Random()
    scaled = []
    for a in actions:
        scaled_prop = a.base_propensity * math.exp(store.gamma * store.level)
        scaled.append(scaled_prop if scaled_prop > 0 else 0.0)

    total = sum(scaled)
    if total == 0.0:
        # fallback to uniform random choice
        return rng.choice(actions)

    cum = 0.0
    threshold = rng.random() * total
    for a, prop in zip(actions, scaled):
        cum += prop
        if cum >= threshold:
            return a
    # numerical safety net
    return actions[-1]


# ----------------------------------------------------------------------
# Hybrid step that ties everything together
# ----------------------------------------------------------------------
def hybrid_step(
    artifact_id: str,
    artifact_kind: str,
    features: np.ndarray,
    target_vram_mb: float,
    W: np.ndarray,
    nlms_w: np.ndarray,
    store: StoreState,
    bandit_actions: List[BanditAction],
    rng: random.Random | None = None,
) -> Tuple[np.ndarray, np.ndarray, StoreState, BanditAction, VramSlotPlan]:
    """
    Executes one hybrid iteration:
    1. Estimate per‑slot VRAM with the linear map (Parent A).
    2. Predict total VRAM with NLMS and adapt the NLMS weights.
    3. Update the diffusion store using the NLMS error.
    4. Select a bandit action whose propensity is shaped by the store.
    5. Apply the chosen action as a corrective update to the VRAM matrix W.
       The action can be:
         - "scale_up":   increase all entries of W by a factor >1.
         - "scale_down": decrease all entries of W by a factor <1.
         - "reset":      re‑initialise W to an identity matrix.
    Returns the updated W, NLMS weights, store, selected action and a
    VramSlotPlan describing the decision.
    """
    rng = rng or random.Random()

    # 1. Linear VRAM estimate
    per_slot = estimate_vram(features, W)
    total_est = float(per_slot.sum())

    # 2. NLMS prediction & adaptation
    nlms_w, nlms_error = nlms_step(nlms_w, features, target_vram_mb)

    # 3. Store diffusion update
    update_store(store, nlms_error)

    # 4. Bandit selection
    selected_action = bandit_select(bandit_actions, store, rng)

    # 5. Apply corrective update to W
    if selected_action.action_id == "scale_up":
        factor = 1.1
        W = W * factor
        reason = "Increase capacity after under‑prediction"
    elif selected_action.action_id == "scale_down":
        factor = 0.9
        W = W * factor
        reason = "Decrease allocation after over‑prediction"
    elif selected_action.action_id == "reset":
        W = np.eye(W.shape[0])
        reason = "Reset matrix to identity due to divergent behaviour"
    else:
        # unknown action → no change
        reason = "No matrix change (unknown action)"
    
    plan = VramSlotPlan(
        artifact_id=artifact_id,
        artifact_kind=artifact_kind,
        action=selected_action.action_id,
        estimated_mb=int(total_est),
        reason=reason,
        detail={
            "per_slot_estimate": per_slot.tolist(),
            "nlms_prediction": total_est,
            "nlms_error": nlms_error,
            "store_level": store.level,
        },
    )
    return W, nlms_w, store, selected_action, plan


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    rng = random.Random(42)

    # Synthetic problem dimension
    dim = 5
    # Initialise linear map (W) close to identity
    W = np.eye(dim) + 0.01 * np.random.randn(dim, dim)

    # NLMS weight vector
    nlms_w = np.zeros(dim)

    # Store state
    store = StoreState(alpha=0.4, beta=0.8, dt=1.0, gamma=0.05)

    # Define a static bandit action set
    bandit_actions = [
        BanditAction(action_id="scale_up", base_propensity=0.3),
        BanditAction(action_id="scale_down", base_propensity=0.5),
        BanditAction(action_id="reset", base_propensity=0.2),
    ]

    # Simulate a few artifacts
    for step in range(10):
        artifact_id = f"art_{step}"
        artifact_kind = "image" if step % 2 == 0 else "text"
        # Random feature vector (e.g., size, type embeddings)
        features = np.abs(np.random.randn(dim))
        # Ground‑truth total VRAM (synthetic)
        target_vram = float(features @ np.arange(1, dim + 1)) + rng.uniform(-5, 5)

        W, nlms_w, store, action, plan = hybrid_step(
            artifact_id=artifact_id,
            artifact_kind=artifact_kind,
            features=features,
            target_vram_mb=target_vram,
            W=W,
            nlms_w=nlms_w,
            store=store,
            bandit_actions=bandit_actions,
            rng=rng,
        )

        # Simple sanity prints (could be suppressed)
        print(f"Step {step}: action={action.action_id}, est={plan.estimated_mb}MB, "
              f"error={plan.detail['nlms_error']:.2f}, store={store.level:.2f}")
    sys.exit(0)