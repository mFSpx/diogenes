# DARWIN HAMMER — match 1465, survivor 4
# gen: 5
# parent_a: hybrid_hybrid_hybrid_model__hybrid_hybrid_regret_m267_s0.py (gen4)
# parent_b: hybrid_hybrid_cockpit_metri_hard_truth_math_m27_s1.py (gen2)
# born: 2026-05-29T23:36:38Z

"""Hybrid Module: fusion of DARWIN HAMMER Parent A (VRAM‑aware regret‑weighted GA) and Parent B (cockpit‑trust‑weighted flow with LSM similarity).

Mathematical Bridge
-------------------
Parent A provides a *regret‑weighted learning‑rate*  η = LR·w_regret, where LR is a VRAM‑scaled base learning rate and w_regret ∈ ℝ⁺ adapts to the instantaneous regret of an action.  
Parent B supplies a *trust scalar* h ∈ [0,1] derived from cockpit honesty metrics and a *similarity factor* s ∈ [−1,1] obtained from hard‑truth LSM vectors.

The hybrid algorithm treats the product  γ = η·h·s as a **global step‑size multiplier** that simultaneously:
* respects GPU memory constraints (η from Parent A),
* discounts updates when cockpit trust is low (h from Parent B),
* amplifies updates for semantically similar inputs (s from Parent B).

The base velocity field of Parent B,
 v₀ = x₁ − x₀,
is first scaled by the trust‑weighted factor (h·v₀), then rotated by a quaternion rotor (Parent A) to introduce geometric flexibility, and finally multiplied by the global scalar γ.  The resulting vector is used in an Euler‑style integration step, yielding a unified update rule that fuses both topologies."""

import os
import sys
import math
import random
from pathlib import Path
from datetime import datetime, timezone
from typing import Tuple, Callable, Iterable, Any

import numpy as np

# ----------------------------------------------------------------------
# VRAM‑related helpers (Parent A)
# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2] if Path(__file__).exists() else Path.cwd()
DEFAULT_BUDGET_MB = int(os.getenv("LUCIDOTA_VRAM_BUDGET_MB", "4096"))
DEFAULT_RESERVE_MB = int(os.getenv("LUCIDOTA_VRAM_RESERVE_MB", "768"))

def now_z() -> str:
    """Current UTC timestamp in ISO‑8601 with trailing “Z”."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def gpu_memory() -> Tuple[int, int]:
    """
    Mocked GPU memory query.
    Returns (free_mb, total_mb). In a real setting this would query the driver.
    """
    total_mb = 16384  # assume 16 GiB total
    free_mb = random.randint(1024, total_mb - DEFAULT_RESERVE_MB)
    return free_mb, total_mb

def budgeted_lr(base_lr: float = 0.01,
                budget_mb: int = DEFAULT_BUDGET_MB,
                reserve_mb: int = DEFAULT_RESERVE_MB) -> float:
    """
    Scale the base learning rate according to currently free GPU memory.
    If free memory ≥ budget, return base_lr; otherwise linearly decay.
    """
    free_mb, _ = gpu_memory()
    if free_mb >= budget_mb:
        return base_lr
    # linear decay to 20 % of base_lr when free memory hits reserve_mb
    min_lr = 0.2 * base_lr
    scale = (free_mb - reserve_mb) / max(1, budget_mb - reserve_mb)
    return max(min_lr, base_lr * scale)

# ----------------------------------------------------------------------
# Quaternion utilities (Parent A)
# ----------------------------------------------------------------------
def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two quaternions (w, x, y, z)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=float)

def quat_conj(q: np.ndarray) -> np.ndarray:
    """Conjugate of a quaternion."""
    w, x, y, z = q
    return np.array([w, -x, -y, -z], dtype=float)

def apply_rotor(v: np.ndarray, rotor: np.ndarray) -> np.ndarray:
    """Rotate vector v ∈ ℝ³ by unit quaternion rotor."""
    qv = np.concatenate([[0.0], v])
    rotated = quat_mul(quat_mul(rotor, qv), quat_conj(rotor))
    return rotated[1:]

def rotor_from_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Create a unit quaternion representing rotation about axis by angle."""
    axis = axis / np.linalg.norm(axis)
    half = angle_rad / 2.0
    return np.concatenate([[math.cos(half)], math.sin(half) * axis])

# ----------------------------------------------------------------------
# Regret‑weighted strategy (Parent A)
# ----------------------------------------------------------------------
def compute_regret_weighted_strategy(predictions: np.ndarray,
                                     rewards: np.ndarray) -> np.ndarray:
    """
    Simple regret: r = reward - prediction.
    Weight = 1 + sigmoid(r) ∈ (0,2). Returns per‑action weight vector.
    """
    regret = rewards - predictions
    return 1.0 + 1.0 / (1.0 + np.exp(-regret))

# ----------------------------------------------------------------------
# Cockpit honesty metrics (Parent B)
# ----------------------------------------------------------------------
def anti_slop_ratio(claims_with_evidence: int, total_claims_emitted: int) -> float:
    """Ratio of supported claims, clamped to [0,1]."""
    if total_claims_emitted <= 0:
        return 1.0
    return max(0.0, min(1.0, claims_with_evidence / total_claims_emitted))

def cockpit_honesty(displayed_ok: int, unknown_displayed_as_ok: int) -> float:
    """Fraction of displayed items that are known‑good, clamped to [0,1]."""
    total = displayed_ok + unknown_displayed_as_ok
    if total <= 0:
        return 1.0
    return max(0.0, min(1.0, displayed_ok / total))

def audit_debt(exports_missing_audit_step: int, total_exports: int) -> float:
    """Debt ratio, clamped to [0,1]; larger debt → lower trust."""
    if total_exports <= 0:
        return 0.0
    debt = exports_missing_audit_step / total_exports
    return max(0.0, min(1.0, 1.0 - debt))

def aggregate_trust(*metrics: float) -> float:
    """Geometric mean of supplied trust metrics (all ∈ [0,1])."""
    if not metrics:
        return 1.0
    prod = 1.0
    for m in metrics:
        prod *= max(0.0, min(1.0, m))
    return prod ** (1.0 / len(metrics))

# ----------------------------------------------------------------------
# Hard‑truth LSM similarity (Parent B)
# ----------------------------------------------------------------------
def lsm_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity in [-1,1] for LSM vectors."""
    if a.size == 0 or b.size == 0:
        return 0.0
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

# ----------------------------------------------------------------------
# Hybrid core functions (fusion of both parents)
# ----------------------------------------------------------------------
def trust_weighted_velocity(x0: np.ndarray,
                            x1: np.ndarray,
                            trust: float) -> np.ndarray:
    """
    Parent B: v_h = trust * (x1 - x0).
    """
    return trust * (x1 - x0)

def regret_weighted_lr(predictions: np.ndarray,
                       rewards: np.ndarray,
                       base_lr: float = 0.01) -> float:
    """
    Combine VRAM‑scaled base learning rate with regret weights.
    Returns a scalar learning rate.
    """
    lr = budgeted_lr(base_lr=base_lr)
    w = compute_regret_weighted_strategy(predictions, rewards)
    # Use mean weight as a scalar factor
    return lr * float(np.mean(w))

def hybrid_euler_step(x0: np.ndarray,
                      x1: np.ndarray,
                      predictions: np.ndarray,
                      rewards: np.ndarray,
                      trust: float,
                      similarity: float,
                      rotor: np.ndarray,
                      base_lr: float = 0.01) -> np.ndarray:
    """
    Single Euler integration step that fuses both parents:

    1. Compute regret‑scaled learning rate η.
    2. Compute trust‑scaled base velocity v₀.
    3. Modulate by similarity s ∈ [-1,1] (shift to [0,1] for positivity).
    4. Rotate the resulting vector by the supplied quaternion rotor.
    5. Apply global scalar γ = η * h * (0.5 + 0.5·s) to obtain the update.
    """
    # 1. Regret‑weighted learning rate
    eta = regret_weighted_lr(predictions, rewards, base_lr=base_lr)

    # 2. Trust‑weighted velocity
    v_base = trust_weighted_velocity(x0, x1, trust)

    # 3. Similarity factor shifted to [0,1]
    sim_factor = 0.5 + 0.5 * similarity

    # 4. Rotate
    v_rot = apply_rotor(v_base, rotor)

    # 5. Global step
    gamma = eta * sim_factor
    return x0 + gamma * v_rot

def hybrid_euler_integrate(x_start: np.ndarray,
                           x_target: np.ndarray,
                           steps: int,
                           pred_fn: Callable[[np.ndarray], np.ndarray],
                           reward_fn: Callable[[np.ndarray], np.ndarray],
                           trust_metrics: Tuple[float, ...],
                           lsm_vec_start: np.ndarray,
                           lsm_vec_target: np.ndarray,
                           axis: np.ndarray,
                           base_lr: float = 0.01) -> np.ndarray:
    """
    Perform `steps` Euler updates using the hybrid step defined above.

    Parameters
    ----------
    x_start, x_target : np.ndarray
        Endpoints of the flow.
    steps : int
        Number of integration steps.
    pred_fn, reward_fn : callables
        Produce predictions and rewards for the current state (used for regret).
    trust_metrics : tuple of floats
        Cockpit metrics that will be aggregated into a trust scalar.
    lsm_vec_start, lsm_vec_target : np.ndarray
        LSM feature vectors for similarity computation.
    axis : np.ndarray
        Axis used to build the rotor (will be normalized internally).
    base_lr : float
        Base learning rate before VRAM & regret scaling.
    """
    x = np.array(x_start, dtype=float)
    trust = aggregate_trust(*trust_metrics)
    similarity = lsm_cosine_similarity(lsm_vec_start, lsm_vec_target)

    for i in range(steps):
        # simple linear schedule for rotor angle
        angle = (i + 1) / steps * math.pi * 0.5  # up to 90°
        rotor = rotor_from_axis_angle(axis, angle)

        preds = pred_fn(x)
        rews = reward_fn(x)

        x = hybrid_euler_step(x, x_target, preds, rews,
                              trust, similarity, rotor,
                              base_lr=base_lr)
    return x

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Dummy start / target vectors in ℝ³
    x0 = np.array([0.0, 0.0, 0.0])
    x1 = np.array([1.0, 2.0, -1.0])

    # Dummy prediction / reward generators
    def dummy_pred(state: np.ndarray) -> np.ndarray:
        # pretend a linear model
        return 0.5 * state

    def dummy_reward(state: np.ndarray) -> np.ndarray:
        # reward higher the closer we are to target
        dist = np.linalg.norm(state - x1)
        return np.array([ -dist ])

    # Cockpit metrics
    trust_vals = (
        anti_slop_ratio(claims_with_evidence=8, total_claims_emitted=10),
        cockpit_honesty(displayed_ok=7, unknown_displayed_as_ok=1),
        audit_debt(exports_missing_audit_step=0, total_exports=5),
    )

    # Random LSM vectors (simulating text embeddings)
    lsm_start = np.random.randn(128)
    lsm_target = np.random.randn(128)

    # Rotation axis (arbitrary)
    rot_axis = np.array([0.0, 0.0, 1.0])

    result = hybrid_euler_integrate(
        x_start=x0,
        x_target=x1,
        steps=10,
        pred_fn=dummy_pred,
        reward_fn=dummy_reward,
        trust_metrics=trust_vals,
        lsm_vec_start=lsm_start,
        lsm_vec_target=lsm_target,
        axis=rot_axis,
        base_lr=0.02,
    )

    print("Final integrated state:", result)
    print("Distance to target:", np.linalg.norm(result - x1))