# DARWIN HAMMER — match 522, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_hybrid_model__hybrid_hybrid_regret_m267_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_geomet_hybrid_bandit_router_m394_s0.py (gen4)
# born: 2026-05-29T23:29:32Z

"""Hybrid Algorithm: DARWIN HAMMER – Fusion of
`hybrid_hybrid_hybrid_model__hybrid_hybrid_regret_m267_s0.py` (Parent A)
and
`hybrid_hybrid_hybrid_geomet_hybrid_bandit_router_m394_s0.py` (Parent B).

Mathematical Bridge
-------------------
Both parents expose an *update* that rescales a raw gradient by a scalar
derived from a model‑specific quantity:

* Parent A computes a **regret‑weighted strategy**  wᵣ  that is later used
  to adapt learning‑rates in a VRAM‑aware scheduler.
* Parent B computes an **Ollivier‑Ricci curvature** κ and a **bandit
  confidence term** c(S,Nₐ) that scale a linear‑model gradient.

The bridge consists of treating the regret weight as an additional
confidence factor and merging the curvature into the VRAM‑budgeted
learning‑rate.  The final scalar multiplier applied to the gradient is


η_eff = η_budget(free_mem) * wᵣ * c(S,Nₐ) / (1 + κ)


where `η_budget` is the VRAM‑aware learning‑rate, `wᵣ` the regret‑weight,
`c` the bandit confidence and `κ` the Ollivier‑Ricci curvature.
The unified update therefore respects memory constraints, regret‑based
exploration, and geometric‑curvature regularisation in a single step.
"""

import os
import sys
import math
import random
import pathlib
import datetime
from typing import List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# VRAM‑aware learning‑rate utilities (derived from Parent A)
# ----------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parents[2] if pathlib.Path(__file__).exists() else pathlib.Path.cwd()
DEFAULT_BUDGET_MB = int(os.environ.get("LUCIDOTA_VRAM_BUDGET_MB", "4096"))
DEFAULT_RESERVE_MB = int(os.environ.get("LUCIDOTA_VRAM_RESERVE_MB", "768"))


def _now_iso_z() -> str:
    """Current UTC timestamp in ISO‑8601 with trailing “Z”."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def _mock_free_vram_mb() -> int:
    """Return a pseudo‑random free‑VRAM estimate (for CPU‑only testing)."""
    # Simulate a system with 8 GiB total; reserve the configured amount.
    total = 8192
    used = random.randint(0, total - DEFAULT_RESERVE_MB)
    return max(total - used - DEFAULT_RESERVE_MB, 0)


def budgeted_lr(base_lr: float,
                free_mb: int,
                budget_mb: int = DEFAULT_BUDGET_MB,
                reserve_mb: int = DEFAULT_RESERVE_MB) -> float:
    """
    Scale ``base_lr`` according to available VRAM.

    If the free memory exceeds the usable budget (budget – reserve) the full
    learning‑rate is returned; otherwise a linear decay down to 10 % is applied.
    """
    usable = max(budget_mb - reserve_mb, 1)
    if free_mb >= usable:
        return base_lr
    # Linear decay to 0.1·base_lr when free_mb → 0
    scale = 0.1 + 0.9 * (free_mb / usable)
    return base_lr * scale


# ----------------------------------------------------------------------
# Regret‑weighted strategy (Parent A)
# ----------------------------------------------------------------------
def compute_regret_weighted_strategy(regrets: List[float]) -> List[float]:
    """
    Convert a list of non‑negative regrets into a probability distribution.

    The classic regret‑matching rule is used:
        p_i ∝ max(regret_i, 0)
    The probabilities sum to one; if all regrets are zero a uniform distribution
    is returned.
    """
    positive = [max(r, 0.0) for r in regrets]
    total = sum(positive)
    if total == 0.0:
        n = len(regrets)
        return [1.0 / n] * n
    return [p / total for p in positive]


# ----------------------------------------------------------------------
# Geometric‑algebra curvature and bandit confidence (Parent B)
# ----------------------------------------------------------------------
def krampus_ollivier_ricci_curvature(W: np.ndarray,
                                    x: np.ndarray,
                                    target: np.ndarray = None) -> float:
    """
    Compute a scalar curvature based on the TTT‑Linear residual.

    The residual r = (W·x) – target is interpreted as a distance; the squared
    Euclidean norm serves as a proxy for Ollivier‑Ricci curvature.
    """
    pred = W @ x
    t = target if target is not None else np.zeros_like(pred)
    residual = pred - t
    return float(residual @ residual)


def confidence_term(store_value: float, pulls: int) -> float:
    """
    Bandit confidence term modulated by a stored value S.

    The formula mirrors the one in Parent B:
        c = (1 + S/(S+1)) / sqrt(1 + N_a)
    """
    return (1.0 + store_value / (store_value + 1.0)) / math.sqrt(1.0 + pulls)


# ----------------------------------------------------------------------
# Hybrid update step – integrates both parents
# ----------------------------------------------------------------------
def hybrid_update_step(W: np.ndarray,
                       x: np.ndarray,
                       target: np.ndarray,
                       regrets: List[float],
                       store_value: float,
                       pulls: int,
                       base_lr: float = 0.01) -> Tuple[np.ndarray, dict]:
    """
    Perform a single weight update that respects VRAM budget, regret weighting,
    curvature regularisation, and bandit confidence.

    Returns the updated weight matrix and a diagnostics dictionary.
    """
    # 1. VRAM‑aware base learning‑rate
    free_mb = _mock_free_vram_mb()
    η_budget = budgeted_lr(base_lr, free_mb)

    # 2. Regret‑derived probability vector (acts as an exploration weight)
    w_regret = compute_regret_weighted_strategy(regrets)
    # For scalar scaling we take the mean probability (any monotone mapping works)
    w_regret_scalar = sum(w_regret) / len(w_regret)

    # 3. Curvature from geometric‑algebra perspective
    κ = krampus_ollivier_ricci_curvature(W, x, target)

    # 4. Bandit confidence term
    c = confidence_term(store_value, pulls)

    # 5. Effective learning‑rate
    η_eff = η_budget * w_regret_scalar * c / (1.0 + κ)

    # 6. Gradient placeholder (in a real system this would be ∇L)
    grad = np.random.randn(*W.shape)  # stochastic surrogate

    # 7. Weight update
    W_new = W - η_eff * grad

    diagnostics = {
        "free_vram_mb": free_mb,
        "η_budget": η_budget,
        "w_regret_scalar": w_regret_scalar,
        "curvature_κ": κ,
        "confidence_c": c,
        "η_eff": η_eff,
    }
    return W_new, diagnostics


# ----------------------------------------------------------------------
# Additional helper demonstrating quaternion rotor composition (Parent A)
# ----------------------------------------------------------------------
def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two quaternions (w, x, y, z)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def rotor_from_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Create a unit quaternion representing rotation about *axis* by *angle*."""
    axis = axis / np.linalg.norm(axis)
    half = angle_rad / 2.0
    return np.concatenate(([math.cos(half)], math.sin(half) * axis))


def apply_rotor(v: np.ndarray, rotor: np.ndarray) -> np.ndarray:
    """Rotate vector *v* by quaternion *rotor*."""
    qv = np.concatenate(([0.0], v))
    r_conj = np.array([rotor[0], -rotor[1], -rotor[2], -rotor[3]])
    return quat_mul(quat_mul(rotor, qv), r_conj)[1:]


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Dummy weight matrix (3 × 3), input vector, and target
    W = np.eye(3) * 0.5
    x = np.array([1.0, -0.5, 0.3])
    target = np.array([0.2, 0.0, -0.1])

    # Regrets for three hypothetical actions
    regrets = [0.1, 0.4, 0.0]

    # Bandit store value and number of pulls
    store_value = 5.0
    pulls = 12

    # Perform hybrid update
    W_updated, diag = hybrid_update_step(W, x, target, regrets,
                                         store_value, pulls, base_lr=0.02)

    print("=== Hybrid Update Diagnostics ===")
    for k, v in diag.items():
        print(f"{k}: {v}")

    print("\nUpdated weight matrix:\n", W_updated)

    # Demonstrate quaternion rotor on a vector
    axis = np.array([0.0, 0.0, 1.0])
    angle = math.pi / 4  # 45 degrees
    rotor = rotor_from_axis_angle(axis, angle)
    v = np.array([1.0, 0.0, 0.0])
    v_rot = apply_rotor(v, rotor)
    print("\nOriginal vector:", v)
    print("Rotated vector :", v_rot)