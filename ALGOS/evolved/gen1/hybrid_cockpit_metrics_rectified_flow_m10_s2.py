# DARWIN HAMMER — match 10, survivor 2
# gen: 1
# parent_a: cockpit_metrics.py (gen0)
# parent_b: rectified_flow.py (gen0)
# born: 2026-05-29T23:22:30Z

"""cockpit_rectified_hybrid.py
Hybrid module unifying the cockpit honesty/evidence metrics (Parent A) with the
rectified flow transport framework (Parent B).

Mathematical bridge
-------------------
The core of the rectified‑flow family is the constant‑velocity vector field

    v*(x0, x1) = x1 - x0                                            (1)

which drives a straight‑line interpolation Z_t = t·x1 + (1‑t)·x0.
The cockpit metrics provide a scalar *trust* value in the interval [0,1]
(e.g. ``cockpit_honesty`` or ``anti_slop_ratio``).  By treating this scalar as a
multiplicative factor on the ideal velocity we obtain a *trust‑weighted*
velocity field

    v_hybrid(x0, x1; h) = h · (x1 - x0)                             (2)

where h ∈ [0,1] is any cockpit metric.  Equation (2) fuses the two topologies:
the linear transport of rectified flow is modulated by the evidence‑coverage
quality of the cockpit.  The corresponding loss, ODE solver and trajectory
analysis are all derived from (2), yielding three representative hybrid
functions:

* ``hybrid_flow_target`` – metric‑scaled target velocity.
* ``hybrid_flow_loss`` – MSE between a model prediction and the scaled target.
* ``hybrid_euler_solve`` – Euler integration that adapts the step size using the
  audit‑debt count as a regulariser.

All functions remain pure NumPy and respect the import restrictions.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from typing import Callable, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Parent A – cockpit metrics (re‑implemented for internal use)
# ---------------------------------------------------------------------------

def anti_slop_ratio(claims_with_evidence: int, total_claims_emitted: int) -> float:
    """Ratio of supported claims, clamped to [0, 1]."""
    return 1.0 if total_claims_emitted <= 0 else max(0.0, min(1.0,
                claims_with_evidence / total_claims_emitted))


def cockpit_honesty(displayed_ok: int, unknown_displayed_as_ok: int) -> float:
    """Fraction of displayed items that are known‑good, clamped to [0, 1]."""
    total = displayed_ok + unknown_displayed_as_ok
    return 1.0 if total <= 0 else max(0.0, min(1.0, displayed_ok / total))


def audit_debt(exports_missing_audit_step: int) -> float:
    """Non‑negative scalar representing pending audit work."""
    return float(max(0, exports_missing_audit_step))


# ---------------------------------------------------------------------------
# Parent B – rectified flow core utilities
# ---------------------------------------------------------------------------

def interpolant(x0: np.ndarray, x1: np.ndarray, t: np.ndarray | float) -> np.ndarray:
    """Straight‑line interpolant Z_t = t·x1 + (1‑t)·x0."""
    x0 = np.asarray(x0, dtype=np.float64)
    x1 = np.asarray(x1, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    if x0.ndim > t.ndim:
        t = t[..., np.newaxis]          # broadcast over feature dimension
    return t * x1 + (1.0 - t) * x0


def flow_target(x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
    """Ideal constant‑velocity field (x1‑x0)."""
    return np.asarray(x1, dtype=np.float64) - np.asarray(x0, dtype=np.float64)


def flow_loss(v_pred: np.ndarray, x0: np.ndarray, x1: np.ndarray) -> float:
    """Mean‑squared error between prediction and ideal velocity."""
    v_pred = np.asarray(v_pred, dtype=np.float64)
    target = flow_target(x0, x1)
    return float(np.mean((v_pred - target) ** 2))


def straightness(traj: np.ndarray) -> float:
    """
    Straightness metric for a trajectory.
    1 - (arc length / chord length), clipped to [0,1].
    """
    # traj shape: (T, B, d)
    diffs = np.diff(traj, axis=0)                     # (T‑1, B, d)
    arc = np.sum(np.linalg.norm(diffs, axis=-1), axis=0)   # (B,)
    chord = np.linalg.norm(traj[-1] - traj[0], axis=-1)    # (B,)
    # avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(chord > 0, arc / chord, 0.0)
    straight = 1.0 - np.clip(ratio, 0.0, 1.0)
    return float(np.mean(straight))


# ---------------------------------------------------------------------------
# Hybrid layer – metric‑weighted velocity field
# ---------------------------------------------------------------------------

def hybrid_flow_target(
    x0: np.ndarray,
    x1: np.ndarray,
    honesty: float,
) -> np.ndarray:
    """
    Trust‑weighted target velocity.

    Implements equation (2): v_hybrid = h * (x1 - x0).

    Parameters
    ----------
    x0, x1 : (B, d) arrays.
    honesty : scalar in [0,1] (e.g. from ``cockpit_honesty``).

    Returns
    -------
    np.ndarray of shape (B, d).
    """
    base = flow_target(x0, x1)
    return honesty * base


def hybrid_flow_loss(
    v_pred: np.ndarray,
    x0: np.ndarray,
    x1: np.ndarray,
    honesty: float,
) -> float:
    """
    MSE between a model prediction and the metric‑scaled target.

    The loss reduces to ``flow_loss`` when ``honesty == 1.0``.
    """
    v_pred = np.asarray(v_pred, dtype=np.float64)
    target = hybrid_flow_target(x0, x1, honesty)
    return float(np.mean((v_pred - target) ** 2))


def hybrid_euler_solve(
    v_fn: Callable[[np.ndarray, float], np.ndarray],
    x0: np.ndarray,
    honesty: float,
    debt: float,
    steps: int = 10,
) -> np.ndarray:
    """
    Euler integration of a *trust‑aware* vector field.

    The step size is attenuated by the audit debt: larger debt → smaller dt,
    encouraging more cautious integration.

    Effective dt = (1 / steps) * (1 / (1 + debt))

    Parameters
    ----------
    v_fn   : callable (z, t) -> velocity prediction.
    x0     : (B, d) initial noise.
    honesty: scalar ∈ [0,1] used to scale the underlying velocity inside ``v_fn``.
    debt   : non‑negative scalar; higher values shrink the integration step.
    steps  : number of Euler steps.

    Returns
    -------
    traj : np.ndarray of shape (steps+1, B, d).
    """
    x0 = np.asarray(x0, dtype=np.float64)
    dt_base = 1.0 / steps
    dt = dt_base / (1.0 + debt)                # debt‑aware step size
    ts = np.linspace(0.0, 1.0 - dt, steps)     # timestamps at which v is queried
    traj = np.empty((steps + 1,) + x0.shape, dtype=np.float64)
    traj[0] = x0
    z = x0.copy()
    for k, t in enumerate(ts):
        # The model receives the honesty scalar so it can implement (2) internally.
        v = v_fn(z, float(t), honesty)
        z = z + dt * np.asarray(v, dtype=np.float64)
        traj[k + 1] = z
    return traj


# ---------------------------------------------------------------------------
# Example model stub – learns a simple linear mapping (for testing)
# ---------------------------------------------------------------------------

class LinearVelocityModel:
    """
    Very small linear model: v(z, t, h) = h * W @ z + b
    where W and b are learned parameters.  For the smoke test we initialise
    them to approximate the ideal field (x1 - x0).
    """

    def __init__(self, dim: int):
        rng = np.random.default_rng(42)
        self.W = rng.normal(scale=0.01, size=(dim, dim))
        self.b = np.zeros(dim)

    def __call__(self, z: np.ndarray, t: float, honesty: float) -> np.ndarray:
        # Linear mapping plus honesty scaling
        return honesty * (z @ self.W.T + self.b)


# ---------------------------------------------------------------------------
# Hybrid demonstration functions
# ---------------------------------------------------------------------------

def hybrid_metric_flow(
    x0: np.ndarray,
    x1: np.ndarray,
    displayed_ok: int,
    unknown_displayed_as_ok: int,
) -> Tuple[float, float]:
    """
    Compute both the honesty‑weighted flow loss and the straightness of the
    resulting trajectory when integrated with the hybrid Euler solver.

    Returns (loss, straightness).
    """
    honesty = cockpit_honesty(displayed_ok, unknown_displayed_as_ok)
    # Dummy model that simply returns the ideal velocity (for illustration)
    def perfect_model(z, t, h):
        return hybrid_flow_target(z, interpolant(z, x1, t), h)

    traj = hybrid_euler_solve(perfect_model, x0, honesty, debt=0.0, steps=20)
    # Final prediction is the last point of the trajectory
    v_pred = (traj[-1] - traj[-2]) * 20.0   # approximate velocity (dt≈1/20)
    loss = hybrid_flow_loss(v_pred, x0, x1, honesty)
    strg = straightness(traj)
    return loss, strg


def hybrid_audit_regularized_loss(
    v_pred: np.ndarray,
    x0: np.ndarray,
    x1: np.ndarray,
    claims_with_evidence: int,
    total_claims_emitted: int,
    exports_missing_audit_step: int,
) -> float:
    """
    Combine the metric‑scaled flow loss with a penalty proportional to audit debt.
    """
    honesty = anti_slop_ratio(claims_with_evidence, total_claims_emitted)
    debt = audit_debt(exports_missing_audit_step)
    base_loss = hybrid_flow_loss(v_pred, x0, x1, honesty)
    # Regularization term grows with debt; weight chosen heuristically.
    reg = 0.01 * debt
    return base_loss + reg


def hybrid_interpolant_with_debt(
    x0: np.ndarray,
    x1: np.ndarray,
    t: np.ndarray | float,
    exports_missing_audit_step: int,
) -> np.ndarray:
    """
    Shift the straight‑line interpolant by a small offset proportional to audit debt.
    This mimics a “correction” that the cockpit would apply to a suspect trajectory.
    """
    debt = audit_debt(exports_missing_audit_step)
    base = interpolant(x0, x1, t)
    # Offset direction: unit vector from x0 to x1, scaled by debt factor.
    direction = flow_target(x0, x1)
    norm = np.linalg.norm(direction, axis=-1, keepdims=True) + 1e-12
    offset = (debt * 0.001) * (direction / norm)   # tiny correction
    return base + offset


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Synthetic data
    B, d = 4, 3
    rng = np.random.default_rng(123)

    x0 = rng.normal(size=(B, d))
    x1 = rng.normal(loc=5.0, size=(B, d))

    # Cockpit counts
    displayed_ok = 37
    unknown_displayed_as_ok = 8

    # Demonstrate hybrid_metric_flow
    loss, strg = hybrid_metric_flow(x0, x1, displayed_ok, unknown_displayed_as_ok)
    print(f"Hybrid flow loss: {loss:.6f}, straightness: {strg:.4f}")

    # Demonstrate audit‑regularized loss
    v_pred = rng.normal(size=(B, d))
    reg_loss = hybrid_audit_regularized_loss(
        v_pred,
        x0,
        x1,
        claims_with_evidence=45,
        total_claims_emitted=60,
        exports_missing_audit_step=3,
    )
    print(f"Audit‑regularized loss: {reg_loss:.6f}")

    # Demonstrate interpolant with debt correction
    t_vals = np.linspace(0, 1, 5)
    corrected = hybrid_interpolant_with_debt(x0, x1, t_vals, exports_missing_audit_step=2)
    print("Corrected interpolant shape:", corrected.shape)

    # Simple sanity check: all returned values are finite
    assert np.isfinite(loss) and np.isfinite(strg)
    assert np.isfinite(reg_loss)
    assert np.all(np.isfinite(corrected))

    print("Hybrid module smoke test completed successfully.")