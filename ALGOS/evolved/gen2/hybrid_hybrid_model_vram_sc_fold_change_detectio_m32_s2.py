# DARWIN HAMMER — match 32, survivor 2
# gen: 2
# parent_a: hybrid_model_vram_scheduler_ttt_linear_m11_s0.py (gen1)
# parent_b: fold_change_detection.py (gen0)
# born: 2026-05-29T23:23:13Z

"""Hybrid Fold-Change Linear Trainer
===================================

This module fuses the *ttt_linear* test‑time training algorithm (Parent A) with the
*fold‑change detection* dynamical system (Parent B).  

Both parents operate on linear algebraic objects:

* **ttt_linear** updates a weight matrix ``W`` by gradient descent on the
  quadratic loss ``‖W x − target‖²``.
* **fold_change_detection** evolves a scalar pair ``(x, y)`` according to a
  feed‑forward ODE that computes a *gain* proportional to the fold‑change
  ``u / |x|`` of an external signal ``u``.

The mathematical bridge is the *gain* produced by the fold‑change detector.
We use this gain to **modulate the effective learning rate** of the matrix
update.  In this way the scalar dynamics directly influence the matrix dynamics,
creating a single unified system that simultaneously evolves a weight matrix
and a pair of fold‑change states.

The core hybrid step therefore consists of:

1. Advance the fold‑change detector (scalar ODE) → obtain ``gain``.
2. Scale the learning rate ``η`` of the ttt step by ``1 + gain`` (or any
   monotonic function of the gain).
3. Perform the usual ttt gradient descent update with the scaled learning rate.

The module provides three public functions that illustrate this hybrid
behaviour, plus a small smoke‑test when executed as a script.
"""

from __future__ import annotations

import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Tuple, List, Dict

import numpy as np

# ----------------------------------------------------------------------
# Parent A – ttt_linear core utilities
# ----------------------------------------------------------------------


def init_ttt(d_in: int, d_out: int | None = None, scale: float = 0.01, seed: int = 0) -> np.ndarray:
    """Create a random weight matrix ``W`` of shape (d_out, d_in)."""
    rng = np.random.default_rng(seed)
    if d_out is None:
        d_out = d_in
    return rng.standard_normal((d_out, d_in)) * scale


def ttt_grad(W: np.ndarray, x: np.ndarray, target: np.ndarray | None = None) -> np.ndarray:
    """Gradient of the quadratic loss ‖W x − target‖² w.r.t. ``W``."""
    pred = W @ x
    t = x if target is None else target
    residual = pred - t
    return 2.0 * np.outer(residual, x)


def ttt_step(W: np.ndarray, x: np.ndarray, eta: float = 0.01, target: np.ndarray | None = None) -> np.ndarray:
    """One gradient‑descent step on ``W``."""
    g = ttt_grad(W, x, target=target)
    return W - eta * g


def ttt_forward(W: np.ndarray, x: np.ndarray, eta: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform a ttt step and return the new hidden activation ``h = W_new @ x``
    together with the updated weight matrix ``W_new``.
    """
    W_new = ttt_step(W, x, eta=eta)
    h = W_new @ x
    return h, W_new


# ----------------------------------------------------------------------
# Parent B – fold‑change detection core utilities
# ----------------------------------------------------------------------


def fold_step(
    u: float,
    x: float,
    y: float,
    dt: float = 1.0,
    gain: float = 1.0,
    decay_x: float = 1.0,
    decay_y: float = 1.0,
    eps: float = 1e-12,
) -> Tuple[float, float]:
    """
    Advance the fold‑change detector by one Euler step.

    ``u`` – external input signal.
    ``x`` – low‑pass filtered version of ``u``.
    ``y`` – output that encodes the fold‑change (gain·u/|x| − decay·y).
    """
    if dt < 0:
        raise ValueError("dt must be non‑negative")
    ratio = u / max(abs(x), eps)
    dy = gain * ratio - decay_y * y
    dx = u - decay_x * x
    return x + dt * dx, y + dt * dy


def fold_response_series(
    inputs: List[float],
    x0: float = 1.0,
    y0: float = 0.0,
    **kw,
) -> List[Tuple[float, float]]:
    """Run ``fold_step`` over a list of inputs."""
    x, y = x0, y0
    out = []
    for u in inputs:
        x, y = fold_step(u, x, y, **kw)
        out.append((x, y))
    return out


# ----------------------------------------------------------------------
# Hybrid data structure
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


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------


def hybrid_init(
    d_in: int,
    d_out: int | None = None,
    scale: float = 0.01,
    seed: int = 0,
    x0: float = 1.0,
    y0: float = 0.0,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Initialise both the weight matrix ``W`` and the fold‑change detector state.

    Returns
    -------
    W : np.ndarray
        Randomly initialised weight matrix.
    fc_state : dict
        ``{'x': float, 'y': float}`` – the internal state of the detector.
    """
    W = init_ttt(d_in, d_out=d_out, scale=scale, seed=seed)
    fc_state = {"x": float(x0), "y": float(y0)}
    return W, fc_state


def hybrid_step(
    W: np.ndarray,
    vec: np.ndarray,
    u: float,
    fc_state: Dict[str, float],
    *,
    base_eta: float = 0.01,
    dt: float = 1.0,
    gain: float = 1.0,
    decay_x: float = 1.0,
    decay_y: float = 1.0,
) -> Tuple[np.ndarray, Dict[str, float], VramSlotPlan]:
    """
    Perform a single hybrid update.

    1. Advance the fold‑change detector with input ``u``.
    2. Compute a scaling factor ``scale = 1 + |y|`` (the absolute output of the detector).
    3. Apply a ttt gradient step with learning rate ``eta = base_eta * scale``.
    4. Emit a dummy ``VramSlotPlan`` that records the size of the new weight matrix.

    Parameters
    ----------
    W : np.ndarray
        Current weight matrix.
    vec : np.ndarray
        Input vector ``x`` for the ttt linear model.
    u : float
        External scalar signal feeding the fold‑change detector.
    fc_state : dict
        Current ``{'x':…, 'y':…}`` state of the detector.
    base_eta, dt, gain, decay_x, decay_y : float
        Hyper‑parameters controlling the two sub‑systems.

    Returns
    -------
    W_new : np.ndarray
        Updated weight matrix.
    fc_state_new : dict
        Updated detector state.
    plan : VramSlotPlan
        A lightweight record mimicking the VRAM‑planning output of Parent A.
    """
    # 1. Fold‑change dynamics
    x_fc, y_fc = fold_step(
        u,
        fc_state["x"],
        fc_state["y"],
        dt=dt,
        gain=gain,
        decay_x=decay_x,
        decay_y=decay_y,
    )
    fc_state_new = {"x": x_fc, "y": y_fc}

    # 2. Learning‑rate modulation
    scale = 1.0 + abs(y_fc)  # ensures positivity
    eta_eff = base_eta * scale

    # 3. ttt matrix update
    W_new = ttt_step(W, vec, eta=eta_eff)

    # 4. Dummy VRAM plan (size in elements ≈ memory usage)
    plan = VramSlotPlan(
        artifact_id="hybrid_step",
        artifact_kind="matrix",
        action="update",
        estimated_mb=W_new.size * 4 // (1024 * 1024),  # assuming 32‑bit floats
        reason="scaled by fold‑change gain",
        detail={"eta_eff": eta_eff, "scale": scale, "u": u},
    )
    return W_new, fc_state_new, plan


def hybrid_series(
    W: np.ndarray,
    vecs: List[np.ndarray],
    us: List[float],
    *,
    base_eta: float = 0.01,
    dt: float = 1.0,
    gain: float = 1.0,
    decay_x: float = 1.0,
    decay_y: float = 1.0,
) -> Tuple[np.ndarray, List[VramSlotPlan]]:
    """
    Run a sequence of hybrid updates.

    ``vecs[i]`` and ``us[i]`` are processed together at step ``i``.
    Returns the final weight matrix and the list of generated ``VramSlotPlan`` objects.
    """
    if len(vecs) != len(us):
        raise ValueError("vecs and us must have the same length")
    fc_state = {"x": 1.0, "y": 0.0}
    plans: List[VramSlotPlan] = []
    for vec, u in zip(vecs, us):
        W, fc_state, plan = hybrid_step(
            W,
            vec,
            u,
            fc_state,
            base_eta=base_eta,
            dt=dt,
            gain=gain,
            decay_x=decay_x,
            decay_y=decay_y,
        )
        plans.append(plan)
    return W, plans


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Simple reproducible demo
    rng = np.random.default_rng(1234)

    d_in = 6
    d_out = 6
    W, fc_state = hybrid_init(d_in, d_out=d_out, seed=42)

    # Generate a short random series
    vec_series = [rng.random(d_in) for _ in range(5)]
    u_series = [float(rng.uniform(0.5, 2.0)) for _ in range(5)]

    W_final, plans = hybrid_series(
        W,
        vec_series,
        u_series,
        base_eta=0.02,
        dt=0.5,
        gain=1.5,
        decay_x=0.9,
        decay_y=0.8,
    )

    print("Initial weight norm :", np.linalg.norm(W))
    print("Final weight norm   :", np.linalg.norm(W_final))
    print("\nGenerated VramSlotPlans:")
    for i, p in enumerate(plans, 1):
        print(f"Step {i}: {p.as_dict()}")