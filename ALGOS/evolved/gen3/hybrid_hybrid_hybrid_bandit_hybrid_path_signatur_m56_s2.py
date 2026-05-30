# DARWIN HAMMER — match 56, survivor 2
# gen: 3
# parent_a: hybrid_hybrid_bandit_router_workshare_allocator_m60_s0.py (gen2)
# parent_b: hybrid_path_signature_kan_m30_s0.py (gen1)
# born: 2026-05-29T23:25:30Z

"""Hybrid Bandit‑Router / Workshare Allocator + Path‑Signature‑KAN Fusion

Parents:
- hybrid_hybrid_bandit_router_workshare_allocator_m60_s0.py
- hybrid_path_signature_kan_m30_s0.py

Mathematical Bridge
-------------------
The bridge is the *store dance* signal produced by the bandit‑router’s
store dynamics.  In the path‑signature/KAN side the iterated‑integral
approximation is obtained by projecting a lead‑lag transformed path onto
a B‑spline basis.  We treat the resulting basis coefficients as a
vector of “flows”.  These flows become the `inflow`/`outflow` arguments
of the store update equation:

    Δ = α·Σ(inflow) – β·Σ(outflow)  
    level_{t+1} = max(0, level_t + Δ·dt)

The scalar `dance = tanh(gain·Δ)` is then fed back to the bandit
policy: it linearly rescales the raw propensities, allowing the
signature‑derived dynamics to modulate the stochastic action selection.
Thus the two parent topologies are fused into a single closed‑loop
system.

The module provides three core functions demonstrating this hybrid
operation:
1. `lead_lag_bspline_signature` – compute B‑spline‑projected signature.
2. `store_update_from_signature` – update the honeybee store using the
   signature coefficients.
3. `adjust_bandit_propensities` – rescale bandit propensities with the
   store’s dance signal.

A tiny smoke test at the bottom exercises the full pipeline."""

import numpy as np
import math
import random
import sys
import pathlib
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

# ----------------------------------------------------------------------
# Data structures from Parent A
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class BanditAction:
    """Result of an action selection."""
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str


@dataclass(frozen=True)
class BanditUpdate:
    """Single observation used to update the policy."""
    context_id: str
    action_id: str
    reward: float
    propensity: float


@dataclass
class StoreState:
    """Honeybee‑style store and derived control signal."""
    level: float = 0.0
    alpha: float = 1.0
    beta: float = 1.0
    dt: float = 1.0
    base: float = 1.0
    gain: float = 1.0
    limit: float = 10.0
    _last_delta: float = 0.0   # internal use

    def update(self, inflow: List[float], outflow: List[float]) -> Tuple[float, float]:
        """Apply the store equation and store the most recent Δ."""
        delta = self.alpha * sum(inflow) - self.beta * sum(outflow)
        self._last_delta = delta
        self.level = max(0.0, self.level + self.dt * delta)
        # clamp to limit for numerical safety
        self.level = min(self.level, self.limit)
        return self.level, delta

    @property
    def dance(self) -> float:
        """Bounded control signal derived from the last Δ."""
        # A smooth bounded non‑linearity; tanh keeps it in (‑1,1)
        return math.tanh(self.gain * self._last_delta)


# ----------------------------------------------------------------------
# Functions from Parent B (path‑signature + KAN)
# ----------------------------------------------------------------------

def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    """
    Lead‑lag transform: interleave (lead, lag) channels for causality encoding.
    Parameters
    ----------
    path : (T, d) array
        Original trajectory.

    Returns
    -------
    out : (2T‑1, 2d) array
        Lead‑lag representation.
    """
    path = np.asarray(path, dtype=float)
    if path.ndim != 2:
        raise ValueError("path must be a 2‑D array (T, d)")
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t] = np.concatenate([path[t], path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out


def bspline_basis(x: np.ndarray, grid: np.ndarray, k: int = 3) -> np.ndarray:
    """
    Evaluate B‑spline basis of order k on a grid.
    Parameters
    ----------
    x : (N,) array
        Evaluation points.
    grid : (M,) array
        Knot positions (must be sorted).
    k : int, default 3
        Spline order (cubic => k=3).

    Returns
    -------
    B : (N, M + k - 2) array
        Basis matrix.
    """
    x = np.asarray(x, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)
    if grid.ndim != 1:
        raise ValueError("grid must be one‑dimensional")
    # Augmented knot vector with clamped ends
    t = np.concatenate([
        np.full(k - 1, grid[0]),
        grid,
        np.full(k - 1, grid[-1]),
    ])
    n_basis = len(grid) + k - 2
    N = x.shape[0]

    # Zeroth order (piecewise constant) basis
    B = np.zeros((N, len(t) - 1), dtype=np.float64)
    for i in range(len(t) - 1):
        B[:, i] = np.where((x >= t[i]) & (x < t[i + 1]), 1.0, 0.0)
    # Ensure the right‑most knot is inclusive
    B[:, -1] = np.where(x == t[-1], 1.0, B[:, -1])

    # Recurrence for higher orders
    for order in range(2, k + 1):
        B_new = np.zeros((N, len(t) - order), dtype=np.float64)
        for i in range(len(t) - order):
            denom_l = t[i + order - 1] - t[i]
            denom_r = t[i + order] - t[i + 1]
            term_l = ((x - t[i]) / denom_l) * B[:, i] if denom_l > 0 else np.zeros(N)
            term_r = ((t[i + order] - x) / denom_r) * B[:, i + 1] if denom_r > 0 else np.zeros(N)
            B_new[:, i] = term_l + term_r
        B = B_new
    # Trim possible numerical excess columns
    return B[:, :n_basis]


def lead_lag_bspline_signature(path: np.ndarray,
                               grid: np.ndarray,
                               spline_order: int = 3) -> np.ndarray:
    """
    Compute a low‑dimensional approximation of the path signature by
    projecting the lead‑lag transformed path onto a B‑spline basis.

    Returns
    -------
    coeffs : (M + k - 2, 2d) array
        Each row corresponds to a basis function; columns are the
        projected lead‑lag channels.
    """
    lead_lag = lead_lag_transform(path)                # (2T‑1, 2d)
    # Use the time axis of the lead‑lag path as the evaluation points
    T_ll = lead_lag.shape[0]
    time = np.linspace(0, 1, T_ll)                    # normalized time
    B = bspline_basis(time, grid, k=spline_order)    # (T_ll, n_basis)
    # Project each channel separately
    coeffs = B.T @ lead_lag                            # (n_basis, 2d)
    return coeffs


# ----------------------------------------------------------------------
# Hybrid Core Functions
# ----------------------------------------------------------------------

def store_update_from_signature(store: StoreState,
                                signature_coeffs: np.ndarray) -> Tuple[float, float]:
    """
    Convert signature coefficients into inflow/outflow streams and update
    the honeybee store.

    The first half of the coefficient matrix is interpreted as inflow,
    the second half as outflow.  If the dimensionality is odd, the extra
    column is discarded for simplicity.

    Returns
    -------
    level, delta : float
        Updated store level and the raw Δ used to compute the dance.
    """
    # Ensure a 2‑D array
    coeffs = np.asarray(signature_coeffs, dtype=float)
    if coeffs.ndim != 2:
        raise ValueError("signature_coeffs must be 2‑D")
    n_basis, n_chan = coeffs.shape
    split = n_chan // 2
    inflow = coeffs[:, :split].flatten().tolist()
    outflow = coeffs[:, split:].flatten().tolist()
    level, delta = store.update(inflow, outflow)
    return level, delta


def adjust_bandit_propensities(actions: List[BanditAction],
                               dance: float,
                               scale: float = 0.5) -> List[BanditAction]:
    """
    Rescale the raw propensities of a list of BanditAction objects using
    the store’s dance signal.

    The new propensity is:
        p_new = p_raw * (1 + scale * dance)

    The function returns new BanditAction instances (dataclasses are frozen).
    """
    factor = 1.0 + scale * dance
    adjusted = []
    for a in actions:
        new_propensity = max(0.0, a.propensity * factor)
        adjusted.append(
            BanditAction(
                action_id=a.action_id,
                propensity=new_propensity,
                expected_reward=a.expected_reward,
                confidence_bound=a.confidence_bound,
                algorithm=a.algorithm,
            )
        )
    return adjusted


def hybrid_step(path: np.ndarray,
                actions: List[BanditAction],
                store: StoreState,
                grid: np.ndarray,
                spline_order: int = 3) -> Tuple[StoreState, List[BanditAction]]:
    """
    Execute one hybrid iteration:
      1. Compute a B‑spline‑projected signature from the path.
      2. Update the store using the signature coefficients.
      3. Adjust bandit propensities with the resulting dance signal.

    Returns
    -------
    updated_store : StoreState
    adjusted_actions : List[BanditAction]
    """
    sig = lead_lag_bspline_signature(path, grid, spline_order)
    _, _ = store_update_from_signature(store, sig)   # updates in‑place
    adjusted = adjust_bandit_propensities(actions, store.dance)
    return store, adjusted


# ----------------------------------------------------------------------
# Simple deterministic allocation (illustrative, not part of the original
# parents but useful to showcase the hybrid dynamics).
# ----------------------------------------------------------------------

def deterministic_target_percentage(store: StoreState,
                                   base_pct: float = 0.5,
                                   amp: float = 0.3) -> float:
    """
    Compute a target allocation percentage that varies with the store
    dance signal.  Result is clipped to [0,1].

    pct = base_pct + amp * tanh(gain * Δ)
    """
    pct = base_pct + amp * store.dance
    return max(0.0, min(1.0, pct))


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic path: 5 time steps, 2‑dimensional
    np.random.seed(42)
    path = np.cumsum(np.random.randn(5, 2), axis=0)

    # B‑spline knot grid (uniform)
    grid = np.linspace(0, 1, 6)   # 6 interior knots

    # Initialise store
    store = StoreState(level=2.0, alpha=0.8, beta=0.5, gain=2.0, limit=15.0)

    # Dummy bandit actions
    actions = [
        BanditAction(action_id="A", propensity=0.4, expected_reward=1.0,
                     confidence_bound=0.1, algorithm="hybrid"),
        BanditAction(action_id="B", propensity=0.6, expected_reward=0.8,
                     confidence_bound=0.2, algorithm="hybrid"),
    ]

    # Run hybrid iteration
    updated_store, adjusted_actions = hybrid_step(
        path=path,
        actions=actions,
        store=store,
        grid=grid,
        spline_order=3,
    )

    # Display results
    print("Updated store level :", updated_store.level)
    print("Store dance signal   :", updated_store.dance)
    print("Deterministic target %:", deterministic_target_percentage(updated_store))
    for a in adjusted_actions:
        print(f"Action {a.action_id}: new propensity = {a.propensity:.4f}")

    # Verify that no NaNs appear
    assert not np.isnan(updated_store.level)
    assert not np.isnan(updated_store.dance)
    for a in adjusted_actions:
        assert not np.isnan(a.propensity)
    print("Smoke test completed successfully.")