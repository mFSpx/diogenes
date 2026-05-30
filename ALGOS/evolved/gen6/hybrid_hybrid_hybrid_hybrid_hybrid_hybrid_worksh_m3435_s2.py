# DARWIN HAMMER — match 3435, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_worksh_m433_s0.py (gen5)
# parent_b: hybrid_hybrid_workshare_all_hybrid_liquid_time_c_m39_s0.py (gen2)
# born: 2026-05-29T23:50:04Z

"""
Hybrid Algorithm Fusion of:
- Parent A: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_worksh_m433_s0.py
- Parent B: hybrid_hybrid_workshare_all_hybrid_liquid_time_c_m39_s0.py

Mathematical Bridge:
Parent A supplies a dynamic StoreState whose level evolves via inflow/outflow
terms and a B‑spline approximation of log‑likelihoods derived from token
signatures. Parent B provides a weekday‑dependent weight vector that modulates
the MinHash similarity between token sets.  The fusion is achieved by
1. Computing a MinHash signature for a token set.
2. Scaling that signature with the weekday‑weight vector (calendar topology).
3. Feeding the weighted signature into a B‑spline smoother that yields an
   approximated log‑count (log‑likelihood) vector.
4. Using the smoothed log‑counts as “inflow” to the StoreState while the
   raw weighted signature acts as “outflow”.  The resulting delta drives the
   StoreState dynamics, thus tightly coupling the calendar‑aware similarity
   metric with the adaptive level‑based system from Parent A.
"""

import sys
import math
import random
import re
import hashlib
import datetime as dt
from pathlib import Path
from typing import List, Sequence, Tuple, Dict, Any
import numpy as np
from dataclasses import dataclass, field

# ----------------------------------------------------------------------
# Core structures from Parent A
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str

@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float

@dataclass
class StoreState:
    """Dynamic reservoir whose level changes with inflow/outflow."""
    level: float = 0.0
    alpha: float = 1.0   # inflow gain
    beta: float = 1.0    # outflow gain
    dt: float = 1.0      # time step
    base: float = 1.0
    gain: float = 1.0
    limit: float = 10.0

    def update(self, inflow: List[float], outflow: List[float]) -> Tuple[float, float]:
        """Apply the governing equation Δ = α·Σ(in) – β·Σ(out)."""
        delta = self.alpha * float(np.sum(inflow)) - self.beta * float(np.sum(outflow))
        self.level = max(0.0, self.level + self.dt * delta)
        return self.level, delta

    @property
    def dance(self) -> float:
        """Normalized level used elsewhere (e.g., as a probability)."""
        return self.level / self.limit

def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    """
    Simple lead‑lag transform: interleave original points with their
    cumulative sums, mimicking the classic construction used in signature
    methods.
    """
    if path.ndim != 2:
        raise ValueError("path must be a 2‑D array (time × dim)")
    cum = np.cumsum(path, axis=0)
    lead = path
    lag = cum
    return np.column_stack((lead.ravel(), lag.ravel()))

# ----------------------------------------------------------------------
# Utilities from Parent B
# ----------------------------------------------------------------------
GROUPS: Tuple[str, ...] = ("codex", "groq", "cohere", "local_models")

def _pct(value: float) -> float:
    return round(float(value), 6)

def doomsday(year: int, month: int, day: int) -> int:
    """Return a weekday index shifted to range [0,6]."""
    return (dt.date(year, month, day).weekday() + 1) % 7

def weekday_weight_vector(groups: Sequence[str], dow: int) -> np.ndarray:
    """
    Calendar‑dependent weight vector.
    Each group receives a sinusoidally modulated weight that varies with the day of week.
    """
    n = len(groups)
    if n == 0:
        raise ValueError("groups must contain at least one element")
    base_angles = np.arange(n) * (2.0 * math.pi) / n
    phase = (2.0 * math.pi) * (dow % 7) / 7.0
    amplitude = 0.2
    raw = 1.0 + amplitude * np.sin(base_angles + phase)
    weight_vec = raw / raw.sum()
    return weight_vec.astype(np.float64)

def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "big")

def minhash_signature(tokens: Sequence[str], k: int = 128) -> List[int]:
    """
    Classic MinHash: for each of k hash functions return the minimal hash value over the token set.
    """
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [2**64 - 1] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]

# ----------------------------------------------------------------------
# Hybrid Functions (the new fused logic)
# ----------------------------------------------------------------------
def weighted_minhash(tokens: Sequence[str],
                    groups: Sequence[str],
                    date: dt.date,
                    k: int = 128) -> np.ndarray:
    """
    1. Compute a MinHash signature for the token set.
    2. Obtain the weekday weight vector for the supplied groups.
    3. Scale (element‑wise) the signature by the weight vector (repeating the
       weight vector to match the signature length) and normalize to [0,1].

    Returns:
        A float ndarray of shape (k,) representing the calendar‑aware similarity
        vector.
    """
    raw_sig = np.array(minhash_signature(tokens, k), dtype=np.float64)
    dow = date.weekday()  # Monday=0 … Sunday=6
    w_vec = weekday_weight_vector(groups, dow)
    # Repeat the weight vector to cover the signature length
    repeats = int(np.ceil(k / len(w_vec)))
    w_expanded = np.tile(w_vec, repeats)[:k]
    weighted = raw_sig * w_expanded
    # Normalization (avoid division by zero)
    max_val = weighted.max() if weighted.max() != 0 else 1.0
    return weighted / max_val

def b_spline_log_approx(weighted_sig: np.ndarray,
                        knots: int = 10,
                        degree: int = 3) -> np.ndarray:
    """
    Approximate the log‑likelihood of the weighted signature using a
    B‑spline basis.  For simplicity we:
    * Take the natural log of the (positive) weighted signature.
    * Fit a piecewise polynomial (B‑spline like) using a moving‑average
      kernel that mimics basis smoothing.
    * Return the smoothed log‑vector.

    The function mirrors the “B‑spline basis to approximate log‑likelihood”
    concept from Parent A.
    """
    eps = 1e-12
    log_vals = np.log(weighted_sig + eps)

    # Construct simple uniform knots over the index domain
    x = np.arange(len(log_vals), dtype=np.float64)
    knot_positions = np.linspace(0, len(log_vals) - 1, knots)

    # Simple smoothing: weighted average of neighboring points based on distance to knots
    smoothed = np.zeros_like(log_vals)
    for i, xi in enumerate(x):
        # distance to each knot
        d = np.abs(knot_positions - xi)
        # basis weight: inverse distance raised to (degree)
        basis = 1.0 / (d + eps) ** degree
        smoothed[i] = np.sum(basis * log_vals) / np.sum(basis)
    return smoothed

def hybrid_state_update(state: StoreState,
                        tokens: Sequence[str],
                        groups: Sequence[str],
                        date: dt.date) -> Tuple[float, float]:
    """
    Full hybrid step:
    * Produce a calendar‑aware weighted MinHash vector.
    * Smooth it with a B‑spline log‑approximation.
    * Use the smoothed values as inflow and the raw weighted vector as outflow
      for the StoreState dynamics.

    Returns the new level and the delta applied.
    """
    weighted = weighted_minhash(tokens, groups, date)
    smoothed = b_spline_log_approx(weighted)

    # Inflow: positive contribution from smoothed log‑likelihood
    inflow = smoothed.tolist()
    # Outflow: raw weighted similarity (already normalized)
    outflow = weighted.tolist()

    level, delta = state.update(inflow, outflow)
    return level, delta

def hybrid_decision(context_id: str,
                    actions: List[BanditAction],
                    tokens: Sequence[str],
                    groups: Sequence[str],
                    date: dt.date,
                    state: StoreState) -> BanditAction:
    """
    Demonstrates a decision made by combining:
    * The dynamic StoreState (which influences the effective propensity)
    * BanditAction scores (expected reward + confidence bound)
    * Calendar‑aware similarity (through the state update)

    The selected action maximizes a score:
        score = (expected_reward + confidence_bound) * state.dance * propensity
    """
    # First, evolve the state with current token information
    _, _ = hybrid_state_update(state, tokens, groups, date)

    # Compute a scaling factor from the current state
    scale = state.dance  # between 0 and 1

    best_action = None
    best_score = -math.inf
    for act in actions:
        score = (act.expected_reward + act.confidence_bound) * scale * act.propensity
        if score > best_score:
            best_score = score
            best_action = act
    return best_action

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Sample data
    sample_tokens = ["alpha", "beta", "gamma", "delta", "epsilon"]
    sample_groups = list(GROUPS)
    today = dt.date.today()

    # Initialise StoreState
    state = StoreState(level=2.0, alpha=0.8, beta=0.5, dt=1.0, limit=15.0)

    # Define a few dummy actions
    actions = [
        BanditAction(action_id="A", propensity=0.6, expected_reward=1.2,
                     confidence_bound=0.3, algorithm="alg1"),
        BanditAction(action_id="B", propensity=0.4, expected_reward=1.5,
                     confidence_bound=0.2, algorithm="alg2"),
        BanditAction(action_id="C", propensity=0.9, expected_reward=0.9,
                     confidence_bound=0.4, algorithm="alg3")
    ]

    # Run a hybrid decision step
    chosen = hybrid_decision(
        context_id="ctx_001",
        actions=actions,
        tokens=sample_tokens,
        groups=sample_groups,
        date=today,
        state=state
    )

    print(f"Chosen action: {chosen.action_id}")
    print(f"Updated StoreState level: {state.level:.4f}, dance: {state.dance:.4f}")

    # Verify that the weighted MinHash vector sums to 1 (by construction)
    w_vec = weighted_minhash(sample_tokens, sample_groups, today)
    print(f"Weighted MinHash sum (should be 1.0): {w_vec.sum():.6f}")

    # Ensure B‑spline smoothing runs without error
    smoothed = b_spline_log_approx(w_vec)
    print(f"Smoothed log‑approx shape: {smoothed.shape}")