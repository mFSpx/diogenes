# DARWIN HAMMER — match 2641, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_hoeffding_tre_hybrid_hybrid_hybrid_m379_s1.py (gen4)
# parent_b: honeybee_store.py (gen0)
# born: 2026-05-29T23:43:14Z

"""Hybrid Hoeffding–Tropical Store Controller

This module fuses two parent algorithms:

* **Parent A** – Hoeffding‑bound based split decision enriched with tropical
  (max‑plus) algebra for evaluating ReLU‑like decision surfaces.
* **Parent B** – A simple store feedback primitive that updates a scalar
  resource store from inflow/outflow streams and maps the store change to a
  “dance duration”.

**Mathematical bridge**

The store value `s` is interpreted as a one‑dimensional state vector and fed
into a tropical neural network.  The network’s max‑plus evaluation produces a
scalar “gain” `g(s)`.  This gain replaces the raw information‑gain estimates
used by the Hoeffding splitter.  Consequently the Hoeffding bound decides
whether to split a leaf **based on the resource‑aware gain** `g(s)`.  The
resource dynamics (update_store) and the split decision are coupled: a split
consumes a fixed amount of the store, while the store evolution influences
future split decisions through the tropical gain.

The implementation provides three core hybrid functions:

1. `tropical_network_eval` – max‑plus forward pass.
2. `compute_gain_from_store` – maps the store to a gain via the tropical net.
3. `hybrid_step` – updates the store, computes the gain, applies the
   Hoeffding split test, and accounts for split‑induced resource consumption.

All code relies only on the standard library and NumPy. """

import numpy as np
import math
import random
import sys
import pathlib
from dataclasses import dataclass

# ----------------------------------------------------------------------
# Parent A – Hoeffding bound utilities
# ----------------------------------------------------------------------
def hoeffding_bound(r: float, delta: float, n: int) -> float:
    """Return the Hoeffding bound ε = sqrt( (r² * ln(1/δ)) / (2n) )."""
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0 required")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))


@dataclass(frozen=True)
class SplitDecision:
    """Result of a Hoeffding split test."""
    should_split: bool
    epsilon: float
    gain_gap: float
    reason: str


def should_split(best_gain: float,
                 second_best_gain: float,
                 r: float,
                 delta: float,
                 n: int,
                 tie_threshold: float = 0.05) -> SplitDecision:
    """Perform the classic Hoeffding split test."""
    eps = hoeffding_bound(r, delta, n)
    gap = best_gain - second_best_gain
    split = gap > eps or eps < tie_threshold
    reason = ("gap_exceeds_bound"
              if gap > eps else
              ("tie_threshold" if eps < tie_threshold else "wait"))
    return SplitDecision(split, eps, gap, reason)


# ----------------------------------------------------------------------
# Tropical (max‑plus) algebra utilities – from Parent A
# ----------------------------------------------------------------------
def t_add(x, y):
    """Tropical addition (max)."""
    return np.maximum(x, y)


def t_mul(x, y):
    """Tropical multiplication (ordinary addition)."""
    return np.add(x, y)


def t_matmul(A, B):
    """
    Max‑plus matrix multiplication.
    Result_{i,k} = max_j (A_{i,j} + B_{j,k})
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    # broadcast to compute all pairwise sums
    return np.max(A[:, :, np.newaxis] + B[np.newaxis, :, :], axis=1)


def t_polyval(coeffs, x):
    """
    Tropical polynomial evaluation.
    coeffs[i] corresponds to the coefficient of x^{i}.
    """
    coeffs = np.asarray(coeffs, dtype=float)
    x = np.asarray(x, dtype=float)
    exponents = np.arange(len(coeffs), dtype=float)
    # term_i = coeff_i + i * x  (tropical multiplication of x^i)
    terms = coeffs.reshape((-1,) + (1,) * x.ndim) + exponents.reshape((-1,) + (1,) * x.ndim) * x
    return np.max(terms, axis=0)


def tropical_layer_forward(x: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Forward pass of a single tropical (max‑plus) layer.
    For each output i:
        y_i = max_j (W_{i,j} + x_j) + b_i
    """
    # W shape (out_dim, in_dim), x shape (in_dim,)
    max_part = np.max(W + x, axis=1)
    return max_part + b


def tropical_network_eval(x: np.ndarray, layers: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """
    Evaluate a feed‑forward tropical network.
    `layers` is a list of (W, b) tuples.
    """
    h = np.asarray(x, dtype=float)
    for W, b in layers:
        h = tropical_layer_forward(h, W, b)
        # optional ReLU‑like clipping (tropical nets are already piecewise linear)
        h = np.maximum(0.0, h)
    return h


# ----------------------------------------------------------------------
# Parent B – Store dynamics
# ----------------------------------------------------------------------
def update_store(store: float,
                 inflow: list[float],
                 outflow: list[float],
                 alpha: float = 1.0,
                 beta: float = 1.0,
                 dt: float = 1.0) -> tuple[float, float]:
    """
    Incrementally update a scalar store.
    Δ = α·Σ(inflow) − β·Σ(outflow)
    store_{t+1} = max(0, store_t + dt·Δ)
    Returns (new_store, Δ).
    """
    delta = alpha * sum(inflow) - beta * sum(outflow)
    new_store = max(0.0, store + dt * delta)
    return new_store, delta


def dance_duration(delta_store: float,
                   base: float = 1.0,
                   gain: float = 1.0,
                   limit: float = 10.0) -> float:
    """
    Map a store change to a duration (the “dance”).
    """
    return max(0.0, min(limit, base + gain * delta_store))


# ----------------------------------------------------------------------
# Hybrid functions (requirement: at least three)
# ----------------------------------------------------------------------
def compute_gain_from_store(store: float,
                            layers: list[tuple[np.ndarray, np.ndarray]]) -> float:
    """
    Encode the scalar `store` as a one‑dimensional vector,
    run it through a tropical network, and collapse the output to a
    single gain value (here we take the maximum component).
    """
    x = np.array([store], dtype=float)          # shape (1,)
    out = tropical_network_eval(x, layers)     # shape (out_dim,)
    gain = float(np.max(out))                  # scalar gain
    return gain


def hybrid_should_split(store: float,
                        layers: list[tuple[np.ndarray, np.ndarray]],
                        r: float,
                        delta: float,
                        n: int,
                        tie_threshold: float = 0.05) -> SplitDecision:
    """
    Perform a Hoeffding split test where the gains are derived from the
    tropical network evaluated on the current store.
    """
    best_gain = compute_gain_from_store(store, layers)
    # A simple proxy for the second best gain: a slight degradation.
    second_best_gain = best_gain * 0.9
    return should_split(best_gain, second_best_gain, r, delta, n, tie_threshold)


def hybrid_step(store: float,
                inflow: list[float],
                outflow: list[float],
                layers: list[tuple[np.ndarray, np.ndarray]],
                r: float,
                delta: float,
                n: int,
                split_cost: float = 1.0,
                alpha: float = 1.0,
                beta: float = 1.0,
                dt: float = 1.0) -> dict:
    """
    Execute one hybrid iteration:
      1. Update the resource store.
      2. Compute a tropical‑network gain.
      3. Test for a Hoeffding split.
      4. If a split occurs, deduct `split_cost` from the store.

    Returns a dictionary summarising the step.
    """
    # 1. Store dynamics
    new_store, delta_store = update_store(store, inflow, outflow, alpha, beta, dt)

    # 2. Gain from tropical net
    gain = compute_gain_from_store(new_store, layers)

    # 3. Split decision
    decision = hybrid_should_split(new_store, layers, r, delta, n)

    # 4. Apply split cost if needed
    final_store = new_store - split_cost if decision.should_split else new_store
    final_store = max(0.0, final_store)  # keep non‑negative

    # Optional side‑channel: duration of a "dance" reflecting the store change
    duration = dance_duration(delta_store)

    return {
        "pre_store": store,
        "post_store": new_store,
        "final_store": final_store,
        "delta_store": delta_store,
        "gain": gain,
        "split_decision": decision,
        "dance_duration": duration,
    }


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Construct a tiny tropical network: 1 → 3 → 1
    rng = np.random.default_rng(seed=42)
    W1 = rng.uniform(-1, 1, size=(3, 1))   # (out=3, in=1)
    b1 = rng.uniform(-0.5, 0.5, size=3)

    W2 = rng.uniform(-1, 1, size=(1, 3))   # (out=1, in=3)
    b2 = rng.uniform(-0.5, 0.5, size=1)

    layers = [(W1, b1), (W2, b2)]

    # Initial conditions
    store = 5.0
    inflow = [0.8, 0.3]
    outflow = [0.2]
    r = 1.0          # range of gain (assumed)
    delta = 0.05    # confidence
    n = 100         # number of examples seen at the leaf
    split_cost = 0.7

    result = hybrid_step(store,
                         inflow,
                         outflow,
                         layers,
                         r,
                         delta,
                         n,
                         split_cost=split_cost)

    # Print a concise summary
    print("Hybrid step result:")
    print(f"  pre_store   = {result['pre_store']:.3f}")
    print(f"  post_store  = {result['post_store']:.3f}")
    print(f"  final_store = {result['final_store']:.3f}")
    print(f"  delta_store = {result['delta_store']:.3f}")
    print(f"  gain        = {result['gain']:.3f}")
    print(f"  split?      = {result['split_decision'].should_split}")
    print(f"  reason      = {result['split_decision'].reason}")
    print(f"  dance_dur   = {result['dance_duration']:.3f}")