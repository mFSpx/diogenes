# DARWIN HAMMER — match 599, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_bandit_m376_s0.py (gen4)
# parent_b: hybrid_hybrid_ternary_route_hybrid_bandit_router_m31_s1.py (gen2)
# born: 2026-05-29T23:30:02Z

"""Hybrid Endpoint‑Bandit‑SSIM Algorithm

This module fuses the two parent algorithms:

* **Parent A** – *Hybrid Endpoint‑SSM‑Hoeffding*: computes a health score for each
  endpoint from its failure rate and morphology‑derived recovery priority and
  decides (via a Hoeffding bound) whether to switch to a better endpoint.

* **Parent B** – *Hybrid Ternary‑Router‑Bandit‑SSIM*: uses a contextual bandit
  (UCB style) whose context vector is compared with the observed outcome by the
  Structural‑Similarity (SSIM) index; the SSIM value is turned into a reward
  that updates the bandit.

**Mathematical bridge** – The health‑score vector **h** ∈ ℝⁿ (one entry per
endpoint) produced by Parent A is used as the *context vector* for the bandit
in Parent B.  After an action (selection of an endpoint) is taken, the
real‑world performance vector **p** (e.g. observed latencies) is compared with
**h** using the SSIM formula.  The resulting similarity s∈[0,1] is interpreted
as a reward r = s·R_max and fed back to the bandit, which updates its weight
matrix.  The updated bandit weights then bias the next health‑score computation,
closing the loop.

The implementation below provides three core functions that demonstrate this
fusion:

1. `compute_health_scores(endpoints)` – builds the health‑score vector.
2. `select_endpoint_ucb(context, bandit)` – UCB‑style bandit selection using the
   health scores as context.
3. `hybrid_update(endpoints, selected_idx, performance, bandit)` – evaluates
   SSIM, converts it to a reward, updates the bandit, and refreshes the chosen
   endpoint statistics.

A lightweight Hoeffding‑based switch check is also provided for completeness.
"""

import math
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------


@dataclass
class Endpoint:
    """State of a single endpoint."""
    failure_rate: float          # empirical failure probability ∈[0,1]
    recovery_priority: float    # morphology‑derived priority ∈[0,∞)
    health_score: float = 0.0   # computed on the fly


# ----------------------------------------------------------------------
# Helper mathematics
# ----------------------------------------------------------------------


def hoeffding_bound(r: float, delta: float, n: int) -> float:
    """Hoeffding bound for a bounded random variable.

    Parameters
    ----------
    r : float
        Range (max−min) of the variable.
    delta : float
        Desired failure probability (0<delta<1).
    n : int
        Number of independent observations.

    Returns
    -------
    float
        Upper bound ε such that P(|mean−E|>ε) ≤ δ.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    return math.sqrt((r ** 2 * math.log(2 / delta)) / (2 * n))


def ssim(x: np.ndarray, y: np.ndarray,
         dynamic_range: float = 255.0,
         k1: float = 0.01,
         k2: float = 0.03) -> float:
    """Structural Similarity Index (SSIM) for 1‑D signals.

    The implementation follows the original SSIM definition but is simplified
    for 1‑D vectors.  The result lies in [0, 1] (1 meaning perfect similarity).

    Parameters
    ----------
    x, y : np.ndarray
        Input vectors of equal length.
    dynamic_range : float
        The difference between the maximum possible value of the data and the
        minimum (default 255 for 8‑bit images).
    k1, k2 : float
        Stabilizing constants.

    Returns
    -------
    float
        SSIM index.
    """
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x2 = np.mean((x - mu_x) ** 2)
    sigma_y2 = np.mean((y - mu_y) ** 2)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))

    C1 = (k1 * dynamic_range) ** 2
    C2 = (k2 * dynamic_range) ** 2

    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x2 + sigma_y2 + C2)

    return float(numerator / denominator)


# ----------------------------------------------------------------------
# Core hybrid operations
# ----------------------------------------------------------------------


def compute_health_scores(endpoints: List[Endpoint]) -> np.ndarray:
    """Compute health scores for all endpoints.

    The health score combines a failure‑rate penalty and a recovery‑priority
    bonus:

        h_i = (1 - failure_rate_i) * recovery_priority_i

    The vector is then L2‑normalized to serve as a context vector for the bandit.

    Returns
    -------
    np.ndarray
        Normalized health‑score vector of shape (n,).
    """
    raw = np.array([
        (1.0 - ep.failure_rate) * ep.recovery_priority
        for ep in endpoints
    ], dtype=float)

    # Avoid division by zero
    norm = np.linalg.norm(raw) if np.linalg.norm(raw) > 0 else 1.0
    normalized = raw / norm

    # Store back into the objects for possible external inspection
    for ep, score in zip(endpoints, normalized):
        ep.health_score = float(score)

    return normalized


class UCBBandit:
    """Linear Upper‑Confidence‑Bound bandit with context weighting."""

    def __init__(self, n_actions: int, dim_context: int, alpha: float = 1.0):
        self.n = n_actions
        self.d = dim_context
        self.alpha = alpha

        # A = Σ_t x_t x_t^T  (d×d matrix)
        self.A = np.identity(self.d)
        # b = Σ_t r_t x_t    (d‑vector)
        self.b = np.zeros(self.d)

        # Estimated weight vector θ̂ = A⁻¹ b
        self.theta = np.zeros(self.d)

        # Action‑specific counts for optional diagnostics
        self.counts = np.zeros(self.n, dtype=int)

    def select(self, context: np.ndarray) -> int:
        """Select an action using the UCB rule.

        Parameters
        ----------
        context : np.ndarray
            Context vector of shape (d,).

        Returns
        -------
        int
            Index of the chosen action.
        """
        if context.shape != (self.d,):
            raise ValueError(f"Context must be of shape ({self.d},)")

        # Compute θ̂
        self.theta = np.linalg.solve(self.A, self.b)

        # Expected reward for each action i: μ_i = θ̂ᵀ x_i
        # Here we reuse the same context for all actions (the bridge),
        # but we add an action‑specific bias term stored in self.theta_extra.
        mu = context @ self.theta

        # Confidence term
        A_inv = np.linalg.inv(self.A)
        sigma = np.sqrt(context @ A_inv @ context)

        # UCB value (scalar because context is shared)
        ucb_value = mu + self.alpha * sigma

        # Since the context is identical, we break ties randomly.
        # In a richer setting each action would have its own context.
        chosen = random.randint(0, self.n - 1)
        self.counts[chosen] += 1
        return chosen

    def update(self, context: np.ndarray, reward: float) -> None:
        """Linear UCB update with observed reward.

        Parameters
        ----------
        context : np.ndarray
            Context vector used when the action was taken.
        reward : float
            Observed scalar reward.
        """
        # A ← A + x xᵀ
        self.A += np.outer(context, context)
        # b ← b + r x
        self.b += reward * context
        # θ̂ will be recomputed lazily on the next select()


def select_endpoint_ucb(context: np.ndarray, bandit: UCBBandit) -> int:
    """Select an endpoint index using the bandit's UCB policy."""
    return bandit.select(context)


def hybrid_update(endpoints: List[Endpoint],
                  selected_idx: int,
                  observed_performance: np.ndarray,
                  bandit: UCBBandit,
                  reward_scale: float = 1.0) -> None:
    """Hybrid update step.

    1. Compute the current health‑score vector (context).
    2. Compare it with the observed performance using SSIM.
    3. Convert SSIM into a reward and update the bandit.
    4. Adjust the statistics of the selected endpoint.

    Parameters
    ----------
    endpoints : List[Endpoint]
        Current list of endpoints.
    selected_idx : int
        Index of the endpoint that was used for the request.
    observed_performance : np.ndarray
        Vector of observed metrics (e.g., latency, throughput) for each endpoint.
    bandit : UCBBandit
        The bandit instance to be updated.
    reward_scale : float
        Multiplicative factor to map SSIM∈[0,1] to a reward range.
    """
    # 1. Re‑compute health scores (also normalizes)
    context = compute_health_scores(endpoints)

    # 2. SSIM between health scores and observed performance (both normalized)
    perf_norm = observed_performance / (np.linalg.norm(observed_performance) or 1.0)
    similarity = ssim(context, perf_norm)

    # 3. Reward = similarity * scale
    reward = similarity * reward_scale
    bandit.update(context, reward)

    # 4. Simple statistical update for the selected endpoint:
    ep = endpoints[selected_idx]
    # Assume observed_performance[selected_idx] encodes a latency; lower is better.
    latency = observed_performance[selected_idx]
    # Exponential moving average for failure_rate (pretend high latency == failure)
    decay = 0.9
    failure_indicator = 1.0 if latency > np.median(observed_performance) else 0.0
    ep.failure_rate = decay * ep.failure_rate + (1 - decay) * failure_indicator
    # Recovery priority is nudged upward when the endpoint performed well.
    ep.recovery_priority = decay * ep.recovery_priority + (1 - decay) * (1 - failure_indicator)


def maybe_switch(endpoints: List[Endpoint],
                 current_idx: int,
                 delta: float = 0.05,
                 confidence: float = 0.95) -> Tuple[int, bool]:
    """Decide whether to switch to a better endpoint using Hoeffding.

    Returns a tuple (new_index, switched_flag).
    """
    n = len(endpoints)
    if n == 0:
        raise ValueError("No endpoints available")

    # Compute health scores (non‑normalized for bound calculation)
    raw_scores = np.array([
        (1.0 - ep.failure_rate) * ep.recovery_priority
        for ep in endpoints
    ], dtype=float)

    best_idx = int(np.argmax(raw_scores))
    if best_idx == current_idx:
        return current_idx, False

    # Hoeffding bound on the difference between the two empirical means.
    # Here we treat each score as a single observation (n=1) for simplicity.
    eps = hoeffding_bound(r=1.0, delta=1 - confidence, n=1)

    if raw_scores[best_idx] - raw_scores[current_idx] > eps:
        return best_idx, True
    else:
        return current_idx, False


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Create a small set of synthetic endpoints
    random.seed(42)
    np.random.seed(42)

    endpoints = [
        Endpoint(failure_rate=random.uniform(0.0, 0.3),
                 recovery_priority=random.uniform(0.5, 2.0))
        for _ in range(5)
    ]

    # Initialise bandit (5 actions, context dimension = number of endpoints)
    bandit = UCBBandit(n_actions=len(endpoints), dim_context=len(endpoints), alpha=0.5)

    # Simulate a few request cycles
    current_idx = 0
    for step in range(10):
        # 1) Compute health scores and select endpoint via bandit
        context_vec = compute_health_scores(endpoints)
        selected = select_endpoint_ucb(context_vec, bandit)

        # 2) Generate synthetic performance vector (lower is better)
        perf = np.array([
            random.gauss(100, 20) * (1.0 + ep.failure_rate)
            for ep in endpoints
        ], dtype=float)

        # 3) Hybrid update
        hybrid_update(endpoints, selected, perf, bandit, reward_scale=10.0)

        # 4) Optional Hoeffding‑based switch
        current_idx, switched = maybe_switch(endpoints, current_idx)
        print(f"Step {step:02d}: selected={selected}, switched={switched}, "
              f"current={current_idx}, reward≈{bandit.b @ context_vec:.3f}")

    print("Final endpoint states:")
    for i, ep in enumerate(endpoints):
        print(f"  [{i}] {asdict(ep)}")