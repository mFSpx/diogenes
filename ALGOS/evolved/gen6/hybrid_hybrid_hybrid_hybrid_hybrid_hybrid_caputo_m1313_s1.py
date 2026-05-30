# DARWIN HAMMER — match 1313, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_regret_hybrid_hybrid_hybrid_m142_s1.py (gen4)
# parent_b: hybrid_hybrid_caputo_fracti_hybrid_hybrid_infota_m618_s0.py (gen5)
# born: 2026-05-29T23:35:08Z

"""Hybrid Algorithm Fusion of:
- hybrid_hybrid_hybrid_regret_hybrid_hybrid_hybrid_m142_s1.py (Parent A)
- hybrid_hybrid_caputo_fracti_hybrid_hybrid_infota_m618_s0.py (Parent B)

Mathematical Bridge
-------------------
Parent A provides a dynamical “store” governed by a first‑order differential
equation (Δ = α·inflow − β·outflow) whose state produces a bounded control
signal called *dance*.  
Parent B supplies a MinHash‑based Jaccard similarity estimator together with a
Caputo fractional derivative that yields a weighted memory of past similarity
values.

The fusion treats the similarity series as a discrete signal *s(t)*.  A
Caputo‑fractional derivative of order α (0 < α ≤ 1) is applied to *s(t)*,
producing a memory‑augmented gradient *g(t)*.  This gradient modulates the
store’s inflow, i.e. inflow = g(t)·base_inflow, while outflow is kept as in
Parent A.  Consequently the store’s dance signal now reflects both the
instantaneous similarity and its fractional history, creating a unified
adaptive allocation mechanism.

The module implements:
1. MinHash signature & similarity (Parent B).
2. Caputo fractional derivative approximation (new).
3. Store dynamics with similarity‑driven inflow (Parent A + bridge).
4. Action selection that combines propensity, store dance, and similarity.

All components are pure Python with only the standard library and NumPy.
"""

import sys
import random
import math
import pathlib
from dataclasses import dataclass, field
from typing import List, Tuple, Iterable, Dict
import numpy as np
import hashlib

# ----------------------------------------------------------------------
# Data structures from Parent A
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class HybridAction:
    """Result of an action selection."""
    id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True)
class HybridUpdate:
    """Single observation used to update the policy."""
    context_id: str
    action_id: str
    reward: float
    propensity: float


@dataclass
class StoreState:
    """Honeybee‑style store with a bounded control signal."""
    level: float = 0.0
    alpha: float = 1.0          # inflow coefficient
    beta: float = 1.0           # outflow coefficient
    dt: float = 1.0
    base: float = 1.0           # base of the dance signal
    gain: float = 1.0           # gain applied to Δ in the dance
    limit: float = 10.0         # max dance value

    def update(self, inflow: List[float], outflow: List[float]) -> Tuple[float, float]:
        """Apply the store equation and recompute the dance duration."""
        delta = self.alpha * sum(inflow) - self.beta * sum(outflow)
        self.level = max(0.0, self.level + self.dt * delta)
        self._store_last_delta(delta)
        return self.level, delta

    @property
    def dance(self) -> float:
        """Bounded control signal derived from the last Δ."""
        delta = getattr(self, "_last_delta", 0.0)
        return max(0.0, min(self.limit, self.base + self.gain * delta))

    def _store_last_delta(self, delta: float) -> None:
        setattr(self, "_last_delta", delta)


# ----------------------------------------------------------------------
# MinHash utilities from Parent B
# ----------------------------------------------------------------------
def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def signature(tokens: Iterable[str], k: int = 128) -> List[int]:
    """Return a MinHash signature of length *k* for the given token set."""
    toks: set[str] = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [2**64 - 1] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]


def similarity(sig_a: List[int], sig_b: List[int]) -> float:
    """Approximate Jaccard similarity via MinHash signatures."""
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have equal length")
    if not sig_a:
        raise ValueError("signatures must not be empty")
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / len(sig_a)


# ----------------------------------------------------------------------
# Caputo fractional derivative approximation (new)
# ----------------------------------------------------------------------
def _caputo_weights(alpha: float, n: int) -> np.ndarray:
    """
    Compute the Caputo weights w_k = (-1)^k * C(alpha, k) for k = 0..n‑1,
    where C(alpha, k) = Γ(alpha+1) / (Γ(k+1)Γ(alpha‑k+1)).
    """
    if not (0 < alpha <= 1):
        raise ValueError("alpha must be in (0, 1]")
    w = np.empty(n, dtype=float)
    gamma = math.gamma
    for k in range(n):
        binom = gamma(alpha + 1) / (gamma(k + 1) * gamma(alpha - k + 1))
        w[k] = (-1) ** k * binom
    return w


def caputo_derivative(series: List[float], alpha: float, dt: float = 1.0) -> float:
    """
    Approximate the Caputo fractional derivative of order *alpha* at the
    last point of *series* using the Grunwald‑Letnikov discretisation.

    Parameters
    ----------
    series : List[float]
        Discrete signal values f(t_0), …, f(t_n).
    alpha : float
        Fractional order (0 < alpha ≤ 1).
    dt : float
        Time step between consecutive samples.

    Returns
    -------
    float
        Approximation of D^α f(t_n).
    """
    n = len(series)
    if n < 2:
        return 0.0
    w = _caputo_weights(alpha, n)
    # The Caputo derivative uses the differences f(t_n‑k) – f(t_n‑k‑1)
    diffs = np.array([series[n - k - 1] - series[n - k - 2] for k in range(1, n)])
    # prepend the first difference (which is zero for k=0) to align lengths
    diffs = np.concatenate(([0.0], diffs))
    derivative = (1 / dt ** alpha) * np.dot(w, diffs)
    return derivative


# ----------------------------------------------------------------------
# Hybrid core functions (bridge between A and B)
# ----------------------------------------------------------------------
def compute_similarity_series(
    token_sets: List[Iterable[str]],
    k: int = 128,
) -> List[float]:
    """
    Given a chronological list of token sets, compute the MinHash similarity
    between each consecutive pair, producing a similarity time series.
    """
    if len(token_sets) < 2:
        return []
    sig_prev = signature(token_sets[0], k=k)
    series = []
    for tokens in token_sets[1:]:
        sig_cur = signature(tokens, k=k)
        sim = similarity(sig_prev, sig_cur)
        series.append(sim)
        sig_prev = sig_cur
    return series


def hybrid_store_update(
    store: StoreState,
    similarity_series: List[float],
    alpha_frac: float,
    base_inflow: float = 1.0,
) -> Tuple[float, float]:
    """
    Update the store where the inflow is driven by the Caputo‑fractional
    derivative of the similarity series.

    Parameters
    ----------
    store : StoreState
        The dynamical store to be updated.
    similarity_series : List[float]
        Historical similarity values (most recent last).
    alpha_frac : float
        Fractional order for the Caputo derivative (0 < α ≤ 1).
    base_inflow : float
        Baseline inflow magnitude before modulation.

    Returns
    -------
    Tuple[float, float]
        (new store level, Δ) after the update.
    """
    # Compute fractional gradient; if not enough history, fall back to plain diff
    if len(similarity_series) >= 2:
        grad = caputo_derivative(similarity_series, alpha_frac, dt=store.dt)
    else:
        grad = similarity_series[-1] if similarity_series else 0.0

    # Positive gradient drives inflow, negative drives outflow
    inflow = [base_inflow * max(0.0, grad)]
    outflow = [base_inflow * max(0.0, -grad)]

    level, delta = store.update(inflow, outflow)
    return level, delta


def select_hybrid_action(
    actions: List[HybridAction],
    store: StoreState,
    similarity_score: float,
) -> HybridAction:
    """
    Choose an action by combining:
    - Propensity (exploration weight)
    - Store dance signal (resource availability)
    - Current similarity score (information relevance)

    The combined score is:
        score = propensity * (1 + store.dance) * (1 + similarity_score)

    Returns the action with the highest score.
    """
    if not actions:
        raise ValueError("action list cannot be empty")
    best = None
    best_score = -math.inf
    for act in actions:
        score = act.propensity * (1.0 + store.dance) * (1.0 + similarity_score)
        if score > best_score:
            best_score = score
            best = act
    return best


# ----------------------------------------------------------------------
# Minimal pheromone system (placeholder from Parent A)
# ----------------------------------------------------------------------
class HybridPheromoneSystem:
    """Simple pheromone tracker used only for demonstration."""
    def __init__(self):
        self.pheromones: Dict[str, float] = {}

    def deposit(self, key: str, amount: float) -> None:
        self.pheromones[key] = self.pheromones.get(key, 0.0) + amount

    def evaporate(self, rate: float = 0.01) -> None:
        for k in list(self.pheromones.keys()):
            self.pheromones[k] *= (1.0 - rate)
            if self.pheromones[k] < 1e-6:
                del self.pheromones[k]

    def strength(self, key: str) -> float:
        return self.pheromones.get(key, 0.0)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a synthetic token history (e.g., document word sets)
    token_history = [
        ["alpha", "beta", "gamma"],
        ["beta", "delta", "epsilon"],
        ["gamma", "delta", "zeta"],
        ["eta", "theta", "iota"],
    ]

    # Compute similarity series
    sim_series = compute_similarity_series(token_history, k=64)
    print("Similarity series:", sim_series)

    # Initialise store and pheromone system
    store = StoreState(alpha=0.8, beta=0.5, dt=1.0, base=1.0, gain=2.0, limit=15.0)
    pher = HybridPheromoneSystem()

    # Perform a few update steps
    for step in range(3):
        # Use the similarity series up to current step+1
        current_series = sim_series[: step + 1]
        level, delta = hybrid_store_update(store, current_series, alpha_frac=0.6, base_inflow=1.0)
        print(f"Step {step}: level={level:.3f}, delta={delta:.3f}, dance={store.dance:.3f}")

        # Deposit pheromone proportional to dance
        pher.deposit(f"step_{step}", store.dance)

    # Define dummy actions
    actions = [
        HybridAction(id="A", propensity=0.9, expected_reward=1.0,
                     confidence_bound=0.2, algorithm="hybrid", expected_value=0.8),
        HybridAction(id="B", propensity=0.5, expected_reward=0.5,
                     confidence_bound=0.1, algorithm="hybrid", expected_value=0.4),
        HybridAction(id="C", propensity=0.7, expected_reward=0.8,
                     confidence_bound=0.15, algorithm="hybrid", expected_value=0.6),
    ]

    # Use the most recent similarity as the relevance signal
    recent_sim = sim_series[-1] if sim_series else 0.0
    chosen = select_hybrid_action(actions, store, recent_sim)
    print("Chosen action:", chosen.id, "with propensity", chosen.propensity)

    # Show pheromone levels
    print("Pheromone strengths:", {k: round(v, 3) for k, v in pher.pheromones.items()})