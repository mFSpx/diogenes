# DARWIN HAMMER — match 1200, survivor 4
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_regret_engine_m8_s3.py (gen4)
# parent_b: hybrid_hybrid_bandit_router_workshare_allocator_m60_s0.py (gen2)
# born: 2026-05-29T23:33:29Z

import numpy as np
import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any


# ----------------------------------------------------------------------
# Core data structures
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
    """Honey‑bee style store with a lazily computed control signal."""

    level: float = 0.0
    alpha: float = 1.0
    beta: float = 1.0
    dt: float = 1.0
    base: float = 1.0
    gain: float = 1.0
    limit: float = 10.0
    _last_delta: float = field(default=0.0, init=False, repr=False)

    def update(self, inflow: List[float], outflow: List[float]) -> Tuple[float, float]:
        """
        Apply the store differential equation and cache the most recent Δ.

        Returns
        -------
        new_level, delta
        """
        delta = self.alpha * sum(inflow) - self.beta * sum(outflow)
        self._last_delta = delta
        self.level = max(0.0, min(self.limit, self.level + self.dt * delta))
        return self.level, delta

    @property
    def dance(self) -> float:
        """
        Bounded control signal derived from the most recent Δ.
        A sigmoid squashes the raw delta into (0, 1) and is then scaled
        by the current level to keep the signal proportional to store size.
        """
        # Sigmoid on the last delta to obtain a smooth, bounded factor.
        sigmoid = 1.0 / (1.0 + math.exp(-self._last_delta))
        return sigmoid * self.level


# ----------------------------------------------------------------------
# Regret‑engine utilities (tropical / max‑plus style)
# ----------------------------------------------------------------------
def compute_health_scores(endpoints: List[Dict[str, Any]]) -> np.ndarray:
    """
    Extract health scores from a list of endpoint dictionaries.
    """
    return np.array([float(ep.get("health_score", 0.0)) for ep in endpoints], dtype=float)


def tropical_regret_gains(health_scores: np.ndarray, actions: List[Dict[str, Any]]) -> np.ndarray:
    """
    Compute a tropical (max‑plus) gain for each action.
    The gain is the difference between the best health score and the
    intrinsic cost, clipped at zero to avoid negative gains.
    """
    best = np.max(health_scores)
    costs = np.array([float(a.get("intrinsic_cost", 0.0)) for a in actions], dtype=float)
    gains = np.maximum(best - costs, 0.0)
    return gains


# ----------------------------------------------------------------------
# Statistics & splitting logic
# ----------------------------------------------------------------------
def _gini_coefficient(values: np.ndarray) -> float:
    """
    Compute the Gini coefficient for a 1‑D array.
    """
    if values.size == 0:
        return 0.0
    sorted_vals = np.sort(values)
    n = values.size
    cumulative = np.cumsum(sorted_vals, dtype=float)
    gini = (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n
    return float(gini)


def update_stats_and_maybe_split(
    gains: np.ndarray,
    stats: Dict[str, float],
    delta: float,
    gini_thr: float,
    confidence: float = 0.95,
) -> bool:
    """
    Update Hoeffding‑type statistics and decide whether a split is warranted.

    Parameters
    ----------
    gains
        Current gain vector.
    stats
        Dictionary holding running statistics.
    delta
        Recent change in store level (used as a proxy for observation variance).
    gini_thr
        Threshold below which the distribution is considered homogeneous enough to split.
    confidence
        Desired confidence level for the Hoeffding bound (default 95 %).

    Returns
    -------
    split_decision : bool
        True if conditions for a split are met.
    """
    # Update a simple exponential moving average of the absolute delta.
    ema_alpha = 0.2
    stats["ema_delta"] = ema_alpha * abs(delta) + (1 - ema_alpha) * stats.get("ema_delta", 0.0)

    # Hoeffding bound for bounded [0, 1] observations.
    n = max(1, stats.get("observations", 1))
    bound = math.sqrt(math.log(2.0 / (1 - confidence)) / (2 * n))
    stats["hoeffding_bound"] = bound
    stats["observations"] = n + 1

    # Compute Gini coefficient on gains.
    stats["gini_coefficient"] = _gini_coefficient(gains)

    # Decision rule: low variance (bound) and sufficiently homogeneous gains.
    return stats["hoeffding_bound"] < stats["ema_delta"] and stats["gini_coefficient"] < gini_thr


# ----------------------------------------------------------------------
# Bandit router – integrates store state and health scores
# ----------------------------------------------------------------------
def _softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Numerically stable softmax."""
    z = x / max(temperature, 1e-8)
    e = np.exp(z - np.max(z))
    return e / e.sum()


def bandit_router(store_state: StoreState, health_scores: np.ndarray) -> BanditAction:
    """
    Select an action using a softmax over health scores, modulated by the
    store's dance signal (propensity). The confidence bound is derived from
    the current Hoeffding bound (if present) or defaults to 1.0.
    """
    # Convert health scores to a probability distribution.
    probs = _softmax(health_scores, temperature=0.5)

    # Sample according to the distribution (deterministic argmax is a special case).
    action_index = np.argmax(probs)

    # Propensity is the dance signal scaled to [0, 1].
    propensity = min(1.0, store_state.dance / store_state.limit)

    expected_reward = float(health_scores[action_index])
    confidence_bound = 1.0  # placeholder – could be linked to stats in a full system.

    return BanditAction(
        action_id=str(action_index),
        propensity=propensity,
        expected_reward=expected_reward,
        confidence_bound=confidence_bound,
        algorithm="bandit_router",
    )


# ----------------------------------------------------------------------
# Workshare allocator – deeper coupling of gains and store dynamics
# ----------------------------------------------------------------------
def workshare_allocator(store_state: StoreState, gains: np.ndarray) -> np.ndarray:
    """
    Allocate workshare using a convex combination of normalized gains and the
    store's current level. This yields a richer signal than a pure gain‑based
    allocation.

    Returns
    -------
    allocation : np.ndarray
        Fractions that sum to 1.0 (or all zeros if gains are all zero).
    """
    if gains.size == 0:
        return np.array([])

    # Normalized gains (avoid division by zero).
    gain_sum = gains.sum()
    norm_gains = gains / gain_sum if gain_sum > 0 else np.full_like(gains, 1.0 / gains.size)

    # Store influence: higher level pushes more work to higher‑gain actions.
    store_factor = store_state.level / store_state.limit  # in [0, 1]
    allocation = (1 - store_factor) * norm_gains + store_factor * norm_gains ** 2
    allocation /= allocation.sum()  # re‑normalize to guarantee sum == 1
    return allocation


# ----------------------------------------------------------------------
# High‑level hybrid step (optional convenience wrapper)
# ----------------------------------------------------------------------
def hybrid_step(
    store_state: StoreState,
    endpoints: List[Dict[str, Any]],
    actions: List[Dict[str, Any]],
    inflow: List[float],
    outflow: List[float],
    stats: Dict[str, float],
    delta: float,
    gini_thr: float,
) -> Tuple[BanditAction, np.ndarray, bool]:
    """
    Execute one iteration of the hybrid algorithm:
    1. Update the store.
    2. Compute health scores and tropical gains.
    3. Possibly trigger a split.
    4. Select an action via the bandit router.
    5. Allocate workshare.

    Returns
    -------
    action, allocation, split_decision
    """
    # 1. Store dynamics
    _, recent_delta = store_state.update(inflow, outflow)

    # 2. Regret‑engine calculations
    health_scores = compute_health_scores(endpoints)
    gains = tropical_regret_gains(health_scores, actions)

    # 3. Splitting logic
    split = update_stats_and_maybe_split(gains, stats, recent_delta, gini_thr)

    # 4. Decision making
    action = bandit_router(store_state, health_scores)

    # 5. Allocation
    allocation = workshare_allocator(store_state, gains)

    return action, allocation, split


# ----------------------------------------------------------------------
# Demo / sanity‑check execution
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Example data
    endpoints = [
        {"health_score": 0.5},
        {"health_score": 0.7},
        {"health_score": 0.3},
    ]
    actions = [
        {"intrinsic_cost": 0.1},
        {"intrinsic_cost": 0.2},
        {"intrinsic_cost": 0.3},
    ]

    # Initialise components
    store_state = StoreState()
    stats = {"hoeffding_bound": 0.0, "gini_coefficient": 0.0, "observations": 0}
    inflow = [0.2, 0.1]
    outflow = [0.05]
    delta = 0.1
    gini_thr = 0.4

    # Run a single hybrid step
    action, allocation, split = hybrid_step(
        store_state,
        endpoints,
        actions,
        inflow,
        outflow,
        stats,
        delta,
        gini_thr,
    )

    print("Selected action:", action)
    print("Workshare allocation:", allocation)
    print("Split triggered:", split)