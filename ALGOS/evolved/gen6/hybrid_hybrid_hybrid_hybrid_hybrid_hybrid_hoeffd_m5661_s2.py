# DARWIN HAMMER — match 5661, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_worksh_hybrid_caputo_fracti_m46_s0.py (gen3)
# parent_b: hybrid_hybrid_hoeffding_tre_hybrid_hybrid_hybrid_m1710_s1.py (gen5)
# born: 2026-05-30T00:03:56Z

"""
Hybrid Allocation‑LTC, Fractional‑Memory Tree Cost, Hoeffding‑Risk & Pheromone Decay

Parents:
- hybrid_hybrid_hybrid_worksh_hybrid_caputo_fracti_m46_s0.py (LTC + Caputo fractional tree)
- hybrid_hybrid_hoeffding_tre_hybrid_hybrid_hybrid_m1710_s1.py (Hoeffding bound risk + pheromone decay)

Mathematical Bridge
-------------------
Both parents contain a *time‑dependent weighting* mechanism:
    • LTC supplies an effective time‑constant τ_sys(t) that modulates allocation.
    • Pheromone entries decay with a factor ½^{Δt/half_life}.
These are unified by defining a joint weight

        w(t) = τ_sys(t) · ½^{Δt/half_life} .

The Hoeffding bound provides a risk score r ∈ [0,1] that attenuates the
pheromone signal, while the Caputo fractional weights α_i weight historic
contributions in the tree‑cost sum.  The hybrid cost for an edge e at time t is

        C_hybrid(e,t) = w(t) · (1‑r) · Σ_i α_i · d_i(e)

where d_i(e) is the incremental distance contributed by edge e at the i‑th
historical step.  This single expression fuses the core topologies of the two
parents into a unified system.
"""

import math
import random
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Iterable

import numpy as np

# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str
    vram_mb: int

TIER_T1_QWEN_0_5B = ModelTier("qwen-0.5b", 512, "T1", 1024)
TIER_T2_REASONING = ModelTier("reasoning-t2", 3000, "T2", 2048)
TIER_T2_TOOL = ModelTier("tool-t2", 2600, "T2", 2048)
TIER_T3_QWEN_7B = ModelTier("qwen-7b", 7000, "T3", 4096)


class PheromoneEntry:
    """Simple pheromone container with exponential decay."""
    __slots__ = ("uuid", "surface_key", "signal_kind", "signal_value",
                 "half_life_seconds", "created_at", "last_decay")

    def __init__(self, surface_key: str, signal_kind: str,
                 signal_value: float, half_life_seconds: int):
        self.uuid = str(random.getrandbits(128))
        self.surface_key = surface_key
        self.signal_kind = signal_kind
        self.signal_value = signal_value
        self.half_life_seconds = half_life_seconds
        self.created_at = datetime.now(timezone.utc)
        self.last_decay = self.created_at

    def decay_factor(self, now: datetime) -> float:
        """Return 0.5^{Δt/half_life}."""
        elapsed = (now - self.last_decay).total_seconds()
        if self.half_life_seconds <= 0:
            return 1.0
        return 0.5 ** (elapsed / self.half_life_seconds)

    def apply_decay(self, now: datetime) -> None:
        factor = self.decay_factor(now)
        self.signal_value *= factor
        self.last_decay = now


# ---------------------------------------------------------------------------
# LTC (Liquid Time‑Constant) utilities – from Parent A
# ---------------------------------------------------------------------------

def init_ltc(day_of_week: int, base_tau: float = 1.0) -> Tuple[float, float]:
    """
    Initialise the effective time constant τ_sys for a given day of week.
    The schedule is sinusoidal to mimic weekly cycles.

    Returns (tau, phase) where phase = 2π * day/7.
    """
    phase = 2.0 * math.pi * (day_of_week % 7) / 7.0
    tau = base_tau * (1.0 + 0.3 * math.sin(phase))
    return tau, phase


def effective_tau(tau: float, phase: float, t: float) -> float:
    """
    Time‑varying τ_sys(t) = τ * (1 + 0.1·cos(t + phase)).
    """
    return tau * (1.0 + 0.1 * math.cos(t + phase))


# ---------------------------------------------------------------------------
# Hoeffding bound – from Parent B
# ---------------------------------------------------------------------------

def hoeffding_risk(mean: float, range_width: float, n: int,
                   delta: float = 0.05) -> float:
    """
    Compute a normalized risk score r ∈ [0,1] using the Hoeffding bound.
    The bound ε = sqrt(R^2 * ln(1/δ) / (2n)).
    Risk is defined as min(1, ε / (range_width/2)).
    """
    if n <= 0:
        return 1.0
    epsilon = math.sqrt((range_width ** 2) * math.log(1.0 / delta) / (2.0 * n))
    risk = min(1.0, epsilon / (range_width / 2.0))
    return risk


# ---------------------------------------------------------------------------
# Caputo fractional weights – from Parent A
# ---------------------------------------------------------------------------

def caputo_weights(times: List[float], alpha: float) -> List[float]:
    """
    Compute Caputo fractional weights α_i for a monotonic time list.
    w_i = (t_i - t_{i-1})^{1-α} / Γ(2-α)
    """
    if not times:
        return []
    from math import gamma
    denom = gamma(2.0 - alpha)
    weights = [0.0]  # first interval has no history
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        w = (dt ** (1.0 - alpha)) / denom
        weights.append(w)
    return weights


# ---------------------------------------------------------------------------
# Hybrid core functions
# ---------------------------------------------------------------------------

def hybrid_allocate_by_dates(dates: List[datetime],
                             groups: List[str],
                             base_tau: float = 1.0,
                             half_life_seconds: int = 86400) -> Dict[datetime, Dict[str, float]]:
    """
    Allocate a synthetic resource per day per group.
    Allocation = LTC‑modulated share * pheromone decay factor * (1‑risk).

    Returns a nested dict: {date: {group: allocation}}.
    """
    # initialise a single pheromone entry that will be shared across days
    pheromone = PheromoneEntry(surface_key="global",
                               signal_kind="allocation",
                               signal_value=1.0,
                               half_life_seconds=half_life_seconds)

    # placeholder statistics for risk (could be replaced by real observations)
    obs_mean = 0.5
    obs_range = 1.0
    obs_n = 100

    allocations: Dict[datetime, Dict[str, float]] = {}

    for dt in dates:
        day_idx = dt.weekday()  # Monday=0 … Sunday=6
        tau, phase = init_ltc(day_idx, base_tau)
        tau_t = effective_tau(tau, phase, dt.timestamp())

        # decay pheromone up to current time
        pheromone.apply_decay(dt)
        decay = pheromone.decay_factor(dt)

        # risk based on synthetic observations (could be dynamic)
        risk = hoeffding_risk(obs_mean, obs_range, obs_n)

        # compute per‑group share (simple equal split)
        base_share = 1.0 / max(1, len(groups))

        for g in groups:
            alloc = base_share * tau_t * decay * (1.0 - risk)
            allocations.setdefault(dt, {})[g] = alloc

    return allocations


def incremental_fractional_tree_cost_with_pheromone(
        edge_list: List[Tuple[int, int, float]],
        timestamps: List[float],
        alpha: float = 0.7,
        delta: float = 0.05,
        base_tau: float = 1.0,
        half_life_seconds: int = 86400) -> float:
    """
    Build a tree incrementally, updating a distance matrix and evaluating the
    hybrid cost:

        C_hybrid = Σ_edge w(t)·(1‑r)·Σ_i α_i·Δd_i

    where w(t) = τ_sys(t)·decay_factor(t).

    Parameters
    ----------
    edge_list : list of (src, dst, length)
    timestamps : list of float timestamps corresponding to each edge addition
    alpha : fractional order for Caputo weighting
    delta : confidence for Hoeffding risk
    base_tau, half_life_seconds : LTC & pheromone parameters

    Returns
    -------
    total hybrid cost (float)
    """
    if len(edge_list) != len(timestamps):
        raise ValueError("edge_list and timestamps must be same length")

    # initialise distance matrix with infinities
    nodes = set()
    for src, dst, _ in edge_list:
        nodes.update([src, dst])
    n = max(nodes) + 1
    dist = np.full((n, n), np.inf)
    np.fill_diagonal(dist, 0.0)

    # initialise pheromone entry (global)
    pheromone = PheromoneEntry(surface_key="tree",
                               signal_kind="cost",
                               signal_value=1.0,
                               half_life_seconds=half_life_seconds)

    # placeholder observation stats for risk (could be updated online)
    obs_mean = 0.5
    obs_range = 1.0
    obs_n = 100

    # keep a history of incremental distance contributions for Caputo weighting
    incremental_contribs: List[float] = []
    times_history: List[float] = []

    total_cost = 0.0

    for (src, dst, length), t in zip(edge_list, timestamps):
        # update distance matrix (Floyd‑Warshall one‑step update)
        if length < dist[src, dst]:
            dist[src, dst] = dist[dst, src] = length
            for i in range(n):
                for j in range(n):
                    via = dist[i, src] + length + dist[dst, j]
                    if via < dist[i, j]:
                        dist[i, j] = via
                    via = dist[i, dst] + length + dist[src, j]
                    if via < dist[i, j]:
                        dist[i, j] = via

        # incremental contribution: sum of all‑pairs distances after this edge
        contrib = float(np.nansum(dist))
        incremental_contribs.append(contrib)
        times_history.append(t)

        # Caputo weights for the history up to now
        wts = caputo_weights(times_history, alpha)

        # weighted sum of contributions
        weighted_sum = sum(w * c for w, c in zip(wts, incremental_contribs))

        # compute LTC τ_sys(t)
        day_idx = datetime.fromtimestamp(t, tz=timezone.utc).weekday()
        tau, phase = init_ltc(day_idx, base_tau)
        tau_t = effective_tau(tau, phase, t)

        # pheromone decay up to current timestamp
        now_dt = datetime.fromtimestamp(t, tz=timezone.utc)
        pheromone.apply_decay(now_dt)
        decay = pheromone.decay_factor(now_dt)

        # risk via Hoeffding bound
        risk = hoeffding_risk(obs_mean, obs_range, obs_n, delta)

        # hybrid edge cost
        edge_cost = tau_t * decay * (1.0 - risk) * weighted_sum
        total_cost += edge_cost

    return total_cost


def fractional_ssm_step(state: np.ndarray,
                        input_vec: np.ndarray,
                        A: np.ndarray,
                        B: np.ndarray,
                        alpha: float,
                        timestamps: List[float],
                        current_time: float,
                        base_tau: float = 1.0,
                        half_life_seconds: int = 86400) -> np.ndarray:
    """
    State‑space update that incorporates:
        • LTC‑modulated dynamics (τ_sys)
        • Pheromone decay (as a multiplicative gain)
        • Caputo fractional memory (via a convolution over past inputs)

    The update rule is

        x_{t+1} = τ_sys(t)·γ(t)·(A·x_t + B·Σ_i α_i·u_{t-i})

    where γ(t) is the pheromone decay factor at time t.
    """
    # compute LTC factor
    day_idx = datetime.fromtimestamp(current_time, tz=timezone.utc).weekday()
    tau, phase = init_ltc(day_idx, base_tau)
    tau_t = effective_tau(tau, phase, current_time)

    # pheromone decay factor (global entry)
    pheromone = PheromoneEntry(surface_key="ssm",
                               signal_kind="gain",
                               signal_value=1.0,
                               half_life_seconds=half_life_seconds)
    now_dt = datetime.fromtimestamp(current_time, tz=timezone.utc)
    pheromone.apply_decay(now_dt)
    decay = pheromone.decay_factor(now_dt)

    # build fractional memory term
    if len(timestamps) == 0:
        mem_term = input_vec
    else:
        times = timestamps + [current_time]
        wts = caputo_weights(times, alpha)
        # align wts with inputs (assume past inputs stored externally)
        # For demonstration we reuse the current input_vec scaled by each weight
        mem_term = sum(w * input_vec for w in wts)

    # linear update
    next_state = tau_t * decay * (A @ state + B @ mem_term)
    return next_state


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 1. Hybrid allocation demo
    today = datetime.now(timezone.utc)
    dates = [today + timedelta(days=i) for i in range(7)]
    groups = ["alpha", "beta", "gamma"]
    alloc = hybrid_allocate_by_dates(dates, groups)
    print("Hybrid allocations (first day):")
    print(alloc[dates[0]])

    # 2. Incremental tree cost demo
    edges = [(0, 1, 1.2), (1, 2, 0.8), (0, 2, 2.5)]
    ts = [today.timestamp() + i * 3600 for i in range(len(edges))]
    cost = incremental_fractional_tree_cost_with_pheromone(edges, ts)
    print(f"Hybrid incremental tree cost: {cost:.4f}")

    # 3. Fractional SSM step demo
    state = np.zeros(3)
    inp = np.array([0.5, -0.2, 0.1])
    A = np.eye(3) * 0.9
    B = np.eye(3) * 0.1
    timestamps = [today.timestamp() - 7200, today.timestamp() - 3600]
    new_state = fractional_ssm_step(state, inp, A, B, alpha=0.6,
                                    timestamps=timestamps,
                                    current_time=today.timestamp())
    print("New state after fractional SSM step:", new_state)