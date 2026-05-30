# DARWIN HAMMER — match 4453, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_bandit_m1221_s4.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_worksh_m1380_s1.py (gen4)
# born: 2026-05-29T23:55:58Z

import numpy as np
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional

# ----------------------------------------------------------------------
# Core mathematical models
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class SchoolfieldParams:
    """Parameters of the full Schoolfield temperature model."""
    rho_25: float = 1.0                     # rate at 25 °C (reference)
    delta_h_activation: float = 12_000.0   # activation enthalpy (J mol⁻¹)
    t_low: float = 283.15                   # low‑temperature threshold (K)
    t_high: float = 307.15                  # high‑temperature threshold (K)
    delta_h_low: float = -45_000.0          # low‑temperature enthalpy (J mol⁻¹)
    delta_h_high: float = 65_000.0          # high‑temperature enthalpy (J mol⁻¹)
    r_cal: float = 1.987                    # gas constant (cal mol⁻¹ K⁻¹)

def c_to_k(celsius: float) -> float:
    """Convert Celsius to Kelvin."""
    return celsius + 273.15

def schoolfield_rate(temp_k: float, p: SchoolfieldParams) -> float:
    """
    Full Schoolfield temperature‑dependent developmental rate.

    r(T) = ρ25 * exp[-ΔH_A/(R) * (1/T - 1/T_ref)]
           / (1 + exp[ΔH_L/(R) * (1/T_L - 1/T)])
           / (1 + exp[ΔH_H/(R) * (1/T - 1/T_H)])

    The reference temperature T_ref is 298.15 K (25 °C).
    """
    if temp_k <= 0:
        raise ValueError("Temperature in Kelvin must be > 0.")
    if p.rho_25 < 0:
        raise ValueError("rho_25 must be non‑negative.")

    t_ref = 298.15  # reference temperature (K)

    # activation term (Arrhenius)
    act = math.exp(
        -p.delta_h_activation / (p.r_cal * 4.184) *
        (1.0 / temp_k - 1.0 / t_ref)
    )

    # low‑temperature inhibition term
    low = 1.0 + math.exp(
        p.delta_h_low / (p.r_cal * 4.184) *
        (1.0 / p.t_low - 1.0 / temp_k)
    )

    # high‑temperature inhibition term
    high = 1.0 + math.exp(
        p.delta_h_high / (p.r_cal * 4.184) *
        (1.0 / temp_k - 1.0 / p.t_high)
    )

    return p.rho_25 * act / (low * high)

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )

# ----------------------------------------------------------------------
# HybridLogLog distinct‑token estimator (lightweight)
# ----------------------------------------------------------------------
class HybridLogLog:
    """
    Very small HybridLogLog sketch for estimating the number of distinct
    activation patterns (tokens).  The implementation follows the classic
    LogLog algorithm with a simple bias correction.
    """
    def __init__(self, b: int = 5):
        if b < 4 or b > 16:
            raise ValueError("b must be in [4, 16]")
        self.b = b
        self.m = 1 << b
        self.registers = np.zeros(self.m, dtype=np.uint8)

    def _hash(self, value: bytes) -> int:
        # Use built‑in hash for speed; mask to 64‑bit.
        return hash(value) & ((1 << 64) - 1)

    def add(self, value: str) -> None:
        h = self._hash(value.encode("utf‑8"))
        idx = h >> (64 - self.b)
        w = (h << self.b) & ((1 << 64) - 1)
        rank = self._clz(w) + 1
        if rank > self.registers[idx]:
            self.registers[idx] = rank

    @staticmethod
    def _clz(x: int) -> int:
        """Count leading zeros in a 64‑bit integer."""
        return (64 - x.bit_length()) if x != 0 else 64

    def estimate(self) -> float:
        """Return the cardinality estimate."""
        alpha = 0.7213 / (1 + 1.079 / self.m)
        Z = 1.0 / np.sum(2.0 ** -self.registers)
        raw = alpha * self.m ** 2 * Z
        # Small‑range correction
        V = np.count_nonzero(self.registers == 0)
        if V > 0:
            return self.m * math.log(self.m / V)
        return raw

# ----------------------------------------------------------------------
# Bandit router with temperature‑aware reward shaping
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    algorithm: str

@dataclass
class BanditState:
    """Tracks cumulative reward and pulls for each action."""
    reward_sum: float = 0.0
    pulls: int = 0

class TemperatureBanditRouter:
    """
    Contextual bandit whose reward is scaled by the Schoolfield developmental
    rate.  Uses Upper Confidence Bound (UCB1) for exploration.
    """
    def __init__(self, actions: Iterable[BanditAction], c: float = 2.0):
        self.actions = list(actions)
        self.c = c
        self._stats: Dict[str, BanditState] = {a.action_id: BanditState() for a in self.actions}
        self.total_pulls = 0

    def select_action(self) -> BanditAction:
        """Select action using UCB1."""
        if self.total_pulls < len(self.actions):
            # Ensure each action is tried at least once
            return self.actions[self.total_pulls]

        ucb_values = {}
        for a in self.actions:
            s = self._stats[a.action_id]
            avg = s.reward_sum / s.pulls if s.pulls > 0 else 0.0
            bonus = self.c * math.sqrt(math.log(self.total_pulls) / s.pulls)
            ucb_values[a] = avg + bonus

        return max(ucb_values, key=ucb_values.get)

    def update(self, action_id: str, reward: float, temp_rate: float) -> None:
        """
        Incorporate observed reward.  The raw reward is multiplied by the
        temperature‑dependent developmental rate to bias learning toward
        biologically plausible regimes.
        """
        if action_id not in self._stats:
            raise KeyError(f"Unknown action_id {action_id}")

        scaled_reward = reward * temp_rate
        state = self._stats[action_id]
        state.reward_sum += scaled_reward
        state.pulls += 1
        self.total_pulls += 1

    def get_estimates(self) -> Dict[str, float]:
        """Return average scaled reward per action."""
        return {
            aid: (st.reward_sum / st.pulls if st.pulls > 0 else 0.0)
            for aid, st in self._stats.items()
        }

# ----------------------------------------------------------------------
# Workshare allocation that fuses bandit propensities and token diversity
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class WorkshareParams:
    total_units: float
    deterministic_target_frac: float  # 0.0 – 1.0
    groups: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self):
        if self.total_units <= 0:
            raise ValueError("total_units must be positive")
        if not (0.0 <= self.deterministic_target_frac <= 1.0):
            raise ValueError("deterministic_target_frac must be in [0, 1]")
        if not self.groups:
            raise ValueError("At least one group must be specified")

def allocate_workshare(
    params: WorkshareParams,
    dev_rate: float,
    bandit_estimates: Dict[str, float],
    token_estimator: HybridLogLog,
) -> Dict[str, float]:
    """
    Allocate workshare units per group.

    1. Compute a deterministic base (fraction of total_units) and modulate it
       by the developmental rate.
    2. Distribute the deterministic part proportionally to the bandit UCB
       estimates (higher estimated reward → more units).
    3. Add a stochastic component proportional to the estimated number of
       distinct tokens, encouraging exploration of under‑used groups.
    """
    # 1. Deterministic, temperature‑scaled core
    deterministic_units = params.total_units * params.deterministic_target_frac * dev_rate
    deterministic_units = max(deterministic_units, 0.0)

    # 2. Proportional split using bandit estimates
    total_est = sum(bandit_estimates.values()) or 1.0
    prop_weights = {
        gid: bandit_estimates.get(gid, 0.0) / total_est
        for gid in params.groups
    }

    deterministic_alloc = {
        gid: deterministic_units * prop_weights.get(gid, 0.0)
        for gid in params.groups
    }

    # 3. Stochastic token‑diversity boost
    distinct_est = token_estimator.estimate()
    stochastic_pool = params.total_units * (1.0 - params.deterministic_target_frac)
    stochastic_pool = max(stochastic_pool, 0.0)

    # Allocate stochastic pool uniformly but perturb by a small random factor
    # scaled by the token diversity (more tokens → larger perturbation)
    epsilon = min(0.2, 0.01 * math.log1p(distinct_est))
    stochastic_alloc = {}
    for gid in params.groups:
        base = stochastic_pool / len(params.groups)
        jitter = base * epsilon * (random.random() - 0.5) * 2  # ±epsilon*base
        stochastic_alloc[gid] = max(base + jitter, 0.0)

    # Combine deterministic and stochastic parts
    final_alloc = {
        gid: deterministic_alloc.get(gid, 0.0) + stochastic_alloc.get(gid, 0.0)
        for gid in params.groups
    }

    # Normalise to total_units (tiny rounding errors may accumulate)
    total_assigned = sum(final_alloc.values())
    if total_assigned > 0:
        scale = params.total_units / total_assigned
        for gid in final_alloc:
            final_alloc[gid] *= scale

    return final_alloc

# ----------------------------------------------------------------------
# High‑level hybrid operation
# ----------------------------------------------------------------------
def hybrid_operation(
    schoolfield_params: SchoolfieldParams,
    workshare_params: WorkshareParams,
    temp_c: float,
    bandit_router: TemperatureBanditRouter,
    token_estimator: HybridLogLog,
) -> Dict[str, float]:
    """
    Execute one hybrid step:
    1. Compute temperature‑dependent developmental rate.
    2. Choose an action via the bandit router (action_id must match a group name).
    3. Simulate a raw reward (placeholder: random normal) and update the router.
    4. Allocate workshare using the updated bandit estimates and token diversity.
    """
    temp_k = c_to_k(temp_c)
    dev_rate = schoolfield_rate(temp_k, schoolfield_params)

    # 2. Bandit selection
    chosen_action = bandit_router.select_action()
    # placeholder reward – in real systems this would be a measured performance metric
    raw_reward = random.gauss(mu=1.0, sigma=0.2)

    # 3. Update bandit with temperature‑scaled reward
    bandit_router.update(chosen_action.action_id, raw_reward, dev_rate)

    # 4. Feed distinct‑token estimator (example: add chosen action id)
    token_estimator.add(chosen_action.action_id)

    # 5. Allocate workshare
    allocation = allocate_workshare(
        workshare_params,
        dev_rate,
        bandit_router.get_estimates(),
        token_estimator,
    )
    return allocation

# ----------------------------------------------------------------------
# Example driver (executed when run as script)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Initialise models
    schoolfield_params = SchoolfieldParams()
    workshare_params = WorkshareParams(
        total_units=100.0,
        deterministic_target_frac=0.9,
        groups=("codex", "groq", "cohere", "local_models")
    )
    actions = [
        BanditAction(action_id="codex", algorithm="A"),
        BanditAction(action_id="groq", algorithm="B"),
        BanditAction(action_id="cohere", algorithm="C"),
        BanditAction(action_id="local_models", algorithm="D"),
    ]
    bandit_router = TemperatureBanditRouter(actions)
    token_estimator = HybridLogLog(b=6)

    # Run a few iterations to demonstrate adaptation
    for step in range(15):
        temp_c = 20.0 + 5.0 * math.sin(step * 0.4)  # slowly varying temperature
        alloc = hybrid_operation(
            schoolfield_params,
            workshare_params,
            temp_c,
            bandit_router,
            token_estimator,
        )
        print(f"Step {step:02d} | Temp °C={temp_c:.2f} | Allocation={alloc}")

    # Final bandit statistics
    print("\nFinal bandit reward estimates:")
    for aid, val in bandit_router.get_estimates().items():
        print(f"  {aid}: {val:.4f}")

    print(f"\nEstimated distinct tokens: {token_estimator.estimate():.2f}")