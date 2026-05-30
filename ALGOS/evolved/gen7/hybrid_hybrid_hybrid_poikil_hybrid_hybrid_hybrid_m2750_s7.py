# DARWIN HAMMER — match 2750, survivor 7
# gen: 7
# parent_a: hybrid_hybrid_poikilotherm__hybrid_regret_engine_m1595_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_ternar_m2473_s2.py (gen6)
# born: 2026-05-29T23:45:48Z

import math
import random
import sys
import pathlib
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Sequence, Iterable
import numpy as np

# ----------------------------------------------------------------------
# Data structures (derived from both parents)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0
    delta_h_activation: float = 12_000.0      # J·mol⁻¹
    t_low: float = 283.15                     # K
    t_high: float = 307.15                    # K
    delta_h_low: float = -45_000.0            # J·mol⁻¹
    delta_h_high: float = 65_000.0            # J·mol⁻¹
    r_cal: float = 1.987                      # cal·mol⁻¹·K⁻¹ (converted internally)

@dataclass(frozen=True)
class MathAction:
    """Action definition used for regret weighting."""
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0

@dataclass(frozen=True)
class BanditAction:
    """Bandit‑style action used by the ternary router."""
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "hybrid"

@dataclass
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float

Point = Tuple[float, float]
Edge = Tuple[str, str]

# ----------------------------------------------------------------------
# Core mathematical primitives
# ----------------------------------------------------------------------
def schoolfield_rate(temp_k: float, params: SchoolfieldParams) -> float:
    """
    Compute the temperature‑dependent rate using the Schoolfield‑Rollinson
    formulation.  Temperature is given in Kelvin.
    """
    R = params.r_cal * 4.184  # convert cal·mol⁻¹·K⁻¹ to J·mol⁻¹·K⁻¹
    # numerator: Arrhenius term at reference temperature (25 °C = 298.15 K)
    num = params.rho_25 * math.exp(
        -params.delta_h_activation / (R * 298.15)
    )
    # denominator: temperature‑dependent modifier
    denom = (1.0 +
             math.exp(params.delta_h_low / (R * params.t_low)) +
             math.exp(params.delta_h_high / (R * params.t_high)))
    # temperature term
    temp_term = (1.0 +
                 math.exp(params.delta_h_low / (R * temp_k)) +
                 math.exp(params.delta_h_high / (R * temp_k)))
    return num * denom / temp_term

def gini_coefficient(values: Iterable[float]) -> float:
    """Gini coefficient for a non‑negative iterable (Parent A)."""
    xs = sorted(float(v) for v in values)
    if not xs or sum(xs) == 0:
        return 0.0
    n = len(xs)
    cumulative = sum((2 * i - n - 1) * x for i, x in enumerate(xs, 1))
    return cumulative / (n * sum(xs))

def euclidean_length(a: Point, b: Point) -> float:
    """Euclidean distance between two points (Parent B)."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def radial_basis_function(x: np.ndarray, centers: np.ndarray, epsilon: float = 1.0) -> np.ndarray:
    """
    Simple Gaussian RBF surrogate.
    Returns φ(x) = exp(-ε * ||x - c||²) for each centre c.
    """
    diff = x[:, None, :] - centers[None, :, :]
    sq_norm = np.sum(diff ** 2, axis=2)
    return np.exp(-epsilon * sq_norm)

# ----------------------------------------------------------------------
# Hybrid system components
# ----------------------------------------------------------------------
class HybridPheromoneGraph:
    """
    Graph where each edge carries a pheromone level.
    Pheromone updates are driven by:
      * temperature‑dependent rate (Schoolfield)
      * regret‑weighted propensities (Bandit/Math actions)
      * Gini‑scaled decay (Do​msday‑Gini)
    """
    def __init__(self,
                 nodes: Dict[str, Point],
                 edges: List[Edge],
                 params: SchoolfieldParams):
        self.nodes = nodes
        self.edges = edges
        self.params = params
        # Initialise pheromones uniformly
        self.pheromones: Dict[Edge, float] = {e: 1.0 for e in edges}
        # Decay half‑life (seconds) – a placeholder constant
        self.half_life = 300.0
        self.lengths: Dict[Edge, float] = {(u, v): euclidean_length(nodes[u], nodes[v]) for u, v in edges}

    def decay_factor(self, dt_seconds: float, gini: float) -> float:
        """
        Exponential decay factor modulated by the Gini coefficient.
        Higher inequality → slower decay (more exploration).
        """
        base = 0.5 ** (dt_seconds / self.half_life)
        return base ** (1.0 - gini)  # gini in [0,1]

    def update_pheromones(self,
                          temperature_c: float,
                          actions: List[MathAction],
                          bandit_updates: List[BanditUpdate],
                          dt_seconds: float = 1.0) -> None:
        """
        Perform a full pheromone update step.
        1. Compute temperature‑dependent scalar φ(T).
        2. Compute regret‑weighted propensities for each action.
        3. Compute Gini coefficient of the regret distribution.
        4. Decay existing pheromones.
        5. Reinforce edges proportional to propensities of actions that
           traversed them (simulated via bandit_updates).
        """
        temp_k = temperature_c + 273.15
        phi_t = schoolfield_rate(temp_k, self.params)

        # ---- Regret weighting ------------------------------------------------
        expected_vals = np.array([a.expected_value for a in actions])
        max_ev = expected_vals.max() if expected_vals.size else 0.0
        regrets = max_ev - expected_vals
        # Avoid division by zero; use temperature as a softening factor
        beta = 1.0 / max(1.0, temperature_c)
        propensities = np.exp(-beta * regrets)
        # Normalise
        if propensities.sum() > 0:
            propensities /= propensities.sum()
        # Map action id -> propensity
        prop_map: Dict[str, float] = {a.id: p for a, p in zip(actions, propensities)}

        # ---- Gini coefficient -------------------------------------------------
        gini = gini_coefficient(propensities)

        # ---- Decay ------------------------------------------------------------
        decay = self.decay_factor(dt_seconds, gini)
        for e in self.pheromones:
            self.pheromones[e] = max(0.0, self.pheromones[e] * decay)

        # ---- Reinforcement ----------------------------------------------------
        for upd in bandit_updates:
            # Find edges that involve the context (here we treat context_id as a node)
            src = upd.context_id
            dst = upd.action_id
            edge = (src, dst) if (src, dst) in self.pheromones else (dst, src)
            if edge in self.pheromones:
                # Reinforcement proportional to:
                #   φ(T) * propensity(action) * reward
                prop = prop_map.get(upd.action_id, 0.0)
                reinforcement = phi_t * prop * max(0.0, upd.reward) / self.lengths[edge]
                self.pheromones[edge] += reinforcement

    def edge_weights(self) -> Dict[Edge, float]:
        """
        Combine geometric length with current pheromone level.
        Weight = length / (1 + pheromone)  – higher pheromone lowers effective cost.
        """
        weights = {}
        for e in self.edges:
            weights[e] = self.lengths[e] / (1.0 + self.pheromones[e])
        return weights