# DARWIN HAMMER — match 2665, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m579_s1.py (gen5)
# parent_b: hybrid_poikilotherm_schoolf_hybrid_hybrid_hybrid_m1080_s0.py (gen4)
# born: 2026-05-29T23:44:59Z

import math
import random
import sys
import pathlib
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Dict, Tuple, Any, Callable, List

import numpy as np

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
R_CAL = 1.987  # cal·mol⁻¹·K⁻¹
KELVIN_REF = 298.15  # 25 °C in Kelvin


# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class SchoolfieldParams:
    """Parameters for the Schoolfield temperature‑dependence model."""
    rho_25: float = 1.0               # rate at 25 °C
    delta_h_activation: float = 12_000.0  # activation enthalpy (cal·mol⁻¹)
    t_low: float = 283.15             # lower temperature bound (K)
    t_high: float = 307.15            # upper temperature bound (K)
    delta_h_low: float = -45_000.0    # low‑temp deactivation enthalpy
    delta_h_high: float = 65_000.0    # high‑temp deactivation enthalpy
    r_cal: float = R_CAL


@dataclass(frozen=True)
class ModelTier:
    """Simple description of a model tier."""
    name: str
    ram_mb: int
    tier: str
    vram_mb: int


# Pre‑defined tiers (example)
TIER_T1_QWEN_0_5B = ModelTier("qwen-0.5b", 512, "T1", 1024)
TIER_T2_REASONING = ModelTier("reasoning-t2", 3000, "T2", 2048)
TIER_T2_TOOL = ModelTier("tool-t2", 2600, "T2", 2048)
TIER_T3_QWEN_7B = ModelTier("qwen-7b", 7000, "T3", 4096)


# ----------------------------------------------------------------------
# Core mathematical utilities
# ----------------------------------------------------------------------
def schoolfield_rate(
    temperature: float,
    params: SchoolfieldParams = SchoolfieldParams()
) -> float:
    """
    Compute the temperature‑dependent rate using the Schoolfield model.
    The formulation follows the classic Arrhenius term with low‑ and high‑
    temperature deactivation factors applied multiplicatively.
    """
    t = temperature
    # Base Arrhenius term
    rate = params.rho_25 * math.exp(-params.delta_h_activation / (params.r_cal * t))

    # Low‑temperature deactivation (if applicable)
    if t < params.t_low:
        rate *= math.exp(-params.delta_h_low / (params.r_cal * t))

    # High‑temperature deactivation (if applicable)
    if t > params.t_high:
        rate *= math.exp(-params.delta_h_high / (params.r_cal * t))

    return rate


def exponential_decay(initial: float, half_life: float, elapsed: float) -> float:
    """
    Return the value after exponential decay given a half‑life.
    """
    if half_life <= 0:
        raise ValueError("half_life must be positive")
    return initial * 0.5 ** (elapsed / half_life)


def current_utc_timestamp() -> float:
    """Return the current UTC time as a POSIX timestamp (seconds)."""
    return datetime.now(timezone.utc).timestamp()


# ----------------------------------------------------------------------
# Pheromone system
# ----------------------------------------------------------------------
class HybridPheromoneSystem:
    """
    Manages pheromone concentrations on arbitrary ``surface_key`` identifiers.
    Each entry stores:
        - ``signal_kind``: a free‑form tag (e.g. "risk", "decay")
        - ``concentration``: current pheromone amount
        - ``half_life``: decay half‑life in seconds
        - ``last_update``: POSIX timestamp of the most recent decay update
    The system automatically decays concentrations on read/write based on elapsed time.
    """

    def __init__(self):
        self._store: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_entry(
        self,
        surface_key: str,
        signal_kind: str,
        initial_conc: float,
        half_life: float,
    ) -> None:
        """Create a new entry if it does not exist."""
        if surface_key not in self._store:
            self._store[surface_key] = {
                "signal_kind": signal_kind,
                "concentration": float(initial_conc),
                "half_life": float(half_life),
                "last_update": current_utc_timestamp(),
            }

    def _apply_decay(self, surface_key: str) -> None:
        """Decay the stored concentration according to elapsed time."""
        entry = self._store[surface_key]
        now = current_utc_timestamp()
        elapsed = now - entry["last_update"]
        if elapsed > 0:
            entry["concentration"] = exponential_decay(
                entry["concentration"], entry["half_life"], elapsed
            )
            entry["last_update"] = now

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def add_pheromone(
        self,
        surface_key: str,
        signal_kind: str,
        amount: float,
        half_life_seconds: float,
    ) -> None:
        """
        Add (or replace) a pheromone entry. If the entry already exists,
        its concentration is first decayed to the present moment, then
        the new amount is added on top of the residual.
        """
        self._ensure_entry(surface_key, signal_kind, amount, half_life_seconds)
        self._apply_decay(surface_key)
        # after decay, add the fresh amount
        self._store[surface_key]["concentration"] += amount
        self._store[surface_key]["signal_kind"] = signal_kind
        self._store[surface_key]["half_life"] = half_life_seconds

    def get_concentration(self, surface_key: str) -> float:
        """
        Return the current concentration for ``surface_key`` after applying
        decay up to the current moment. If the key does not exist, ``0.0`` is returned.
        """
        if surface_key not in self._store:
            return 0.0
        self._apply_decay(surface_key)
        return self._store[surface_key]["concentration"]

    def get_entry(self, surface_key: str) -> Dict[str, Any]:
        """Return a shallow copy of the stored entry (including decay)."""
        if surface_key not in self._store:
            raise KeyError(f"No pheromone entry for key {surface_key!r}")
        self._apply_decay(surface_key)
        return dict(self._store[surface_key])


# ----------------------------------------------------------------------
# Ollivier‑Ricci curvature placeholder
# ----------------------------------------------------------------------
def ollivier_ricci_curvature(
    adjacency: np.ndarray,
    weight_func: Callable[[int, int], float] = lambda i, j: 1.0,
) -> float:
    """
    Very lightweight placeholder that returns a scalar curvature estimate.
    In a real implementation this would compute edge‑wise Ollivier‑Ricci curvature
    on a graph. Here we approximate it by the average weighted degree deviation.
    """
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("Adjacency must be a square matrix")
    n = adjacency.shape[0]
    degrees = adjacency.sum(axis=1)
    weighted_degrees = np.array(
        [
            sum(weight_func(i, j) * adjacency[i, j] for j in range(n))
            for i in range(n)
        ]
    )
    # curvature proxy: variance of weighted degree normalized
    if n == 0:
        return 0.0
    variance = np.var(weighted_degrees)
    curvature = 1.0 - variance / (np.mean(weighted_degrees) + 1e-12)
    return float(curvature)


# ----------------------------------------------------------------------
# Model‑level risk / health calculations
# ----------------------------------------------------------------------
def calculate_model_risk_score(
    pheromone_system: HybridPheromoneSystem,
    model_tier: ModelTier,
    temperature: float,
    schoolfield_params: SchoolfieldParams,
    adjacency: np.ndarray,
) -> float:
    """
    Compute a risk score that blends:
        • pheromone concentration (risk‑type signal)
        • temperature‑dependent rate (Schoolfield)
        • structural curvature (Ollivier‑Ricci)
    The score is normalised to the interval [0, 1] by a sigmoid‑like scaling.
    """
    # 1️⃣ Retrieve (and decay) the risk pheromone concentration
    risk_conc = pheromone_system.get_concentration(model_tier.name)

    # 2️⃣ Temperature‑dependent kinetic factor
    temp_factor = schoolfield_rate(temperature, schoolfield_params)

    # 3️⃣ Curvature influence (higher curvature → more robust → lower risk)
    curvature = ollivier_ricci_curvature(adjacency)
    curvature_factor = 1.0 / (1.0 + math.exp(-curvature))  # maps ℝ → (0, 1)

    # 4️⃣ Raw risk product
    raw_risk = risk_conc * temp_factor * (1.0 - curvature_factor)

    # 5️⃣ Normalise to [0, 1] using a soft‑clamp
    risk_score = 1.0 - math.exp(-raw_risk)
    return float(risk_score)


def calculate_pheromone_decay_factor(
    pheromone_system: HybridPheromoneSystem,
    model_tier: ModelTier,
    temperature: float,
    schoolfield_params: SchoolfieldParams,
) -> float:
    """
    Produce a decay multiplier that will be applied to the pheromone
    concentration each evaluation step. It is a function of:
        • current decay‑type pheromone concentration
        • temperature‑dependent rate
    The factor is guaranteed to lie in (0, 1].
    """
    decay_conc = pheromone_system.get_concentration(model_tier.name + "_decay")
    temp_factor = schoolfield_rate(temperature, schoolfield_params)

    # Combine multiplicatively, then bound
    raw_factor = decay_conc * temp_factor
    decay_factor = max(min(raw_factor, 1.0), 0.0)
    return float(decay_factor)


def calculate_model_health_score(
    pheromone_system: HybridPheromoneSystem,
    model_tier: ModelTier,
    temperature: float,
    schoolfield_params: SchoolfieldParams,
    adjacency: np.ndarray,
) -> float:
    """
    Health is defined as the complement of risk, attenuated by the
    decay factor. The result is forced into the [0, 1] interval.
    """
    risk = calculate_model_risk_score(
        pheromone_system, model_tier, temperature, schoolfield_params, adjacency
    )
    decay = calculate_pheromone_decay_factor(
        pheromone_system, model_tier, temperature, schoolfield_params
    )
    health = (1.0 - risk) * (1.0 - decay)
    # Clamp for numerical safety
    health = max(min(health, 1.0), 0.0)
    return float(health)


# ----------------------------------------------------------------------
# Example usage (can be removed or adapted for unit tests)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Initialise systems
    pheromone_system = HybridPheromoneSystem()
    schoolfield_params = SchoolfieldParams()
    model_tier = TIER_T1_QWEN_0_5B
    temperature = 298.15  # 25 °C

    # Dummy adjacency matrix for curvature (simple 3‑node line graph)
    adj = np.array([[0, 1, 0],
                    [1, 0, 1],
                    [0, 1, 0]], dtype=float)

    # Seed pheromones (risk and decay types)
    pheromone_system.add_pheromone(
        surface_key=model_tier.name,
        signal_kind="risk",
        amount=0.8,
        half_life_seconds=7200,
    )
    pheromone_system.add_pheromone(
        surface_key=model_tier.name + "_decay",
        signal_kind="decay",
        amount=0.4,
        half_life_seconds=3600,
    )

    # Compute metrics
    risk_score = calculate_model_risk_score(
        pheromone_system, model_tier, temperature, schoolfield_params, adj
    )
    decay_factor = calculate_pheromone_decay_factor(
        pheromone_system, model_tier, temperature, schoolfield_params
    )
    health_score = calculate_model_health_score(
        pheromone_system, model_tier, temperature, schoolfield_params, adj
    )

    print(f"Risk Score   : {risk_score:.4f}")
    print(f"Decay Factor : {decay_factor:.4f}")
    print(f"Health Score : {health_score:.4f}")