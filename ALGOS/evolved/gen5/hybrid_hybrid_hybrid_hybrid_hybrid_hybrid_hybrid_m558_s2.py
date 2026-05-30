# DARWIN HAMMER — match 558, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hard_t_hybrid_endpoint_circ_m199_s6.py (gen3)
# parent_b: hybrid_hybrid_hybrid_doomsd_hybrid_hybrid_bandit_m48_s5.py (gen4)
# born: 2026-05-29T23:29:41Z

"""Hybrid Model Resource Scheduler

This module fuses two distinct parent algorithms:

* **Parent A** – a RAM‑constrained `ModelPool` with a `Morphology` descriptor.
* **Parent B** – statistical utilities (`weekday_sakamoto`, `gini_coefficient`,
  `schoolfield_rate`) and a contextual bandit data model.

The mathematical bridge is built on three common concepts:

1. **Resource distribution** – the RAM usage of loaded models is treated as a
   1‑D vector; its inequality is quantified with the Gini coefficient from
   Parent B.
2. **Environmental scaling** – the `schoolfield_rate` temperature‑dependence
   provides a multiplicative factor that adjusts the RAM ceiling of the pool.
3. **Decision‑making under uncertainty** – a lightweight contextual bandit uses
   morphology‑derived sphericity as a propensity, the Gini‑adjusted fairness
   as a confidence bound, and weekday‑derived weights to select which model to
   load next.

The three functions below demonstrate this hybrid operation. """

import math
import random
import sys
import pathlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Model pool and morphology utilities
# ----------------------------------------------------------------------
class ModelTier:
    """Lightweight descriptor of a model."""
    def __init__(self, name: str, ram_mb: int, tier: str):
        self.name = name
        self.ram_mb = ram_mb
        self.tier = tier

class ModelPool:
    """Manages loaded models under a RAM ceiling."""
    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded: Dict[str, ModelTier] = {}

    def is_loaded(self, name: str) -> bool:
        return name in self.loaded

    def _used(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def can_load(self, model: ModelTier) -> bool:
        """Return True if the model fits within the remaining RAM."""
        return (self._used() + model.ram_mb) <= self.ram_ceiling_mb

    def load(self, model: ModelTier) -> None:
        """Load a model if RAM permits; raise otherwise."""
        if self.is_loaded(model.name):
            return  # already loaded
        if not self.can_load(model):
            raise RuntimeError(
                f"Insufficient RAM to load {model.name}: "
                f"required {model.ram_mb} MB, available {self.ram_ceiling_mb - self._used()} MB."
            )
        self.loaded[model.name] = model

    def unload(self, name: str) -> None:
        """Remove a model from the pool."""
        self.loaded.pop(name, None)

    def ram_usage_vector(self) -> np.ndarray:
        """Return a 1‑D array of RAM usages of the currently loaded models."""
        return np.array([m.ram_mb for m in self.loaded.values()], dtype=np.float64)


@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

def sphericity_index(length: float, width: float, height: float) -> float:
    """Corrected sphericity index: geometric mean of dimensions divided by the
    longest side (a simple proxy for roundness)."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    geometric_mean = (length * width * height) ** (1.0 / 3.0)
    longest = max(length, width, height)
    return geometric_mean / longest

# ----------------------------------------------------------------------
# Parent B – Date, inequality, temperature, and bandit utilities
# ----------------------------------------------------------------------
def weekday_sakamoto(
    years: np.ndarray,
    months: np.ndarray,
    days: np.ndarray,
) -> np.ndarray:
    y = years.astype(np.int64)
    m = months.astype(np.int64)
    d = days.astype(np.int64)

    m_adj = np.where(m < 3, m + 12, m)
    y_adj = np.where(m < 3, y - 1, y)

    t = np.array([0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4], dtype=np.int64)

    w = (y_adj + y_adj // 4 - y_adj // 100 + y_adj // 400 + t[m_adj - 1] + d) % 7
    return w.astype(np.int8)  # 0 = Sunday, …, 6 = Saturday


def gini_coefficient(values: np.ndarray) -> float:
    """Compute the Gini coefficient of a 1‑D non‑negative array."""
    if values.ndim != 1:
        raise ValueError("values must be a 1‑D array")
    x = np.sort(values.astype(np.float64))
    if x.size == 0 or np.isclose(x.sum(), 0.0):
        return 0.0
    if np.any(x < 0):
        raise ValueError("values must be non‑negative")
    n = x.size
    i = np.arange(1, n + 1, dtype=np.float64)
    numerator = np.sum((2 * i - n - 1) * x)
    denominator = n * x.sum()
    return float(numerator / denominator)


@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0
    delta_h_activation: float = 12_000.0
    t_low: float = 283.15
    t_high: float = 307.15
    delta_h_low: float = -45_000.0
    delta_h_high: float = 65_000.0
    r_cal: float = 1.987


def schoolfield_rate(params: SchoolfieldParams, temperature: np.ndarray) -> np.ndarray:
    """Temperature‑dependent rate from the Schoolfield model."""
    T = temperature.astype(np.float64)
    R = params.r_cal * 4.184  # convert cal·K⁻¹·mol⁻¹ to J·K⁻¹·mol⁻¹

    num = np.exp(-params.delta_h_activation / R * (1.0 / T - 1.0 / 298.15))
    low = np.exp(params.delta_h_low / R * (1.0 / params.t_low - 1.0 / T))
    high = np.exp(params.delta_h_high / R * (1.0 / params.t_high - 1.0 / T))

    denominator = 1.0 + low + high
    return params.rho_25 * num / denominator


@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float          # probability‑like weight (from morphology)
    expected_reward: float     # model‑specific reward estimate
    confidence_bound: float    # fairness term derived from Gini
    algorithm: str = "HybridScheduler"


@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float


# ----------------------------------------------------------------------
# Hybrid Functions
# ----------------------------------------------------------------------
def compute_ram_gini(pool: ModelPool) -> float:
    """Return the Gini coefficient of the current RAM distribution."""
    usage = pool.ram_usage_vector()
    if usage.size == 0:
        return 0.0
    return gini_coefficient(usage)


def adjusted_ram_ceiling(base_ceiling: int, temperatures: np.ndarray) -> int:
    """
    Scale the RAM ceiling by the mean Schoolfield rate over the supplied
    temperature array.  The result is rounded to the nearest integer.
    """
    rates = schoolfield_rate(SchoolfieldParams(), temperatures)
    scaling = rates.mean()
    # Ensure a sensible lower bound (e.g., 10 % of base) to avoid pathological zero.
    scaling = max(scaling, 0.1)
    return int(round(base_ceiling * scaling))


def prioritize_models(
    pool: ModelPool,
    candidates: List[Tuple[ModelTier, Morphology]],
    temperature: float,
    date_tuple: Tuple[int, int, int],
) -> List[BanditAction]:
    """
    Convert a list of candidate models into BanditAction objects.
    Propensity is derived from the sphericity index,
    expected_reward from inverse RAM usage (smaller models are cheaper),
    and confidence_bound from the current Gini coefficient (promotes fairness).
    Weekday weighting biases the propensity: weekdays (Mon‑Fri) get a 1.2× boost.
    """
    year, month, day = date_tuple
    weekday = weekday_sakamoto(
        np.array([year]), np.array([month]), np.array([day])
    )[0]  # 0 = Sunday … 6 = Saturday
    weekday_factor = 1.2 if weekday in (1, 2, 3, 4, 5) else 0.9  # Mon‑Fri vs. weekend

    gini = compute_ram_gini(pool)

    actions: List[BanditAction] = []
    for model, morph in candidates:
        # Propensity from morphology (higher sphericity → higher propensity)
        sph = sphericity_index(morph.length, morph.width, morph.height)
        propensity = sph * weekday_factor

        # Expected reward: we reward low RAM consumption and high temperature‑adjusted
        # performance.  Temperature influences reward via the Schoolfield rate.
        temp_factor = schoolfield_rate(SchoolfieldParams(), np.array([temperature]))[0]
        reward = (1.0 / (model.ram_mb + 1.0)) * temp_factor

        # Confidence bound: larger Gini means less fairness, so we penalise.
        confidence = 1.0 / (1.0 + gini)

        actions.append(
            BanditAction(
                action_id=model.name,
                propensity=propensity,
                expected_reward=reward,
                confidence_bound=confidence,
                algorithm="HybridScheduler",
            )
        )
    return actions


def select_and_load_model(
    pool: ModelPool,
    candidates: List[Tuple[ModelTier, Morphology]],
    temperatures: np.ndarray,
    dates: np.ndarray,
) -> ModelTier | None:
    """
    Perform a single selection step:
    1. Adjust the pool's RAM ceiling using the mean temperature.
    2. Build BanditAction objects for each candidate.
    3. Choose the action maximising (propensity * expected_reward * confidence_bound).
    4. Attempt to load the selected model; return it on success, None otherwise.
    """
    # 1. Adjust RAM ceiling
    pool.ram_ceiling_mb = adjusted_ram_ceiling(pool.ram_ceiling_mb, temperatures)

    # 2. Build actions – we use the first date entry for simplicity
    year, month, day = dates[0]
    actions = prioritize_models(pool, candidates, float(temperatures.mean()), (year, month, day))

    # 3. Score actions
    def score(a: BanditAction) -> float:
        return a.propensity * a.expected_reward * a.confidence_bound

    best_action = max(actions, key=score, default=None)
    if best_action is None:
        return None

    # 4. Load the selected model
    model_dict = {m.name: m for m, _ in candidates}
    chosen_model = model_dict.get(best_action.action_id)
    if chosen_model is None:
        return None
    try:
        pool.load(chosen_model)
        return chosen_model
    except RuntimeError:
        # Not enough RAM after ceiling adjustment; skip.
        return None

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a pool with a nominal ceiling.
    pool = ModelPool(ram_ceiling_mb=5000)

    # Generate synthetic candidate models with associated morphology.
    random.seed(42)
    candidates: List[Tuple[ModelTier, Morphology]] = []
    for i in range(6):
        name = f"model_{i}"
        ram = random.randint(200, 1500)  # MB
        tier = random.choice(["small", "medium", "large"])
        model = ModelTier(name=name, ram_mb=ram, tier=tier)

        # Random morphology dimensions (cm) and mass (kg)
        morph = Morphology(
            length=random.uniform(5.0, 20.0),
            width=random.uniform(5.0, 20.0),
            height=random.uniform(5.0, 20.0),
            mass=random.uniform(0.5, 5.0),
        )
        candidates.append((model, morph))

    # Temperature array (Kelvin) – simulate a day‑long profile.
    temperatures = np.linspace(295.0, 310.0, num=12)  # 12 samples

    # Dates array – use today's date repeated.
    today = np.datetime64('today')
    y, m, d = today.astype('datetime64[D]').astype(object).year, \
              today.astype('datetime64[M]').astype(object).month, \
              today.astype('datetime64[D]').astype(object).day
    dates = np.array([[y, m, d]])

    # Run the hybrid selector.
    selected = select_and_load_model(pool, candidates, temperatures, dates)

    print("Adjusted RAM ceiling:", pool.ram_ceiling_mb, "MB")
    print("Current Gini of RAM usage:", compute_ram_gini(pool))
    if selected:
        print(f"Selected and loaded model: {selected.name} ({selected.ram_mb} MB)")
    else:
        print("No model could be loaded under current constraints.")

    # Show final pool state.
    print("Loaded models:", list(pool.loaded.keys()))
    print("Total RAM used:", pool._used(), "MB")