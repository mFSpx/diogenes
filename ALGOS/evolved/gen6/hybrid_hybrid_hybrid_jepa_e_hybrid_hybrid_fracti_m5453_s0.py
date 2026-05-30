# DARWIN HAMMER — match 5453, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_jepa_energy_h_hybrid_hybrid_hybrid_m1517_s0.py (gen5)
# parent_b: hybrid_hybrid_fractional_hd_hybrid_hoeffding_tre_m12_s0.py (gen2)
# born: 2026-05-30T00:01:59Z

"""Hybrid JEPA‑Workshare‑Hoeffding‑Gini algorithm.

This module fuses the *model‑pool* topology of
`hybrid_hybrid_jepa_energy_h_hybrid_hybrid_hybrid_m1517_s0.py` with the
statistical primitives of
`hybrid_hybrid_fractional_hd_hybrid_hoeffding_tre_m12_s0.py`.

Mathematical bridge
-------------------
* Each `ModelTier` is given a *hyper‑vector* signature (`random_hv`).  
* The *uncertainty* of a model’s performance is estimated with a
  Hoeffding bound (`hoeffding_bound`).  
* The *inequality* of the current memory allocation is measured by the
  Gini coefficient (`gini_coefficient`).  
* A *fractional causal effect* (`fractional_power`) of loading a model
  on the pool’s free‑energy is computed.  

The hybrid decision score combines these three quantities:


score = -ΔE_free                # reduction in free‑energy (reward)
        - α * hoeffding_bound   # penalise high uncertainty
        + β * gini_memory       # reward balanced memory distribution


A model is loaded (with possible eviction) if its score improves the
overall free‑energy of the pool.  The three core functions
`hybrid_score`, `hybrid_load` and `allocate_workshare` demonstrate this
integration."""

import math
import random
import sys
import pathlib
from dataclasses import dataclass, asdict
from itertools import chain
from typing import Optional, List, Dict

import numpy as np

# ---------------------------------------------------------------------------
# Parent A – model pool and workshare structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str  # e.g. "T1", "T2", "T3"


@dataclass(frozen=True)
class WorkshareLane:
    group: str
    llm_units: float
    llm_share_pct: float
    proof_required: bool


class ModelPool:
    """Manages a pool of loaded models, tracks a variational free‑energy."""
    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded: Dict[str, ModelTier] = {}
        self._energy: float = 0.0
        self.workshare_lanes: Dict[str, WorkshareLane] = {}

    # -----------------------------------------------------------------------
    # Basic bookkeeping
    # -----------------------------------------------------------------------
    def is_loaded(self, name: str) -> bool:
        return name in self.loaded

    def _used(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    # -----------------------------------------------------------------------
    # Loading / eviction with energetic bookkeeping
    # -----------------------------------------------------------------------
    def add_model(self, model: ModelTier) -> None:
        # Tier conflict penalty
        if model.tier == "T3" and any(m.tier == "T2" for m in self.loaded.values()):
            self._energy += 1e10
        # Memory‑ceiling penalty
        if model.ram_mb + self._used() > self.ram_ceiling_mb:
            self._energy += 1e6
        self.loaded[model.name] = model

    def load(self, model: ModelTier) -> None:
        self._energy -= 1e4          # reward for loading
        self.add_model(model)

    def load_with_eviction(self, model: ModelTier) -> None:
        self._energy -= 1e3          # reward for eviction step
        # Evict FIFO until there is room
        while self.loaded and model.ram_mb + self._used() > self.ram_ceiling_mb:
            evicted_name = next(iter(self.loaded))
            del self.loaded[evicted_name]
        self.load(model)

    def free_energy(self) -> float:
        return self._energy

    # -----------------------------------------------------------------------
    # Helper for hybrid calculations
    # -----------------------------------------------------------------------
    def memory_distribution(self) -> List[int]:
        """Return a list of RAM usages of currently loaded models."""
        return [m.ram_mb for m in self.loaded.values()]

    def gini_memory(self) -> float:
        """Gini coefficient of the current memory distribution."""
        return gini_coefficient(self.memory_distribution())

# ---------------------------------------------------------------------------
# Parent B – fractional hyper‑vector algebra and statistical primitives
# ---------------------------------------------------------------------------

def random_hv(d: int = 128, kind: str = "real", seed: Optional[int] = None) -> np.ndarray:
    """Generate a random hyper‑vector."""
    rng = np.random.default_rng(seed)
    if kind == "complex":
        theta = rng.uniform(0.0, 2.0 * np.pi, size=d)
        return np.exp(1j * theta)
    if kind == "bipolar":
        return rng.choice(np.array([-1.0, 1.0]), size=d)
    if kind == "real":
        v = rng.standard_normal(d)
        return v / (np.linalg.norm(v) + 1e-30)
    raise ValueError(f"Unsupported kind {kind!r}")


def bind(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Circular convolution binding (frequency‑domain multiplication)."""
    return np.fft.ifft(np.fft.fft(X) * np.fft.fft(Y))


def unbind(Z: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Inverse of bind using division in the Fourier domain."""
    FY = np.fft.fft(Y)
    # Avoid division by zero
    FY = np.where(np.abs(FY) < 1e-12, 1e-12, FY)
    return np.fft.ifft(np.fft.fft(Z) / FY)


def fractional_power(x: float, alpha: float) -> float:
    """Fractional (signed) power used for causal‑effect scaling."""
    if x == 0.0:
        return 0.0
    return math.copysign(abs(x) ** alpha, x)


def hoeffding_bound(mean: float, n: int, delta: float = 0.05) -> float:
    """Hoeffding bound for a bounded random variable in [0,1]."""
    if n <= 0:
        return float('inf')
    return math.sqrt(math.log(2.0 / delta) / (2 * n))


def gini_coefficient(values: List[float]) -> float:
    """Compute the Gini coefficient of a non‑negative list."""
    if not values:
        return 0.0
    arr = np.array(values, dtype=float)
    if np.any(arr < 0):
        raise ValueError("Gini coefficient is undefined for negative values")
    sorted_arr = np.sort(arr)
    n = len(arr)
    cumulative = np.cumsum(sorted_arr)
    sum_y = cumulative[-1]
    if sum_y == 0:
        return 0.0
    gini = (n + 1 - 2 * np.sum(cumulative) / sum_y) / n
    return gini

# ---------------------------------------------------------------------------
# Hybrid core functions (the mathematical bridge)
# ---------------------------------------------------------------------------

def model_signature(model: ModelTier, dim: int = 128) -> np.ndarray:
    """Deterministic hyper‑vector for a model derived from its name."""
    seed = (hash(model.name) & 0xffffffff)
    return random_hv(d=dim, kind="real", seed=seed)


def hybrid_score(
    model: ModelTier,
    pool: ModelPool,
    alpha: float = 0.5,
    delta: float = 0.05,
) -> float:
    """
    Compute a hybrid decision score for loading *model* into *pool*.

    The score combines:
      • ΔE_free   – energetic reward of loading the model (fixed -1e4)
      • Hoeffding bound on a proxy performance metric (ram_mb as sample size)
      • Gini coefficient of the current memory distribution
      • Fractional causal scaling (alpha)

    Higher scores indicate a more favourable loading decision.
    """
    # Energetic reward (same magnitude as ModelPool.load)
    delta_E = -1e4

    # Uncertainty term – treat ram_mb as number of observations
    bound = hoeffding_bound(mean=0.5, n=model.ram_mb, delta=delta)
    uncertainty_penalty = fractional_power(bound, alpha)

    # Inequality term – encourage balanced memory usage
    gini = pool.gini_memory()
    inequality_reward = fractional_power(gini, alpha)

    # Composite score (larger is better)
    score = delta_E - uncertainty_penalty + inequality_reward
    return score


def hybrid_load(
    pool: ModelPool,
    model: ModelTier,
    alpha: float = 0.5,
    delta: float = 0.05,
) -> None:
    """
    Load *model* into *pool* using the hybrid decision criterion.

    If the model improves the pool's free‑energy according to
    ``hybrid_score`` it is loaded (with eviction if necessary);
    otherwise the operation is ignored.
    """
    current_energy = pool.free_energy()
    prospective_score = hybrid_score(model, pool, alpha=alpha, delta=delta)

    # Simulate the energy change that would occur if we loaded the model
    # (ModelPool.load adds -1e4, but also may add penalties via add_model)
    temp_pool = ModelPool(ram_ceiling_mb=pool.ram_ceiling_mb)
    temp_pool.loaded = dict(pool.loaded)  # shallow copy
    temp_pool._energy = current_energy

    # Perform a tentative load with eviction to see the resulting energy
    temp_pool.load_with_eviction(model)
    new_energy = temp_pool.free_energy()

    # Accept the load only if energy is reduced (i.e., free energy lower)
    if new_energy < current_energy:
        pool.load_with_eviction(model)
    else:
        # No load; optionally log the decision (omitted for brevity)
        pass


def allocate_workshare(pool: ModelPool, alpha: float = 0.5) -> None:
    """
    Allocate LLM work‑share units to groups based on the hybrid scores
    of the currently loaded models.

    For each loaded model we compute a weight proportional to its
    fractional power of the Hoeffding bound; the total weight is normalised
    and split among existing workshare lanes (or created on‑the‑fly).
    """
    if not pool.loaded:
        return

    # Compute raw weights
    raw_weights = {}
    for name, model in pool.loaded.items():
        bound = hoeffding_bound(mean=0.5, n=model.ram_mb)
        raw_weights[name] = fractional_power(1.0 - bound, alpha)

    total = sum(raw_weights.values()) + 1e-12
    for name, weight in raw_weights.items():
        pct = weight / total
        lane = WorkshareLane(
            group=name,
            llm_units=pct * 100.0,
            llm_share_pct=pct * 100.0,
            proof_required=False,
        )
        pool.workshare_lanes[name] = lane

# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Initialise a pool with a modest RAM ceiling
    pool = ModelPool(ram_ceiling_mb=2000)

    # Define a few candidate models
    candidates = [
        ModelTier(name="bert_small", ram_mb=300, tier="T1"),
        ModelTier(name="gpt_medium", ram_mb=800, tier="T2"),
        ModelTier(name="opt_large", ram_mb=1200, tier="T3"),
        ModelTier(name="llama_xl", ram_mb=600, tier="T2"),
    ]

    # Attempt hybrid loading for each candidate
    for mdl in candidates:
        hybrid_load(pool, mdl, alpha=0.6, delta=0.01)

    # Allocate workshare based on the final pool composition
    allocate_workshare(pool, alpha=0.6)

    # Output results
    print("Loaded models:")
    for m in pool.loaded.values():
        print(f" - {m.name} ({m.ram_mb} MB, tier={m.tier})")
    print(f"Total RAM used: {pool._used()} / {pool.ram_ceiling_mb} MB")
    print(f"Free‑energy: {pool.free_energy():.2e}")

    print("\nWorkshare lanes:")
    for lane in pool.workshare_lanes.values():
        print(f" * {lane.group}: {lane.llm_units:.2f} units ({lane.llm_share_pct:.1f} %)")