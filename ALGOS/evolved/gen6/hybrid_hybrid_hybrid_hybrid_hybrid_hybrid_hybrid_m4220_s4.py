# DARWIN HAMMER — match 4220, survivor 4
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2311_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_jepa_e_hybrid_hybrid_hybrid_m2401_s3.py (gen5)
# born: 2026-05-29T23:54:15Z

"""HybridPheromoneModelSystem
Parent A: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2311_s0.py
Parent B: hybrid_hybrid_hybrid_jepa_e_hybrid_hybrid_hybrid_m2401_s3.py

Mathematical bridge:
- The pheromone signal strength ϕ is decayed over time (Parent A) and then multiplied by a
  risk‑modulation factor ρ derived from the reconstruction‑risk scores of the models in the
  ModelPool (Parent B). ρ∈[0,1] attenuates the signal proportionally to the estimated risk.
- The curvature vector κ computed from the pheromone feature map is used as a context‑dependent
  prior shift Δα for the Beta posteriors of a Thompson‑Bandit that selects models.
- Energy penalties from the ModelPool (Parent B) are added to the pheromone decay exponent,
  creating a unified update equation:

    ϕ_{t+1} = ϕ_t·½^{(Δt·(1+E))/τ}·(1−ρ)

  where Δt is elapsed seconds, τ is the half‑life, E is the current energy of the ModelPool,
  and ρ is the normalized reconstruction risk of the candidate model.

The following implementation fuses both topologies into a single executable module.
"""

import sys
import math
import random
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from dataclasses import dataclass
from typing import Iterable, Dict, List, Tuple

# ----------------------------------------------------------------------
# Parent B – ModelPool and related data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str
    recon_risk: float  # reconstruction risk score ∈[0,1]

@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

class ModelPool:
    """Energy‑aware container for ModelTier objects."""
    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded: Dict[str, ModelTier] = {}
        self._energy: float = 0.0

    def is_loaded(self, name: str) -> bool:
        return name in self.loaded

    def _used(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def add_model(self, model: ModelTier) -> None:
        # Tier conflict penalty
        if model.tier == "T3" and any(m.tier == "T2" for m in self.loaded.values()):
            self._energy += 1e10
        # Memory ceiling penalty
        if model.ram_mb + self._used() > self.ram_ceiling_mb:
            self._energy += 1e6
        self.loaded[model.name] = model

    def load(self, model: ModelTier) -> None:
        self._energy -= 1e4          # reward for loading
        self.add_model(model)

    def load_with_eviction(self, model: ModelTier) -> None:
        self._energy -= 1e3          # reward for eviction
        while self.loaded and model.ram_mb + self._used() > self.ram_ceiling_mb:
            # evict the largest‑RAM model
            evicted_name = max(self.loaded, key=lambda n: self.loaded[n].ram_mb)
            self.loaded.pop(evicted_name)
        self.add_model(model)

    @property
    def energy(self) -> float:
        return self._energy

    def risk_vector(self) -> np.ndarray:
        """Return a normalized vector of reconstruction risks for loaded models."""
        if not self.loaded:
            return np.array([0.0])
        risks = np.array([m.recon_risk for m in self.loaded.values()], dtype=float)
        # Normalise to [0,1]
        return risks / (risks.max() + 1e-9)

# ----------------------------------------------------------------------
# Parent A – Pheromone & Thompson‑Bandit core
# ----------------------------------------------------------------------
class HybridPheromoneBrainmapSystem:
    """Tracks pheromone signals on arbitrary surface keys."""
    def __init__(self):
        self.pheromones: Dict[str, Dict] = {}
        self.actions: List[str] = []
        self.rewards: List[float] = []
        self.action_counts: Dict[str, int] = {}
        self.action_values: Dict[str, Tuple[float, float]] = {}  # (α, β) for Beta posterior

    def _ensure_action(self, action: str) -> None:
        if action not in self.action_counts:
            self.action_counts[action] = 0
            self.action_values[action] = (1.0, 1.0)  # uniform prior α=β=1

    def calculate_pheromone_signal(self,
                                   surface_key: str,
                                   signal_kind: str,
                                   signal_value: float,
                                   half_life_seconds: float,
                                   risk_factor: float = 0.0,
                                   pool_energy: float = 0.0) -> float:
        """
        Update (or create) a pheromone entry and return the new signal value.

        The decay factor incorporates:
        - elapsed time Δt,
        - half‑life τ,
        - current ModelPool energy E,
        - external risk factor ρ (from reconstruction risk).

        New signal: ϕ' = ϕ·½^{(Δt·(1+E))/τ}·(1−ρ)
        """
        now = datetime.now(timezone.utc)
        if surface_key not in self.pheromones:
            self.pheromones[surface_key] = {
                "signal_kind": signal_kind,
                "signal_value": signal_value,
                "half_life_seconds": half_life_seconds,
                "created_time": now,
            }
            return signal_value

        entry = self.pheromones[surface_key]
        prev_val = entry["signal_value"]
        prev_half = entry["half_life_seconds"]
        prev_created = entry["created_time"]
        elapsed = (now - prev_created).total_seconds()

        # decay exponent with energy coupling
        decay_exponent = (elapsed * (1.0 + pool_energy)) / prev_half
        decayed = prev_val * math.pow(0.5, decay_exponent)

        # risk attenuation
        attenuated = decayed * (1.0 - risk_factor)

        # store updated values
        entry.update({
            "signal_value": attenuated,
            "created_time": now,
        })
        return attenuated

    # ------------------------------------------------------------------
    # Thompson‑Bandit utilities (Beta posterior)
    # ------------------------------------------------------------------
    def sample_beta(self, action: str) -> float:
        """Draw a sample from the Beta(α,β) posterior for the given action."""
        α, β = self.action_values[action]
        # Use numpy's beta sampler – works with float parameters
        return np.random.beta(α, β)

    def select_action(self, candidate_actions: Iterable[str]) -> str:
        """Select an action using Thompson sampling."""
        best_action = None
        best_sample = -1.0
        for act in candidate_actions:
            self._ensure_action(act)
            sample = self.sample_beta(act)
            if sample > best_sample:
                best_sample = sample
                best_action = act
        return best_action

    def update_posterior(self, action: str, reward: float) -> None:
        """Increment α for success, β for failure (reward∈{0,1})."""
        self._ensure_action(action)
        α, β = self.action_values[action]
        if reward >= 0.5:
            α += 1.0
        else:
            β += 1.0
        self.action_values[action] = (α, β)

# ----------------------------------------------------------------------
# Fusion utilities – mathematical bridge
# ----------------------------------------------------------------------
def compute_curvature_vector(feature_map: np.ndarray) -> np.ndarray:
    """
    Approximate a curvature vector κ from a 2‑D feature map.
    κ is taken as the Laplacian (second‑order differences) flattened.
    """
    if feature_map.ndim != 2:
        raise ValueError("feature_map must be 2‑dimensional")
    # Pad to keep dimensions
    padded = np.pad(feature_map, 1, mode='edge')
    laplacian = (
        -4 * padded[1:-1, 1:-1]
        + padded[:-2, 1:-1] + padded[2:, 1:-1]
        + padded[1:-1, :-2] + padded[1:-1, 2:]
    )
    return laplacian.ravel()

def curvature_to_prior_shift(curvature: np.ndarray) -> float:
    """
    Map curvature magnitude to a prior shift Δα for the Beta posterior.
    Larger curvature → larger shift (more confident prior).
    Normalised to [0, 2].
    """
    mag = np.linalg.norm(curvature)
    shift = 2.0 * (mag / (mag + 1e-6))  # sigmoid‑like scaling
    return shift

def hybrid_model_selection(model_pool: ModelPool,
                           pheromone_system: HybridPheromoneBrainmapSystem,
                           feature_map: np.ndarray) -> str:
    """
    Perform a hybrid selection:
    1. Compute curvature κ from the pheromone feature map.
    2. Convert κ to a prior shift Δα and apply it to all actions.
    3. Use Thompson sampling (with shifted priors) to pick a model name.
    4. Return the selected model name.
    """
    curvature = compute_curvature_vector(feature_map)
    delta_alpha = curvature_to_prior_shift(curvature)

    # Apply Δα to all loaded models' posteriors
    for name in model_pool.loaded:
        pheromone_system._ensure_action(name)
        α, β = pheromone_system.action_values[name]
        pheromone_system.action_values[name] = (α + delta_alpha, β)

    selected = pheromone_system.select_action(model_pool.loaded.keys())
    return selected

def update_pheromone_via_risk(surface_key: str,
                              pheromone_system: HybridPheromoneBrainmapSystem,
                              model_pool: ModelPool,
                              base_signal: float = 1.0,
                              half_life: float = 300.0) -> float:
    """
    Update a pheromone entry using the average reconstruction risk of the
    currently loaded models as the risk factor ρ.
    """
    risk_vec = model_pool.risk_vector()
    avg_risk = float(risk_vec.mean())
    new_signal = pheromone_system.calculate_pheromone_signal(
        surface_key=surface_key,
        signal_kind="hybrid",
        signal_value=base_signal,
        half_life_seconds=half_life,
        risk_factor=avg_risk,
        pool_energy=model_pool.energy,
    )
    return new_signal

def hybrid_step(model_pool: ModelPool,
                pheromone_system: HybridPheromoneBrainmapSystem,
                feature_map: np.ndarray) -> Tuple[str, float]:
    """
    Execute one hybrid iteration:
    - Update pheromone with risk.
    - Select a model via the curvature‑adjusted Thompson bandit.
    - Simulate a binary reward (random) and update the posterior.
    Returns the selected model name and the updated pheromone signal.
    """
    # 1️⃣ Pheromone update
    signal = update_pheromone_via_risk(
        surface_key="global",
        pheromone_system=pheromone_system,
        model_pool=model_pool,
        base_signal=1.0,
        half_life=300.0,
    )

    # 2️⃣ Model selection
    selected_model = hybrid_model_selection(model_pool, pheromone_system, feature_map)

    # 3️⃣ Simulated reward
    reward = random.choice([0.0, 1.0])
    pheromone_system.update_posterior(selected_model, reward)

    return selected_model, signal

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Initialise components
    pool = ModelPool(ram_ceiling_mb=8000)
    pheromone = HybridPheromoneBrainmapSystem()

    # Populate ModelPool with synthetic models
    for i in range(5):
        model = ModelTier(
            name=f"model_{i}",
            ram_mb=random.randint(500, 1500),
            tier=random.choice(["T1", "T2", "T3"]),
            recon_risk=random.random(),  # risk ∈[0,1]
        )
        pool.load_with_eviction(model)

    # Dummy feature map (e.g., 8×8 grid of random values)
    feature_map = np.random.rand(8, 8)

    # Run a few hybrid steps
    for step in range(3):
        selected, signal = hybrid_step(pool, pheromone, feature_map)
        print(f"Step {step+1}: selected={selected}, pheromone_signal={signal:.6f}, pool_energy={pool.energy:.2e}")

    sys.exit(0)