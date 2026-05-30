# DARWIN HAMMER — match 4828, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_decisi_m2663_s0.py (gen4)
# parent_b: hybrid_physarum_network_hybrid_hybrid_bandit_m11_s2.py (gen3)
# born: 2026-05-29T23:58:20Z

import math
import numpy as np
import random
import sys
import pathlib
from dataclasses import dataclass

class ModelTier:
    def __init__(self, name: str, ram_mb: int, tier: str):
        self.name = name
        self.ram_mb = ram_mb
        self.tier = tier

class ModelPool:
    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded: dict = {}

    def is_loaded(self, name: str) -> bool:
        return name in self.loaded

    def _used(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def can_load(self, model: ModelTier) -> bool:
        return (self._used() + model.ram_mb) <= self.ram_ceiling_mb

    def load(self, model: ModelTier) -> None:
        if self.is_loaded(model.name):
            return  
        if not self.can_load(model):
            raise RuntimeError(
                f"Insufficient RAM to load {model.name}: "
                f"required {model.ram_mb} MB, available {self.ram_ceiling_mb - self._used()} MB."
            )
        self.loaded[model.name] = model

    def unload(self, name: str) -> None:
        self.loaded.pop(name, None)

@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length

def flatness_index(length: float, width: float, height: float) -> float:
    return max(length, width, height) / min(length, width, height)

def flux(conductance: float, edge_length: float, pressure_a: float, pressure_b: float, eps: float = 1e-12) -> float:
    if edge_length <= 0:
        raise ValueError('edge_length must be positive')
    return conductance / max(edge_length, eps) * (pressure_a - pressure_b)

def update_conductance(conductance: float, q: float, dt: float = 1.0, gain: float = 1.0, decay: float = 0.05) -> float:
    if dt < 0 or decay < 0:
        raise ValueError('dt and decay must be non-negative')
    return max(0.0, conductance + dt * (gain * abs(q) - decay * conductance))

def update_conductance_with_morphology(conductance: float, morphology: Morphology, q: float, dt: float = 1.0, gain: float = 1.0, decay: float = 0.05) -> float:
    sphericity, flatness = calculate_morphology_features(morphology)
    return max(0.0, conductance + dt * (gain * abs(q) * sphericity * flatness - decay * conductance))

def calculate_morphology_features(morphology: Morphology) -> tuple:
    sphericity = sphericity_index(morphology.length, morphology.width, morphology.height)
    flatness = flatness_index(morphology.length, morphology.width, morphology.height)
    return sphericity, flatness

class HybridBanditPhysarum:
    def __init__(
        self,
        d_in: int,
        d_out: int,
        seed: int = 0,
        base_eta: float = 0.01,
        alpha: float = 1.0,
        beta: float = 1.0,
        dt: float = 1.0,
        store_decay: float = 0.99,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.base_eta = base_eta
        self.alpha = alpha
        self.beta = beta
        self.dt = dt
        self.store_decay = store_decay
        self.d_in = d_in
        self.d_out = d_out
        self.conductance = np.ones((d_in, d_out))
        self.propensity = np.ones((d_in, d_out))
        self.morphologies = np.empty((d_in, d_out), dtype=object)

    def update_bandit(self, action_id: int, reward: float, propensity: float, morphology: Morphology) -> None:
        self.propensity[action_id] = propensity
        self.morphologies[action_id] = morphology
        self.conductance[action_id] = update_conductance_with_morphology(self.conductance[action_id], morphology, reward, self.dt, self.alpha, self.beta)

    def update_physarum(self, edge_length: float, pressure_a: float, pressure_b: float, action_id: int) -> None:
        q = flux(self.conductance[action_id, 0], edge_length, pressure_a, pressure_b)
        self.conductance[action_id, 0] = update_conductance_with_morphology(self.conductance[action_id, 0], self.morphologies[action_id, 0], q, self.dt, self.alpha, self.beta)

def hybrid_update(hybrid_bandit_physarum: HybridBanditPhysarum, edge_length: float, pressure_a: float, pressure_b: float, action_id: int) -> None:
    q = flux(hybrid_bandit_physarum.conductance[action_id, 0], edge_length, pressure_a, pressure_b)
    hybrid_bandit_physarum.conductance[action_id, 0] = update_conductance_with_morphology(hybrid_bandit_physarum.conductance[action_id, 0], hybrid_bandit_physarum.morphologies[action_id, 0], q, hybrid_bandit_physarum.dt, hybrid_bandit_physarum.alpha, hybrid_bandit_physarum.beta)

if __name__ == "__main__":
    hybrid_bandit_physarum = HybridBanditPhysarum(10, 10)
    morphology = Morphology(1.0, 2.0, 3.0, 4.0)
    hybrid_bandit_physarum.morphologies[0, 0] = morphology
    hybrid_update(hybrid_bandit_physarum, 1.0, 2.0, 3.0, 0)