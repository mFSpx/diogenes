# DARWIN HAMMER — match 2934, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_voronoi_parti_hybrid_sparse_wta_hy_m228_s0.py (gen3)
# parent_b: hybrid_hybrid_hybrid_bandit_state_space_duality_m60_s0.py (gen3)
# born: 2026-05-29T23:46:53Z

import math
import random
import sys
import pathlib
from dataclasses import dataclass
from typing import Any, Iterable, Dict, List, Tuple
import numpy as np

Point = Tuple[float, float]

def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def nearest(point: Point, seeds: List[Point]) -> int:
    if not seeds:
        raise ValueError('seeds required')
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))

def assign(points: List[Point], seeds: List[Point]) -> Dict[int, List[Point]]:
    regions: Dict[int, List[Point]] = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions

@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0
    delta_h_activation: float = 12_000.0
    t_low: float = 283.15
    t_high: float = 307.15
    delta_h_low: float = -45_000.0
    delta_h_high: float = 65_000.0
    r_cal: float = 1.987  # cal mol^-1 K^-1

def developmental_rate(temp_k: float, params: SchoolfieldParams = SchoolfieldParams()) -> float:
    if temp_k <= 0 or params.rho_25 < 0:
        raise ValueError("temperature must be Kelvin-positive and rho_25 non-negative")
    numerator = params.rho_25 * (temp_k / 298.15) * math.exp(
        (params.delta_h_activation / params.r_cal) * ((1.0 / 298.15) - (1.0 / temp_k))
    )
    low = math.exp((params.delta_h_low / params.r_cal) * ((1.0 / params.t_low) - (1.0 / temp_k)))
    high = math.exp((params.delta_h_high / params.r_cal) * ((1.0 / params.t_high) - (1.0 / temp_k)))
    return numerator / (1.0 + low + high)

def temperature_dependent_state_transition(A: np.ndarray, temp_k: float,
                                            params: SchoolfieldParams = SchoolfieldParams()) -> np.ndarray:
    rate = developmental_rate(temp_k, params)
    return rate * A

@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str   # e.g., "T1", "T2", "T3"

class ModelPool:
    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded: Dict[str, ModelTier] = {}

    def is_loaded(self, name: str) -> bool:
        return name in self.loaded

    def _used(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def load(self, model: ModelTier) -> None:
        if model.tier == "T3" and any(m.tier == "T2" for m in self.loaded.values()):
            raise Exception("T3 mutually exclusive with T2 resident")
        if model.ram_mb + self._used() > self.ram_ceiling_mb:
            raise Exception("RAM ceiling exceeded")
        self.loaded[model.name] = model

    def _evict_one(self) -> None:
        if not self.loaded:
            return
        evict_name = min(self.loaded, key=lambda name: self.loaded[name].ram_mb)
        del self.loaded[evict_name]

    def load_with_eviction(self, model: ModelTier) -> None:
        while model.ram_mb + self._used() > self.ram_ceiling_mb:
            self._evict_one()
        if model.tier == "T3" and any(m.tier == "T2" for m in self.loaded.values()):
            for mname in [n for n, m in self.loaded.items() if m.tier == "T2"]:
                del self.loaded[mname]
        self.loaded[model.name] = model

def compute_utility(model: ModelTier, temp_k: float,
                    params: SchoolfieldParams = SchoolfieldParams()) -> float:
    rate = developmental_rate(temp_k, params)
    return rate / max(model.ram_mb, 1e-6)

def wta_select(models: List[ModelTier], temp_k: float,
               params: SchoolfieldParams = SchoolfieldParams()) -> ModelTier:
    if not models:
        raise ValueError("No candidate models for WTA")
    scores = [(compute_utility(m, temp_k, params), m) for m in models]
    best_score, best_model = max(scores, key=lambda pair: (pair[0], pair[1].name))
    return best_model

def hybrid_assign_and_select(points: List[Point],
                             seeds: List[Point],
                             temps_for_points: List[float],
                             seed_to_models: Dict[int, List[ModelTier]],
                             pool: ModelPool,
                             params: SchoolfieldParams = SchoolfieldParams()) -> Dict[int, ModelTier]:
    if len(points) != len(temps_for_points):
        raise ValueError("points and temps must have same length")
    regions = assign(points, seeds)
    selected: Dict[int, ModelTier] = {}
    for seed_index, points_in_seed in regions.items():
        if not points_in_seed:
            continue
        temps_in_seed = [temps_for_points[points.index(p)] for p in points_in_seed]
        avg_temp = sum(temps_in_seed) / len(temps_in_seed)
        models_in_seed = seed_to_models.get(seed_index, [])
        if not models_in_seed:
            continue
        best_model = wta_select(models_in_seed, avg_temp, params)
        pool.load_with_eviction(best_model)
        selected[seed_index] = best_model
    return selected

def hybrid_state_space_step(selected_models: Dict[int, ModelTier],
                           points: List[Point],
                           temps_for_points: List[float],
                           state: np.ndarray,
                           params: SchoolfieldParams = SchoolfieldParams()) -> np.ndarray:
    new_state = state.copy()
    for seed_index, model in selected_models.items():
        points_in_seed = [p for i, p in enumerate(points) if nearest(p, [s for i, s in enumerate(points) if i in selected_models]) == seed_index]
        if not points_in_seed:
            continue
        temps_in_seed = [temps_for_points[points.index(p)] for p in points_in_seed]
        avg_temp = sum(temps_in_seed) / len(temps_in_seed)
        transition_matrix = temperature_dependent_state_transition(np.eye(state.shape[0]), avg_temp, params)
        new_state = np.dot(transition_matrix, new_state)
    return new_state

def hybrid_fusion(points: List[Point],
                  seeds: List[Point],
                  temps_for_points: List[float],
                  seed_to_models: Dict[int, List[ModelTier]],
                  pool: ModelPool,
                  state: np.ndarray,
                  params: SchoolfieldParams = SchoolfieldParams()) -> Tuple[Dict[int, ModelTier], np.ndarray]:
    selected_models = hybrid_assign_and_select(points, seeds, temps_for_points, seed_to_models, pool, params)
    new_state = hybrid_state_space_step(selected_models, points, temps_for_points, state, params)
    return selected_models, new_state