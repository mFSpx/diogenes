# DARWIN HAMMER — match 2934, survivor 0
# gen: 4
# parent_a: hybrid_hybrid_voronoi_parti_hybrid_sparse_wta_hy_m228_s0.py (gen3)
# parent_b: hybrid_hybrid_hybrid_bandit_state_space_duality_m60_s0.py (gen3)
# born: 2026-05-29T23:46:53Z

"""
Hybrid Voronoi‑State‑Space Model Pool

Parents:
- hybrid_hybrid_voronoi_parti_hybrid_sparse_wta_hy_m228_s0.py (Voronoi partitioning + sparse WTA tags for model pool)
- hybrid_hybrid_hybrid_bandit_state_space_duality_m60_s0.py (Temperature‑dependent developmental rate embedded in a state‑space model)

Mathematical bridge:
The Voronoi partition assigns each incoming data point (e.g., a sensor reading) to the nearest *seed* that represents a model location.
Each seed is associated with a concrete ModelTier residing in a ModelPool.
The temperature‑dependent developmental rate ρ(T) from the Schoolfield poikilotherm model provides a scalar *risk/utility* score for the model that lives at a seed.
A sparse winner‑take‑all (WTA) tag selects, within each Voronoi cell, the model with the highest ρ(T)·(1/ram) utility.
Thus the Voronoi geometry drives *where* a model can be used, while the temperature‑driven rate drives *which* model wins the WTA competition, and the ModelPool enforces RAM ceiling and tier constraints.
The fused system therefore:
1. Partitions points → cells.
2. Computes temperature‑dependent scores for candidate models.
3. Performs WTA per cell to decide which model to load/evict.
4. Runs a temperature‑scaled state‑space step using the selected model’s dynamics.

All core equations from both parents are retained and combined.
"""

import math
import random
import sys
import pathlib
from dataclasses import dataclass
from typing import Any, Iterable, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Geometry (Voronoi) utilities – from Parent A
# ----------------------------------------------------------------------
Point = Tuple[float, float]

def distance(a: Point, b: Point) -> float:
    """Euclidean distance."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def nearest(point: Point, seeds: List[Point]) -> int:
    """Return index of the nearest seed (ties broken by index)."""
    if not seeds:
        raise ValueError('seeds required')
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))

def assign(points: List[Point], seeds: List[Point]) -> Dict[int, List[Point]]:
    """Assign each point to the index of its nearest seed."""
    regions: Dict[int, List[Point]] = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions

# ----------------------------------------------------------------------
# Developmental rate (temperature) – from Parent B
# ----------------------------------------------------------------------
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
    """Schoolfield poikilotherm developmental rate."""
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
    """Scale state‑transition matrix A by the developmental rate at temp_k."""
    rate = developmental_rate(temp_k, params)
    return rate * A

# ----------------------------------------------------------------------
# Model description and pool – from Parent A (completed)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str   # e.g., "T1", "T2", "T3"

class ModelPool:
    """Manages loaded models under RAM ceiling and tier exclusivity."""
    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded: Dict[str, ModelTier] = {}

    def is_loaded(self, name: str) -> bool:
        return name in self.loaded

    def _used(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def load(self, model: ModelTier) -> None:
        """Load a model; raise if constraints violated."""
        if model.tier == "T3" and any(m.tier == "T2" for m in self.loaded.values()):
            raise Exception("T3 mutually exclusive with T2 resident")
        if model.ram_mb + self._used() > self.ram_ceiling_mb:
            raise Exception("RAM ceiling exceeded")
        self.loaded[model.name] = model

    def _evict_one(self) -> None:
        """Evict a random model (simple policy)."""
        if not self.loaded:
            return
        evict_name = random.choice(list(self.loaded.keys()))
        del self.loaded[evict_name]

    def load_with_eviction(self, model: ModelTier) -> None:
        """Attempt to load; evict models until enough space."""
        while model.ram_mb + self._used() > self.ram_ceiling_mb:
            self._evict_one()
        # after freeing space, check tier exclusivity again
        if model.tier == "T3" and any(m.tier == "T2" for m in self.loaded.values()):
            # evict all T2 models
            for mname in [n for n, m in self.loaded.items() if m.tier == "T2"]:
                del self.loaded[mname]
        self.loaded[model.name] = model

# ----------------------------------------------------------------------
# Sparse Winner‑Take‑All tag based on temperature utility
# ----------------------------------------------------------------------
def compute_utility(model: ModelTier, temp_k: float,
                    params: SchoolfieldParams = SchoolfieldParams()) -> float:
    """
    Utility = developmental_rate(temp) / ram_mb.
    Higher rate and lower RAM give a larger score.
    """
    rate = developmental_rate(temp_k, params)
    return rate / max(model.ram_mb, 1e-6)

def wta_select(models: List[ModelTier], temp_k: float,
               params: SchoolfieldParams = SchoolfieldParams()) -> ModelTier:
    """
    Sparse winner‑take‑all: return the model with maximal utility.
    If list empty, raise.
    """
    if not models:
        raise ValueError("No candidate models for WTA")
    scores = [(compute_utility(m, temp_k, params), m) for m in models]
    # Sparse: pick the single highest; ties broken by name order
    best_score, best_model = max(scores, key=lambda pair: (pair[0], pair[1].name))
    return best_model

# ----------------------------------------------------------------------
# Fusion functions (at least three)
# ----------------------------------------------------------------------
def hybrid_assign_and_select(points: List[Point],
                             seeds: List[Point],
                             temps_for_points: List[float],
                             seed_to_models: Dict[int, List[ModelTier]],
                             pool: ModelPool,
                             params: SchoolfieldParams = SchoolfieldParams()) -> Dict[int, ModelTier]:
    """
    1. Voronoi assign points → cells.
    2. For each cell, collect temperatures of its points, average them.
    3. Run WTA among models attached to that seed using the averaged temperature.
    4. Load the winning model into the pool (evicting if necessary).
    Returns a mapping seed_index -> selected ModelTier.
    """
    if len(points) != len(temps_for_points):
        raise ValueError("points and temps must have same length")
    regions = assign(points, seeds)

    selected: Dict[int, ModelTier] = {}
    for seed_idx, pts in regions.items():
        if not pts:
            continue
        # average temperature of points in this region
        idxs = [points.index(p) for p in pts]  # simple lookup (small data)
        avg_temp = sum(temps_for_points[i] for i in idxs) / len(idxs)
        candidates = seed_to_models.get(seed_idx, [])
        if not candidates:
            continue
        winner = wta_select(candidates, avg_temp, params)
        # Load with eviction policy
        if not pool.is_loaded(winner.name):
            pool.load_with_eviction(winner)
        selected[seed_idx] = winner
    return selected

def hybrid_state_space_step(h: np.ndarray,
                            x: np.ndarray,
                            A: np.ndarray,
                            B: np.ndarray,
                            C: np.ndarray,
                            temp_k: float,
                            params: SchoolfieldParams = SchoolfieldParams()) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform a single temperature‑scaled state‑space step.
    Mirrors `ssm_step` from Parent B but is kept as a separate callable
    to emphasise the hybrid nature.
    """
    A_temp = temperature_dependent_state_transition(A, temp_k, params)
    h_new = A_temp @ h + B @ x
    y = C @ h_new
    return h_new, y

def hybrid_sequential_process(x_seq: np.ndarray,
                              A: np.ndarray,
                              B: np.ndarray,
                              C: np.ndarray,
                              temp_seq: np.ndarray,
                              h0: np.ndarray | None = None,
                              params: SchoolfieldParams = SchoolfieldParams()) -> np.ndarray:
    """
    Run a sequence of inputs through the temperature‑aware state‑space model.
    Returns the concatenated outputs y_t for each time step.
    """
    T, n_in = x_seq.shape
    if temp_seq.shape != (T,):
        raise ValueError("temp_seq must have shape (T,)")
    n_state = A.shape[0]
    h = np.zeros((n_state,)) if h0 is None else h0.copy()
    outputs = []
    for t in range(T):
        x_t = x_seq[t]
        temp_k = temp_seq[t]
        h, y = hybrid_state_space_step(h, x_t, A, B, C, temp_k, params)
        outputs.append(y)
    return np.vstack(outputs)

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Define seeds (model locations) and points with temperatures
    seeds = [(0.0, 0.0), (10.0, 0.0), (5.0, 8.0)]
    points = [(1, 1), (2, 0.5), (9, 1), (8, -1), (5, 7), (6, 9)]
    temps_celsius = [20, 22, 25, 24, 18, 19]          # in °C
    temps_kelvin = [temp + 273.15 for temp in temps_celsius]

    # 2. Create a few models per seed
    model_catalog = {
        0: [ModelTier(name="M0_A", ram_mb=500, tier="T1"),
            ModelTier(name="M0_B", ram_mb=800, tier="T2")],
        1: [ModelTier(name="M1_A", ram_mb=300, tier="T1"),
            ModelTier(name="M1_B", ram_mb=1200, tier="T3")],
        2: [ModelTier(name="M2_A", ram_mb=400, tier="T2"),
            ModelTier(name="M2_B", ram_mb=700, tier="T1")]
    }

    # 3. Initialise pool
    pool = ModelPool(ram_ceiling_mb=2500)

    # 4. Hybrid Voronoi‑WTA selection & pool loading
    selected_models = hybrid_assign_and_select(
        points=points,
        seeds=seeds,
        temps_for_points=temps_kelvin,
        seed_to_models=model_catalog,
        pool=pool
    )
    print("Selected models per seed:")
    for sid, mdl in selected_models.items():
        print(f"  Seed {sid}: {mdl.name} (RAM {mdl.ram_mb} MB)")

    print("\nCurrent pool contents:")
    for name, mdl in pool.loaded.items():
        print(f"  {name}: tier={mdl.tier}, ram={mdl.ram_mb}")

    # 5. Build a simple state‑space system (same dimensions for all models)
    A = np.array([[0.9, 0.1],
                  [0.05, 0.95]])
    B = np.array([[0.1],
                  [0.05]])
    C = np.array([[1.0, 0.0]])

    # 6. Input sequence and temperature sequence (use same temps as above)
    x_seq = np.random.randn(len(points), 1) * 0.5
    temp_seq = np.array(temps_kelvin)

    # 7. Run the hybrid sequential process
    y_out = hybrid_sequential_process(x_seq, A, B, C, temp_seq)
    print("\nState‑space outputs (first 3 rows):")
    print(y_out[:3])
    print("\nSmoke test completed without errors.")