# DARWIN HAMMER — match 5591, survivor 5
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hard_t_hybrid_endpoint_circ_m199_s5.py (gen3)
# parent_b: hybrid_hybrid_fold_change_d_hybrid_hybrid_pherom_m306_s1.py (gen4)
# born: 2026-05-30T00:03:12Z

"""HybridModelBanditPool
Integrates:
- Parent A (hybrid_hybrid_hybrid_hard_t_hybrid_endpoint_circ_m199_s5): linear
  rectified‑flow allocation a(α) = (1‑α)a₀ + αa₁ and the priority field  
  f(endpoint, model) = p(m)·(1‑r/R_max) where p(m) is a morphology‑derived
  priority and r is the model RAM size.
- Parent B (hybrid_hybrid_fold_change_d_hybrid_hybrid_pherom_m306_s1): bandit‑style
  action scoring using reward, store‑factor, fold‑change detection and pheromone
  infotaxis.

Mathematical bridge
------------------
For a given endpoint‑model pair we compute a *combined score*  

    S = f(endpoint, model) · ( reward(action) + store_factor(action) )

where the action corresponds to “load model X for endpoint Y”.  
The target allocation a₁ for each endpoint is set proportionally to S,
normalised across all candidate models, and the rectified‑flow interpolant
moves the current RAM allocation a₀ toward a₁.  This fuses the morphology‑aware
resource budgeting of Parent A with the adaptive bandit‑driven desirability
of Parent B.

The module provides three core hybrid operations:
1. `linear_interpolant` – generic rectified‑flow interpolation.
2. `hybrid_priority` – computes the priority field f(endpoint, model).
3. `hybrid_allocate` – determines target allocations from bandit scores,
   applies the interpolant, and loads models respecting the RAM ceiling.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Data structures (lightweight versions of the parents)
# ----------------------------------------------------------------------
@dataclass
class Endpoint:
    """Morphology‑driven endpoint."""
    id: str
    righting_time: float  # raw right‑ing time index (larger → slower)
    # derived priority p(m) ∈ [0,1] will be normalised later
    def priority(self, max_rt: float) -> float:
        """Normalize righting time to a priority (fast righting → high priority)."""
        if max_rt == 0:
            return 0.0
        return 1.0 - (self.righting_time / max_rt)


@dataclass
class Model:
    """Lightweight model descriptor."""
    name: str
    ram_mb: int
    tier: str = "default"


@dataclass(frozen=True)
class BanditAction:
    """Action used by the bandit component."""
    action_id: str          # convention: f"{endpoint_id}:{model_name}"
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str


# ----------------------------------------------------------------------
# Parent‑A utilities (linear schedule & RAM ceiling)
# ----------------------------------------------------------------------
def linear_interpolant(a0: float, a1: float, alpha: float) -> float:
    """Rectified‑flow linear interpolation between current (a0) and target (a1)."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0,1]")
    return (1.0 - alpha) * a0 + alpha * a1


class ModelPool:
    """Manages loaded models under a RAM ceiling."""
    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded: Dict[str, Model] = {}          # endpoint_id → Model
        self.allocation_mb: Dict[str, float] = {}   # endpoint_id → current RAM usage

    def current_total(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def can_allocate(self, model: Model) -> bool:
        return self.current_total() + model.ram_mb <= self.ram_ceiling_mb

    def load(self, endpoint_id: str, model: Model) -> bool:
        """Attempt to load a model for an endpoint respecting the RAM ceiling."""
        if not self.can_allocate(model):
            return False
        self.loaded[endpoint_id] = model
        self.allocation_mb[endpoint_id] = model.ram_mb
        return True

    def unload(self, endpoint_id: str) -> None:
        """Remove a model from an endpoint."""
        if endpoint_id in self.loaded:
            del self.loaded[endpoint_id]
            del self.allocation_mb[endpoint_id]


# ----------------------------------------------------------------------
# Parent‑B utilities (bandit scoring)
# ----------------------------------------------------------------------
_POLICY: Dict[str, Tuple[float, float]] = {}   # action_id → (cumulative_reward, count)


def reset_policy() -> None:
    _POLICY.clear()


def _reward(action_id: str) -> float:
    total, n = _POLICY.get(action_id, (0.0, 0.0))
    return total / n if n else 0.0


def _count(action_id: str) -> float:
    return _POLICY.get(action_id, (0.0, 0.0))[1]


def _store_factor(action_id: str, log_count_ratio: float) -> float:
    """Hybrid store factor = log_count_ratio * count."""
    return log_count_ratio * _count(action_id)


def _fold_change_detection(x: float, eps: float = 1e-9) -> float:
    """Fold‑change detection used for entropy‑like adjustments."""
    return math.log(max(x / eps, 1.0))


def _pheromone_infotaxis(pheromone: float, log_count_ratio: float) -> float:
    return pheromone * log_count_ratio


def _decision_hygiene_entropy(pheromone: float, log_count_ratio: float) -> float:
    infotaxis = _pheromone_infotaxis(pheromone, log_count_ratio)
    return -infotaxis * math.log(max(infotaxis, 1e-10))


def hybrid_select_action(actions: List[BanditAction], log_count_ratio: float) -> str:
    """Select action maximizing reward + store factor."""
    best_id = None
    best_val = -math.inf
    for act in actions:
        val = _store_factor(act.action_id, log_count_ratio) + _reward(act.action_id)
        if val > best_val:
            best_val = val
            best_id = act.action_id
    return best_id


def update_policy(action_id: str, reward: float) -> None:
    """Record observed reward for an action."""
    total, cnt = _POLICY.get(action_id, (0.0, 0.0))
    _POLICY[action_id] = (total + reward, cnt + 1.0)


# ----------------------------------------------------------------------
# Hybrid bridge: combine priority f with bandit scores to drive allocation
# ----------------------------------------------------------------------
def hybrid_priority(endpoint: Endpoint, model: Model, R_max: int, max_rt: float) -> float:
    """
    Compute f(endpoint, model) = p(m) * (1 - r/R_max)

    - p(m) is derived from endpoint.righting_time (fast → high priority).
    - r is model.ram_mb.
    """
    p_m = endpoint.priority(max_rt)
    ram_factor = 1.0 - (model.ram_mb / R_max)
    return max(p_m * ram_factor, 0.0)


def compute_combined_scores(
    endpoints: List[Endpoint],
    models: List[Model],
    actions: List[BanditAction],
    R_max: int,
    log_count_ratio: float,
) -> Dict[Tuple[str, str], float]:
    """
    For each (endpoint, model) pair, locate the corresponding action_id
    (format: f"{endpoint.id}:{model.name}") and compute

        S = f(endpoint, model) * ( reward + store_factor )

    Returns a dict keyed by (endpoint_id, model_name) → score.
    """
    # Determine max righting time for normalisation
    max_rt = max(e.righting_time for e in endpoints) if endpoints else 0.0

    # Pre‑compute f for all pairs
    f_vals: Dict[Tuple[str, str], float] = {}
    for ep in endpoints:
        for mdl in models:
            f_vals[(ep.id, mdl.name)] = hybrid_priority(ep, mdl, R_max, max_rt)

    # Compute bandit component per action
    bandit_vals: Dict[str, float] = {}
    for act in actions:
        reward = _reward(act.action_id)
        store = _store_factor(act.action_id, log_count_ratio)
        bandit_vals[act.action_id] = reward + store

    # Combine
    scores: Dict[Tuple[str, str], float] = {}
    for (ep_id, mdl_name), f_val in f_vals.items():
        act_id = f"{ep_id}:{mdl_name}"
        bandit = bandit_vals.get(act_id, 0.0)
        scores[(ep_id, mdl_name)] = f_val * bandit
    return scores


def hybrid_allocate(
    pool: ModelPool,
    endpoints: List[Endpoint],
    models: List[Model],
    actions: List[BanditAction],
    alpha: float = 0.5,
) -> None:
    """
    Perform one allocation step:
    1. Compute log‑count ratio across all actions (log(N_total / N_min+1)).
    2. Derive combined scores S for each endpoint‑model pair.
    3. For each endpoint, set target allocation a1 proportional to the
       highest‑scoring model (if any) while respecting RAM ceiling.
    4. Apply linear_interpolant between current allocation a0 and a1.
    5. Load/unload models accordingly.
    """
    # 1. Log‑count ratio (global statistic)
    counts = np.array([_count(act.action_id) for act in actions]) + 1.0  # avoid zero
    log_count_ratio = math.log(counts.max() / counts.min())

    # 2. Combined scores
    scores = compute_combined_scores(endpoints, models, actions, pool.ram_ceiling_mb, log_count_ratio)

    # 3. Determine best model per endpoint
    best_per_endpoint: Dict[str, Tuple[Model, float]] = {}
    for ep in endpoints:
        # collect scores for this endpoint
        ep_scores = {mdl_name: scores[(ep.id, mdl_name)] for mdl_name in [m.name for m in models]}
        # pick model with max score
        best_model_name = max(ep_scores, key=ep_scores.get, default=None)
        if best_model_name is None:
            continue
        best_score = ep_scores[best_model_name]
        if best_score <= 0.0:
            continue  # no attractive candidate
        best_model = next(m for m in models if m.name == best_model_name)
        best_per_endpoint[ep.id] = (best_model, best_score)

    # 4. Apply rectified‑flow allocation
    for ep_id, (target_model, _) in best_per_endpoint.items():
        a0 = pool.allocation_mb.get(ep_id, 0.0)
        a1 = float(target_model.ram_mb)
        a_new = linear_interpolant(a0, a1, alpha)

        # Decide whether to load/unload based on interpolation result
        if a_new >= target_model.ram_mb * 0.9:  # close enough → load
            success = pool.load(ep_id, target_model)
            if success:
                # Simulate a reward proportional to priority * alpha
                reward = a_new / pool.ram_ceiling_mb
                update_policy(f"{ep_id}:{target_model.name}", reward)
        else:
            # If we moved away from a previously loaded model, unload it
            if ep_id in pool.loaded:
                pool.unload(ep_id)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Reset bandit policy
    reset_policy()

    # Create a tiny pool (2 GB ceiling)
    pool = ModelPool(ram_ceiling_mb=2000)

    # Define endpoints with varying righting times
    endpoints = [
        Endpoint(id="epA", righting_time=0.8),
        Endpoint(id="epB", righting_time=1.5),
        Endpoint(id="epC", righting_time=0.3),
    ]

    # Define candidate models
    models = [
        Model(name="M_small", ram_mb=400),
        Model(name="M_medium", ram_mb=800),
        Model(name="M_large", ram_mb=1500),
    ]

    # Create bandit actions for each endpoint‑model pair
    actions = []
    for ep in endpoints:
        for mdl in models:
            act_id = f"{ep.id}:{mdl.name}"
            actions.append(
                BanditAction(
                    action_id=act_id,
                    propensity=random.random(),
                    expected_reward=random.random(),
                    confidence_bound=random.random(),
                    algorithm="hybrid"
                )
            )
            # Seed some initial policy entries to avoid zero counts
            update_policy(act_id, reward=random.uniform(0.0, 1.0))

    # Run a few allocation steps
    for step in range(5):
        hybrid_allocate(pool, endpoints, models, actions, alpha=0.4)
        total = pool.current_total()
        print(f"Step {step+1}: total RAM used = {total} MB, loaded models = { {k: v.name for k, v in pool.loaded.items()} }")

    sys.exit(0)