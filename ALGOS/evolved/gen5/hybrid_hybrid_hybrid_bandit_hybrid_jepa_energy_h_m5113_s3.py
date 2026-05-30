# DARWIN HAMMER — match 5113, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_bandit_router_hybrid_hybrid_hybrid_m547_s2.py (gen4)
# parent_b: hybrid_jepa_energy_hybrid_sparse_wta_hy_m181_s3.py (gen3)
# born: 2026-05-29T23:59:51Z

"""
Hybrid Algorithm integrating:
- Parent A: Bandit router with Schoolfield temperature‑dependent propensities and Fisher‑information‑based scoring.
- Parent B: ModelPool managed by variational free‑energy (VFE) with tier‑aware RAM constraints.

Mathematical bridge:
Each *BanditAction* corresponds to a *ModelTier* operation (load or unload).
The action propensity 𝜋(T) is computed with the Schoolfield temperature model
(temperature‑dependent kinetic factor). The expected reward of an action is the
change ΔF in the ModelPool's variational free energy if the action were executed.
A Fisher‑information‑like confidence bound is approximated by the inverse
square‑root of the number of times the action has been taken, giving a UCB‑style
selection rule:
    score = ΔF + c·√(log(N)/n_i)
where N is total action count, n_i is the count for action i, and c is a tunable
exploration constant. The hybrid thus fuses thermodynamic propensity,
information‑theoretic confidence, and free‑energy optimisation in a single
decision‑making loop.
"""

import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Constants and helper regular expressions (kept from Parent A)
# ----------------------------------------------------------------------
R_CAL = 1.987  # cal·mol⁻¹·K⁻¹
K25 = 298.15  # reference temperature (K)

# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ModelTier:
    """Immutable descriptor of a model tier."""
    name: str
    ram_mb: int
    tier: str  # e.g. "T1", "T2", "T3"


@dataclass(frozen=True, slots=True)
class BanditAction:
    """An action that manipulates the ModelPool."""
    action_id: str                 # e.g. "load:bert-base"
    model: ModelTier               # model involved in the action
    load: bool                     # True → load, False → unload
    propensity: float = 0.0       # temperature‑dependent propensity (filled later)
    expected_reward: float = 0.0  # ΔF (filled later)
    confidence_bound: float = 0.0  # UCB term (filled later)


@dataclass
class SchoolfieldParams:
    rho_25: float = 1.0
    delta_h_activation: float = 12_000.0
    t_low: float = 283.15
    t_high: float = 307.15
    delta_h_low: float = -45_000.0
    delta_h_high: float = 65_000.0
    r_cal: float = R_CAL


# ----------------------------------------------------------------------
# ModelPool (from Parent B) with VFE management
# ----------------------------------------------------------------------
class ModelPool:
    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb: int = ram_ceiling_mb
        self.loaded: Dict[str, ModelTier] = {}
        self._vfe: float = 0.0                     # variational free energy (lower is better)
        self._action_counts: Dict[str, int] = {}   # for confidence bound

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _used_ram(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def _vfe_penalty(self, delta: float) -> None:
        self._vfe += delta

    def _tier_conflict(self, candidate: ModelTier) -> bool:
        """Return True if adding candidate would violate tier constraints."""
        if candidate.tier == "T3":
            return any(m.tier == "T2" for m in self.loaded.values())
        if candidate.tier == "T2":
            return any(m.tier == "T3" for m in self.loaded.values())
        return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def is_loaded(self, name: str) -> bool:
        return name in self.loaded

    def free_energy(self) -> float:
        return self._vfe

    def add_model(self, model: ModelTier) -> None:
        """Attempt to load a model; apply penalties for conflicts or RAM overflow."""
        conflict = self._tier_conflict(model)
        ram_over = model.ram_mb + self._used_ram() - self.ram_ceiling_mb

        if conflict:
            self._vfe_penalty(1e6)          # heavy penalty for tier conflict
            return
        if ram_over > 0:
            self._vfe_penalty(ram_over * 0.5)  # soft RAM overflow penalty

        self.loaded[model.name] = model
        self._vfe_penalty(-model.ram_mb * 0.01)  # reward for efficient RAM use

    def remove_model(self, name: str) -> None:
        """Unload a model and adjust VFE."""
        if name in self.loaded:
            model = self.loaded.pop(name)
            self._vfe_penalty(model.ram_mb * 0.02)  # small penalty for eviction

    def simulate_action(self, action: BanditAction) -> float:
        """
        Return the hypothetical ΔF (change in VFE) if `action` were applied,
        without mutating the pool.
        """
        original_vfe = self._vfe
        # copy state
        backup_loaded = dict(self.loaded)
        backup_vfe = self._vfe

        if action.load:
            self.add_model(action.model)
        else:
            self.remove_model(action.model.name)

        delta_f = self._vfe - original_vfe

        # restore original state
        self.loaded = backup_loaded
        self._vfe = backup_vfe
        return delta_f

    def record_action(self, action_id: str) -> None:
        self._action_counts[action_id] = self._action_counts.get(action_id, 0) + 1

    def action_count(self, action_id: str) -> int:
        return self._action_counts.get(action_id, 0)


# ----------------------------------------------------------------------
# Schoolfield temperature model (from Parent A)
# ----------------------------------------------------------------------
def schoolfield_propensity(
    temperature: float,
    params: SchoolfieldParams = SchoolfieldParams()
) -> float:
    """
    Compute the temperature‑dependent propensity factor π(T) using the
    Schoolfield model.
    """
    inv_T = 1.0 / temperature
    inv_T25 = 1.0 / K25
    inv_Tlow = 1.0 / params.t_low
    inv_Thigh = 1.0 / params.t_high

    # activation term
    act = math.exp(
        -params.delta_h_activation / params.r_cal * (inv_T - inv_T25)
    )
    # low‑temperature inhibition
    low = 1.0 + math.exp(
        params.delta_h_low / params.r_cal * (inv_Tlow - inv_T)
    )
    # high‑temperature inhibition
    high = 1.0 + math.exp(
        params.delta_h_high / params.r_cal * (inv_T - inv_Thigh)
    )
    propensity = params.rho_25 * act / (low * high)
    return propensity


# ----------------------------------------------------------------------
# Core hybrid functions
# ----------------------------------------------------------------------
def generate_actions(pool: ModelPool, catalog: List[ModelTier]) -> List[BanditAction]:
    """
    Produce a list of possible BanditActions (load or unload) given the current pool.
    """
    actions: List[BanditAction] = []
    for model in catalog:
        # Load actions for models not already present
        if not pool.is_loaded(model.name):
            actions.append(
                BanditAction(
                    action_id=f"load:{model.name}",
                    model=model,
                    load=True,
                )
            )
        # Unload actions for models that are present
        else:
            actions.append(
                BanditAction(
                    action_id=f"unload:{model.name}",
                    model=model,
                    load=False,
                )
            )
    return actions


def evaluate_actions(
    actions: List[BanditAction],
    pool: ModelPool,
    temperature: float,
    exploration_coef: float = 1.0,
) -> List[BanditAction]:
    """
    For each action compute:
    - propensity via Schoolfield model,
    - expected reward as ΔF (change in VFE),
    - confidence bound using a Fisher‑information‑like term.
    Returns a new list with populated fields.
    """
    total_counts = sum(pool.action_count(a.action_id) for a in actions) + 1e-9
    evaluated: List[BanditAction] = []

    for act in actions:
        prop = schoolfield_propensity(temperature)
        delta_f = pool.simulate_action(act)  # ΔF can be negative (reward) or positive (penalty)
        n_i = pool.action_count(act.action_id) + 1  # add-one smoothing
        confidence = exploration_coef * math.sqrt(math.log(total_counts) / n_i)

        evaluated.append(
            BanditAction(
                action_id=act.action_id,
                model=act.model,
                load=act.load,
                propensity=prop,
                expected_reward=-delta_f,  # we *want* lower VFE, so reward = -ΔF
                confidence_bound=confidence,
            )
        )
    return evaluated


def select_action(evaluated_actions: List[BanditAction]) -> BanditAction:
    """
    Choose the action with the highest UCB‑style score:
        score = propensity * (expected_reward + confidence_bound)
    """
    best = None
    best_score = -float("inf")
    for act in evaluated_actions:
        score = act.propensity * (act.expected_reward + act.confidence_bound)
        if score > best_score:
            best_score = score
            best = act
    assert best is not None, "No actions available to select."
    return best


def apply_action(action: BanditAction, pool: ModelPool) -> None:
    """
    Execute the selected action on the ModelPool and record it for future confidence updates.
    """
    if action.load:
        pool.add_model(action.model)
    else:
        pool.remove_model(action.model.name)
    pool.record_action(action.action_id)


# ----------------------------------------------------------------------
# Example catalog of models (could be loaded from a config file)
# ----------------------------------------------------------------------
_EXAMPLE_CATALOG: List[ModelTier] = [
    ModelTier(name="bert-base", ram_mb=1500, tier="T1"),
    ModelTier(name="gpt-small", ram_mb=2500, tier="T2"),
    ModelTier(name="vision-x", ram_mb=3000, tier="T3"),
    ModelTier(name="audio-lite", ram_mb=800, tier="T1"),
]


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    # Initialise pool and temperature schedule
    pool = ModelPool(ram_ceiling_mb=6000)
    temperature = 295.0  # Kelvin, typical room temperature

    # Run a few decision steps
    for step in range(5):
        actions = generate_actions(pool, _EXAMPLE_CATALOG)
        evaluated = evaluate_actions(actions, pool, temperature, exploration_coef=0.5)
        chosen = select_action(evaluated)
        apply_action(chosen, pool)

        print(
            f"Step {step+1}: Chosen {chosen.action_id} (load={chosen.load}) | "
            f"Propensity={chosen.propensity:.4f} | Reward={chosen.expected_reward:.4f} | "
            f"VFE={pool.free_energy():.2f}"
        )

    print("Final loaded models:", list(pool.loaded.keys()))
    print("Final VFE:", pool.free_energy())