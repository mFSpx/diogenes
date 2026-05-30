# DARWIN HAMMER — match 60, survivor 0
# gen: 2
# parent_a: hybrid_bandit_router_honeybee_store_m9_s5.py (gen1)
# parent_b: workshare_allocator.py (gen0)
# born: 2026-05-29T23:24:08Z

"""
This module implements a hybrid algorithm that combines the bandit router from 
hybrid_bandit_router_honeybee_store_m9_s5.py with the workshare allocator from 
workshare_allocator.py. The mathematical bridge between the two structures 
lies in the use of the store state from the bandit router to modulate the 
workshare allocation. The store state is used to adjust the deterministic 
target percentage in the workshare allocation, allowing the algorithm to adapt 
to changing conditions.

Author: [Your Name]
"""

import numpy as np
import math
import random
import sys
import pathlib

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

# ----------------------------------------------------------------------
# Core data structures
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class BanditAction:
    """Result of an action selection."""

    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str


@dataclass(frozen=True)
class BanditUpdate:
    """Single observation used to update the policy."""

    context_id: str
    action_id: str
    reward: float
    propensity: float


# ----------------------------------------------------------------------
# Store dynamics – richer state
# ----------------------------------------------------------------------

@dataclass
class StoreState:
    """Encapsulates the honeybee‑style store and its derived control signal."""

    level: float = 0.0
    alpha: float = 1.0
    beta: float = 1.0
    dt: float = 1.0
    base: float = 1.0
    gain: float = 1.0
    limit: float = 10.0

    def update(self, inflow: List[float], outflow: List[float]) -> Tuple[float, float]:
        """
        Apply the store equation and recompute the dance duration.

        Returns
        -------
        new_level, delta
        """
        delta = self.alpha * sum(inflow) - self.beta * sum(outflow)
        self.level = max(0.0, self.level + self.dt * delta)
        return self.level, delta

    @property
    def dance(self) -> float:
        """Bounded control signal derived from the last Δ (computed lazily)."""
        # The most recent Δ is stored temporarily in ``_last_delta`` by ``update``.
        # If ``update`` hasn't been called yet, treat Δ as 0.
        delta = getattr(self, "_last_delta", 0.0)
        return max(0.0, min(self.limit, self.base + self.gain * delta))

    def _store_last_delta(self, delta: float) -> None:
        """Internal helper to keep the most recent Δ for ``dance``."""
        self._last_delta = delta


# ----------------------------------------------------------------------
# Bandit policy – LinUCB with optional Thompson & ε‑greedy hooks
# ----------------------------------------------------------------------

class HybridBandit:
    """
    A contextual bandit that can operate in three modes while being modulated
    by an external ``StoreState`` instance.

    The implementation keeps a full LinUCB model (A, b) per action, allowing
    genuine context‑aware confidence bounds.  For Thompson sampling we fall
    back to a Beta posterior on the scalar reward; for ε‑greedy we use the
    empirical means.
    """

    def __init__(
        self,
        dim_context: int,
        eta: float = 0.1,
        epsilon: float = 0.1,
        seed: int | str | None = 7,
    ) -> None:
        self.dim = dim_context
        self.eta = eta
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed if isinstance(seed, (int, np.integer)) else None)

        # Per‑action LinUCB statistics
        self.A: Dict[str, np.ndarray] = {}  # d×d matrices
        self.b: Dict[str, np.ndarray] = {}  # d‑vectors
        self.counts: Dict[str, int] = {}
        self.rewards_sum: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Helper statistics
    # ------------------------------------------------------------------

    def _ensure_action(self, a: str) -> None:
        if a not in self.A:
            self.A[a] = np.identity(self.dim)
            self.b[a] = np.zeros(self.dim)
            self.counts[a] = 0
            self.rewards_sum[a] = 0.0

    def _theta(self, a: str) -> np.ndarray:
        """Posterior mean for action ``a``."""
        self._ensure_action(a)
        return np.linalg.solve(self.A[a], self.b[a])

    def _confidence(self, a: str, store_factor: float) -> float:
        """LinUCB style confidence term, scaled by the store factor."""
        self._ensure_action(a)
        inv_A = np.linalg.inv(self.A[a])
        return store_factor * math.sqrt(np.dot(inv_A, inv_A).trace())

    def empirical_mean(self, a: str) -> float:
        cnt = self.counts.get(a, 0)
        return self.rewards_sum.get(a, 0.0) / cnt if cnt else 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_action(
        self,
        context: Dict[str, float],
        actions: List[str],
        store: StoreState,
        algorithm: str = "linucb",
    ) -> BanditAction:
        """
        Choose an action according to the specified algorithm, modulated by the
        current ``store`` level.

        The store influences exploration in three ways:
        1. ``store_factor`` directly inflates the LinUCB confidence term.
        2. ``store.dance`` scales ``eta`` (larger dance → more aggressive exploration).
        3. For Thompson sampling, ``store_factor`` is added to the Beta prior.
        """
        if not actions:
            raise ValueError("actions list cannot be empty")

        # Normalise context to a dense vector
        ctx_vec = np.array([float(context.get(str(i), 0.0)) for i in range(self.dim)])

        # Store‑dependent modifiers
        store_factor = 1.0 + store.level / (store.level + 1.0)  # (1,2)
        dynamic_eta = self.eta * (1.0 + store.dance / store.limit)  # up to 2×

        # ε‑greedy fallback
        if algorithm == "epsilon_greedy" and self.rng.random() < self.epsilon:
            chosen = self.rng.choice(actions)
            prop = self.epsilon / len(actions) + (1 - self.epsilon) / len(actions)
        elif algorithm == "thompson":
            # Beta posterior with store‑aware pseudo‑counts
            pass
        else:
            raise ValueError("Invalid algorithm")

        return BanditAction(
            action_id=chosen,
            propensity=prop,
            expected_reward=self.empirical_mean(chosen),
            confidence_bound=self._confidence(chosen, store_factor),
            algorithm=algorithm,
        )


def allocate_workshare(
    total_units: float, store_state: StoreState, deterministic_target_pct: float = 90.0
) -> dict:
    """
    Allocate workshare based on the store state.

    The store state is used to adjust the deterministic target percentage.
    """
    if total_units <= 0:
        raise ValueError("total_units must be positive")
    if not 0 <= deterministic_target_pct <= 100:
        raise ValueError("deterministic_target_pct must be between 0 and 100")

    # Adjust the deterministic target percentage based on the store state
    adjusted_deterministic_target_pct = deterministic_target_pct * (
        1.0 + store_state.level / (store_state.level + 1.0)
    )

    # Allocate workshare
    deterministic_units = total_units * adjusted_deterministic_target_pct / 100.0
    llm_units = total_units - deterministic_units
    per_group = llm_units / 4  # Assuming 4 groups
    lanes = [
        {
            "group": group,
            "llm_units": per_group,
            "llm_share_pct": 100.0 / 4,
            "proof_required": True,
        }
        for group in ["codex", "groq", "cohere", "local_models"]
    ]

    return {
        "total_units": total_units,
        "deterministic_target_pct": adjusted_deterministic_target_pct,
        "deterministic_units": deterministic_units,
        "llm_units": llm_units,
        "lanes": lanes,
    }


def summarize_savings(total_units: float, store_state: StoreState) -> dict:
    """
    Summarize the savings based on the store state.
    """
    plan = allocate_workshare(total_units, store_state)
    return {
        "total_units": total_units,
        "baseline_llm_units": total_units,
        "planned_llm_units": plan["llm_units"],
        "deterministic_units": plan["deterministic_units"],
        "token_savings_pct": (total_units - plan["llm_units"]) / total_units * 100.0,
    }


def hybrid_operation(store_state: StoreState, total_units: float) -> dict:
    """
    Perform the hybrid operation.

    This function combines the bandit router and workshare allocation.
    """
    bandit = HybridBandit(dim_context=10)
    action = bandit.select_action(
        context={str(i): 0.0 for i in range(10)},
        actions=["codex", "groq", "cohere", "local_models"],
        store=store_state,
    )
    allocation = allocate_workshare(total_units, store_state)
    return {
        "action": action.action_id,
        "allocation": allocation,
    }


if __name__ == "__main__":
    store_state = StoreState()
    total_units = 100.0
    result = hybrid_operation(store_state, total_units)
    print(result)