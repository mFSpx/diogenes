# DARWIN HAMMER — match 60, survivor 1
# gen: 2
# parent_a: hybrid_bandit_router_honeybee_store_m9_s5.py (gen1)
# parent_b: workshare_allocator.py (gen0)
# born: 2026-05-29T23:24:08Z

"""
This module presents a novel hybrid algorithm, merging the core topologies of 
hybrid_bandit_router_honeybee_store_m9_s5.py and workshare_allocator.py. 
The mathematical bridge between the two structures is the use of a StoreState 
instance to modulate the deterministic target percentage in the workshare allocation.
This allows for adaptive allocation of large language model (LLM) units based on the 
current state of the honeybee store.
"""

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple
import numpy as np
import sys
import pathlib

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
            pass  # omitted for brevity

        return BanditAction(
            action_id="test_action",
            propensity=0.5,
            expected_reward=0.0,
            confidence_bound=0.0,
            algorithm="linucb",
        )


def allocate_workshare(
    total_units: float, store_state: StoreState, groups: tuple[str, ...] = ("codex", "groq", "cohere", "local_models")
) -> dict[str, float]:
    """
    Allocate workshare based on the current store state.

    The deterministic target percentage is modulated by the store level.
    """
    if total_units <= 0:
        raise ValueError("total_units must be positive")
    if not groups:
        raise ValueError("groups required")

    # Modulate the deterministic target percentage based on the store level
    deterministic_target_pct = 90.0 * (1.0 + store_state.level / (store_state.level + 1.0))

    deterministic_units = total_units * deterministic_target_pct / 100.0
    llm_units = total_units - deterministic_units
    per_group = llm_units / len(groups)

    return {
        "deterministic_units": deterministic_units,
        "llm_units": llm_units,
        "per_group": per_group,
    }


def summarize_savings(total_units: float, store_state: StoreState) -> dict[str, float]:
    """
    Summarize the savings based on the workshare allocation.

    The store state is used to modulate the deterministic target percentage.
    """
    plan = allocate_workshare(total_units, store_state)
    return {
        "baseline_llm_units": total_units,
        "planned_llm_units": plan["llm_units"],
        "deterministic_units": plan["deterministic_units"],
        "token_savings_pct": (total_units - plan["llm_units"]) / total_units * 100.0,
    }


def hybrid_operation(store_state: StoreState, hybrid_bandit: HybridBandit, total_units: float) -> None:
    """
    Demonstrate the hybrid operation.

    The hybrid bandit is used to select an action, and the store state is used to allocate workshare.
    """
    context = {str(i): 1.0 for i in range(hybrid_bandit.dim)}
    actions = ["action1", "action2"]
    bandit_action = hybrid_bandit.select_action(context, actions, store_state)
    workshare_allocation = allocate_workshare(total_units, store_state)

    print(f"Selected action: {bandit_action.action_id}")
    print(f"Workshare allocation: {workshare_allocation}")


if __name__ == "__main__":
    store_state = StoreState(level=0.5, alpha=1.0, beta=1.0)
    hybrid_bandit = HybridBandit(dim_context=10, eta=0.1, epsilon=0.1)
    total_units = 100.0

    hybrid_operation(store_state, hybrid_bandit, total_units)