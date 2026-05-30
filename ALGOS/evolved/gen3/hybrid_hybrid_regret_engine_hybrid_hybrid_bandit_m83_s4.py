# DARWIN HAMMER — match 83, survivor 4
# gen: 3
# parent_a: hybrid_regret_engine_hybrid_doomsday_cale_m19_s3.py (gen2)
# parent_b: hybrid_hybrid_bandit_router_koopman_operator_m64_s2.py (gen2)
# born: 2026-05-29T23:26:44Z

"""
Hybrid Regret‑Bandit‑Koopman Engine
----------------------------------

Parent algorithms:
* **A** – regret_engine + hybrid_doomsday_calendar (Gini on regret‑weighted values)
* **B** – hybrid_bandit_router + Koopman operator (store‑scaled confidence, linear forecast)

Mathematical bridge:
The regret‑weighted probability distribution `p_t` over actions is interpreted as the
observable vector `μ_t` (empirical mean rewards) for the Koopman operator.
A Gini coefficient `G_t` computed from `p_t` quantifies the inequality of the
distribution and modulates the *store* `S_t`, which in turn scales the confidence
multiplier used by the contextual bandit.  The forecast `μ̂_{t+h}=K^h μ_t` supplied
by the Koopman operator provides the exploitation term, while the store‑adjusted
confidence supplies exploration.  The resulting index

    U_a(t) = μ̂_a(t) + η·‖context‖·c_a(t)

combines both parents in a single unified decision rule.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Iterable

import numpy as np

# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class MathAction:
    """Action definition used by the regret engine."""
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True)
class MathCounterfactual:
    """Counterfactual outcome for an action."""
    action_id: str
    outcome_value: float
    probability: float = 1.0


@dataclass(frozen=True)
class BanditAction:
    """Result of a bandit selection."""
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "HybridRegretBanditKoopman"


# ----------------------------------------------------------------------
# Core components from Parent A
# ----------------------------------------------------------------------


def compute_regret_weighted_strategy(
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual],
) -> Dict[str, float]:
    """Return a softmax‑like probability distribution over actions."""
    if not actions:
        return {}
    cf = {c.action_id: c.outcome_value * c.probability for c in counterfactuals}
    vals = {
        a.id: a.expected_value - a.cost - a.risk + cf.get(a.id, 0.0) for a in actions
    }
    best = max(vals.values())
    w = {k: math.exp(v - best) for k, v in vals.items()}
    total = sum(w.values()) or 1.0
    return {k: v / total for k, v in w.items()}


def gini_coefficient(values: Iterable[float]) -> float:
    """Standard Gini coefficient for a non‑negative sequence."""
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0.0:
        return 0.0
    if xs[0] < 0:
        raise ValueError("values must be non‑negative")
    n = len(xs)
    return sum((2 * i - n - 1) * x for i, x in enumerate(xs, 1)) / (n * sum(xs))


# ----------------------------------------------------------------------
# Core components from Parent B
# ----------------------------------------------------------------------


def fit_koopman_operator(history: List[np.ndarray]) -> np.ndarray:
    """
    Fit a Koopman operator K such that μ_{t+1} ≈ K μ_t.
    Uses the least‑squares solution K = Y X⁺ where X = [μ_0 … μ_{T‑1}],
    Y = [μ_1 … μ_T].
    """
    if len(history) < 2:
        dim = history[0].shape[0] if history else 1
        return np.eye(dim)
    X = np.column_stack(history[:-1])          # shape (A, T‑1)
    Y = np.column_stack(history[1:])           # shape (A, T‑1)
    K = Y @ np.linalg.pinv(X)
    return K


def forecast_rewards(K: np.ndarray, mu: np.ndarray, steps: int = 1) -> np.ndarray:
    """Return μ̂_{t+steps} = K^{steps} μ_t."""
    if steps <= 0:
        return mu.copy()
    K_power = np.linalg.matrix_power(K, steps)
    return K_power @ mu


# ----------------------------------------------------------------------
# Hybrid mechanics
# ----------------------------------------------------------------------


@dataclass
class HybridEngine:
    """
    Maintains the state required for the hybrid algorithm:
    * store S_t (scalar)
    * counts N_a(t) per action
    * history of μ vectors (regret‑weighted probabilities)
    * current Koopman operator K
    """
    actions: List[MathAction]
    store: float = 0.0
    eta: float = 1.0                     # exploration coefficient
    counts: Dict[str, int] = field(default_factory=dict)
    history: List[np.ndarray] = field(default_factory=list)
    K: np.ndarray = field(init=False)

    def __post_init__(self):
        # initialise counts
        self.counts = {a.id: 0 for a in self.actions}
        # initialise Koopman as identity (will be updated after enough data)
        dim = len(self.actions)
        self.K = np.eye(dim)

    def _action_index(self, aid: str) -> int:
        """Map action id to integer index."""
        for i, a in enumerate(self.actions):
            if a.id == aid:
                return i
        raise KeyError(f"Unknown action id {aid}")

    def update_store(self, reward_sum: float, cost_sum: float, gini: float) -> None:
        """
        Store dynamics:
            S_{t+1} = S_t + reward_sum - cost_sum
        The Gini coefficient modulates the effective store:
            S̃_t = S_t * (1 + G_t)
        """
        self.store += reward_sum - cost_sum
        # keep store non‑negative for stability
        self.store = max(self.store, 0.0)
        self.store *= 1.0 + gini

    def step(
        self,
        counterfactuals: List[MathCounterfactual],
        observed_rewards: Dict[str, float],
    ) -> BanditAction:
        """
        Perform a full hybrid step:
        1. Compute regret‑weighted distribution → μ_t.
        2. Append μ_t to history and (re)fit Koopman if enough data.
        3. Forecast μ̂_{t+1}.
        4. Compute Gini, update store.
        5. Build confidence multiplier c_a(t).
        6. Select action via UCB‑style index.
        7. Update counts with the selected action.
        Returns the selected BanditAction.
        """
        # 1. Regret‑weighted strategy
        p = compute_regret_weighted_strategy(self.actions, counterfactuals)
        mu_t = np.array([p.get(a.id, 0.0) for a in self.actions])

        # 2. History & Koopman update
        self.history.append(mu_t)
        if len(self.history) >= 5:               # arbitrary warm‑up length
            self.K = fit_koopman_operator(self.history)

        # 3. Forecast one step ahead
        mu_hat = forecast_rewards(self.K, mu_t, steps=1)

        # 4. Gini and store update
        G = gini_coefficient(p.values())
        reward_sum = sum(observed_rewards.get(a.id, 0.0) for a in self.actions)
        cost_sum = sum(a.cost for a in self.actions)
        self.update_store(reward_sum, cost_sum, G)

        # 5. Confidence multiplier per action
        conf = {}
        for a in self.actions:
            N = self.counts[a.id]
            conf[a.id] = (1.0 + self.store / (self.store + 1.0)) / math.sqrt(1.0 + N)

        # 6. Index computation and selection
        indices = {}
        for a in self.actions:
            idx = a.id
            exploitation = mu_hat[self._action_index(idx)]
            exploration = self.eta * conf[idx]  # context norm assumed 1
            indices[idx] = exploitation + exploration

        selected_id = max(indices, key=indices.get)
        selected_propensity = p.get(selected_id, 0.0)
        selected_exploitation = mu_hat[self._action_index(selected_id)]
        selected_confidence = conf[selected_id]

        # 7. Update counts
        self.counts[selected_id] += 1

        return BanditAction(
            action_id=selected_id,
            propensity=selected_propensity,
            expected_reward=selected_exploitation,
            confidence_bound=selected_confidence,
        )

    def forecast_future(self, steps: int = 3) -> np.ndarray:
        """Public helper to obtain a multi‑step forecast from the latest μ."""
        if not self.history:
            raise RuntimeError("No history to forecast from.")
        latest_mu = self.history[-1]
        return forecast_rewards(self.K, latest_mu, steps=steps)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Define a tiny action set
    actions = [
        MathAction(id="A", expected_value=5.0, cost=1.0, risk=0.2),
        MathAction(id="B", expected_value=3.0, cost=0.5, risk=0.1),
        MathAction(id="C", expected_value=4.0, cost=0.8, risk=0.3),
    ]

    # No counterfactuals for the first round
    cf: List[MathCounterfactual] = []

    engine = HybridEngine(actions=actions, eta=0.5)

    # Simulate 10 interaction rounds
    for t in range(10):
        # Randomly generated observed rewards (could come from environment)
        observed = {
            "A": random.uniform(0, 5),
            "B": random.uniform(0, 5),
            "C": random.uniform(0, 5),
        }

        selected = engine.step(counterfactuals=cf, observed_rewards=observed)
        print(
            f"Round {t+1}: selected {selected.action_id} | "
            f"propensity={selected.propensity:.3f} | "
            f"exp_reward={selected.expected_reward:.3f} | "
            f"conf={selected.confidence_bound:.3f}"
        )

    # Demonstrate multi‑step forecast
    forecast = engine.forecast_future(steps=4)
    print("\n4‑step forecast of regret‑weighted probabilities:", forecast)