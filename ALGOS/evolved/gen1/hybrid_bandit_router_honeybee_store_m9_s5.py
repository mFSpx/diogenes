# DARWIN HAMMER — match 9, survivor 5
# gen: 1
# parent_a: bandit_router.py (gen0)
# parent_b: honeybee_store.py (gen0)
# born: 2026-05-29T23:16:48Z

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

import numpy as np

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
            def sample(a: str) -> float:
                mean = self.empirical_mean(a)
                a_param = 1.0 + max(0.0, mean) * store_factor
                b_param = 1.0 + max(0.0, 1.0 - mean) * store_factor
                return self.rng.beta(a_param, b_param)

            chosen = max(actions, key=sample)
            prop = 1.0 / len(actions)  # Thompson does not expose exact propensities
        else:  # default: LinUCB
            def ucb_score(a: str) -> float:
                theta = self._theta(a)
                mean = float(theta @ ctx_vec)
                conf = self._confidence(a, store_factor)
                return mean + dynamic_eta * conf

            chosen = max(actions, key=ucb_score)
            prop = 1.0 / len(actions)

        # Diagnostic quantities
        exp_reward = self.empirical_mean(chosen)
        conf_bound = self._confidence(chosen, store_factor)

        return BanditAction(
            action_id=chosen,
            propensity=prop,
            expected_reward=exp_reward,
            confidence_bound=conf_bound,
            algorithm=algorithm,
        )

    def update(
        self,
        updates: List[BanditUpdate],
        context: Dict[str, float],
    ) -> None:
        """
        Apply a batch of observations to the LinUCB statistics.
        For Thompson/ε‑greedy we only need the scalar aggregates.
        """
        ctx_vec = np.array([float(context.get(str(i), 0.0)) for i in range(self.dim)])

        for u in updates:
            self._ensure_action(u.action_id)
            self.counts[u.action_id] += 1
            self.rewards_sum[u.action_id] += u.reward

            # LinUCB matrix update
            self.A[u.action_id] += np.outer(ctx_vec, ctx_vec)
            self.b[u.action_id] += u.reward * ctx_vec

    def reset(self) -> None:
        """Clear all learned statistics."""
        self.A.clear()
        self.b.clear()
        self.counts.clear()
        self.rewards_sum.clear()


# ----------------------------------------------------------------------
# Hybrid step – full loop
# ----------------------------------------------------------------------


def hybrid_step(
    bandit: HybridBandit,
    store: StoreState,
    context: Dict[str, float],
    actions: List[str],
    true_reward_fn: Callable[[str, Dict[str, float]], float],
    algorithm: str = "linucb",
) -> Tuple[BanditAction, float, StoreState]:
    """
    Perform a single iteration of the hybrid system:

    1. Select an action using the bandit, modulated by the current store.
    2. Sample a stochastic reward via ``true_reward_fn``.
    3. Feed the reward back to the bandit (LinUCB update).
    4. Propagate the reward as *inflow* to the store and apply store dynamics.

    Parameters
    ----------
    bandit : HybridBandit
        The contextual bandit instance.
    store : StoreState
        Current store; will be mutated in‑place.
    context : dict
        Feature vector (numeric values) for the decision.
    actions : list[str]
        Candidate actions.
    true_reward_fn : callable(action_id, context) -> float
        Ground‑truth stochastic reward generator.
    algorithm : {'linucb', 'epsilon_greedy', 'thompson'}
        Bandit algorithm to employ.

    Returns
    -------
    (action, reward, updated_store)
    """
    # 1️⃣ Action selection
    action = bandit.select_action(context, actions, store, algorithm)

    # 2️⃣ Observe reward
    reward = float(true_reward_fn(action.action_id, context))

    # 3️⃣ Bandit update (single observation)
    update = BanditUpdate(
        context_id="default",
        action_id=action.action_id,
        reward=reward,
        propensity=action.propensity,
    )
    bandit.update([update], context)

    # 4️⃣ Store dynamics – reward as inflow, a fixed unit cost as outflow
    inflow = [reward]
    outflow = [1.0]  # could be parameterised later
    new_level, delta = store.update(inflow, outflow)
    store._store_last_delta(delta)  # keep Δ for the dance signal

    return action, reward, store


# ----------------------------------------------------------------------
# Example usage (smoke test)
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Simple synthetic reward: Bernoulli with arm‑dependent probability
    def bernoulli_reward(action_id: str, _: Dict[str, float]) -> float:
        probs = {"A": 0.8, "B": 0.5, "C": 0.2}
        return 1.0 if random.random() < probs.get(action_id, 0.0) else 0.0

    dim = 5
    bandit = HybridBandit(dim_context=dim, eta=0.2, epsilon=0.15, seed=42)
    store = StoreState(level=0.0, alpha=1.0, beta=0.5, dt=0.5, base=1.0, gain=2.0, limit=8.0)

    actions = ["A", "B", "C"]
    ctx = {str(i): random.random() for i in range(dim)}

    for t in range(30):
        act, rew, store = hybrid_step(
            bandit=bandit,
            store=store,
            context=ctx,
            actions=actions,
            true_reward_fn=bernoulli_reward,
            algorithm="linucb",
        )
        print(
            f"t={t:02d} | act={act.action_id} | rew={rew:.1f} | "
            f"store={store.level:.2f} | dance={store.dance:.2f}"
        )