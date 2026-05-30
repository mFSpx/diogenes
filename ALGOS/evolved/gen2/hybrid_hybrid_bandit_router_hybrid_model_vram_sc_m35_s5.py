# DARWIN HAMMER — match 35, survivor 5
# gen: 2
# parent_a: hybrid_bandit_router_honeybee_store_m9_s1.py (gen1)
# parent_b: hybrid_model_vram_scheduler_ttt_linear_m11_s3.py (gen1)
# born: 2026-05-29T23:23:32Z

from __future__ import annotations

import math
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Any, Optional

import numpy as np

# ----------------------------------------------------------------------
# Core data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    """Result of a bandit decision."""
    action_id: str
    propensity: float            # interpreted as inflow rate
    expected_reward: float
    confidence_bound: float      # interpreted as outflow rate
    algorithm: str


@dataclass(frozen=True)
class BanditUpdate:
    """Observed reward for a given action."""
    context_id: str
    action_id: str
    reward: float
    propensity: float


# ----------------------------------------------------------------------
# Hybrid scheduler – maintains policy, virtual store and weight matrix
# ----------------------------------------------------------------------
class HybridBanditTTT:
    """
    A tighter integration of a contextual bandit (Parent A) and a linear
    TTT model (Parent B).  The virtual VRAM store influences the learning
    rate *and* the bandit’s propensity, creating a deeper feedback loop.
    """

    DEFAULT_BUDGET_MB = 8192  # assumed total VRAM budget for reporting

    def __init__(
        self,
        d_in: int,
        d_out: Optional[int] = None,
        seed: int = 0,
        base_eta: float = 0.01,
        alpha: float = 1.0,
        beta: float = 1.0,
        dt: float = 1.0,
        store_decay: float = 0.99,
    ) -> None:
        """
        Parameters
        ----------
        d_in, d_out : dimensions of the TTT weight matrix.
        seed        : RNG seed for reproducibility.
        base_eta    : Baseline learning rate before modulation.
        alpha, beta : Coefficients for the store differential equation.
        dt          : Time step for store integration.
        store_decay : Exponential decay applied to the store each step
                      (simulates memory eviction).
        """
        self.rng = np.random.default_rng(seed)
        self.base_eta = base_eta
        self.alpha = alpha
        self.beta = beta
        self.dt = dt
        self.store_decay = store_decay

        # policy statistics: action_id -> [cumulative_reward, count]
        self._policy: Dict[str, List[float]] = {}

        # virtual VRAM store per *key* (e.g. per context or per action)
        self._store: Dict[str, float] = {}

        # initialise weight matrix once; it persists across calls
        self.W = self._init_ttt(d_in, d_out, seed)

    # ------------------------------------------------------------------
    # Bandit utilities
    # ------------------------------------------------------------------
    def _reward(self, a: str) -> float:
        total, n = self._policy.get(a, [0.0, 0.0])
        return total / n if n else 0.0

    def _count(self, a: str) -> float:
        return self._policy.get(a, [0.0, 0.0])[1]

    def select_action(
        self,
        context: Dict[str, float],
        actions: List[str],
        algorithm: str = "linucb",
        epsilon: float = 0.1,
    ) -> BanditAction:
        """
        Choose an action using a soft‑max over LinUCB‑style scores.
        The resulting propensity is a proper probability (sums to 1).
        """
        if not actions:
            raise ValueError("actions list cannot be empty")

        # compute a score for each action
        if algorithm == "epsilon_greedy" and random.random() < epsilon:
            chosen = random.choice(actions)
            scores = {a: 0.0 for a in actions}
        else:
            # LinUCB‑style surrogate score
            scale = math.sqrt(
                sum(float(v) * float(v) for v in context.values())
            ) if context else 1.0
            scores = {}
            for a in actions:
                reward_est = self._reward(a)
                exploration = 0.1 * scale / math.sqrt(1 + self._count(a))
                scores[a] = reward_est + exploration

            # soft‑max to obtain a probability distribution
            max_score = max(scores.values())
            exp_vals = {a: math.exp(s - max_score) for a, s in scores.items()}
            total = sum(exp_vals.values())
            probs = {a: v / total for a, v in exp_vals.items()}
            chosen = max(actions, key=lambda a: probs[a])

        # propensity is the probability of the chosen action under the soft‑max
        # (if epsilon_greedy was used we fall back to uniform)
        if algorithm == "epsilon_greedy" and random.random() < epsilon:
            propensity = 1.0 / len(actions)
        else:
            propensity = probs[chosen]

        confidence = 1.0 / math.sqrt(1 + self._count(chosen))

        return BanditAction(
            action_id=chosen,
            propensity=propensity,
            expected_reward=self._reward(chosen),
            confidence_bound=confidence,
            algorithm=algorithm,
        )

    def update_policy(self, updates: List[BanditUpdate]) -> None:
        """Incorporate observed rewards into the bandit statistics."""
        for u in updates:
            stats = self._policy.setdefault(u.action_id, [0.0, 0.0])
            stats[0] += float(u.reward)
            stats[1] += 1.0

    # ------------------------------------------------------------------
    # Store dynamics
    # ------------------------------------------------------------------
    def _update_store(
        self,
        key: str,
        inflow: float,
        outflow: float,
    ) -> Tuple[float, float]:
        """
        Update the virtual VRAM store for ``key``.
        Returns (new_store, delta).
        """
        store = self._store.get(key, 0.0)
        delta = self.alpha * inflow - self.beta * outflow
        new_store = max(0.0, store + self.dt * delta)
        # apply exponential decay to model eviction / aging
        new_store *= self.store_decay
        self._store[key] = new_store
        return new_store, delta

    # ------------------------------------------------------------------
    # TTT core (Parent B)
    # ------------------------------------------------------------------
    @staticmethod
    def _init_ttt(d_in: int, d_out: Optional[int] = None, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        if d_out is None:
            d_out = d_in
        return rng.standard_normal((d_out, d_in)) * 0.01

    @staticmethod
    def _ttt_grad(W: np.ndarray, x: np.ndarray, target: Optional[np.ndarray] = None) -> np.ndarray:
        pred = W @ x
        t = x if target is None else target
        residual = pred - t
        return 2.0 * np.outer(residual, x)

    @staticmethod
    def _ttt_step(W: np.ndarray, x: np.ndarray, eta: float, target: Optional[np.ndarray] = None) -> np.ndarray:
        g = HybridBanditTTT._ttt_grad(W, x, target)
        return W - eta * g

    @staticmethod
    def weight_matrix_memory_mb(W: np.ndarray, dtype: np.dtype = np.float64) -> int:
        element_bytes = np.dtype(dtype).itemsize
        total_bytes = W.size * element_bytes
        return int(round(total_bytes / (1024 * 1024), 0))

    # ------------------------------------------------------------------
    # Hybrid operations
    # ------------------------------------------------------------------
    def _compute_eta(self, action: BanditAction, store: float) -> float:
        """
        Learning rate modulation that blends:
        - baseline ``base_eta``,
        - bandit propensity,
        - current store level (more store → slightly larger steps, but bounded).
        """
        store_factor = store / (store + 1.0)          # ∈ (0,1)
        return self.base_eta * (1.0 + action.propensity) * (1.0 + store_factor)

    def hybrid_step(
        self,
        x: np.ndarray,
        action: BanditAction,
        store_key: str,
        target: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float, float]:
        """
        Execute a single hybrid iteration:
        1. Update virtual store using action.propensity / action.confidence_bound.
        2. Compute an adaptive learning rate.
        3. Perform a TTT gradient‑descent step.
        Returns the new hidden state, the updated store value and the store delta.
        """
        new_store, delta = self._update_store(
            key=store_key,
            inflow=action.propensity,
            outflow=action.confidence_bound,
        )
        eta = self._compute_eta(action, new_store)
        self.W = self._ttt_step(self.W, x, eta=eta, target=target)
        hidden = self.W @ x
        return hidden, new_store, delta

    def run_sequence(
        self,
        context: Dict[str, float],
        actions: List[str],
        x_seq: np.ndarray,
        algorithm: str = "linucb",
        epsilon: float = 0.1,
        store_key: Optional[str] = None,
        target_seq: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Full end‑to‑end hybrid run over a sequence of inputs.

        Parameters
        ----------
        context, actions : bandit inputs.
        x_seq            : shape (T, d_in) – input sequence.
        algorithm, epsilon : bandit configuration.
        store_key        : identifier for the virtual store; defaults to
                          the first action id (stable across runs).
        target_seq       : optional target for each step (same shape as x_seq).

        Returns
        -------
        dict with keys:
            'hidden_states' : (T, d_out) array,
            'final_W'       : weight matrix after the last step,
            'store'         : final store value,
            'store_delta'   : last store delta,
            'bandit_action' : the BanditAction used,
        """
        if store_key is None:
            store_key = actions[0]

        # 1. Bandit decision (single action for the whole sequence)
        bandit_action = self.select_action(context, actions, algorithm, epsilon)

        # 2. Iterate through the sequence, updating store and weights each step
        T = x_seq.shape[0]
        hidden_states = np.empty((T, self.W.shape[0]), dtype=float)
        last_store = 0.0
        last_delta = 0.0

        for t in range(T):
            target = None if target_seq is None else target_seq[t]
            hidden, last_store, last_delta = self.hybrid_step(
                x_seq[t],
                bandit_action,
                store_key,
                target=target,
            )
            hidden_states[t] = hidden

        return {
            "hidden_states": hidden_states,
            "final_W": self.W.copy(),
            "store": last_store,
            "store_delta": last_delta,
            "bandit_action": bandit_action,
        }

    # ------------------------------------------------------------------
    # Reporting utilities
    # ------------------------------------------------------------------
    def report_vram_plan(self, reserve_mb: int = 768) -> Dict[str, Any]:
        """
        Produce a VRAM‑allocation plan based on the current weight matrix
        and the virtual store.  The plan mirrors the intent of Parent B
        while taking the hybrid store into account.
        """
        needed = self.weight_matrix_memory_mb(self.W)
        # approximate store usage as a fraction of reserve (purely illustrative)
        store_mb = int(self._store.get("global", 0.0) * 1024)  # treat store as GB→MB proxy
        available = self.DEFAULT_BUDGET_MB - reserve_mb - store_mb
        status = "fit" if needed <= available else "exceed"
        return {
            "matrix_mb": needed,
            "store_mb_est": store_mb,
            "available_mb": max(0, available),
            "status": status,
        }

    # ------------------------------------------------------------------
    # Reset utilities
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear policy, store and re‑initialise the weight matrix."""
        self._policy.clear()
        self._store.clear()
        # re‑initialise with same dimensions
        d_in = self.W.shape[1]
        d_out = self.W.shape[0]
        self.W = self._init_ttt(d_in, d_out, seed=self.rng.integers(0, 2**31 - 1))


# ----------------------------------------------------------------------
# Example usage (can be removed in production)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # synthetic demo
    d_in = 8
    T = 20
    hybrid = HybridBanditTTT(d_in=d_in, seed=42)

    ctx = {"feature1": 0.3, "feature2": 1.2}
    act_list = [f"a{i}" for i in range(5)]
    xs = np.random.randn(T, d_in)

    result = hybrid.run_sequence(
        context=ctx,
        actions=act_list,
        x_seq=xs,
        algorithm="linucb",
        epsilon=0.05,
    )

    print("Final store:", result["store"])
    print("VRAM plan:", hybrid.report_vram_plan())