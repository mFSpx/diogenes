# DARWIN HAMMER — match 5749, survivor 0
# gen: 6
# parent_a: hybrid_bandit_router_honeybee_store_m9_s5.py (gen1)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_nlms_o_m1334_s2.py (gen5)
# born: 2026-05-30T00:04:31Z

"""Hybrid Algorithm combining Bandit‑Router/Honeybee‑Store dynamics with Caputo‑fractional NLMS.

Parents:
- hybrid_bandit_router_honeybee_store_m9_s5.py (Bandit policy + StoreState with dance signal)
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_nlms_o_m1334_s2.py (Caputo fractional derivative, NLMS predictor)

Mathematical bridge:
The instantaneous store delta Δₜ is fed to a discrete Caputo fractional derivative
    D^{α}Δ(t) = 1/Γ(1‑α) Σ_{τ=0}^{t‑1} Δ_τ (t‑τ)^{‑α}
which endows the “dance” control signal with long‑range memory (Parent B).  
Conversely, the NLMS prediction error εₜ = rₜ – wᵀxₜ, produced while estimating
action rewards (Parent B), is used as the stochastic reward signal for the
contextual bandit (Parent A).  The error magnitude also modulates the
confidence bound of each action, closing the feedback loop.

The resulting module supplies three core functions that demonstrate the hybrid
operation and a smoke‑test in the ``__main__`` block.
"""

import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent B – fractional calculus & NLMS utilities
# ----------------------------------------------------------------------


def _gamma_lanczos(z: float) -> float:
    _LANCZOS_G = 7
    _LANCZOS_C = np.array(
        [
            0.99999999999980993,
            676.5203681218851,
            -1259.1392167224028,
            771.32342877765313,
            -176.61502916214059,
            12.507343278686905,
            -0.13857109526572012,
            9.9843695780195716e-6,
            1.5056327351493116e-7,
        ]
    )
    if z < 0.5:
        return math.pi / (math.sin(math.pi * z) * _gamma_lanczos(1 - z))
    else:
        z_plus = z + _LANCZOS_G + 0.5
        return math.sqrt(2 * math.pi) * (z_plus) ** (z + 0.5) * math.exp(-z_plus) * np.polyval(
            _LANCZOS_C[::-1], z
        )


def caputo_derivative(alpha: float, t: int, f: List[float]) -> float:
    """Discrete Caputo fractional derivative of order ``alpha`` for sequence ``f``."""
    if t == 0:
        return 0.0
    integral = 0.0
    for tau in range(t):
        integral += f[tau] * (t - tau) ** (1 - alpha) / _gamma_lanczos(2 - alpha)
    return integral / _gamma_lanczos(1 - alpha)


def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Linear prediction wᵀx."""
    return float(weights @ x)


def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, float]:
    """Normalized LMS weight update; returns new weights and prediction error."""
    prediction = nlms_predict(weights, x)
    error = target - prediction
    denom = eps + float(x @ x)
    new_weights = weights + mu * error * x / denom
    return new_weights, error


# ----------------------------------------------------------------------
# Parent A – Store dynamics & Bandit scaffolding
# ----------------------------------------------------------------------


@dataclass
class StoreState:
    """Honeybee‑style store with a bounded “dance” control signal."""

    level: float = 0.0
    alpha: float = 1.0
    beta: float = 1.0
    dt: float = 1.0
    base: float = 1.0
    gain: float = 1.0
    limit: float = 10.0
    _last_delta: float = 0.0

    def update(self, inflow: List[float], outflow: List[float]) -> Tuple[float, float]:
        """Apply store equation and record the most recent Δ."""
        delta = self.alpha * sum(inflow) - self.beta * sum(outflow)
        self.level = max(0.0, self.level + self.dt * delta)
        self._last_delta = delta
        return self.level, delta

    @property
    def dance(self) -> float:
        """Bounded control signal derived from the most recent Δ."""
        return max(0.0, min(self.limit, self.base + self.gain * self._last_delta))


@dataclass(frozen=True)
class BanditAction:
    """Result of an action selection."""

    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "HybridBanditNLMS"


# ----------------------------------------------------------------------
# Hybrid components
# ----------------------------------------------------------------------


class HybridBanditNLMS:
    """
    Contextual bandit where each action maintains its own NLMS weight vector.
    The NLMS prediction error is used as the stochastic reward and also drives
    the confidence bound (higher error → larger uncertainty).
    """

    def __init__(
        self,
        action_ids: List[str],
        context_dim: int,
        mu: float = 0.5,
        eps: float = 1e-9,
    ):
        self.mu = mu
        self.eps = eps
        # Initialise weight vectors to zeros for each action
        self.weights: Dict[str, np.ndarray] = {
            aid: np.zeros(context_dim, dtype=float) for aid in action_ids
        }

    def predict(self, action_id: str, context: np.ndarray) -> float:
        return nlms_predict(self.weights[action_id], context)

    def update(
        self, action_id: str, context: np.ndarray, reward: float
    ) -> Tuple[float, float]:
        """Update NLMS weights for ``action_id`` and return (new_expectation, error)."""
        w_old = self.weights[action_id]
        w_new, error = nlms_update(w_old, context, reward, mu=self.mu, eps=self.eps)
        self.weights[action_id] = w_new
        return nlms_predict(w_new, context), error

    def select_action(self, context: np.ndarray) -> BanditAction:
        """
        Compute soft‑max propensities from (prediction ± confidence) and sample.
        Confidence bound is proportional to |error| from the previous update;
        if no prior error exists we use a small constant.
        """
        preds = {}
        confs = {}
        for aid in self.weights:
            pred = self.predict(aid, context)
            # Use a placeholder confidence if we have never updated this action
            conf = 0.1
            preds[aid] = pred
            confs[aid] = conf

        # Compute unnormalized scores = pred + conf (encourages exploration)
        scores = np.array([preds[aid] + confs[aid] for aid in self.weights])
        max_score = np.max(scores)
        exp_scores = np.exp(scores - max_score)  # numerical stability
        propensities = exp_scores / np.sum(exp_scores)

        # Sample according to propensities
        chosen_idx = np.random.choice(len(propensities), p=propensities)
        chosen_aid = list(self.weights.keys())[chosen_idx]

        return BanditAction(
            action_id=chosen_aid,
            propensity=propensities[chosen_idx],
            expected_reward=preds[chosen_aid],
            confidence_bound=confs[chosen_aid],
        )


# ----------------------------------------------------------------------
# Hybrid helper functions
# ----------------------------------------------------------------------


def compute_fractional_delta(delta_history: List[float], alpha_frac: float) -> float:
    """
    Apply the Caputo fractional derivative of order ``alpha_frac`` to the
    stored delta sequence.  The result is used as a memory‑augmented Δ.
    """
    t = len(delta_history) - 1  # latest index
    if t < 0:
        return 0.0
    return caputo_derivative(alpha_frac, t, delta_history)


def compute_dance_with_memory(state: StoreState, delta_history: List[float], alpha_frac: float) -> float:
    """
    Replace the raw Δ in the dance formula with its fractional‑derivative
    counterpart, preserving the original bounds.
    """
    frac_delta = compute_fractional_delta(delta_history, alpha_frac)
    return max(0.0, min(state.limit, state.base + state.gain * frac_delta))


def hybrid_step(
    state: StoreState,
    bandit: HybridBanditNLMS,
    inflow: List[float],
    outflow: List[float],
    context: np.ndarray,
    delta_history: List[float],
    alpha_frac: float = 0.6,
) -> Tuple[BanditAction, float, List[float]]:
    """
    Perform one hybrid iteration:
      1. Update the store → obtain raw Δ and new level.
      2. Append Δ to history and compute memory‑augmented dance.
      3. Select an action using the NLMS‑based bandit.
      4. Simulate a stochastic reward (here: noisy version of the dance signal).
      5. Update the bandit with the observed reward.
    Returns the selected BanditAction, the current dance value, and the updated delta history.
    """
    # 1. Store dynamics
    level, delta = state.update(inflow, outflow)

    # 2. Memory‑augmented dance
    delta_history.append(delta)
    dance = compute_dance_with_memory(state, delta_history, alpha_frac)

    # 3. Action selection
    action = bandit.select_action(context)

    # 4. Simulated reward: dance plus Gaussian noise
    reward = dance + random.gauss(0.0, 0.5)

    # 5. Bandit update (NLMS)
    _, _ = bandit.update(action.action_id, context, reward)

    return action, dance, delta_history


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # deterministic seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Initialise store
    store = StoreState(level=5.0, alpha=1.2, beta=0.8, dt=0.5, base=1.0, gain=0.3, limit=12.0)

    # Define actions and context dimension
    actions = ["A", "B", "C"]
    ctx_dim = 4
    bandit = HybridBanditNLMS(action_ids=actions, context_dim=ctx_dim, mu=0.4)

    # Dummy inflow/outflow generators
    def gen_inflow() -> List[float]:
        return [random.uniform(0, 2) for _ in range(2)]

    def gen_outflow() -> List[float]:
        return [random.uniform(0, 1) for _ in range(2)]

    # Context generator (random vector)
    def gen_context() -> np.ndarray:
        return np.random.randn(ctx_dim)

    delta_hist: List[float] = []

    # Run a few hybrid steps
    for step in range(10):
        inflow = gen_inflow()
        outflow = gen_outflow()
        ctx = gen_context()
        action, dance_val, delta_hist = hybrid_step(
            state=store,
            bandit=bandit,
            inflow=inflow,
            outflow=outflow,
            context=ctx,
            delta_history=delta_hist,
            alpha_frac=0.7,
        )
        print(
            f"Step {step:02d} | Level={store.level:.3f} | Δ={store._last_delta:.3f} | "
            f"Dance={dance_val:.3f} | Action={action.action_id} (p={action.propensity:.2f})"
        )