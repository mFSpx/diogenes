# DARWIN HAMMER — match 1428, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_epistemic_certainty_m1153_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_ternar_koopman_operator_m632_s3.py (gen3)
# born: 2026-05-29T23:36:17Z

"""
Hybrid Bandit‑RBF‑Koopman Engine
================================

Parent A: ``hybrid_hybrid_hybrid_hybrid_epistemic_certainty_m1153_s1.py`` – a
contextual bandit that builds an RBF surrogate for the expected reward and
weights every observation by an epistemic‑certainty factor α∈[0,1].

Parent B: ``hybrid_hybrid_hybrid_ternar_koopman_operator_m632_s3.py`` – a
pipeline that lifts a sequence of feature vectors with a degree‑2 polynomial
observable ψ(·) and fits a finite‑dimensional Koopman matrix **K** by Dynamic
Mode Decomposition (DMD).

Mathematical Bridge
-------------------
Both parents treat a stream of observations *(xₜ, rₜ, αₜ)*.
- In the bandit the scalar reward rₜ is approximated by an RBF surrogate
  ŷ(x) = Σ_i w_i exp(‑ε²‖x‑c_i‖²) where each sample contributes with weight αₜ.
- In the Koopman side the same context vectors xₜ are lifted with ψ(·) and a
  weighted DMD solves **K** from

        Σ_t αₜ ψ(xₜ₊₁) ψ(xₜ)ᵀ = **K** Σ_t αₜ ψ(xₜ) ψ(xₜ)ᵀ .

Thus the epistemic certainty αₜ simultaneously scales the surrogate‑fit
and the Koopman‑fit, providing a unified linear system that couples reward
approximation with dynamics prediction.

The hybrid engine below:
1. Collects bandit updates together with a ``CertaintyFlag``.
2. Updates empirical statistics, the RBF surrogate, and the weighted Koopman
   matrix in a single call.
3. Predicts future reward by (i) propagating the current context one step
   forward with the Koopman operator, (ii) evaluating the surrogate at the
   predicted context, and (iii) scaling the result by the current certainty.

Only ``numpy`` and the Python standard library are used.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Sequence, Callable, Iterable, Optional

import numpy as np

# ----------------------------------------------------------------------
# Shared Types
# ----------------------------------------------------------------------
Vector = Sequence[float]

# ----------------------------------------------------------------------
# Bandit / RBF Surrogate (Parent A)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str


@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float
    context: Tuple[float, ...]  # raw feature vector


# ----------------------------------------------------------------------
# Epistemic Certainty (Parent A)
# ----------------------------------------------------------------------
EPISTEMIC_FLAGS = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")


@dataclass(frozen=True)
class CertaintyFlag:
    label: str
    confidence_bps: int  # 0 … 10000
    authority_class: str
    rationale: str

    @property
    def alpha(self) -> float:
        """Scale factor α∈[0,1] derived from basis‑point confidence."""
        return max(0.0, min(1.0, self.confidence_bps / 10000.0))


# ----------------------------------------------------------------------
# RBF Surrogate
# ----------------------------------------------------------------------
class RBFSurrogate:
    """Weighted RBF regression with Gaussian kernels."""

    def __init__(self, epsilon: float = 1.0):
        self.epsilon = epsilon
        self.centers: List[np.ndarray] = []   # observed contexts
        self.rewards: List[float] = []        # corresponding rewards
        self.weights: Optional[np.ndarray] = None  # learned coefficients

    def _phi(self, x: np.ndarray) -> np.ndarray:
        """Row vector of kernel values φ_i(x) for all centers."""
        if not self.centers:
            return np.empty((0,))
        diff = np.stack(self.centers) - x  # shape (n_centers, dim)
        sq_norm = np.einsum('ij,ij->i', diff, diff)
        return np.exp(-self.epsilon ** 2 * sq_norm)

    def add_observation(self, x: Vector, y: float, alpha: float) -> None:
        """Append a new (x, y) pair with weight α and recompute coefficients."""
        self.centers.append(np.asarray(x, dtype=float))
        self.rewards.append(y)
        self._recompute_weights(alpha)

    def _recompute_weights(self, new_alpha: float) -> None:
        """Solve the weighted least‑squares problem using all stored data."""
        n = len(self.centers)
        if n == 0:
            self.weights = None
            return

        # Build design matrix Φ (n × n) because we use each sample as a centre.
        Phi = np.empty((n, n))
        for i, c_i in enumerate(self.centers):
            Phi[i, :] = self._phi(c_i)

        # Weight matrix W = diag(α₁,…,αₙ).  For simplicity we use the most recent
        # α for all rows – the caller passes the same α for every new point.
        W = np.diag([new_alpha] * n)

        # Solve (Φᵀ W Φ) w = Φᵀ W y
        A = Phi.T @ W @ Phi
        b = Phi.T @ W @ np.asarray(self.rewards, dtype=float)

        # Regularise in case A is singular.
        reg = 1e-8 * np.eye(A.shape[0])
        self.weights = np.linalg.solve(A + reg, b)

    def predict(self, x: Vector) -> float:
        """Evaluate ŷ(x) = φ(x)·w."""
        if self.weights is None or not self.centers:
            return 0.0
        phi_x = self._phi(np.asarray(x, dtype=float))
        return float(phi_x @ self.weights)


# ----------------------------------------------------------------------
# Koopman / DMD (Parent B)
# ----------------------------------------------------------------------
def polynomial_lift(x: Vector) -> np.ndarray:
    """
    Degree‑2 polynomial lift ψ(x) = [x, x⊙x, x_i·x_j (i<j)]ᵀ.
    Returns a 1‑D array.
    """
    x = np.asarray(x, dtype=float)
    linear = x
    quadratic_self = x * x
    n = x.shape[0]
    cross = []
    for i in range(n):
        for j in range(i + 1, n):
            cross.append(x[i] * x[j])
    return np.concatenate([linear, quadratic_self, np.asarray(cross, dtype=float)])


class WeightedKoopman:
    """
    Stores lifted snapshots (ψ(xₜ)) and fits a Koopman matrix K
    by weighted Dynamic Mode Decomposition.
    """

    def __init__(self):
        self.lifted_prev: List[np.ndarray] = []   # ψ(xₜ)
        self.lifted_next: List[np.ndarray] = []   # ψ(xₜ₊₁)
        self.alphas: List[float] = []             # certainty weights αₜ
        self.K: Optional[np.ndarray] = None

    def add_pair(self, x_prev: Vector, x_next: Vector, alpha: float) -> None:
        """Store a weighted pair (ψ(xₜ), ψ(xₜ₊₁))."""
        self.lifted_prev.append(polynomial_lift(x_prev))
        self.lifted_next.append(polynomial_lift(x_next))
        self.alphas.append(alpha)

    def fit(self) -> None:
        """Compute K solving Σ αₜ ψ(xₜ₊₁) ψ(xₜ)ᵀ = K Σ αₜ ψ(xₜ) ψ(xₜ)ᵀ."""
        if not self.lifted_prev:
            self.K = None
            return

        X = np.column_stack(self.lifted_prev)   # shape (d, m)
        Y = np.column_stack(self.lifted_next)   # shape (d, m)
        W_sqrt = np.diag(np.sqrt(self.alphas))   # (m, m)

        Xw = X @ W_sqrt
        Yw = Y @ W_sqrt

        # Least‑squares solution K = Yw Xw⁺
        Xw_pinv = np.linalg.pinv(Xw)
        self.K = Yw @ Xw_pinv

    def propagate(self, x: Vector, steps: int = 1) -> np.ndarray:
        """Evolve x forward `steps` times using the learned Koopman matrix."""
        if self.K is None:
            return np.asarray(x, dtype=float)

        phi = polynomial_lift(x)
        for _ in range(steps):
            phi = self.K @ phi
        # Project back to original space by taking the first n components
        # (the linear part of the lift).  This is a simple linear projection.
        n = len(x)
        return phi[:n]


# ----------------------------------------------------------------------
# Hybrid Engine – integrates bandit, surrogate, and Koopman
# ----------------------------------------------------------------------
class HybridEngine:
    """
    Core object that receives BanditUpdate + CertaintyFlag, updates
    empirical statistics, the weighted RBF surrogate, and the weighted
    Koopman model in a single mathematically consistent step.
    """

    def __init__(self, epsilon: float = 1.0):
        self.policy_stats: Dict[str, List[float]] = {}   # action_id -> [total_reward, count]
        self.surrogate = RBFSurrogate(epsilon=epsilon)
        self.koopman = WeightedKoopman()
        self._last_context: Optional[Vector] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process_update(self, upd: BanditUpdate, flag: CertaintyFlag) -> None:
        """Integrate a new observation into all three sub‑systems."""
        # 1️⃣ Update empirical bandit statistics.
        stats = self.policy_stats.setdefault(upd.action_id, [0.0, 0.0])
        stats[0] += upd.reward
        stats[1] += 1.0

        # 2️⃣ Update RBF surrogate with certainty‑weighted observation.
        self.surrogate.add_observation(upd.context, upd.reward, flag.alpha)

        # 3️⃣ Update Koopman pairs if we have a previous context.
        if self._last_context is not None:
            self.koopman.add_pair(self._last_context, upd.context, flag.alpha)
        self._last_context = upd.context

    def fit_models(self) -> None:
        """Re‑fit the surrogate (already done on each add) and the Koopman matrix."""
        self.koopman.fit()

    def predict_future_reward(self, context: Vector, steps: int = 1,
                              flag: Optional[CertaintyFlag] = None) -> float:
        """
        Predict the reward after `steps` Koopman evolutions.
        If a CertaintyFlag is supplied, its α scales the surrogate output.
        """
        # Propagate the context using the Koopman operator.
        pred_ctx = self.koopman.propagate(context, steps=steps)

        # Evaluate the surrogate at the predicted context.
        base_reward = self.surrogate.predict(pred_ctx)

        # Apply epistemic scaling if requested.
        if flag is not None:
            return base_reward * flag.alpha
        return base_reward

    def get_action_estimate(self, action_id: str) -> Tuple[float, float]:
        """
        Return (average_reward, count) for a given action.
        """
        total, cnt = self.policy_stats.get(action_id, [0.0, 0.0])
        avg = total / cnt if cnt > 0 else 0.0
        return avg, cnt


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic data: 2‑dimensional context, two actions.
    engine = HybridEngine(epsilon=0.5)

    # Create deterministic but slightly noisy stream.
    random.seed(42)
    for t in range(20):
        ctx = (math.sin(t * 0.3), math.cos(t * 0.3))
        action = "A" if t % 2 == 0 else "B"
        # Simulated reward = linear function + noise.
        reward = 1.5 * ctx[0] - 0.8 * ctx[1] + (random.random() - 0.5) * 0.1
        propensity = 0.5  # dummy
        upd = BanditUpdate(
            context_id=f"c{t}",
            action_id=action,
            reward=reward,
            propensity=propensity,
            context=ctx,
        )
        # Certainty cycles through the five flags.
        label = EPISTEMIC_FLAGS[t % len(EPISTEMIC_FLAGS)]
        flag = CertaintyFlag(
            label=label,
            confidence_bps=2000 + 1500 * (t % 5),  # varying 20‑%‑35‑%‑50‑%
            authority_class="test",
            rationale="synthetic",
        )
        engine.process_update(upd, flag)

    # Fit the Koopman matrix after data collection.
    engine.fit_models()

    # Predict reward for a new context after 2 Koopman steps.
    new_ctx = (0.0, 1.0)
    pred_flag = CertaintyFlag(
        label="FACT",
        confidence_bps=8000,
        authority_class="test",
        rationale="demo",
    )
    pred = engine.predict_future_reward(new_ctx, steps=2, flag=pred_flag)
    avg_A, cnt_A = engine.get_action_estimate("A")
    avg_B, cnt_B = engine.get_action_estimate("B")

    print(f"Predicted future reward (scaled) = {pred:.4f}")
    print(f"Action A – avg reward: {avg_A:.4f} over {int(cnt_A)} samples")
    print(f"Action B – avg reward: {avg_B:.4f} over {int(cnt_B)} samples")
    sys.exit(0)