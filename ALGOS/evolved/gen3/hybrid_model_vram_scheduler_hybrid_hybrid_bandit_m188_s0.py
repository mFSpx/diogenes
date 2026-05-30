# DARWIN HAMMER — match 188, survivor 0
# gen: 3
# parent_a: model_vram_scheduler.py (gen0)
# parent_b: hybrid_hybrid_bandit_router_hybrid_model_vram_sc_m35_s2.py (gen2)
# born: 2026-05-29T23:27:25Z

"""Hybrid VRAM‑Bandit Scheduler

This module fuses the **VRAM scheduler** (Parent A) with the **bandit‑TTT
scheduler** (Parent B).  The mathematical bridge is the *store equation* of
the honeybee primitive:

    Δstore = α·Σ(inflow) – β·Σ(outflow)
    storeₜ₊₁ = max(0, storeₜ + Δstore·dt)

We interpret the bandit‑produced ``propensity`` as an inflow of virtual
memory and the ``confidence_bound`` as an outflow.  Simultaneously the
bandit action modulates the learning‑rate of a linear weight matrix ``W``
that estimates the VRAM cost of an artifact:

    η = η₀·(1 + propensity)
    W ← W – η·∇_W L,   where L = (estimate – target)²

Thus a single ``BanditAction`` drives both the *resource‑store* dynamics
and the *matrix‑learning* dynamics, yielding a unified hybrid system that
produces advisory ``VramSlotPlan`` objects while continuously learning a
VRAM‑size model."""

from __future__ import annotations

import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Constants (derived from Parent A)
# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2] if Path(__file__).exists() else Path(".")
DEFAULT_BUDGET_MB = 4096
DEFAULT_RESERVE_MB = 768
DEFAULT_BASE_MODEL_MB = 1800
DEFAULT_ADAPTER_MB = 128
DEFAULT_EMBEDDING_MB = 384

# ----------------------------------------------------------------------
# Data structures (merged)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class VramSlotPlan:
    artifact_id: str
    artifact_kind: str
    action: str
    estimated_mb: int
    reason: str
    detail: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float          # interpreted as inflow rate
    expected_reward: float
    confidence_bound: float    # interpreted as outflow rate
    algorithm: str


# ----------------------------------------------------------------------
# Hybrid core
# ----------------------------------------------------------------------
class HybridVramBanditScheduler:
    """
    Maintains:
      * a virtual VRAM store (in MB)
      * a linear model W that maps feature vectors → VRAM estimate
      * a policy dictionary for bandit statistics (optional)
    """

    def __init__(
        self,
        feature_dim: int,
        alpha: float = 0.5,
        beta: float = 0.5,
        eta0: float = 0.01,
        dt: float = 1.0,
        budget_mb: int = DEFAULT_BUDGET_MB,
        reserve_mb: int = DEFAULT_RESERVE_MB,
    ) -> None:
        self.alpha = alpha                # inflow scaling
        self.beta = beta                  # outflow scaling
        self.eta0 = eta0                  # base learning rate
        self.dt = dt                      # time step for store dynamics
        self.budget_mb = budget_mb
        self.reserve_mb = reserve_mb

        self.store_mb = 0.0               # current virtual store usage
        self.W = np.zeros((feature_dim,))  # weight vector for linear estimator
        self._policy: Dict[str, List[float]] = {}  # optional bandit stats

    # ------------------------------------------------------------------
    # Store dynamics (bandit → resource)
    # ------------------------------------------------------------------
    def _inflow(self, action: BanditAction) -> float:
        """Influx of virtual memory (MB) derived from propensity."""
        return self.alpha * action.propensity

    def _outflow(self, action: BanditAction) -> float:
        """Outflux of virtual memory (MB) derived from confidence bound."""
        return self.beta * action.confidence_bound

    def update_store(self, action: BanditAction) -> None:
        """Apply the honeybee store equation using the given bandit action."""
        delta = self._inflow(action) - self._outflow(action)
        self.store_mb = max(0.0, self.store_mb + delta * self.dt)

    # ------------------------------------------------------------------
    # Linear VRAM estimator (TTT → model)
    # ------------------------------------------------------------------
    def estimate_vram(self, features: np.ndarray) -> float:
        """Linear estimate of VRAM consumption for a given feature vector."""
        return float(np.dot(self.W, features))

    def _learning_rate(self, action: BanditAction) -> float:
        """Learning rate modulated by bandit propensity."""
        return self.eta0 * (1.0 + action.propensity)

    def update_weights(
        self,
        features: np.ndarray,
        target_mb: float,
        action: BanditAction,
    ) -> None:
        """
        Perform a single gradient‑descent step on the squared‑error loss:
            L = (estimate - target)²
        """
        estimate = self.estimate_vram(features)
        error = estimate - target_mb
        grad = 2.0 * error * features
        eta = self._learning_rate(action)
        self.W -= eta * grad

    # ------------------------------------------------------------------
    # High‑level hybrid operation
    # ------------------------------------------------------------------
    def schedule_artifact(
        self,
        artifact_id: str,
        artifact_kind: str,
        base_mb: int,
        features: np.ndarray,
        action: BanditAction,
    ) -> VramSlotPlan:
        """
        Produce an advisory VramSlotPlan using the current estimator,
        update the virtual store, and adapt the weight matrix.
        """
        # 1. Estimate VRAM for this artifact
        estimate = self.estimate_vram(features)
        # Blend base model size with learned estimate (simple linear blend)
        blended_est = 0.6 * base_mb + 0.4 * estimate
        estimated_mb = max(0, int(round(blended_est)))

        # 2. Update virtual store based on bandit inflow/outflow
        self.update_store(action)

        # 3. Compute a synthetic reward: negative of residual budget pressure
        residual = self.budget_mb - (self.reserve_mb + self.store_mb + estimated_mb)
        reward = -abs(residual) / self.budget_mb  # normalized penalty

        # 4. Update weight matrix using the reward as target signal
        target_mb = base_mb  # we treat the known base size as ground truth
        self.update_weights(features, target_mb, action)

        # 5. Record policy statistics (optional, for completeness)
        stats = self._policy.setdefault(action.action_id, [0.0, 0.0])
        stats[0] += reward
        stats[1] += 1.0

        reason = (
            f"Hybrid estimate (blend) vs budget. Store={self.store_mb:.1f} MB, "
            f"Residual={residual:.1f} MB"
        )
        detail = {
            "base_mb": base_mb,
            "estimate_mb": estimated_mb,
            "store_mb": self.store_mb,
            "reward": reward,
            "learning_rate": self._learning_rate(action),
        }
        return VramSlotPlan(
            artifact_id=artifact_id,
            artifact_kind=artifact_kind,
            action=action.action_id,
            estimated_mb=estimated_mb,
            reason=reason,
            detail=detail,
        )

    # ------------------------------------------------------------------
    # Utility for external inspection
    # ------------------------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        """Return a serialisable snapshot of internal state."""
        return {
            "store_mb": self.store_mb,
            "weights": self.W.tolist(),
            "policy": {k: v[:] for k, v in self._policy.items()},
        }


# ----------------------------------------------------------------------
# Helper functions (demonstrate the hybrid operation)
# ----------------------------------------------------------------------
def compute_flow_rates(action: BanditAction, alpha: float, beta: float) -> Tuple[float, float]:
    """Return inflow and outflow (in MB) for a given bandit action."""
    return alpha * action.propensity, beta * action.confidence_bound


def linear_vram_estimate(W: np.ndarray, features: np.ndarray) -> float:
    """Stateless version of the linear VRAM estimator."""
    return float(np.dot(W, features))


def hybrid_step(
    scheduler: HybridVramBanditScheduler,
    artifact_id: str,
    artifact_kind: str,
    base_mb: int,
    features: np.ndarray,
    action: BanditAction,
) -> VramSlotPlan:
    """
    Execute one hybrid scheduling step:
      * update store,
      * adapt weights,
      * emit a VramSlotPlan.
    """
    return scheduler.schedule_artifact(
        artifact_id=artifact_id,
        artifact_kind=artifact_kind,
        base_mb=base_mb,
        features=features,
        action=action,
    )


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    # Create a scheduler for a 5‑dimensional feature space
    scheduler = HybridVramBanditScheduler(feature_dim=5)

    # Dummy feature vectors for three hypothetical artifacts
    feature_vectors = {
        "model_core": np.array([1.0, 0.2, 0.1, 0.0, 0.0]),
        "lora_adapter": np.array([0.0, 1.0, 0.3, 0.0, 0.0]),
        "embedding": np.array([0.0, 0.0, 1.0, 0.5, 0.2]),
    }

    # Simulated bandit actions (propensity/outflow are arbitrary)
    actions = [
        BanditAction(
            action_id="load_core",
            propensity=2.5,
            expected_reward=0.0,
            confidence_bound=1.0,
            algorithm="linucb",
        ),
        BanditAction(
            action_id="load_lora",
            propensity=1.2,
            expected_reward=0.0,
            confidence_bound=0.8,
            algorithm="linucb",
        ),
        BanditAction(
            action_id="load_embed",
            propensity=0.5,
            expected_reward=0.0,
            confidence_bound=0.3,
            algorithm="linucb",
        ),
    ]

    # Run a few hybrid steps
    for i, (aid, feats) in enumerate(feature_vectors.items()):
        plan = hybrid_step(
            scheduler=scheduler,
            artifact_id=aid,
            artifact_kind="artifact",
            base_mb=DEFAULT_BASE_MODEL_MB if i == 0 else (
                DEFAULT_ADAPTER_MB if i == 1 else DEFAULT_EMBEDDING_MB
            ),
            features=feats,
            action=actions[i],
        )
        print(f"Step {i+1} – VramSlotPlan:")
        print(plan.as_dict())
        print("Snapshot:", scheduler.snapshot())
        print("-" * 60)

    sys.exit(0)