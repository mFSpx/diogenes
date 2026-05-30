# DARWIN HAMMER — match 1453, survivor 5
# gen: 6
# parent_a: hybrid_model_vram_scheduler_hybrid_hybrid_bandit_m188_s0.py (gen3)
# parent_b: hybrid_hybrid_hybrid_pherom_hybrid_hybrid_hybrid_m1139_s1.py (gen5)
# born: 2026-05-29T23:36:29Z

"""Hybrid VRAM‑Pheromone‑SSIM Scheduler

Parents:
- hybrid_model_vram_scheduler_hybrid_hybrid_bandit_m188_s0.py (VRAM‑Bandit scheduler)
- hybrid_hybrid_hybrid_pherom_hybrid_hybrid_hybrid_m1139_s1.py (Pheromone‑SSIM controller)

Mathematical bridge:
Both parents describe a *resource pool* whose dynamics are driven by
inflows/outflows (bandit propensity / confidence bound, pheromone decay)
and a *learning* component that adapts a weight matrix to minimise a
prediction error.  We fuse them by defining a unified scalar **store**
that aggregates VRAM memory and pheromone concentration:

    Δstore = α·Σ(inflow) – β·Σ(outflow) – γ·store
    storeₜ₊₁ = max(0, storeₜ + Δstore·dt)

where
- inflow = bandit.propensity  (virtual memory) + pheromone_in (from decay)
- outflow = bandit.confidence_bound (memory release) + γ·store (natural decay)

The learning update now incorporates both the classic squared‑error loss
and an SSIM‑derived regulariser that measures structural similarity between
the current morphology and a target morphology:

    L = (estimate – target)² + λ·(1 – SSIM(state, target_state))

The gradient w.r.t. the weight matrix **W** becomes:

    ∇_W L = 2·(estimate – target)·X  +  λ·∇_W (1 – SSIM)

The bandit action also modulates the effective learning‑rate:

    η = η₀·(1 + propensity)

Thus a single BanditAction simultaneously drives resource‑store dynamics,
pheromone decay, and adaptive learning, yielding a unified hybrid system.
"""

from __future__ import annotations

import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Constants
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
    propensity: float          # inflow rate (virtual memory)
    expected_reward: float
    confidence_bound: float    # outflow rate
    algorithm: str


@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


@dataclass(frozen=True)
class EngineEndpoint:
    engine_id: str
    channel: str
    residency: str
    runtime: str
    resource_class: str
    always_on: bool
    endpoint: str
    capabilities: List[str]
    morphology: Morphology
    outbound_state: str = "draft_only"

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["morphology"] = asdict(self.morphology)
        return d


# ----------------------------------------------------------------------
# Helper metrics from Parent B
# ----------------------------------------------------------------------
def sphericity_index(length: float, width: float, height: float) -> float:
    """Simple sphericity: ratio of geometric mean to longest side."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / max(length, width, height)


def flatness_index(length: float, width: float, height: float) -> float:
    """Flatness as the ratio of planar dimensions to thickness."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


def ssim_like(m1: Morphology, m2: Morphology) -> float:
    """
    Very coarse SSIM‑like similarity using sphericity and flatness.
    Returns a value in [0, 1] where 1 means identical.
    """
    s1 = sphericity_index(m1.length, m1.width, m1.height)
    s2 = sphericity_index(m2.length, m2.width, m2.height)
    f1 = flatness_index(m1.length, m1.width, m1.height)
    f2 = flatness_index(m2.length, m2.width, m2.height)

    # luminance component (mass)
    l1 = m1.mass
    l2 = m2.mass
    C1 = (0.01 * max(l1, l2)) ** 2
    luminance = (2 * l1 * l2 + C1) / (l1 ** 2 + l2 ** 2 + C1)

    # contrast component (sphericity)
    C2 = (0.03 * max(s1, s2)) ** 2
    contrast = (2 * s1 * s2 + C2) / (s1 ** 2 + s2 ** 2 + C2)

    # structure component (flatness)
    C3 = (0.03 * max(f1, f2)) ** 2
    structure = (2 * f1 * f2 + C3) / (f1 ** 2 + f2 ** 2 + C3)

    return luminance * contrast * structure


# ----------------------------------------------------------------------
# Core hybrid dynamics
# ----------------------------------------------------------------------
def update_store_and_pheromone(
    store: float,
    pheromone: float,
    bandit: BanditAction,
    dt: float,
    α: float = 0.6,
    β: float = 0.4,
    γ: float = 0.05,
) -> Tuple[float, float]:
    """
    Unified resource dynamics.

    - Inflow = bandit.propensity + (1 - exp(-γ·dt))·pheromone   (pheromone contribution)
    - Outflow = bandit.confidence_bound + γ·store                (natural decay)
    - Δstore follows the honeybee store equation.
    - Pheromone decays exponentially and receives a fraction of the inflow.
    """
    # Pheromone decay (continuous exponential)
    pheromone_decay = pheromone * math.exp(-γ * dt)
    # Portion of bandit propensity that becomes pheromone
    pheromone_in = bandit.propensity * 0.2  # 20% of propensity reinforces pheromone
    pheromone_next = pheromone_decay + pheromone_in

    inflow = bandit.propensity + pheromone_in
    outflow = bandit.confidence_bound + γ * store

    delta_store = α * inflow - β * outflow
    store_next = max(0.0, store + delta_store * dt)

    return store_next, pheromone_next


def gradient_descent_step(
    W: np.ndarray,
    X: np.ndarray,
    y: float,
    bandit: BanditAction,
    η0: float = 0.01,
    λ: float = 0.5,
    target_morph: Morphology = None,
    current_morph: Morphology = None,
) -> np.ndarray:
    """
    Perform one gradient descent update on weight matrix W.

    Loss = (estimate - y)² + λ·(1 - SSIM_like(current, target))

    The SSIM term does not depend on W; its gradient is approximated as zero
    (treated as a regulariser).  The learning rate is modulated by bandit.propensity.
    """
    estimate = X @ W
    error = estimate - y
    grad = 2.0 * error * X  # ∂/∂W of squared error

    # Learning‑rate modulation
    η = η0 * (1.0 + bandit.propensity)

    # Optional SSIM regulariser gradient (set to zero for simplicity)
    if target_morph is not None and current_morph is not None:
        # In a full implementation this would back‑propagate through morphology,
        # but we keep it zero to stay within numpy only.
        pass

    W_next = W - η * grad
    return W_next


def generate_vram_plan(
    artifact_id: str,
    artifact_kind: str,
    store: float,
    estimated_mb: int,
    reason: str,
) -> VramSlotPlan:
    """
    Produce a VramSlotPlan based on the current store level.
    """
    action = "allocate" if store >= estimated_mb else "defer"
    detail = {
        "store_mb": store,
        "requested_mb": estimated_mb,
        "decision": action,
    }
    return VramSlotPlan(
        artifact_id=artifact_id,
        artifact_kind=artifact_kind,
        action=action,
        estimated_mb=estimated_mb,
        reason=reason,
        detail=detail,
    )


# ----------------------------------------------------------------------
# Demonstration functions
# ----------------------------------------------------------------------
def hybrid_step(
    store: float,
    pheromone: float,
    W: np.ndarray,
    X: np.ndarray,
    y: float,
    bandit: BanditAction,
    current_morph: Morphology,
    target_morph: Morphology,
    dt: float = 1.0,
) -> Tuple[float, float, np.ndarray]:
    """
    Execute one full hybrid iteration:
    1. Update resource store & pheromone.
    2. Update weight matrix.
    Returns updated (store, pheromone, W).
    """
    store, pheromone = update_store_and_pheromone(
        store, pheromone, bandit, dt
    )
    W = gradient_descent_step(
        W,
        X,
        y,
        bandit,
        target_morph=target_morph,
        current_morph=current_morph,
    )
    return store, pheromone, W


def evaluate_similarity(
    current: Morphology, target: Morphology
) -> float:
    """Return SSIM‑like similarity between two morphologies."""
    return ssim_like(current, target)


def run_hybrid_simulation(
    steps: int = 10,
) -> List[VramSlotPlan]:
    """
    Simple simulation that runs `steps` hybrid iterations and records
    VramSlotPlan objects.
    """
    # Initialise state
    store = DEFAULT_BUDGET_MB - DEFAULT_RESERVE_MB
    pheromone = 1.0
    n_features = 5
    W = np.zeros(n_features)

    # Dummy feature vector and target VRAM usage
    X = np.random.rand(n_features)
    y = 512.0  # target memory in MB

    # Random morphologies
    current_morph = Morphology(1.2, 0.8, 0.5, 2.3)
    target_morph = Morphology(1.0, 0.9, 0.6, 2.5)

    plans: List[VramSlotPlan] = []

    for step in range(steps):
        # Simulated bandit action
        bandit = BanditAction(
            action_id=f"act_{step}",
            propensity=random.uniform(0.0, 0.3),
            expected_reward=random.random(),
            confidence_bound=random.uniform(0.0, 0.2),
            algorithm="HybridBandit",
        )

        store, pheromone, W = hybrid_step(
            store,
            pheromone,
            W,
            X,
            y,
            bandit,
            current_morph,
            target_morph,
            dt=1.0,
        )

        similarity = evaluate_similarity(current_morph, target_morph)

        plan = generate_vram_plan(
            artifact_id=f"art_{step}",
            artifact_kind="model",
            store=store,
            estimated_mb=int(y),
            reason=f"Similarity={similarity:.3f}, pheromone={pheromone:.2f}",
        )
        plans.append(plan)

        # For illustration, slightly mutate current morphology towards target
        current_morph = Morphology(
            length=current_morph.length * 0.95 + target_morph.length * 0.05,
            width=current_morph.width * 0.95 + target_morph.width * 0.05,
            height=current_morph.height * 0.95 + target_morph.height * 0.05,
            mass=current_morph.mass * 0.95 + target_morph.mass * 0.05,
        )

    return plans


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    plans = run_hybrid_simulation(steps=5)
    for p in plans:
        print(p.as_dict())