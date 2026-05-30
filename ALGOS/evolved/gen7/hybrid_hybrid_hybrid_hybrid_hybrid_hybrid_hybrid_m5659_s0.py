# DARWIN HAMMER — match 5659, survivor 0
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1387_s0.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_perceptual_de_m264_s3.py (gen4)
# born: 2026-05-30T00:03:55Z

"""Hybrid Algorithm Fusion of DARWIN HAMMER Match 1387 and Match 264

Parents:
- **Match 1387**: Hybrid Bandit‑Router / Workshare Allocator + Path‑Signature‑KAN + Regret Engine.
- **Match 264**: Regex‑feature extraction → RBF surrogate → Liquid‑Time‑Constant (LTC) recurrent cell with diffusion forcing.

Mathematical Bridge
------------------
The bridge is built on the observation that both parents maintain a *dynamic state* that
modulates downstream computation:

* In Match 1387 the **store** (`StoreState`) evolves under signature‑derived flows and a
  regret‑driven developmental rate.
* In Match 264 the **LTC hidden state** (`LTCState`) evolves under a similarity‑driven
  mixing coefficient `α` and a diffusion coefficient `λ` supplied by an RBF surrogate.

We fuse the two by treating the signature vector `s` (output of the path‑signature
module) as the feature vector `x_t` for the LTC cell.  The similarity `α` computed
from successive signatures becomes the liquid‑time‑constant mixing term, while the
RBF surrogate, still driven by `s`, supplies the diffusion `λ`.  The resulting LTC
update feeds back into the store update, and the updated store level rescales the
bandit propensities.  Regret computed from the bandit reward further adapts the
store’s `alpha` parameter, closing the loop.

The unified dynamics are:


# Signature extraction
s_t = signature(text_t)

# Similarity (liquid‑time‑constant)
α_t = exp( -||s_t - s_{t-1}||² / (2σ²) )

# Diffusion from RBF surrogate
λ_t = Σ_i w_i · exp( -||s_t - c_i||² / (2γ²) )

# LTC hidden state update
h_{t+1} = (1-α_t)·h_t + α_t·tanh(W·s_t + U·h_t + b) + λ_t·η_t

# Store update (signature‑driven flow)
Δ = φ·s_t                     # φ – flow coefficients
level_{t+1} = level_t + dt·Δ·(1 + mean(h_{t+1}))

# Regret‑driven α‑parameter update
α_store_{t+1} = α_store_t · exp( -regret·dt )


The module below implements this fusion with three public functions that
demonstrate the hybrid operation.
"""

import re
import math
import random
import sys
import pathlib
from dataclasses import dataclass, field
from typing import List, Tuple, Sequence, Dict

import numpy as np

# ----------------------------------------------------------------------
# Regex feature extraction (shared with Parent 264)
# ----------------------------------------------------------------------
EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|"
    r"hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)
PLANNING_RE = re.compile(
    r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|"
    r"triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
    re.I,
)
DELAY_RE = re.compile(r"\b(?:pause|sleep|wait|later|hold)\b", re.I)

# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------
@dataclass
class StoreState:
    """Honeybee‑style store."""
    level: float = 0.0
    alpha: float = 1.0   # developmental rate parameter
    beta: float = 1.0
    dt: float = 1.0
    base: float = 1.0

@dataclass
class LTCState:
    """Hidden state of the Liquid‑Time‑Constant recurrent cell."""
    h: np.ndarray = field(default_factory=lambda: np.zeros(3))

@dataclass(frozen=True)
class BanditAction:
    """Result of an action selection."""
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "HybridBandit"

# ----------------------------------------------------------------------
# Core hybrid components
# ----------------------------------------------------------------------
def lead_lag_bspline_signature(text: str) -> np.ndarray:
    """
    Compute a simple 3‑dimensional signature from regex counts.
    The vector is optionally smoothed with a cumulative‑sum “lead‑lag” filter
    that mimics a B‑spline projection.
    """
    counts = np.array([
        len(EVIDENCE_RE.findall(text)),
        len(PLANNING_RE.findall(text)),
        len(DELAY_RE.findall(text)),
    ], dtype=float)

    # Lead‑lag smoothing (cumulative sum then differencing)
    lead = np.cumsum(counts)
    lag = np.concatenate(([0.0], lead[:-1]))
    signature = (lead + lag) / 2.0
    return signature


def rbf_surrogate(x: np.ndarray,
                  centers: np.ndarray,
                  weights: np.ndarray,
                  gamma: float = 1.0) -> float:
    """
    Radial‑Basis Function surrogate predicting a scalar diffusion coefficient λ.
    λ = Σ_i w_i · exp( -||x - c_i||² / (2·γ²) )
    """
    diff = x - centers  # shape (n_centers, dim)
    sq_norm = np.einsum('ij,ij->i', diff, diff)
    kernels = np.exp(-sq_norm / (2.0 * gamma ** 2))
    return float(np.dot(weights, kernels))


def hybrid_step(text: str,
                prev_signature: np.ndarray,
                store: StoreState,
                ltc: LTCState,
                rbf_centers: np.ndarray,
                rbf_weights: np.ndarray,
                sigma: float = 1.0,
                gamma: float = 1.0,
                flow_coeffs: np.ndarray = np.array([0.1, 0.2, 0.3]),
                W: np.ndarray = np.eye(3) * 0.5,
                U: np.ndarray = np.eye(3) * 0.3,
                b: np.ndarray = np.zeros(3)) -> Tuple[np.ndarray, LTCState, StoreState]:
    """
    Perform one hybrid iteration:
      1. Extract signature s_t from the input text.
      2. Compute similarity α_t between s_t and s_{t‑1}.
      3. Obtain diffusion λ_t from the RBF surrogate.
      4. Update the LTC hidden state.
      5. Update the store level using the signature flow and the new LTC state.
      6. Return the new signature, updated LTCState and StoreState.
    """
    # 1. Signature
    s_t = lead_lag_bspline_signature(text)

    # 2. Similarity α (liquid‑time‑constant mixing coefficient)
    dist_sq = np.linalg.norm(s_t - prev_signature) ** 2
    alpha_t = math.exp(-dist_sq / (2.0 * sigma ** 2))

    # 3. Diffusion λ from RBF surrogate
    lambda_t = rbf_surrogate(s_t, rbf_centers, rbf_weights, gamma)

    # 4. LTC hidden state update
    noise = np.random.randn(s_t.shape[0])
    tanh_arg = W @ s_t + U @ ltc.h + b
    h_new = (1 - alpha_t) * ltc.h + alpha_t * np.tanh(tanh_arg) + lambda_t * noise
    ltc_new = LTCState(h=h_new)

    # 5. Store update (signature‑driven flow, modulated by LTC activity)
    flow = np.dot(flow_coeffs, s_t)            # scalar Δ
    modulation = 1.0 + np.tanh(np.mean(h_new)) # mild scaling from LTC
    store.level += store.dt * flow * modulation

    # Store's other parameters remain unchanged in this step
    return s_t, ltc_new, store


def bandit_action_selection(actions: List[BanditAction],
                            store: StoreState,
                            ltc: LTCState) -> BanditAction:
    """
    Select an action by scaling raw propensities with the current store level
    and a factor derived from the LTC hidden state.
    The action with the highest scaled propensity is returned.
    """
    ltc_factor = 1.0 + np.tanh(np.mean(ltc.h))  # between 0 and 2
    scale = (1.0 + store.level) * ltc_factor

    best = max(
        actions,
        key=lambda a: a.propensity * scale
    )
    # Return a copy with updated propensity for transparency
    return BanditAction(
        action_id=best.action_id,
        propensity=best.propensity * scale,
        expected_reward=best.expected_reward,
        confidence_bound=best.confidence_bound,
        algorithm=best.algorithm,
    )


def regret_engine_honeybee_alpha_update(store: StoreState,
                                        reward: float,
                                        expected_reward: float) -> StoreState:
    """
    Update the store's α parameter using a simple regret formulation:
        regret = max(0, expected_reward - reward)
        α_{t+1} = α_t * exp( -regret·dt )
    """
    regret = max(0.0, expected_reward - reward)
    store.alpha *= math.exp(-regret * store.dt)
    return store


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Initialise components
    rng = np.random.default_rng(42)
    rbf_centers = rng.normal(size=(5, 3))          # 5 RBF centres, 3‑dimensional
    rbf_weights = rng.uniform(-1, 1, size=5)       # corresponding weights

    store = StoreState(level=0.5, alpha=1.0, dt=0.1)
    ltc = LTCState(h=np.zeros(3))

    prev_sig = np.zeros(3)

    # Example texts
    texts = [
        "The evidence was verified and the plan is to schedule the next checkpoint.",
        "Please wait while we source the hash and upload the screenshot.",
        "No action needed now, hold the current state."
    ]

    # Dummy bandit actions
    actions = [
        BanditAction("A", propensity=0.3, expected_reward=0.8, confidence_bound=0.1),
        BanditAction("B", propensity=0.5, expected_reward=0.6, confidence_bound=0.2),
        BanditAction("C", propensity=0.2, expected_reward=0.9, confidence_bound=0.15),
    ]

    for i, txt in enumerate(texts):
        sig, ltc, store = hybrid_step(
            text=txt,
            prev_signature=prev_sig,
            store=store,
            ltc=ltc,
            rbf_centers=rbf_centers,
            rbf_weights=rbf_weights,
            sigma=1.0,
            gamma=1.0,
        )
        selected = bandit_action_selection(actions, store, ltc)

        # Simulate a stochastic reward (for demo purposes)
        reward = random.random() * selected.expected_reward
        store = regret_engine_honeybee_alpha_update(store, reward, selected.expected_reward)

        print(f"Step {i+1}:")
        print(f"  Signature: {sig}")
        print(f"  α (similarity): {math.exp(-np.linalg.norm(sig - prev_sig)**2 / 2):.4f}")
        print(f"  LTC state: {ltc.h}")
        print(f"  Store level: {store.level:.4f}, alpha: {store.alpha:.4f}")
        print(f"  Selected action: {selected.action_id} with scaled propensity {selected.propensity:.4f}")
        print(f"  Simulated reward: {reward:.4f}\n")

        prev_sig = sig.copy()