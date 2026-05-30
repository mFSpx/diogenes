# DARWIN HAMMER — match 1253, survivor 4
# gen: 3
# parent_a: hybrid_hybrid_bandit_router_hybrid_model_vram_sc_m35_s5.py (gen2)
# parent_b: hybrid_hybrid_ternary_route_hybrid_bandit_router_m31_s4.py (gen2)
# born: 2026-05-29T23:34:46Z

"""Hybrid Bandit‑TTT with SSIM‑based Reward
================================================

Parent A: ``hybrid_hybrid_bandit_router_hybrid_model_vram_sc_m35_s5.py`` – a contextual
bandit coupled to a linear “TTT” weight matrix and a virtual‑VRAM store that follows
a differential equation


dS/dt = α·inflow − β·outflow − γ·S   (γ = 1−store_decay)


Parent B: ``hybrid_hybrid_ternary_route_hybrid_bandit_router_m31_s4.py`` – provides a
structural similarity (SSIM) metric and a packet‑routing routine.

**Mathematical Bridge**

The bridge is the *reward* fed back to the bandit.  Instead of a scalar reward
originating from an external environment, we compute the SSIM between the
TTT model’s prediction and a target signal.  This SSIM value (∈ [0, 1]) serves as
the observed reward for the selected bandit action.  The bandit’s propensity
modulates the *inflow* term of the VRAM store, while the store’s current level
rescales the bandit’s propensity, closing a feedback loop that ties together the
three core components:


inflow  = propensity * store_factor
outflow = confidence_bound
S(t+Δt) = S(t) + Δt·(α·inflow − β·outflow − γ·S(t))
η(t)    = base_eta / (1 + S(t)/DEFAULT_BUDGET_MB)   # learning‑rate modulation


The following module implements this fused system, exposing three public
functions that illustrate the hybrid operation.  A small smoke test at the
bottom runs the loop without external dependencies."""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Core data structures (shared by both parents)
# ----------------------------------------------------------------------
class BanditAction:
    """Result of a bandit decision."""
    __slots__ = ("action_id", "propensity", "expected_reward", "confidence_bound", "algorithm")

    def __init__(self, action_id: str, propensity: float, expected_reward: float,
                 confidence_bound: float, algorithm: str = "hybrid"):
        self.action_id = action_id
        self.propensity = propensity          # interpreted as inflow rate
        self.expected_reward = expected_reward
        self.confidence_bound = confidence_bound  # interpreted as outflow rate
        self.algorithm = algorithm

    def __repr__(self) -> str:
        return (f"BanditAction(id={self.action_id!r}, prop={self.propensity:.3f}, "
                f"exp={self.expected_reward:.3f}, bound={self.confidence_bound:.3f})")


class BanditUpdate:
    """Observed reward for a given action."""
    __slots__ = ("context_id", "action_id", "reward", "propensity")

    def __init__(self, context_id: str, action_id: str, reward: float, propensity: float):
        self.context_id = context_id
        self.action_id = action_id
        self.reward = reward
        self.propensity = propensity


# ----------------------------------------------------------------------
# SSIM implementation (Parent B)
# ----------------------------------------------------------------------
def ssim(x: np.ndarray, y: np.ndarray,
         dynamic_range: float = 255.0,
         k1: float = 0.01,
         k2: float = 0.03) -> float:
    """Structural Similarity Index between two 1‑D signals."""
    if x.shape != y.shape:
        raise ValueError("Input arrays must have the same shape")
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x = np.sqrt(np.var(x))
    sigma_y = np.sqrt(np.var(y))
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))
    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x ** 2 + sigma_y ** 2 + c2)
    return float(numerator / denominator)


# ----------------------------------------------------------------------
# Virtual VRAM store dynamics (from Parent A)
# ----------------------------------------------------------------------
def update_store(store: float,
                 inflow: float,
                 outflow: float,
                 dt: float,
                 alpha: float,
                 beta: float,
                 store_decay: float) -> float:
    """
    Euler integration of the VRAM store differential equation:

        dS/dt = α·inflow − β·outflow − γ·S   with γ = 1−store_decay
    """
    gamma = 1.0 - store_decay
    dS = alpha * inflow - beta * outflow - gamma * store
    return max(store + dt * dS, 0.0)   # store cannot be negative


# ----------------------------------------------------------------------
# Hybrid Bandit‑TTT model
# ----------------------------------------------------------------------
class HybridBanditSSIMTTT:
    """
    Combines:
      * Contextual bandit (propensity / confidence bound)
      * Linear TTT weight matrix (prediction)
      * VRAM store that modulates learning‑rate and receives bandit inflow
      * SSIM as the reward signal
    """
    DEFAULT_BUDGET_MB = 8192.0

    def __init__(self,
                 d_in: int,
                 d_out: int,
                 seed: int = 0,
                 base_eta: float = 0.01,
                 alpha: float = 1.0,
                 beta: float = 1.0,
                 dt: float = 1.0,
                 store_decay: float = 0.99):
        self.rng = np.random.default_rng(seed)
        self.base_eta = base_eta
        self.alpha = alpha
        self.beta = beta
        self.dt = dt
        self.store_decay = store_decay

        # Linear weight matrix W ∈ ℝ^{d_out × d_in}
        self.W = self.rng.normal(scale=0.1, size=(d_out, d_in))

        # Initialise virtual VRAM store (scalar)
        self.store = self.DEFAULT_BUDGET_MB * 0.5

        # Initialise a tiny pool of bandit actions (could be expanded)
        self.actions: List[BanditAction] = [
            BanditAction(action_id="A", propensity=0.5, expected_reward=0.0,
                         confidence_bound=0.1),
            BanditAction(action_id="B", propensity=0.5, expected_reward=0.0,
                         confidence_bound=0.1)
        ]

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------
    def predict(self, context: np.ndarray) -> np.ndarray:
        """Linear TTT prediction: ŷ = W·x."""
        if context.shape[0] != self.W.shape[1]:
            raise ValueError("Context dimension mismatch")
        return self.W @ context

    def select_action(self) -> BanditAction:
        """
        Choose an action using a softmax over propensities that are
        scaled by the current store level (more store → higher exploration).
        """
        scale = 1.0 + self.store / self.DEFAULT_BUDGET_MB
        logits = np.array([a.propensity * scale for a in self.actions])
        probs = np.exp(logits - np.max(logits))
        probs /= probs.sum()
        idx = self.rng.choice(len(self.actions), p=probs)
        return self.actions[idx]

    def receive_update(self, upd: BanditUpdate) -> None:
        """Simple bandit update: move expected_reward toward observed reward."""
        for act in self.actions:
            if act.action_id == upd.action_id:
                lr = 0.1  # bandit learning rate (fixed)
                act.expected_reward = (1 - lr) * act.expected_reward + lr * upd.reward
                # Slightly adapt propensity based on reward magnitude
                act.propensity = max(0.01, min(1.0, act.propensity + 0.05 * (upd.reward - 0.5)))
                break

    def step(self, context: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Perform one hybrid iteration:
          1. Predict with TTT.
          2. Compute SSIM reward against target.
          3. Select bandit action → define inflow/outflow.
          4. Update VRAM store.
          5. Modulate learning rate with store and apply a gradient step.
        Returns the prediction and the obtained reward.
        """
        # 1. prediction
        pred = self.predict(context)

        # 2. reward via SSIM (clip to [0,1] just in case)
        reward = float(np.clip(ssim(pred, target), 0.0, 1.0))

        # 3. bandit decision
        act = self.select_action()
        inflow = act.propensity * (self.store / self.DEFAULT_BUDGET_MB)
        outflow = act.confidence_bound

        # 4. store dynamics
        self.store = update_store(self.store, inflow, outflow,
                                  self.dt, self.alpha, self.beta, self.store_decay)

        # 5. learning‑rate modulation & weight update
        eta = self.base_eta / (1.0 + self.store / self.DEFAULT_BUDGET_MB)
        error = pred - target
        grad = np.outer(error, context)  # ∂L/∂W for L = ½‖error‖²
        self.W -= eta * grad

        # 6. bandit learning
        upd = BanditUpdate(context_id="ctx", action_id=act.action_id,
                          reward=reward, propensity=act.propensity)
        self.receive_update(upd)

        return pred, reward


# ----------------------------------------------------------------------
# Public helper functions demonstrating the hybrid operation
# ----------------------------------------------------------------------
def hybrid_initialize(d_in: int, d_out: int) -> HybridBanditSSIMTTT:
    """Create a fresh HybridBanditSSIMTTT instance with deterministic seed."""
    return HybridBanditSSIMTTT(d_in=d_in, d_out=d_out, seed=42)


def hybrid_run_one_step(model: HybridBanditSSIMTTT,
                        context: np.ndarray,
                        target: np.ndarray) -> Dict[str, Any]:
    """
    Execute a single hybrid step and return a diagnostic dictionary.
    """
    pred, reward = model.step(context, target)
    diagnostics = {
        "prediction": pred.tolist(),
        "reward": reward,
        "store": model.store,
        "learning_rate": model.base_eta / (1.0 + model.store / model.DEFAULT_BUDGET_MB),
        "actions": [repr(a) for a in model.actions]
    }
    return diagnostics


def hybrid_simulation(num_steps: int = 5,
                      d_in: int = 8,
                      d_out: int = 8) -> List[Dict[str, Any]]:
    """
    Run a short simulation of the hybrid system on synthetic data.
    Returns the list of per‑step diagnostics.
    """
    model = hybrid_initialize(d_in, d_out)
    logs: List[Dict[str, Any]] = []

    for step in range(num_steps):
        # synthetic context and target (random but correlated)
        ctx = np.random.rand(d_in).astype(np.float64)
        true_W = np.ones((d_out, d_in)) * 0.5   # hidden ground‑truth matrix
        tgt = true_W @ ctx + np.random.normal(scale=0.01, size=d_out)

        logs.append(hybrid_run_one_step(model, ctx, tgt))

    return logs


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    logs = hybrid_simulation(num_steps=10, d_in=4, d_out=4)
    for i, entry in enumerate(logs, 1):
        print(f"Step {i}: reward={entry['reward']:.4f}, store={entry['store']:.2f}, "
              f"lr={entry['learning_rate']:.5f}")
        # Show actions briefly
        print("  Actions:", ", ".join(entry["actions"]))