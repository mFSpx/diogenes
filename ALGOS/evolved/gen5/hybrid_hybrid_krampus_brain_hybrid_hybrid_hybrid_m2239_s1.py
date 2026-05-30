# DARWIN HAMMER — match 2239, survivor 1
# gen: 5
# parent_a: hybrid_krampus_brainmap_bandit_router_m129_s2.py (gen1)
# parent_b: hybrid_hybrid_hybrid_regret_hybrid_hybrid_krampu_m63_s0.py (gen4)
# born: 2026-05-29T23:41:28Z

"""Hybrid Krampus‑Bandit Regret Engine

This module fuses the *Krampus* feature‑map (Parent A) with the *Regret‑Weighted
MinHash‑curvature* ideas of Parent B.  

Mathematical bridge  
-------------------

* The Krampus extractor yields a deterministic dictionary of real‑valued
  features **f**(text) → **x ∈ ℝᵈ** (d = 12).  
* The Regret‑Weighted engine hashes the same raw context (the input text) with
  a Blake2b‑based MinHash, maps the hash to a scalar **c ∈ (0,1)** via a
  sigmoid, and treats **c** as a *curvature* that modulates the confidence
  term of a LinUCB bandit.

For every admissible action *a* we keep a regularised Gram matrix
**Aₐ ∈ ℝ^{d×d}** and a response vector **bₐ ∈ ℝᵈ**.  The ordinary LinUCB
estimate is

    θₐ = Aₐ⁻¹ bₐ ,  UCBₐ(x) = θₐ·x + α·√(xᵀ Aₐ⁻¹ x)

The hybrid replaces the exploration bonus by a product of three factors:

* **α** – global exploration scale,
* **√(xᵀ Aₐ⁻¹ x)** – standard LinUCB uncertainty,
* **c·wₐ** – curvature **c** (MinHash) multiplied by a regret‑weight
  **wₐ = StoreState.dance** that evolves according to the observed regret.

Thus the final bound is  

    UCBₐ^hyb(x) = θₐ·x + α·c·wₐ·√(xᵀ Aₐ⁻¹ x)

The algorithm therefore couples the feature topology of Krampus with the
regret‑driven adaptive scaling of the hybrid Regret engine.

Only the Python standard library and NumPy are used.
"""

from __future__ import annotations

import hashlib
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – deterministic pseudo‑feature extractor (stub)
# ----------------------------------------------------------------------
def extract_full_features(text: str) -> Dict[str, float]:
    """Deterministic pseudo‑features derived from the hash of the words."""
    if not text.strip():
        return {}
    words = text.split()
    base = sum(hash(w) for w in words) % 1000
    return {
        "operator_visceral_ratio": (base % 10) / 10.0,
        "operator_tech_ratio": ((base // 10) % 10) / 10.0,
        "operator_legal_osint_ratio": ((base // 100) % 10) / 10.0,
        "operator_ledger_density": ((base // 1000) % 10) / 10.0,
        "operator_recursion_score": ((base // 2) % 5) / 5.0,
        "operator_directive_ratio": ((base // 3) % 7) / 7.0,
        "operator_target_density": ((base // 5) % 9) / 9.0,
        "psyche_forensic_shield_ratio": ((base // 7) % 8) / 8.0,
        "psyche_poetic_entropy": ((base // 11) % 6) / 6.0,
        "psyche_dissociative_index": ((base // 13) % 4) / 4.0,
        "psyche_wrath_velocity": ((base // 17) % 3) / 3.0,
        "resilience_bureaucratic_weaponiz": ((base // 19) % 5) / 5.0,
    }

# ----------------------------------------------------------------------
# Parent B – regret‑weighted utilities
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathAction:
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0

@dataclass(frozen=True)
class MathCounterfactual:
    action_id: str
    outcome_value: float
    probability: float = 1.0

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

@dataclass
class StoreState:
    """Simple regret‑driven accumulator used as a curvature weight."""
    level: float = 0.0
    alpha: float = 1.0
    beta: float = 1.0
    dt: float = 1.0
    base: float = 1.0
    gain: float = 1.0
    limit: float = 10.0
    _last_delta: float = 0.0

    def update(self, inflow: List[float], outflow: List[float]) -> Tuple[float, float]:
        delta = self.alpha * sum(inflow) - self.beta * sum(outflow)
        self.level = max(0.0, self.level + self.dt * delta)
        self._last_delta = delta
        return self.level, delta

    @property
    def dance(self) -> float:
        """Scaled, bounded version of the internal level."""
        return max(0.0, min(self.limit, self.base + self.gain * self._last_delta))

def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def minhash_curvature(context: str, seed: int = 0) -> float:
    """Deterministic MinHash → scalar curvature in (0,1)."""
    h = _hash(seed, context)
    # Normalise to [0,1] and pass through a sigmoid for smoothness
    norm = h / (2 ** 64 - 1)
    return float(sigmoid(np.array([norm * 12.0 - 6.0])))  # map roughly to (0,1)

# ----------------------------------------------------------------------
# Hybrid core – LinUCB with regret‑weighted curvature
# ----------------------------------------------------------------------
FeatureKeys = sorted(
    [
        "operator_visceral_ratio",
        "operator_tech_ratio",
        "operator_legal_osint_ratio",
        "operator_ledger_density",
        "operator_recursion_score",
        "operator_directive_ratio",
        "operator_target_density",
        "psyche_forensic_shield_ratio",
        "psyche_poetic_entropy",
        "psyche_dissociative_index",
        "psyche_wrath_velocity",
        "resilience_bureaucratic_weaponiz",
    ]
)

def vector_from_features(feat: Dict[str, float]) -> np.ndarray:
    """Convert the feature dict into a deterministic d‑dimensional vector."""
    return np.array([feat.get(k, 0.0) for k in FeatureKeys], dtype=float)

def init_models(action_ids: List[str], d: int = len(FeatureKeys), reg: float = 1.0) -> Dict[str, dict]:
    """Create per‑action LinUCB matrices and a StoreState."""
    models: Dict[str, dict] = {}
    for a in action_ids:
        models[a] = {
            "A": np.eye(d) * reg,          # regularised Gram matrix
            "b": np.zeros(d),              # response vector
            "state": StoreState(),         # regret‑weighted accumulator
        }
    return models

def compute_ucb(
    x: np.ndarray,
    model: dict,
    alpha: float,
    curvature: float,
) -> float:
    """Hybrid UCB = exploitation + α·curvature·w·uncertainty."""
    A = model["A"]
    b = model["b"]
    state: StoreState = model["state"]
    # Solve for theta = A⁻¹ b
    theta = np.linalg.solve(A, b)
    # Exploitation term
    exploit = float(theta @ x)
    # Uncertainty term
    invA = np.linalg.inv(A)
    uncertainty = math.sqrt(float(x @ invA @ x))
    # Regret‑weighted curvature factor
    bonus = alpha * curvature * state.dance * uncertainty
    return exploit + bonus

def select_action(
    context: str,
    models: Dict[str, dict],
    alpha: float = 1.0,
) -> Tuple[str, float]:
    """Return the action id with the highest hybrid UCB and the associated bound."""
    # Feature extraction → vector x
    feats = extract_full_features(context)
    x = vector_from_features(feats)
    # Curvature from MinHash
    curvature = minhash_curvature(context)
    # Evaluate all actions
    best_a = None
    best_val = -math.inf
    for aid, mdl in models.items():
        ucb = compute_ucb(x, mdl, alpha, curvature)
        if ucb > best_val:
            best_val = ucb
            best_a = aid
    return best_a, best_val

def update_models(
    models: Dict[str, dict],
    chosen_action: str,
    reward: float,
    context: str,
) -> None:
    """Standard LinUCB update plus regret‑weighted state evolution."""
    feats = extract_full_features(context)
    x = vector_from_features(feats)
    # Update the linear model for the chosen action
    A = models[chosen_action]["A"]
    b = models[chosen_action]["b"]
    A += np.outer(x, x)
    b += reward * x
    models[chosen_action]["A"] = A
    models[chosen_action]["b"] = b
    # Regret computation: compare received reward with the highest estimated reward
    # (using current parameters, not the stochastic bound)
    estimates = {}
    for aid, mdl in models.items():
        theta = np.linalg.solve(mdl["A"], mdl["b"])
        estimates[aid] = float(theta @ x)
    best_est = max(estimates.values())
    regret = max(0.0, best_est - estimates[chosen_action])
    # Feed regret as inflow to the StoreState of the chosen action
    state: StoreState = models[chosen_action]["state"]
    state.update([regret], [])
    models[chosen_action]["state"] = state

# ----------------------------------------------------------------------
# Demonstration helpers (three distinct functions)
# ----------------------------------------------------------------------
def simulate_one_round(
    context: str,
    models: Dict[str, dict],
    true_reward_fn,
    alpha: float = 1.0,
) -> Tuple[str, float]:
    """Run a single decision‑update cycle."""
    action, ucb = select_action(context, models, alpha)
    reward = true_reward_fn(action, context)
    update_models(models, action, reward, context)
    return action, reward

def batch_simulation(
    contexts: List[str],
    models: Dict[str, dict],
    true_reward_fn,
    alpha: float = 1.0,
) -> List[Tuple[str, float]]:
    """Execute a series of rounds and return the (action,reward) history."""
    history: List[Tuple[str, float]] = []
    for ctx in contexts:
        act, rew = simulate_one_round(ctx, models, true_reward_fn, alpha)
        history.append((act, rew))
    return history

def report_model_state(models: Dict[str, dict]) -> None:
    """Print a concise summary of each action's learned parameters."""
    for aid, mdl in models.items():
        theta = np.linalg.solve(mdl["A"], mdl["b"])
        print(f"Action {aid}: ||θ||={np.linalg.norm(theta):.3f}, w={mdl['state'].dance:.3f}")

# ----------------------------------------------------------------------
# Simple synthetic reward function for testing
# ----------------------------------------------------------------------
def synthetic_reward(action_id: str, context: str) -> float:
    """Reward is higher when the hash of (action, context) is even."""
    h = _hash(42, f"{action_id}:{context}")
    return 1.0 if h % 2 == 0 else 0.0

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)

    actions = ["alpha", "beta", "gamma"]
    models = init_models(actions)

    # Generate a few random textual contexts
    sample_texts = [
        "investigate the ledger and reconcile the operator ratios",
        "deploy forensic shield against poetic entropy",
        "legal osint reveals hidden recursion scores",
        "target density spikes during crisis",
        "wrath velocity exceeds resilience thresholds",
    ]

    history = batch_simulation(sample_texts, models, synthetic_reward, alpha=0.5)

    print("=== Simulation History ===")
    for i, (act, rew) in enumerate(history):
        print(f"Round {i+1}: action={act}, reward={rew}")

    print("\n=== Final Model Summary ===")
    report_model_state(models)