# DARWIN HAMMER — match 2298, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m630_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_gliner_hybrid_hybrid_fisher_m432_s0.py (gen4)
# born: 2026-05-29T23:41:41Z

"""Hybrid NLMS‑Bandit‑Pheromone‑Fisher Fusion
Parents:
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m630_s0.py (NLMS‑Omni‑Chaotic‑Sprint + Bandit + Sheaf)
- hybrid_hybrid_hybrid_gliner_hybrid_hybrid_fisher_m432_s0.py (Pheromone decay + Fisher information + Ternary logic)

Mathematical Bridge
-------------------
The NLMS adaptive filter produces a weight vector **w** and a prediction error ϵ.
We interpret the instantaneous “velocity” v = ϵ·x (scalar product of error and input) as a
scalar *information‑energy* that simultaneously:
1. Scales the decay factor of pheromone entries (the higher the velocity, the faster the
   pheromone evaporates, mirroring the diffusion‑forcing schedule of the sheaf model).
2. Serves as a surrogate for the Fisher information I = (∂μ/∂θ)²/σ² of a Gaussian
   observation model μ = w·x, σ² fixed.  I is used to re‑weight bandit propensities.

Thus the NLMS dynamics feed directly into the pheromone‑based decision process and the
bandit‑based action selection.  A ternary logic gate combines the sign of the pheromone
signal with the sign of the Fisher‑information‑scaled propensity to yield a discrete
decision {‑1, 0, +1}.
"""

import numpy as np
import math
import random
import sys
import pathlib
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple
from datetime import datetime, timezone
import uuid

NodeId = str
Edge = Tuple[NodeId, NodeId, int]

# ----------------------------------------------------------------------
# Data structures (light‑weight versions of the parents)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float = 0.0
    confidence_bound: float = 0.0

@dataclass
class StoreState:
    level: float = 0.0
    alpha: float = 1.0
    beta: float = 1.0
    dt: float = 1.0
    base: float = 1.0

class PheromoneEntry:
    __slots__ = ("uuid", "key", "value", "half_life_seconds",
                 "created_at", "last_decay")

    def __init__(self, key: str, value: float, half_life_seconds: int = 30):
        self.uuid = str(uuid.uuid4())
        self.key = key
        self.value = value
        self.half_life_seconds = half_life_seconds
        now = datetime.now(timezone.utc)
        self.created_at = now
        self.last_decay = now

    def age_seconds(self) -> float:
        return (datetime.now(timezone.utc) - self.last_decay).total_seconds()

    def decay_factor(self, velocity: float) -> float:
        """Velocity‑modulated decay: faster NLMS velocity → stronger decay."""
        base_factor = 0.5 ** (self.age_seconds() / self.half_life_seconds)
        # clamp velocity to a reasonable range to avoid overflow
        vel = max(min(velocity, 10.0), 0.0)
        return base_factor ** (1.0 + vel)

    def apply_decay(self, velocity: float) -> None:
        factor = self.decay_factor(velocity)
        self.value *= factor
        self.last_decay = datetime.now(timezone.utc)

# ----------------------------------------------------------------------
# Core hybrid functions
# ----------------------------------------------------------------------
def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, float, float]:
    """
    Perform one NLMS adaptation step.
    Returns updated weights, prediction, and instantaneous velocity (error·x norm).
    """
    pred = float(weights @ x)
    error = target - pred
    norm = float(x @ x) + eps
    step = (mu / norm) * error
    new_weights = weights + step * x
    velocity = abs(error) * np.linalg.norm(x)  # scalar information‑energy
    return new_weights, pred, velocity

def compute_fisher_information(weights: np.ndarray, x: np.ndarray, sigma2: float = 1.0) -> float:
    """
    Fisher information for a Gaussian model N(μ= w·x, σ²) with respect to the scalar
    parameter θ = w·x.  I = (∂μ/∂θ)² / σ² = 1/σ² .
    For a vector‑parameter view we use the squared norm of the gradient:
        I = (x·x) / σ²
    """
    grad = x  # ∂μ/∂w = x
    return float((grad @ grad) / sigma2)

def update_pheromones(
    pheromones: Dict[str, PheromoneEntry],
    signature: np.ndarray,
    velocity: float,
    decay_rate: float = 1.0,
) -> None:
    """
    Map a graph‑signature vector to pheromone entries.
    The signature magnitude creates/updates entries; NLMS velocity modulates decay.
    """
    # simple mapping: each dimension -> a key
    for i, val in enumerate(signature):
        key = f"dim_{i}"
        if key not in pheromones:
            pheromones[key] = PheromoneEntry(key, value=val, half_life_seconds=30)
        else:
            # reinforcement proportional to |val|
            pheromones[key].value += decay_rate * float(val)
        pheromones[key].apply_decay(velocity)

def ternary_logic_gate(pheromone: float, fisher_scaled_propensity: float) -> int:
    """
    Combine pheromone signal and Fisher‑scaled propensity into a ternary decision.
    Returns -1, 0, or +1.
    """
    sign_ph = np.sign(pheromone)
    sign_fi = np.sign(fisher_scaled_propensity)
    combined = sign_ph + sign_fi
    if combined > 0:
        return 1
    if combined < 0:
        return -1
    return 0

def select_action(
    actions: List[BanditAction],
    pheromones: Dict[str, PheromoneEntry],
    fisher_info: float,
) -> Tuple[BanditAction, int]:
    """
    Choose an action by weighting propensities with Fisher information and
    perturbing with pheromone influence via ternary logic.
    Returns the chosen action and the ternary decision value.
    """
    # aggregate pheromone influence (sum of all current values)
    total_pheromone = sum(p.value for p in pheromones.values())
    # scale propensities
    scaled = []
    for a in actions:
        fi_weight = 1.0 + fisher_info * 0.1  # modest scaling
        scaled_prop = a.propensity * fi_weight
        scaled.append((a, scaled_prop))

    # softmax‑like selection (but deterministic for the smoke test)
    scaled.sort(key=lambda tup: tup[1], reverse=True)
    chosen = scaled[0][0]

    # ternary decision based on chosen action's propensity vs pheromone level
    decision = ternary_logic_gate(total_pheromone, scaled[0][1])
    return chosen, decision

# ----------------------------------------------------------------------
# Example hybrid system encapsulating all components
# ----------------------------------------------------------------------
class HybridSystem:
    def __init__(self, dim: int, num_actions: int = 3):
        self.dim = dim
        self.weights = np.zeros(dim, dtype=float)
        self.store = StoreState()
        self.pheromones: Dict[str, PheromoneEntry] = {}
        self.actions = [
            BanditAction(action_id=f"act_{i}", propensity=random.random())
            for i in range(num_actions)
        ]

    def step(self, x: np.ndarray, target: float) -> None:
        # 1. NLMS adaptation
        self.weights, pred, velocity = nlms_update(self.weights, x, target)

        # 2. Graph‑signature (here simply the absolute weights)
        signature = np.abs(self.weights)

        # 3. Update pheromones with velocity‑modulated decay
        update_pheromones(self.pheromones, signature, velocity)

        # 4. Compute Fisher information for the current input
        fisher = compute_fisher_information(self.weights, x)

        # 5. Select an action using the hybrid decision rule
        action, decision = select_action(self.actions, self.pheromones, fisher)

        # 6. Simple bandit‑like update of the chosen action's propensity
        reward = -abs(pred - target)  # negative error as reward
        lr = 0.1
        new_propensity = action.propensity + lr * (reward - action.propensity)
        # replace the action (BanditAction is frozen)
        self.actions = [
            BanditAction(a.action_id,
                         new_propensity if a.action_id == action.action_id else a.propensity)
            for a in self.actions
        ]

        # 7. Store dynamics (toy example: level follows a damped update)
        self.store.level = (
            self.store.base
            + self.store.alpha * velocity
            - self.store.beta * self.store.level
        ) * self.store.dt

    def summary(self) -> str:
        return (
            f"Weights: {self.weights}\\n"
            f"Store level: {self.store.level:.3f}\\n"
            f"Pheromones: {{k: v.value:.3f for k,v in self.pheromones.items()}}\\n"
            f"Actions: {[ (a.action_id, a.propensity) for a in self.actions ]}"
        )

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(0)
    random.seed(0)

    dim = 5
    system = HybridSystem(dim=dim, num_actions=4)

    # generate a synthetic regression stream
    true_w = np.array([1.2, -0.7, 0.3, 0.0, 0.5])
    for t in range(20):
        x = np.random.randn(dim)
        noise = np.random.normal(scale=0.1)
        target = float(true_w @ x + noise)
        system.step(x, target)

    # print a concise summary
    print("Final system state:")
    print(system.summary())