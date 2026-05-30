# DARWIN HAMMER — match 512, survivor 2
# gen: 4
# parent_a: hybrid_hybrid_hybrid_ternar_ttt_linear_m5_s0.py (gen3)
# parent_b: hybrid_bandit_router_honeybee_store_m9_s3.py (gen1)
# born: 2026-05-29T23:29:12Z

"""
Hybrid Ternary‑Bandit Router

This module fuses the core mathematics of two parent algorithms:

* **Parent A – hybrid_hybrid_hybrid_ternar_ttt_linear_m5_s0.py**  
  – Provides a structural similarity (SSIM) measure between two signal
    vectors and a linear recurrent weight‑update (the “TTT‑Linear” rule).

* **Parent B – hybrid_bandit_router_honeybee_store_m9_s3.py**  
  – Implements a contextual multi‑armed bandit with a store that evolves
    according to inflow/outflow dynamics and supplies an action‑selection
    mechanism (ε‑greedy / Thompson / LinUCB‑style).

**Mathematical bridge** – The SSIM value computed on the ternary router’s
input‑output pair is interpreted as a *reward* for the bandit.  The store
value, updated by the honey‑bee dynamics, modulates both the exploration
parameter of the bandit and the learning‑rate of the linear weight update.
Thus the similarity‑driven reward, the adaptive store, and the linear weight
recurrence are tightly coupled into a single hybrid system.
"""

import json
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Utility – SSIM (from Parent A)
# ----------------------------------------------------------------------
def ssim(x: np.ndarray, y: np.ndarray, dynamic_range: float = 255.0,
         k1: float = 0.01, k2: float = 0.03) -> float:
    """Structural Similarity Index Measure for 1‑D signals."""
    if x.shape != y.shape:
        raise ValueError("samples must have equal shape")
    if x.size == 0:
        raise ValueError("samples must not be empty")
    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x = np.var(x)
    sigma_y = np.var(y)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    return float(numerator / denominator)


# ----------------------------------------------------------------------
# Linear recurrent weight update – “TTT‑Linear” (simplified)
# ----------------------------------------------------------------------
def ttt_linear_update(weights: np.ndarray, grad: np.ndarray,
                     lr: float, momentum: float = 0.0,
                     velocity: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform a single TTT‑Linear update:
        v_{t+1} = momentum * v_t + lr * grad
        w_{t+1} = w_t - v_{t+1}
    Returns the updated weights and the new velocity.
    """
    if velocity is None:
        velocity = np.zeros_like(weights)
    velocity = momentum * velocity + lr * grad
    weights = weights - velocity
    return weights, velocity


# ----------------------------------------------------------------------
# Bandit data structures – from Parent B
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


# ----------------------------------------------------------------------
# Global policy store (simple mean reward per action)
# ----------------------------------------------------------------------
_POLICY: Dict[str, List[float]] = {}


def reset_policy() -> None:
    _POLICY.clear()


def _reward(action: str) -> float:
    total, n = _POLICY.get(action, [0.0, 0.0])
    return total / n if n else 0.0


def _count(action: str) -> float:
    return _POLICY.get(action, [0.0, 0.0])[1]


def update_policy(updates: List[BanditUpdate]) -> None:
    for u in updates:
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        stats[0] += float(u.reward)
        stats[1] += 1.0


# ----------------------------------------------------------------------
# Honey‑bee store dynamics – from Parent B
# ----------------------------------------------------------------------
def update_store(store: float,
                 inflow: List[float],
                 outflow: List[float],
                 alpha: float = 1.0,
                 beta: float = 1.0,
                 dt: float = 1.0) -> Tuple[float, float]:
    """
    Store evolves as:
        dS/dt = α Σ inflow - β Σ outflow
    Returns (new_store, delta) where delta is the raw change before clipping.
    """
    delta = alpha * sum(inflow) - beta * sum(outflow)
    new_store = max(0.0, store + dt * delta)
    return new_store, delta


# ----------------------------------------------------------------------
# Hybrid action selection – merges bandit choice with SSIM‑derived reward
# ----------------------------------------------------------------------
def hybrid_select_action(context: Dict[str, float],
                         actions: List[str],
                         store: float,
                         observation: np.ndarray,
                         prediction: np.ndarray,
                         algorithm: str = "epsilon_greedy",
                         epsilon: float = 0.1,
                         seed: int | str | None = 7) -> Tuple[BanditAction, float]:
    """
    1. Compute similarity = SSIM(observation, prediction).  
    2. Treat similarity as an instantaneous reward signal.  
    3. Update the global policy with this reward (so future expectations reflect
       the structural match).  
    4. Use the store to bias exploration/exploitation and to scale the learning
       rate for the linear weight update (performed elsewhere).  
    5. Return the chosen BanditAction and the raw similarity reward.
    """
    if not actions:
        raise ValueError("actions required")
    rng = random.Random(seed)

    # 1. Similarity as reward
    reward = ssim(observation, prediction)

    # 2. Update policy (single‑step update)
    updates = [BanditUpdate(context_id="hybrid",
                            action_id=actions[0],  # placeholder, will be overwritten later
                            reward=reward,
                            propensity=1.0)]
    # We'll update after the action is chosen

    # 3. Store‑dependent exploration factor
    store_factor = 1.0 + store / (store + 1.0)   # grows from 1 to 2

    # 4. Choose action according to algorithm
    if algorithm == "epsilon_greedy" and rng.random() < epsilon * (1.0 / store_factor):
        chosen_id = rng.choice(actions)
    elif algorithm == "thompson":
        def sample(a: str) -> float:
            r = _reward(a)
            n = _count(a)
            a_param = 1.0 + max(0.0, r) * store_factor
            b_param = 1.0 + max(0.0, 1.0 - r) * store_factor
            return rng.betavariate(a_param, b_param)
        chosen_id = max(actions, key=sample)
    else:  # default: LinUCB‑style (simplified)
        ctx_norm = np.linalg.norm(list(context.values())) + 1e-8
        def ucb_score(a: str) -> float:
            mean = _reward(a)
            confidence = math.sqrt(math.log(1 + ctx_norm) / (_count(a) + 1e-6))
            return mean + store_factor * confidence
        chosen_id = max(actions, key=ucb_score)

    # 5. Final policy update with correct action_id
    updates[0] = updates[0]._replace(action_id=chosen_id)
    update_policy(updates)

    # 6. Construct BanditAction object
    prop = 1.0 / len(actions)  # uniform propensity for illustration
    exp_reward = _reward(chosen_id)
    conf = math.sqrt(2 * math.log(1 + store) / (_count(chosen_id) + 1e-6))
    action = BanditAction(action_id=chosen_id,
                          propensity=prop,
                          expected_reward=exp_reward,
                          confidence_bound=conf,
                          algorithm=algorithm)
    return action, reward


# ----------------------------------------------------------------------
# Example hybrid pipeline (three demonstration functions)
# ----------------------------------------------------------------------
def embed_text(text: str, dim: int = 32) -> np.ndarray:
    """Deterministic pseudo‑embedding: hash → vector."""
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    return rng.random(dim)


def process_packet(packet: Dict[str, Any],
                   actions: List[str],
                   store: float,
                   weights: np.ndarray,
                   velocity: np.ndarray | None = None) -> Tuple[BanditAction,
                                                                np.ndarray,
                                                                np.ndarray,
                                                                float]:
    """
    Full hybrid step:
    * embed incoming text,
    * produce a linear prediction = weights·embedding,
    * compute SSIM reward,
    * select a bandit action,
    * update linear weights with a learning rate scaled by store.
    Returns (action, new_weights, new_velocity, new_store).
    """
    text = str(packet.get("text", ""))
    obs = embed_text(text)
    pred = weights @ obs  # scalar prediction
    # For SSIM we need two vectors of same shape; replicate scalar
    pred_vec = np.full_like(obs, pred)
    action, reward = hybrid_select_action(
        context={"len": len(text)},
        actions=actions,
        store=store,
        observation=obs,
        prediction=pred_vec,
        algorithm="epsilon_greedy",
        epsilon=0.2,
        seed=packet.get("seed", 7)
    )
    # Gradient of a simple squared error loss w.r.t. weights
    grad = 2.0 * (pred - reward) * obs
    lr = 0.01 * (1.0 + store)  # store‑scaled learning rate
    new_weights, new_velocity = ttt_linear_update(weights, grad, lr, momentum=0.9, velocity=velocity)
    # Update store with reward as inflow and negative reward as outflow (illustrative)
    new_store, _ = update_store(store, inflow=[reward], outflow=[-reward])
    return action, new_weights, new_velocity, new_store


def initialise_hybrid(dim: int = 32) -> Tuple[np.ndarray, np.ndarray, float]:
    """Create initial linear weights, zero velocity, and zero store."""
    rng = np.random.default_rng(42)
    weights = rng.normal(scale=0.1, size=dim)
    velocity = np.zeros_like(weights)
    store = 0.0
    return weights, velocity, store


def demo_hybrid_loop(num_steps: int = 5) -> None:
    """Run a short demo loop to prove that the hybrid works without error."""
    actions = ["alpha", "beta", "gamma"]
    weights, velocity, store = initialise_hybrid()
    for step in range(num_steps):
        packet = {"text": f"sample message {step}", "seed": step}
        action, weights, velocity, store = process_packet(
            packet, actions, store, weights, velocity
        )
        print(f"Step {step}: action={action.action_id}, reward≈{action.expected_reward:.3f}, store={store:.3f}")


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    demo_hybrid_loop()