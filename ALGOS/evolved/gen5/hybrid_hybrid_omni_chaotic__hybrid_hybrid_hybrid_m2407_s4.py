# DARWIN HAMMER — match 2407, survivor 4
# gen: 5
# parent_a: hybrid_omni_chaotic_sprint_jepa_energy_m80_s0.py (gen1)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m39_s1.py (gen4)
# born: 2026-05-29T23:42:18Z

"""Hybrid Algorithm: Omni-Graph Fisher Bandit Fusion

This module fuses the core topology of **hybrid_omni_chaotic_sprint_jepa_energy_m80_s0.py**
(parent A) and **hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m39_s1.py**
(parent B).

* From parent A we take a *graph‑based representation learner* that produces
  node embeddings 𝑅∈ℝ^{N×d} and a *latent predictor* that maps these embeddings
  to a target space 𝑌̂.  The learning objective is the L2 prediction error
  ‖𝑌̂−𝑌‖₂².

* From parent B we take the *Fisher information* computation for a scalar
  angle θ and the *contextual bandit* that yields a confidence scalar
  (propensity) for each action.  The bandit’s propensity is interpreted as a
  confidence weight that modulates the Fisher information.

**Mathematical bridge** – the bandit confidence (propensity) scales the Fisher
information term that is added to the L2 loss of the graph‑latent pipeline.
Thus the total hybrid loss is  

    L = ‖Ŷ−Y‖₂²  +  λ · (propensity) · F(θ;μ,σ)  

where μ and σ are the mean and standard deviation of the latent predictions,
λ is a tunable scalar, and F is the Fisher information defined in parent B.
The fused system therefore learns graph representations while the bandit
guides the optimisation through a confidence‑weighted Fisher regulariser.

The module provides three public functions that demonstrate the hybrid
operation and a small smoke test.
"""

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import re

# ----------------------------------------------------------------------
# Parent A – Graph representation + latent predictor
# ----------------------------------------------------------------------
def graph_representation(adj: np.ndarray, feats: np.ndarray, steps: int = 5) -> np.ndarray:
    """
    Simple diffusion‑based graph embedding.
    R^{(0)} = feats
    R^{(t+1)} = A @ R^{(t)}   (A is row‑normalised adjacency)
    Returns the final embedding normalised to unit length per node.
    """
    if adj.shape[0] != adj.shape[1]:
        raise ValueError("Adjacency matrix must be square")
    if adj.shape[0] != feats.shape[0]:
        raise ValueError("Adjacency and feature row dimensions must match")
    # Row‑normalise adjacency to avoid blow‑up
    row_sums = adj.sum(axis=1, keepdims=True) + 1e-12
    A = adj / row_sums
    R = feats.astype(float)
    for _ in range(steps):
        R = A @ R
    # Normalise each node vector
    norm = np.linalg.norm(R, axis=1, keepdims=True) + 1e-12
    return R / norm


def latent_predict(embeddings: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """
    Linear latent predictor: Ŷ = R @ W
    embeddings : (N, d)
    weight     : (d, o)   where o is output dimension
    Returns (N, o)
    """
    return embeddings @ weight


# ----------------------------------------------------------------------
# Parent B – Fisher information + contextual bandit
# ----------------------------------------------------------------------
def compute_fisher_information(theta: float, mu: float, sigma: float, v: float) -> Tuple[float, float]:
    """
    Returns (I, F) where
        I = v * exp(-((θ-μ)/σ)^2)               Gaussian intensity
        F = v * ((2*(θ-μ)/σ^2)^2 / I)           Fisher information
    """
    I = np.exp(-((theta - mu) / sigma) ** 2)
    F = (2 * (theta - mu) / sigma ** 2) ** 2 / (I + 1e-12)
    return v * I, v * F


@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float          # confidence scalar
    expected_reward: float
    confidence_bound: float
    algorithm: str


@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float


# Simple LinUCB‑style policy storage
_POLICY: Dict[str, List[float]] = {}   # action_id -> [cumulative_reward, count]
_STORE: Dict[str, float] = {}          # virtual store for any auxiliary data


def reset_policy() -> None:
    """Clear all learned statistics and the virtual store."""
    _POLICY.clear()
    _STORE.clear()


def select_action(
    context: Dict[str, float],
    actions: List[str],
    algorithm: str = "linucb",
    epsilon: float = 0.1,
    seed: int | str | None = 7,
) -> BanditAction:
    """
    Epsilon‑greedy selector.
    - With probability epsilon, picks a random action.
    - Otherwise picks the action with highest UCB score.
    The returned propensity is the normalized UCB score (in [0,1]).
    """
    if not actions:
        raise ValueError("No actions provided")
    random_state = random.Random(seed)
    if random_state.random() < epsilon:
        chosen = random_state.choice(actions)
        propensity = 0.5
        exp_reward = _POLICY.get(chosen, [0.0, 0])[0] / max(1, _POLICY.get(chosen, [0.0, 0])[1])
        bound = 0.0
        return BanditAction(chosen, propensity, exp_reward, bound, algorithm)

    # Compute UCB for each action
    total_counts = sum(cnt for _, cnt in _POLICY.values()) + 1e-12
    best_score = -float("inf")
    best_action = None
    for a in actions:
        reward_sum, cnt = _POLICY.get(a, [0.0, 0])
        avg = reward_sum / max(1, cnt)
        # classic LinUCB confidence term
        confidence = math.sqrt(2 * math.log(total_counts) / max(1, cnt))
        ucb = avg + confidence
        if ucb > best_score:
            best_score = ucb
            best_action = a
    # Normalise propensity to [0,1] using sigmoid
    propensity = 1 / (1 + math.exp(-best_score))
    exp_reward = _POLICY.get(best_action, [0.0, 0])[0] / max(1, _POLICY.get(best_action, [0.0, 0])[1])
    return BanditAction(best_action, propensity, exp_reward, confidence, algorithm)


def update_policy(update: BanditUpdate) -> None:
    """
    Incremental update of the bandit statistics.
    """
    stats = _POLICY.setdefault(update.action_id, [0.0, 0])
    stats[0] += update.reward
    stats[1] += 1


# ----------------------------------------------------------------------
# Hybrid core – combines both families
# ----------------------------------------------------------------------
def hybrid_loss(
    adj: np.ndarray,
    feats: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
    theta: float,
    v: float = 1.0,
    lambda_fisher: float = 0.1,
    bandit_actions: List[str] = None,
) -> Tuple[float, BanditAction]:
    """
    Executes one hybrid optimisation step and returns the scalar loss together
    with the bandit action that supplied the confidence scaling.

    Steps:
    1. Compute graph embeddings R.
    2. Predict Ŷ = R @ W.
    3. Derive μ and σ from the predictions (mean & std over all nodes).
    4. Compute Fisher information (I, F) for the supplied angle θ.
    5. Obtain a bandit action; its propensity acts as a confidence weight.
    6. Assemble the hybrid loss:
           L = ||Ŷ - Y||_2^2 + λ * propensity * F
    """
    # 1‑2: representation + prediction
    R = graph_representation(adj, feats)
    Y_hat = latent_predict(R, weight)

    # 3: statistics of predictions (flatten across nodes and output dim)
    flat = Y_hat.ravel()
    mu = float(np.mean(flat))
    sigma = float(np.std(flat) + 1e-12)

    # 4: Fisher term
    _, F = compute_fisher_information(theta, mu, sigma, v)

    # 5: bandit confidence
    if bandit_actions is None:
        bandit_actions = ["inc", "dec", "stay"]
    # Context can be any dict; here we pass simple stats
    context = {"mu": mu, "sigma": sigma, "theta": theta}
    action = select_action(context, bandit_actions)

    # 6: hybrid loss
    l2 = np.linalg.norm(Y_hat - target) ** 2
    loss = l2 + lambda_fisher * action.propensity * F

    return loss, action


def hybrid_step(
    adj: np.ndarray,
    feats: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
    theta: float,
    v: float = 1.0,
    lr: float = 0.01,
) -> Tuple[float, np.ndarray]:
    """
    Performs a single gradient‑descent style update on the predictor weight
    matrix using the hybrid loss. Returns the loss value and the updated weight.
    The gradient is approximated analytically for the L2 term; the Fisher term
    does not depend on the weight (it depends on predictions only), so its
    gradient w.r.t. the weight is zero.
    """
    # Compute forward pass and loss
    loss, _ = hybrid_loss(adj, feats, target, weight, theta, v)

    # Back‑propagate L2 part: dL/dW = 2 * R^T (R W - Y)
    R = graph_representation(adj, feats)
    grad = 2.0 * (R.T @ (R @ weight - target))

    # Simple SGD update
    weight_new = weight - lr * grad
    return loss, weight_new


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a tiny synthetic graph (5 nodes)
    np.random.seed(42)
    N = 5
    d = 4   # node feature dimension
    o = 3   # output dimension

    # Random adjacency (symmetric, no self‑loops)
    A = np.random.rand(N, N)
    A = (A + A.T) / 2
    np.fill_diagonal(A, 0)

    # Random node features and target matrix
    X = np.random.randn(N, d)
    Y = np.random.randn(N, o)

    # Random predictor weight
    W = np.random.randn(d, o)

    # Hyper‑parameters
    theta_val = 0.7
    v_val = 1.0

    # Reset bandit stats
    reset_policy()

    # Run a few hybrid steps to ensure everything executes
    for epoch in range(3):
        loss, W = hybrid_step(A, X, Y, W, theta_val, v=v_val, lr=0.05)
        print(f"Epoch {epoch+1}: loss = {loss:.6f}")

    # Show final predictor weight shape
    print("Final weight shape:", W.shape)