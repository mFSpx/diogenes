# DARWIN HAMMER — match 2407, survivor 3
# gen: 5
# parent_a: hybrid_omni_chaotic_sprint_jepa_energy_m80_s0.py (gen1)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m39_s1.py (gen4)
# born: 2026-05-29T23:42:18Z

"""Hybrid Algorithm: Graph‑Bandit Fisher Fusion

Parents:
- hybrid_omni_chaotic_sprint_jepa_energy_m80_s0.py (graph‑based representation learning + latent variable prediction)
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m39_s1.py (Fisher‑information angle selection + contextual bandit)

Mathematical Bridge:
The bandit’s *propensity* (a confidence scalar) is used as a weighting factor for the
L2 prediction error of the latent model, while the Fisher information computed for a
chosen “angle” (θ) scales the learning rate of the latent parameters.  Thus the loss
L = w_propensity · ‖Y – Ŷ‖₂² is minimized with a step size η·I_Fisher, where
I_Fisher is the Fisher information returned by the Fisher core.  This couples the
graph‑based representation Z, the latent prediction Ŷ = fθ(Z), and the bandit‑driven
confidence/Fisher dynamics into a single unified training step.
"""

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Fisher core (parent B)
# ----------------------------------------------------------------------
def compute_fisher_information(theta: float, mu: float, sigma: float, v: float) -> Tuple[float, float]:
    """
    Returns (intensity, Fisher information) for a Gaussian‑shaped angle model.
    """
    I = np.exp(-((theta - mu) / sigma) ** 2)                # Gaussian intensity
    F = (2 * (theta - mu) / sigma ** 2) ** 2 / (I + 1e-12)  # Fisher information (avoid div‑0)
    return v * I, v * F

# ----------------------------------------------------------------------
# Bandit core (parent B)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float          # confidence scalar
    expected_reward: float
    confidence_bound: float
    algorithm: str

# Simple in‑memory policy store (stateless for this demo)
_POLICY: Dict[str, List[float]] = {}
_STORE: Dict[str, float] = {}

def reset_policy() -> None:
    """Clear all learned statistics and the virtual store."""
    _POLICY.clear()
    _STORE.clear()

def select_action(
    context: Dict[str, float],
    actions: List[str],
    algorithm: str = "epsilon_greedy",
    epsilon: float = 0.1,
    seed: int | str | None = 7,
) -> BanditAction:
    """Choose an action and return its descriptor."""
    if not actions:
        raise ValueError("No actions provided")
    rng = random.Random(seed)
    if rng.random() < epsilon:
        chosen = rng.choice(actions)
        reward_est = 0.0
    else:
        # naive greedy: pick action with highest stored expected reward
        reward_est = -math.inf
        chosen = actions[0]
        for a in actions:
            val = _STORE.get(a, 0.0)
            if val > reward_est:
                reward_est = val
                chosen = a
    # generate synthetic confidence and bound
    propensity = max(0.1, min(1.0, rng.random()))          # confidence ∈ (0,1]
    confidence_bound = rng.random() * 0.5 + 0.5          # between 0.5 and 1.0
    return BanditAction(
        action_id=chosen,
        propensity=propensity,
        expected_reward=reward_est,
        confidence_bound=confidence_bound,
        algorithm=algorithm,
    )

def update_policy(action: BanditAction, reward: float) -> None:
    """Simple incremental update of the expected reward."""
    prev = _STORE.get(action.action_id, 0.0)
    # exponential moving average with fixed decay
    decay = 0.9
    _STORE[action.action_id] = decay * prev + (1 - decay) * reward

# ----------------------------------------------------------------------
# Graph‑based representation learning (parent A)
# ----------------------------------------------------------------------
def build_adjacency(X: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Constructs a symmetric adjacency matrix using an RBF kernel.
    X : (n_samples, n_features)
    """
    sq_dists = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)
    A = np.exp(-sq_dists / (2 * sigma ** 2))
    np.fill_diagonal(A, 0.0)
    return A

def graph_representation_learning(X: np.ndarray, dim: int = 4, iters: int = 10) -> np.ndarray:
    """
    Simple graph‑based embedding via power iteration on the normalized Laplacian.
    Returns Z ∈ ℝ^{n_samples × dim}.
    """
    n = X.shape[0]
    A = build_adjacency(X)
    D = np.diag(A.sum(axis=1) + 1e-12)
    L = D - A                       # unnormalized Laplacian
    # Normalize
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D)))
    L_norm = D_inv_sqrt @ L @ D_inv_sqrt
    # Power iteration
    Z = np.random.randn(n, dim)
    for _ in range(iters):
        Z = L_norm @ Z
        # re‑orthogonalize
        Q, _ = np.linalg.qr(Z)
        Z = Q
    return Z

# ----------------------------------------------------------------------
# Latent variable prediction (parent A)
# ----------------------------------------------------------------------
def latent_predict(Z: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Linear latent model: Ŷ = Z·W + b
    Z : (n, d_z)
    W : (d_z, d_y)
    b : (d_y,)
    """
    return Z @ W + b

# ----------------------------------------------------------------------
# Hybrid training step (fusion of A & B)
# ----------------------------------------------------------------------
def hybrid_train_step(
    X: np.ndarray,
    Y: np.ndarray,
    theta: float,
    mu: float,
    sigma_f: float,
    v: float,
    actions: List[str],
    context: Dict[str, float],
) -> Tuple[np.ndarray, float]:
    """
    Performs one hybrid training iteration:
    1. Graph‑based representation Z = f_graph(X)
    2. Bandit selects an action → provides propensity w_p
    3. Fisher information (I, F) is computed for the chosen angle θ
    4. Latent prediction Ŷ = f_latent(Z; W,b)
    5. Weighted L2 loss L = w_p * ‖Y-Ŷ‖₂²
    6. Parameter update with step size η = η₀ * F (Fisher scaling)
    Returns updated parameters W and the scalar loss.
    """
    # 1. representation
    Z = graph_representation_learning(X)

    # 2. bandit decision
    ba = select_action(context, actions)
    w_p = ba.propensity

    # 3. Fisher scaling
    _, fisher = compute_fisher_information(theta, mu, sigma_f, v)
    eta0 = 1e-3                     # base learning rate
    eta = eta0 * fisher

    # 4. latent parameters (initialized if not present)
    d_z = Z.shape[1]
    d_y = Y.shape[1]
    if not hasattr(hybrid_train_step, "W"):
        hybrid_train_step.W = np.random.randn(d_z, d_y) * 0.01
        hybrid_train_step.b = np.zeros(d_y)

    # prediction
    Y_hat = latent_predict(Z, hybrid_train_step.W, hybrid_train_step.b)

    # 5. weighted loss
    diff = Y - Y_hat
    loss = w_p * np.mean(np.linalg.norm(diff, axis=1) ** 2)

    # 6. gradient descent with Fisher‑scaled step
    grad_W = -2 * w_p * (Z.T @ diff) / X.shape[0]
    grad_b = -2 * w_p * np.mean(diff, axis=0)

    hybrid_train_step.W -= eta * grad_W
    hybrid_train_step.b -= eta * grad_b

    # optional: update bandit statistics with pseudo‑reward = -loss
    update_policy(ba, -loss)

    return Y_hat, loss

# ----------------------------------------------------------------------
# Example usage (smoke test)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # synthetic dataset
    rng = np.random.default_rng(42)
    X = rng.normal(size=(50, 6))          # 50 samples, 6 features
    true_W = rng.normal(size=(4, 3))
    true_b = rng.normal(size=(3,))

    # generate ground‑truth Y via a hidden graph+latent pipeline
    Z_true = graph_representation_learning(X, dim=4, iters=5)
    Y = latent_predict(Z_true, true_W, true_b) + rng.normal(scale=0.05, size=(50, 3))

    # bandit configuration
    actions = ["angle_-30", "angle_0", "angle_30"]
    context = {"epoch": 0.0}

    # run a few hybrid steps
    for epoch in range(5):
        theta = {"angle_-30": -30.0, "angle_0": 0.0, "angle_30": 30.0}[select_action(context, actions).action_id]
        Y_hat, loss = hybrid_train_step(
            X,
            Y,
            theta=theta,
            mu=0.0,
            sigma_f=15.0,
            v=1.0,
            actions=actions,
            context=context,
        )
        print(f"Epoch {epoch}: loss={loss:.6f}")

    print("Hybrid training completed without errors.")