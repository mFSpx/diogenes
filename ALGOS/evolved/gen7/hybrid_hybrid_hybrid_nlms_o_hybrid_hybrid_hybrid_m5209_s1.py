# DARWIN HAMMER — match 5209, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_nlms_omni_cha_hybrid_gliner_zero_s_m26_s2.py (gen2)
# parent_b: hybrid_hybrid_hybrid_bandit_hybrid_hybrid_hybrid_m2045_s1.py (gen6)
# born: 2026-05-30T00:00:37Z

"""
Hybrid NLMS‑Tree & Gaussian‑Fisher Bandit

This module fuses the core mathematics of two parent algorithms:

* **Parent A** – a normalized least‑mean‑squares (NLMS) update that adapts the
  edge‑weight vector of a minimum‑cost tree.
* **Parent B** – a Gaussian‑beam similarity measure together with the Fisher
  score, used to modulate a honey‑bee‑style store (policy) based on the quality
  of the NLMS update.

**Mathematical bridge**

1. The NLMS rule produces an error  e = y_target – ŷ  where ŷ = w·x.
2. The absolute error is interpreted as an angular deviation θ between the
   current weight vector and an ideal direction (center = 0).  
   `gaussian_beam(θ, 0, σ)` yields a similarity score S ∈ (0,1].
3. The Fisher score F = (∂S/∂θ)² / S  quantifies the sensitivity of the
   similarity to changes in θ.  This scalar drives the adaptive update of the
   `StoreState` (gain, α, β) which in turn scales the NLMS learning rate μ.
4. The updated NLMS weights are fed back to the (implicit) minimum‑cost tree,
   whose cost is approximated by the ℓ₁‑norm of the weight vector.

Thus the NLMS adaptation, Gaussian‑beam similarity and Fisher‑score‑driven
store update form a closed loop that continuously reshapes both the tree
weights and the policy state.

The module provides three high‑level functions that illustrate this hybrid
behaviour.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Tuple, List

import numpy as np

# ----------------------------------------------------------------------
# Parent A – NLMS core
# ----------------------------------------------------------------------
def predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Linear prediction ŷ = w·x."""
    return float(np.dot(weights, x))


def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, float]:
    """
    Perform one NLMS adaptation step.

    Returns the updated weight vector and the scalar prediction error.
    """
    y_hat = predict(weights, x)
    error = target - y_hat
    power = float(np.dot(x, x) + eps)
    step = mu * error / power
    new_weights = weights + step * x
    return new_weights, error


# ----------------------------------------------------------------------
# Parent B – Gaussian beam & Fisher score
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian beam similarity (unnormalised)."""
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """
    Fisher information of the Gaussian beam w.r.t. its angle parameter.
    """
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


# ----------------------------------------------------------------------
# StoreState – honey‑bee style policy container
# ----------------------------------------------------------------------
@dataclass
class StoreState:
    """State that controls the NLMS learning dynamics."""
    level: float = 0.0          # abstract store level
    alpha: float = 1.0          # scaling for learning rate
    beta: float = 1.0           # secondary scaling (e.g., momentum)
    dt: float = 1.0             # discrete time step
    base: float = 1.0           # baseline gain
    gain: float = 1.0           # multiplicative gain applied to μ
    limit: float = 10.0         # upper bound for level

    def adapt(self, fisher: float, error: float) -> None:
        """
        Update the internal parameters using the Fisher score.
        A larger Fisher score (high sensitivity) increases the gain,
        while the magnitude of the error modulates the store level.
        """
        # Simple proportional adaptation
        self.gain = min(self.limit, self.base + self.alpha * fisher)
        self.level = max(0.0, self.level + self.beta * abs(error) * self.dt)


# ----------------------------------------------------------------------
# Helper – pseudo minimum‑cost tree cost
# ----------------------------------------------------------------------
def tree_cost(weights: np.ndarray) -> float:
    """
    Approximate the minimum‑cost tree cost as the L1 norm of the edge weights.
    In a full implementation this would be the sum of selected edges after
    solving a spanning‑tree optimisation.
    """
    return float(np.sum(np.abs(weights)))


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def hybrid_nlms_step(
    state: StoreState,
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    beam_width: float = 1.0,
) -> Tuple[np.ndarray, StoreState, float]:
    """
    One hybrid iteration:

    1. Scale the NLMS learning rate μ by the current store gain.
    2. Perform the NLMS weight update.
    3. Compute a Gaussian‑beam similarity between the absolute error and 0.
    4. Derive the Fisher score and let the store adapt.
    5. Return the new weights, updated store and the current tree cost.
    """
    # 1 – adaptive learning rate
    mu_adapt = state.gain * 0.5  # base μ = 0.5 scaled by gain

    # 2 – NLMS adaptation
    new_weights, error = nlms_update(weights, x, target, mu=mu_adapt)

    # 3 – similarity (θ is taken as |error|)
    theta = abs(error)
    similarity = gaussian_beam(theta, center=0.0, width=beam_width)

    # 4 – Fisher information drives store adaptation
    fisher = fisher_score(theta, center=0.0, width=beam_width)
    state.adapt(fisher, error)

    # 5 – cost metric (optional for downstream use)
    cost = tree_cost(new_weights)

    # Debug trace (could be toggled by a verbosity flag)
    print(
        f"[HybridStep] μ={mu_adapt:.4f}, err={error:.4f}, sim={similarity:.4f}, "
        f"Fisher={fisher:.4e}, gain={state.gain:.4f}, level={state.level:.4f}, cost={cost:.4f}"
    )

    return new_weights, state, cost


def batch_hybrid_training(
    init_weights: np.ndarray,
    data: List[Tuple[np.ndarray, float]],
    epochs: int = 5,
    beam_width: float = 1.0,
) -> Tuple[np.ndarray, StoreState]:
    """
    Run a mini‑batch training loop over the supplied (x, target) pairs.
    Returns the final weight vector and store state.
    """
    state = StoreState()
    weights = init_weights.copy()

    for epoch in range(epochs):
        print(f"\n=== Epoch {epoch + 1}/{epochs} ===")
        random.shuffle(data)
        for i, (x, target) in enumerate(data):
            weights, state, _ = hybrid_nlms_step(state, weights, x, target, beam_width)

    return weights, state


def evaluate_hybrid(
    weights: np.ndarray,
    test_set: List[Tuple[np.ndarray, float]],
) -> Tuple[float, float]:
    """
    Evaluate the hybrid model on a test set.
    Returns (MSE, average_tree_cost).
    """
    errors = []
    costs = []
    for x, target in test_set:
        pred = predict(weights, x)
        errors.append((target - pred) ** 2)
        costs.append(tree_cost(weights))

    mse = float(np.mean(errors))
    avg_cost = float(np.mean(costs))
    print(f"[Evaluation] MSE={mse:.6f}, AvgTreeCost={avg_cost:.4f}")
    return mse, avg_cost


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic linear regression problem
    dim = 8
    true_weights = np.random.randn(dim)

    # Generate training data
    train_data = []
    for _ in range(200):
        x = np.random.randn(dim)
        noise = np.random.normal(scale=0.1)
        y = float(np.dot(true_weights, x) + noise)
        train_data.append((x, y))

    # Initial guess
    init_w = np.zeros(dim)

    # Train
    final_w, final_state = batch_hybrid_training(init_w, train_data, epochs=3, beam_width=0.5)

    # Generate test data
    test_data = []
    for _ in range(50):
        x = np.random.randn(dim)
        y = float(np.dot(true_weights, x) + np.random.normal(scale=0.1))
        test_data.append((x, y))

    # Evaluate
    evaluate_hybrid(final_w, test_data)

    print("\nFinal StoreState:", final_state)
    print("True weights vs. learned weights (first 5 entries):")
    print("True :", true_weights[:5])
    print("Learned:", final_w[:5])