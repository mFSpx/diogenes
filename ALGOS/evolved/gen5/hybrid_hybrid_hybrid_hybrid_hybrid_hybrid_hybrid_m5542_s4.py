# DARWIN HAMMER — match 5542, survivor 4
# gen: 5
# parent_a: hybrid_hybrid_hybrid_path_s_hybrid_hybrid_nlms_o_m668_s3.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_bandit_router_m999_s2.py (gen4)
# born: 2026-05-30T00:02:42Z

"""
Hybrid Algorithm: Path‑Signature NLMS + Ternary‑Router / Bandit‑Temperature Fusion

Parents
-------
* **Parent A** – `hybrid_hybrid_hybrid_path_s_hybrid_hybrid_nlms_o_m668_s3.py`
  - Builds a lead‑lag augmented discrete path from master vectors `x_i`.
  - Computes level‑1 and level‑2 iterated‑integral signatures `σ_i`.
  - Learns a global weight vector `w` with the Normalised Least‑Mean‑Squares (NLMS)
    rule and later uses `w` to weight graph edges.

* **Parent B** – `hybrid_hybrid_hybrid_hybrid_hybrid_bandit_router_m999_s2.py`
  - Implements a ternary router (output –1, 0, +1) driven by a linear mapping
    `y = w·σ`.
  - Updates routing probabilities with a bandit‑style exponential‑weight rule.
  - Modulates the bandit learning rate by a temperature computed from the
    Schoolfield developmental‑rate model.

Mathematical Bridge
-------------------
Both parents manipulate a **global weight vector** `w` that acts on the same
feature space – the path‑signature vectors `σ_i`.  The bridge therefore consists
of a unified update that

1. **NLMS part** – drives `w` toward a desired importance `d_i = ‖σ_i‖₂`.
2. **Ternary‑Router part** – extracts a discrete decision
   `τ_i = sign_ternary(w·σ_i)` (‑1, 0, +1) which serves as a stochastic bandit
   reward.
3. **Bandit‑Temperature part** – updates a probability vector `p` over the
   three ternary actions using an exponential‑weight rule whose learning rate
   is scaled by a temperature `T` from the Schoolfield model.

The resulting hybrid step for a single sample `σ` is


# NLMS update
y   = w·σ
e   = d - y
w← w + μ·τ·e·σ / (‖σ‖² + ε)          # τ (ternary decision) modulates NLMS gain

# Bandit update
p← softmax( log(p) + (reward·τ)/T )   # reward is +1 for correct ternary sign,
                                      # –1 otherwise


The three public functions below expose this fused pipeline:
`compute_signature`, `hybrid_nlms_ternary_bandit_step`, and `run_hybrid_pipeline`. 
"""

import math
import random
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Utility building blocks
# ----------------------------------------------------------------------


def lead_lag_transform(X: np.ndarray) -> np.ndarray:
    """
    Apply the lead‑lag augmentation to a sequence of vectors.

    Parameters
    ----------
    X : np.ndarray, shape (N, D)
        Original discrete path.

    Returns
    -------
    X̃ : np.ndarray, shape (2·N‑1, D)
        Lead‑lag path: (x₁, x₁, x₂, x₂, …, x_N, x_N) with the duplicate
        of the first element removed to keep the length 2·N‑1.
    """
    if X.ndim != 2:
        raise ValueError("X must be a 2‑D array (N, D)")
    lead = X
    lag = np.vstack([X[0:1], X[:-1]])  # shift right, prepend first element
    return np.concatenate([lead, lag], axis=0)


def compute_signature(X: np.ndarray) -> np.ndarray:
    """
    Compute level‑1 and level‑2 signatures of a lead‑lag path.

    Level‑1  : σ¹ = Σ Δ_k
    Level‑2  : σ² = Σ_{k<ℓ} Δ_k ⊗ Δ_ℓ   (flattened outer products)

    Parameters
    ----------
    X : np.ndarray, shape (N, D)
        Original master‑vector sequence.

    Returns
    -------
    σ : np.ndarray, shape (D + D*D,)
        Concatenated level‑1 vector and flattened level‑2 matrix.
    """
    X̃ = lead_lag_transform(X)                     # (2N‑1, D)
    deltas = np.diff(X̃, axis=0)                  # (2N‑2, D)

    # Level‑1 signature (vector sum of increments)
    sigma1 = deltas.sum(axis=0)                    # (D,)

    # Level‑2 signature (sum of outer products of all ordered pairs)
    # Efficient computation via broadcasting:
    #   Σ_{k<ℓ} Δ_k ⊗ Δ_ℓ = 0.5 * ( (Σ Δ_k) ⊗ (Σ Δ_k) - Σ (Δ_k ⊗ Δ_k) )
    sum_delta = sigma1
    sum_outer = np.einsum('i,j->ij', sum_delta, sum_delta)  # (D, D)
    diag_outer = np.einsum('ki,ki->ij', deltas, deltas)    # (D, D)
    sigma2 = 0.5 * (sum_outer - diag_outer)                # (D, D)

    return np.concatenate([sigma1, sigma2.ravel()])


def nlms_update(
    w: np.ndarray,
    sigma: np.ndarray,
    d: float,
    mu: float = 0.1,
    eps: float = 1e-6,
    modulation: float = 1.0,
) -> np.ndarray:
    """
    One NLMS adaptation step with optional modulation factor.

    Parameters
    ----------
    w          : np.ndarray, shape (M,)
        Current weight vector.
    sigma      : np.ndarray, shape (M,)
        Input feature (signature).
    d          : float
        Desired output (here ‖sigma‖₂).
    mu         : float
        Base learning rate.
    eps        : float
        Regularisation term to avoid division by zero.
    modulation : float
        Multiplicative factor (e.g., ternary decision) that scales the step.

    Returns
    -------
    w_new : np.ndarray, shape (M,)
        Updated weight vector.
    """
    y = w.dot(sigma)
    e = d - y
    norm_sq = sigma.dot(sigma) + eps
    step = mu * modulation * e * sigma / norm_sq
    return w + step


def ternary_decision(y: float, delta: float = 1e-3) -> int:
    """
    Map a scalar prediction to a ternary action.

    Returns
    -------
    -1  if y < -δ
     0  if |y| ≤ δ
    +1  if y >  δ
    """
    if y > delta:
        return 1
    if y < -delta:
        return -1
    return 0


def softmax_log_update(log_p: np.ndarray, reward: float, tau: float) -> np.ndarray:
    """
    Perform an exponential‑weight (bandit) update in log‑space.

    Parameters
    ----------
    log_p : np.ndarray, shape (K,)
        Log‑probabilities of actions (K = 3 for ternary).
    reward: float
        Scalar reward (positive encourages the selected action).
    tau   : float
        Temperature controlling the magnitude of the update.

    Returns
    -------
    new_log_p : np.ndarray, shape (K,)
        Updated log‑probabilities (still unnormalised).
    """
    # Add reward scaled by temperature to the log‑probability of the taken action
    # The caller must have already added the reward to the correct index.
    # Here we simply shift all entries by reward/tau for numerical stability.
    return log_p + reward / tau


def schoolfield_temperature(
    T: float,
    T_opt: float = 25.0,
    H: float = 5000.0,
    dH: float = 10000.0,
) -> float:
    """
    Simple Schoolfield‐type temperature scaling.

    rate = exp(-H / (R * (T+273.15))) /
           (1 + exp((dH) / (R * (T+273.15))))

    For our purposes we return the denominator term as a temperature scaling
    factor (higher temperature ⇒ slower bandit learning).

    Parameters
    ----------
    T      : float
        Ambient temperature in °C.
    T_opt  : float, unused placeholder for optimal temperature.
    H, dH  : float
        Enthalpy constants.

    Returns
    -------
    temperature : float
        Positive scaling factor.
    """
    R = 8.314  # J·mol⁻¹·K⁻¹
    K = math.exp(-H / (R * (T + 273.15)))
    denominator = 1.0 + math.exp(dH / (R * (T + 273.15)))
    return K / denominator + 1e-3  # avoid zero


# ----------------------------------------------------------------------
# Core hybrid step
# ----------------------------------------------------------------------


def hybrid_nlms_ternary_bandit_step(
    w: np.ndarray,
    log_p: np.ndarray,
    sigma: np.ndarray,
    mu: float = 0.1,
    eps: float = 1e-6,
    temperature: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Perform one hybrid adaptation step for a single signature vector.

    The step consists of:
      1. NLMS update of `w` (modulated by ternary decision).
      2. Ternary routing decision based on the updated `w`.
      3. Bandit probability update using a reward derived from the decision.

    Parameters
    ----------
    w          : np.ndarray, shape (M,)
        Current global weight vector.
    log_p      : np.ndarray, shape (3,)
        Log‑probabilities for actions (‑1, 0, +1).
    sigma      : np.ndarray, shape (M,)
        Signature vector of the current sample.
    mu         : float
        Base NLMS learning rate.
    eps        : float
        Regularisation for NLMS.
    temperature: float
        Current temperature from the Schoolfield model.

    Returns
    -------
    w_new      : np.ndarray, shape (M,)
        Updated weight vector.
    log_p_new  : np.ndarray, shape (3,)
        Updated log‑probabilities (still unnormalised).
    action     : int
        Ternary action taken (‑1, 0, +1).
    """
    # Desired output is the ℓ₂ norm of the signature (mirrors Parent A)
    d = float(np.linalg.norm(sigma))

    # Linear prediction and ternary decision (Parent B)
    y = w.dot(sigma)
    action = ternary_decision(y)

    # Modulate NLMS step with the ternary action (acts as a sign)
    modulation = float(action) if action != 0 else 1.0
    w_new = nlms_update(w, sigma, d, mu=mu, eps=eps, modulation=modulation)

    # Simple reward: +1 if action matches sign of (d - y), else -1
    reward = 1.0 if (d - y) * action > 0 else -1.0

    # Bandit exponential‑weight update in log‑space
    # Map action (‑1,0,1) → index (0,1,2)
    idx = action + 1
    log_p_updated = log_p.copy()
    log_p_updated[idx] += reward / temperature
    # Renormalise for numerical stability (subtract max)
    log_p_new = log_p_updated - np.max(log_p_updated)

    return w_new, log_p_new, action


# ----------------------------------------------------------------------
# Demonstration pipeline
# ----------------------------------------------------------------------


def run_hybrid_pipeline(
    master_vectors: np.ndarray,
    num_iterations: int = 5,
    mu: float = 0.05,
    init_temp: float = 20.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Execute the full hybrid algorithm on a collection of master vectors.

    The pipeline:
      * Compute a single signature for the whole sequence (Parent A).
      * Initialise a global weight vector `w` and ternary bandit log‑probs.
      * Iterate `num_iterations` times, each time applying the hybrid step.

    Parameters
    ----------
    master_vectors : np.ndarray, shape (N, D)
        Input master‑vector sequence (e.g., extracted from texts).
    num_iterations : int
        Number of adaptation cycles.
    mu             : float
        Base NLMS learning rate.
    init_temp      : float
        Ambient temperature (°C) fed to the Schoolfield model.

    Returns
    -------
    w_final   : np.ndarray
        Learned weight vector after the last iteration.
    log_p_final : np.ndarray
        Final log‑probabilities of ternary actions.
    """
    # Compute the path‑signature once (could be recomputed per iteration if desired)
    sigma = compute_signature(master_vectors)          # (M,)

    # Initialise weight vector (zero) and uniform log‑probabilities
    w = np.zeros_like(sigma)
    log_p = np.log(np.full(3, 1.0 / 3.0))

    # Temperature evolves with the Schoolfield model (could be dynamic)
    temperature = schoolfield_temperature(init_temp)

    for _ in range(num_iterations):
        w, log_p, _ = hybrid_nlms_ternary_bandit_step(
            w, log_p, sigma, mu=mu, temperature=temperature
        )
        # Optionally update temperature (e.g., slowly cooling)
        temperature *= 0.99  # simple annealing

    return w, log_p


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Create a synthetic master‑vector matrix (N=10, D=4)
    rng = np.random.default_rng(seed=42)
    master_vecs = rng.normal(size=(10, 4))

    w_final, logp_final = run_hybrid_pipeline(master_vecs, num_iterations=10)

    # Display results (no external dependencies)
    print("Final weight vector (first 5 components):", w_final[:5])
    probs = np.exp(logp_final)
    probs /= probs.sum()
    print("Final ternary action probabilities:", probs.tolist())
    sys.exit(0)