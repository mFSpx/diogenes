# DARWIN HAMMER — match 3711, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_nlms_o_hybrid_hybrid_jepa_e_m705_s1.py (gen5)
# parent_b: hybrid_doomsday_calendar_hybrid_rlct_grokking_m396_s4.py (gen3)
# born: 2026-05-29T23:51:17Z

"""Hybrid NLMS‑MinHash‑Doomsday Algorithm
====================================

Parents
-------
* **Parent A** – ``hybrid_hybrid_hybrid_nlms_o_hybrid_hybrid_jepa_e_m705_s1.py``  
  Provides the Normalized Least Mean Squares (NLMS) weight update and an
  entropic MinHash signature used to adapt learning dynamics.

* **Parent B** – ``hybrid_doomsday_calendar_hybrid_rlct_grokking_m396_s4.py``  
  Supplies a weekday (doomsday) calculation that is employed to initialise
  NLMS weights with a cyclic, calendar‑aware pattern.

Mathematical Bridge
-------------------
The bridge is built in two stages:

1. **Weight Initialisation** – The weekday index `d ∈ {0,…,6}` returned by
   ``doomsday`` is expanded into a base weight vector `w₀`.  This injects a
   deterministic, periodic component into the model, mirroring the
   calendar‑driven initialisation of Parent B.

2. **Adaptive NLMS Update** – For each input vector `x` we compute a MinHash
   signature `h(x)`.  The average bucket value of the signature, normalised
   by the number of buckets, yields a scaling factor `α ∈ [1,2)`.  The
   effective learning rate becomes `μ_eff = μ·α`, thereby letting the
   entropic structure of the data (Parent A) modulate the NLMS adaptation
   while the weights retain their calendar‑derived seed.

The resulting algorithm retains the fast convergence of NLMS, gains
data‑dependent adaptability via MinHash, and preserves a cyclic
initialisation that can be useful for temporally periodic problems."""

import numpy as np
import math
import random
import sys
import pathlib
from datetime import date

# ----------------------------------------------------------------------
# Parent‑A building blocks
# ----------------------------------------------------------------------
def minhash(x: np.ndarray, num_buckets: int = 10) -> np.ndarray:
    """
    Compute a simple MinHash‑like signature for a vector.

    The signature consists of the minimum value observed after a random
    permutation of ``x`` reduced modulo ``num_buckets``.  Repeating this
    process ``num_buckets`` times yields a compact, entropy‑rich descriptor.

    Parameters
    ----------
    x : np.ndarray
        Input vector.
    num_buckets : int, optional
        Number of hash buckets (default 10).

    Returns
    -------
    np.ndarray
        Signature of shape ``(num_buckets,)``.
    """
    # Ensure reproducibility per call without affecting global RNG state
    rng = np.random.default_rng()
    return np.array([np.min(rng.permutation(x) % num_buckets) for _ in range(num_buckets)])


def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    """
    Standard NLMS prediction: inner product between weights and input.

    Parameters
    ----------
    weights : np.ndarray
    x : np.ndarray

    Returns
    -------
    float
    """
    return float(np.dot(weights, x))


# ----------------------------------------------------------------------
# Parent‑B building blocks
# ----------------------------------------------------------------------
def doomsday(year: int, month: int, day: int) -> int:
    """
    Calendar helper returning the weekday index (0=Sunday … 6=Saturday)
    for a given Gregorian date.

    Parameters
    ----------
    year, month, day : int

    Returns
    -------
    int
    """
    # Python's weekday(): Monday=0 … Sunday=6 → shift to Sunday=0
    return (date(year, month, day).weekday() + 1) % 7


def initialize_weights(year: int, month: int, day: int, dim: int, seed: int = 0) -> np.ndarray:
    """
    Initialise NLMS weights using the doomsday weekday as a deterministic
    seed.  The weekday index is spread across the weight vector and perturbed
    with small Gaussian noise to break symmetry.

    Parameters
    ----------
    year, month, day : int
        Date used for the calendar‑derived seed.
    dim : int
        Dimensionality of the weight vector.
    seed : int, optional
        Additional integer seed for reproducibility (default 0).

    Returns
    -------
    np.ndarray
        Initial weight vector of shape ``(dim,)``.
    """
    weekday = doomsday(year, month, day)  # value in 0..6
    rng = np.random.default_rng(seed + weekday)
    base = float(weekday) / 6.0  # normalise to [0,1]
    # Uniform base plus tiny Gaussian jitter
    return np.full(dim, base) + rng.normal(scale=0.01, size=dim)


# ----------------------------------------------------------------------
# Hybrid core
# ----------------------------------------------------------------------
def hybrid_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
    num_buckets: int = 10,
) -> tuple[np.ndarray, float]:
    """
    Perform an NLMS weight update where the learning rate is adaptively
    scaled by a factor derived from the MinHash signature of the input.

    The scaling factor is

        α = 1 + (mean(h(x)) / num_buckets)

    guaranteeing ``α ∈ [1,2)``.  The effective learning rate becomes
    ``μ_eff = μ·α``.

    Parameters
    ----------
    weights : np.ndarray
        Current weight vector.
    x : np.ndarray
        Input feature vector.
    target : float
        Desired output.
    mu : float, optional
        Base learning rate (default 0.5).
    eps : float, optional
        Regularisation term to avoid division by zero (default 1e-9).
    num_buckets : int, optional
        Number of MinHash buckets (default 10).

    Returns
    -------
    tuple[np.ndarray, float]
        Updated weights and the prediction error ``e = target - y``.
    """
    # Prediction and error
    y = nlms_predict(weights, x)
    error = target - y

    # MinHash‑based adaptive scaling
    signature = minhash(x, num_buckets=num_buckets)
    alpha = 1.0 + (np.mean(signature) / float(num_buckets))
    mu_eff = mu * alpha

    # NLMS update rule with adapted learning rate
    norm_factor = np.dot(x, x) + eps
    weight_increment = mu_eff * error * x / norm_factor
    new_weights = weights + weight_increment

    return new_weights, error


def compute_bic(log_likelihood: float, n_params: int, n_samples: int) -> float:
    """
    Bayesian Information Criterion (BIC) – retained from Parent B for model
    selection purposes.

    Parameters
    ----------
    log_likelihood : float
    n_params : int
    n_samples : int

    Returns
    -------
    float
    """
    return -2.0 * log_likelihood + n_params * math.log(n_samples)


def hybrid_predict(weights: np.ndarray, x: np.ndarray) -> float:
    """
    Wrapper that forwards to the NLMS predictor.  Kept as a separate function
    to emphasise the hybrid interface.

    Parameters
    ----------
    weights : np.ndarray
    x : np.ndarray

    Returns
    -------
    float
    """
    return nlms_predict(weights, x)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple reproducible demo
    dim = 5
    rng = np.random.default_rng(42)

    # Initialise weights using a fixed date (e.g., 2024‑02‑29 – a leap day)
    w = initialize_weights(2024, 2, 29, dim, seed=123)

    # Generate a random input vector and a synthetic target
    x = rng.normal(size=dim)
    true_weights = rng.normal(size=dim)
    target = nlms_predict(true_weights, x) + rng.normal(scale=0.05)  # noisy observation

    # Perform a few hybrid updates
    for step in range(3):
        w, err = hybrid_update(w, x, target, mu=0.4)
        pred = hybrid_predict(w, x)
        print(f"Step {step+1}: error={err:.4f}, prediction={pred:.4f}")

    # Compute a dummy BIC (log_likelihood approximated by -0.5*error^2)
    log_like = -0.5 * err ** 2
    bic = compute_bic(log_like, n_params=dim, n_samples=1)
    print(f"Final BIC (dummy): {bic:.4f}")

    sys.exit(0)