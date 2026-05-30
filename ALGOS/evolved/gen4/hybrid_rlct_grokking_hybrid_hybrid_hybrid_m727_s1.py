# DARWIN HAMMER — match 727, survivor 1
# gen: 4
# parent_a: rlct_grokking.py (gen0)
# parent_b: hybrid_hybrid_hybrid_model__hybrid_hard_truth_ma_m23_s3.py (gen3)
# born: 2026-05-29T23:30:38Z

"""
Hybrid RLCT‑Grokking / Stylometry‑Fold‑Change Algorithm
====================================================

Parents
-------
* **rlct_grokking.py** – Provides the Real Log Canonical Threshold (RLCT) theory,
  free‑energy asymptotics and the notion of a *grokking* transition in singular
  learning models.

* **hybrid_hybrid_hybrid_model__hybrid_hard_truth_ma_m23_s3.py** – Defines a
  recurrent weight‑matrix dynamics  

      dW/dt = -α·W + β·dX/dt  

  and a stylometry feature extractor (LSM) that turns a text corpus into a
  numeric vector.

Mathematical Bridge
-------------------
The bridge is the **stylometry vector** `X`.  In the original fold‑change model it
appears only through its time derivative `dX/dt`.  Here we treat `X` as the
output of the stylometry extractor and feed it directly into the weight‑matrix
update.  Conversely, the *singular* geometry of the weight matrix (its rank
deficiency) determines the RLCT `λ`.  We let `λ` modulate the decay rate `α`,
so that a more singular weight matrix (smaller `λ`) yields slower decay – a
mechanism that mirrors the “collapse” of a complex memorisation manifold into a
simple generalising one during grokking.

The resulting hybrid system therefore intertwines:

1. **Stylometry → Fold‑change dynamics**  
   `X = stylometry(corpus)` → `dX/dt` drives `W`.

2. **Weight‑matrix singularity → RLCT**  
   `λ = estimate_rlct(W, losses)` → modifies `α` in the dynamics.

3. **RLCT & losses → Free‑energy & grokking threshold**  
   Standard singular‑learning formulas are reused to assess when the system
   has “grokked”.

The code below implements this fused mathematics in a self‑contained,
executable module.
"""

import sys
import math
import random
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Stylometry utilities (parent B)
# ----------------------------------------------------------------------
def stylometry_vector(corpus: List[str]) -> np.ndarray:
    """
    Compute a simple stylometry feature vector (LSM) for a list of texts.

    The vector contains normalized frequencies for four word‑categories:
    pronoun, article, preposition, auxiliary.  Frequencies are summed over the
    whole corpus and then divided by the total token count.

    Parameters
    ----------
    corpus : List[str]
        List of raw text strings.

    Returns
    -------
    np.ndarray
        1‑D array of shape (4,) containing the normalized frequencies.
    """
    FUNCTION_CATS: Dict[str, set] = {
        "pronoun": set(
            "i me my mine myself you your yours yourself he him his she her hers "
            "they them their theirs we us our ours".split()
        ),
        "article": set("a an the".split()),
        "preposition": set(
            "about above after against around as at before behind below between by "
            "during for from in into of off on onto over through to under with without".split()
        ),
        "auxiliary": set(
            "am are be been being can could did do does had has have is may might "
            "must shall should was were will would".split()
        ),
    }

    # Tokenise naïvely on whitespace and lower‑case.
    tokens = [tok.lower() for doc in corpus for tok in doc.split()]
    total = len(tokens) if tokens else 1

    freqs = []
    for cat in ["pronoun", "article", "preposition", "auxiliary"]:
        count = sum(tok in FUNCTION_CATS[cat] for tok in tokens)
        freqs.append(count / total)

    return np.array(freqs, dtype=float)


def stylometry_derivative(prev: np.ndarray, cur: np.ndarray, dt: float) -> np.ndarray:
    """
    Discrete approximation of dX/dt = (X(t) - X(t‑dt)) / dt.

    Parameters
    ----------
    prev, cur : np.ndarray
        Stylometry vectors at consecutive time steps.
    dt : float
        Time step size.

    Returns
    -------
    np.ndarray
        Approximate derivative.
    """
    if dt == 0:
        raise ValueError("dt must be non‑zero")
    return (cur - prev) / dt


# ----------------------------------------------------------------------
# RLCT / Grokking utilities (parent A)
# ----------------------------------------------------------------------
def activation_pattern_count(W: np.ndarray) -> int:
    """
    Rough proxy for the number of distinct linear regions a ReLU network can
    represent, based on the sign pattern of the weight matrix rows.

    Parameters
    ----------
    W : np.ndarray
        Weight matrix.

    Returns
    -------
    int
        Number of unique sign patterns across rows.
    """
    signs = np.sign(W)  # -1, 0, +1
    # Convert each row to a tuple so it can be hashed.
    unique_patterns = {tuple(row) for row in signs}
    return len(unique_patterns)


def estimate_rlct(W: np.ndarray, losses: np.ndarray) -> float:
    """
    Estimate the Real Log Canonical Threshold λ from the singular geometry of
    the weight matrix and the empirical loss values.

    The estimate blends two heuristics:
    * Geometry: rank deficiency → smaller λ.
    * Empirical: higher average loss → larger λ.

    The formula is

        λ = 0.5 * (rank(W) / dim(W)) + 0.5 * (mean(loss) / (mean(loss) + var(loss)))

    This yields λ ∈ (0, 0.5] for typical deep nets, respecting the theoretical
    bound λ < d/2 where d is the number of parameters.

    Parameters
    ----------
    W : np.ndarray
        Weight matrix.
    losses : np.ndarray
        Vector of loss values observed during training.

    Returns
    -------
    float
        Estimated RLCT.
    """
    if losses.size == 0:
        raise ValueError("losses array cannot be empty")

    rank = np.linalg.matrix_rank(W)
    dim = W.size
    geom_part = rank / dim  # ∈ (0,1]

    mean_loss = losses.mean()
    var_loss = losses.var()
    emp_part = mean_loss / (mean_loss + var_loss + 1e-12)  # avoid div‑by‑zero

    lambda_est = 0.5 * (geom_part + emp_part)
    return max(min(lambda_est, 0.5), 1e-6)  # clamp to a sensible range


def free_energy_asymptotic(losses: np.ndarray, n_params: int, n_samples: int, lam: float) -> float:
    """
    Compute Watanabe's free‑energy asymptotic expression:

        F_n ≈ n·L_n + λ·log n - (m-1)·log log n

    For simplicity we set the multiplicity m = 1 (common in practice).

    Parameters
    ----------
    losses : np.ndarray
        Per‑sample loss values; L_n is their mean.
    n_params : int
        Number of trainable parameters (size of W).
    n_samples : int
        Dataset size.
    lam : float
        RLCT estimate.

    Returns
    -------
    float
        Approximate free energy.
    """
    if n_samples <= 1:
        raise ValueError("n_samples must be > 1")
    L_n = losses.mean()
    free_energy = n_samples * L_n + lam * math.log(n_samples)  # m‑1 term omitted
    return free_energy


def grokking_threshold(lam: float, base_n: int = 1000) -> int:
    """
    Estimate a critical dataset size n_crit at which grokking is expected.
    Solving λ·log n ≈ n·ΔL where ΔL is an assumed constant loss gap (set to 0.01).

    Parameters
    ----------
    lam : float
        RLCT estimate.
    base_n : int, optional
        Starting point for a simple fixed‑point iteration.

    Returns
    -------
    int
        Approximate critical sample count.
    """
    delta_L = 0.01
    n = base_n
    for _ in range(20):
        n = int(max(2, (lam / delta_L) * math.log(n)))
    return n


# ----------------------------------------------------------------------
# Hybrid dynamics
# ----------------------------------------------------------------------
def fold_change_update(
    W: np.ndarray,
    dX: np.ndarray,
    alpha: float,
    beta: float,
    dt: float,
    lam: float,
) -> np.ndarray:
    """
    Discrete integration of the fold‑change differential equation with an
    RLCT‑adjusted decay term:

        dW/dt = -α·(1+λ)·W + β·dX/dt

    The factor (1+λ) slows decay when the model is highly singular (small λ).

    Parameters
    ----------
    W : np.ndarray
        Current weight matrix.
    dX : np.ndarray
        Derivative of the stylometry vector (shape must broadcast to W).
    alpha, beta : float
        Hyperparameters controlling decay and fold‑change strength.
    dt : float
        Time step.
    lam : float
        RLCT estimate.

    Returns
    -------
    np.ndarray
        Updated weight matrix.
    """
    decay = -alpha * (1.0 + lam) * W
    # Broadcast dX to the shape of W (e.g., outer product)
    fold_change = beta * np.outer(dX, np.ones(W.shape[1]))
    dW = decay + fold_change
    return W + dW * dt


def hybrid_step(
    W: np.ndarray,
    prev_X: np.ndarray,
    cur_X: np.ndarray,
    alpha: float,
    beta: float,
    dt: float,
    losses: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Perform a single hybrid iteration:
    1. Estimate RLCT λ from the current weight matrix and loss history.
    2. Compute dX/dt from stylometry vectors.
    3. Update the weight matrix using the λ‑modulated fold‑change rule.
    4. Return the new matrix and the λ used.

    Parameters
    ----------
    W : np.ndarray
        Current weight matrix.
    prev_X, cur_X : np.ndarray
        Stylometry vectors at times t‑dt and t.
    alpha, beta, dt : float
        Hyperparameters for the dynamics.
    losses : np.ndarray
        Historical per‑sample losses (used for λ estimation).

    Returns
    -------
    Tuple[np.ndarray, float]
        Updated weight matrix and the RLCT estimate employed.
    """
    lam = estimate_rlct(W, losses)
    dX = stylometry_derivative(prev_X, cur_X, dt)
    W_new = fold_change_update(W, dX, alpha, beta, dt, lam)
    return W_new, lam


def run_hybrid_demo() -> None:
    """
    Smoke‑test that exercises the hybrid algorithm on synthetic data.
    """
    # Synthetic corpus (two time steps)
    corpus_t0 = [
        "I am writing a short test document.",
        "You are reading it with attention.",
    ]
    corpus_t1 = [
        "We have learned about grokking and RLCT.",
        "They will apply it to stylometry soon.",
    ]

    # Initialise weight matrix (e.g., 4‑dim stylometry → 6 hidden units)
    np.random.seed(42)
    W = np.random.randn(4, 6) * 0.1

    # Hyperparameters
    alpha = 0.05
    beta = 0.02
    dt = 1.0

    # Simulated loss history (decreasing exponential)
    n_samples = 500
    losses = np.exp(-0.01 * np.arange(100)) + 0.01 * np.random.rand(100)

    # Compute stylometry vectors
    X0 = stylometry_vector(corpus_t0)
    X1 = stylometry_vector(corpus_t1)

    # Perform one hybrid update
    W_new, lam = hybrid_step(W, X0, X1, alpha, beta, dt, losses)

    # Evaluate free energy before and after the update
    F_before = free_energy_asymptotic(losses, W.size, n_samples, estimate_rlct(W, losses))
    F_after = free_energy_asymptotic(losses, W_new.size, n_samples, lam)

    # Grokking threshold estimate
    n_crit = grokking_threshold(lam)

    # Simple sanity prints (no external I/O beyond stdout)
    print("RLCT estimate (λ):", lam)
    print("Free energy before update:", F_before)
    print("Free energy after update :", F_after)
    print("Estimated grokking sample size (n_crit):", n_crit)
    print("Weight matrix norm change:", np.linalg.norm(W_new - W))


if __name__ == "__main__":
    run_hybrid_demo()