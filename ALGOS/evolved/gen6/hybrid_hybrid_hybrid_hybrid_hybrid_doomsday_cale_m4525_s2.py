# DARWIN HAMMER — match 4525, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_distri_hybrid_hybrid_xgboos_m642_s3.py (gen5)
# parent_b: hybrid_doomsday_calendar_hybrid_rlct_grokking_m396_s1.py (gen3)
# born: 2026-05-29T23:56:20Z

"""Hybrid Leader-Tree NLMS with Calendar‑Modulated Entropy Boosting

Parents
-------
* **hybrid_hybrid_hybrid_distri_hybrid_hybrid_xgboos_m642_s3.py** – provides the
  probabilistic acceptance function based on an energy difference ΔE and an
  entropy term derived from MinHash (Jaccard) similarity of token sets.
* **hybrid_doomsday_calendar_hybrid_rlct_grokking_m396_s1.py** – supplies a
  calendar‑driven periodic activation, a Normalised Least‑Mean‑Squares (NLMS)
  predictor/update routine and a Bayesian Information Criterion (BIC) evaluator.

Mathematical Bridge
-------------------
Both parents expose a *scalar modulation factor* that influences a learning
process:

* The acceptance probability `p_acc = exp(-ΔE / (T·(1+S)))` where `S` is the
  entropy‑like term computed from MinHash similarity.
* The periodic activation `a(date) ∈ [0,1]` derived from the day‑of‑week,
  month and year.

In the hybrid system these two factors are multiplied to obtain an
effective NLMS step‑size  


μ_eff = μ_base · p_acc · a(date)


Thus the NLMS weight update simultaneously respects the energy‑based leader
election dynamics, the similarity‑driven entropy regularisation, and the
cyclical calendar rhythm.  The updated weights can then be evaluated with the
standard BIC to assess model quality.

The module implements this fused mathematics together with supporting
utilities (sigmoid, MinHash similarity, etc.)."""

import math
import random
import sys
import pathlib
from datetime import datetime
from collections.abc import Mapping, Hashable

import numpy as np

# ----------------------------------------------------------------------
# Shared primitive types
Node = Hashable
Graph = Mapping[Node, set[Node]]

# ----------------------------------------------------------------------
# Core utilities from parent A
def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    """Numerically stable sigmoid."""
    x_arr = np.asarray(x, dtype=float)
    pos_mask = x_arr >= 0
    neg_mask = ~pos_mask
    out = np.empty_like(x_arr, dtype=float)
    out[pos_mask] = 1.0 / (1.0 + np.exp(-x_arr[pos_mask]))
    exp_x = np.exp(x_arr[neg_mask])
    out[neg_mask] = exp_x / (1.0 + exp_x)
    if np.isscalar(x):
        return float(out)
    return out


def minhash_similarity(tokens_current: set, tokens_ref: set) -> float:
    """
    Exact Jaccard similarity used as a surrogate for MinHash similarity.

    Parameters
    ----------
    tokens_current, tokens_ref : set
        Token collections to compare.

    Returns
    -------
    float
        Jaccard similarity in [0, 1]; 0 if both sets are empty.
    """
    if not tokens_current and not tokens_ref:
        return 0.0
    intersection = tokens_current & tokens_ref
    union = tokens_current | tokens_ref
    return len(intersection) / len(union)


def acceptance_probability(delta_e: float, temperature: float, entropy_term: float) -> float:
    """
    Acceptance probability used in the leader‑tree election.

    If the energy change is favorable (ΔE < 0) the move is always accepted.
    Otherwise a Boltzmann‑like factor tempered by the entropy term is applied.
    """
    if delta_e < 0:
        return 1.0
    # Guard against division by zero
    denom = max(temperature * (1.0 + entropy_term), 1e-12)
    return math.exp(-delta_e / denom)

# ----------------------------------------------------------------------
# Core utilities from parent B
def periodic_activation(date: datetime) -> float:
    """
    Calendar‑driven periodic activation in the range [0, 1).

    The function folds weekday, month and year into a 7‑day cycle.
    """
    weekday = date.weekday() + 1          # 1‑7
    month = date.month                    # 1‑12
    year = date.year
    return ((weekday + month + year) % 7) / 7.0


def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Linear prediction using current NLMS weights."""
    return float(weights @ x)


def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
) -> np.ndarray:
    """
    Standard Normalised Least‑Mean‑Squares weight update.

    Parameters
    ----------
    weights : np.ndarray
        Current weight vector.
    x : np.ndarray
        Input feature vector.
    target : float
        Desired output.
    mu : float, optional
        Base step‑size (default 0.5).

    Returns
    -------
    np.ndarray
        Updated weight vector.
    """
    eps = 1e-12
    prediction = nlms_predict(weights, x)
    error = target - prediction
    norm_sq = np.dot(x, x) + eps
    return weights + (mu / norm_sq) * error * x


def bayesian_information_criterion(log_likelihood: float, n_params: int, n_samples: int) -> float:
    """
    Standard BIC:  -2·LL + k·log(n).

    Lower values indicate a better trade‑off between fit and complexity.
    """
    return -2.0 * log_likelihood + n_params * math.log(n_samples)

# ----------------------------------------------------------------------
# Hybrid operations
def nlms_update_hybrid(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    delta_e: float,
    temperature: float,
    tokens_current: set,
    tokens_ref: set,
    date: datetime,
    mu_base: float = 0.5,
) -> np.ndarray:
    """
    NLMS weight update where the effective learning rate is modulated by:

    * the acceptance probability derived from ΔE, temperature and the
      entropy term (MinHash similarity), and
    * the calendar‑driven periodic activation.

    The resulting step‑size is

        μ_eff = μ_base · p_acc · a(date)

    Parameters
    ----------
    weights : np.ndarray
        Current weight vector.
    x : np.ndarray
        Input feature vector.
    target : float
        Desired output.
    delta_e : float
        Energy difference from the leader‑tree perspective.
    temperature : float
        Temperature parameter for the Boltzmann factor.
    tokens_current, tokens_ref : set
        Token sets used to compute the entropy term.
    date : datetime
        Current calendar date for periodic activation.
    mu_base : float, optional
        Baseline NLMS step‑size.

    Returns
    -------
    np.ndarray
        Updated weight vector.
    """
    # 1. Entropy term via MinHash (Jaccard) similarity
    entropy_term = minhash_similarity(tokens_current, tokens_ref)

    # 2. Acceptance probability from leader‑tree dynamics
    p_acc = acceptance_probability(delta_e, temperature, entropy_term)

    # 3. Calendar‑driven activation
    activation = periodic_activation(date)

    # 4. Effective learning rate
    mu_eff = mu_base * p_acc * activation

    # Guard against pathological zero step‑size
    if mu_eff < 1e-12:
        return weights.copy()

    # 5. Perform NLMS update with the hybrid step‑size
    return nlms_update(weights, x, target, mu=mu_eff)


def evaluate_bic_hybrid(
    predictions: np.ndarray,
    targets: np.ndarray,
    n_params: int,
) -> float:
    """
    Compute BIC assuming independent Gaussian residuals with constant variance.

    The log‑likelihood for Gaussian residuals (σ² unknown) simplifies to
    `-0.5 * n * (log(2π) + 1 + log(RSS/n))`, where RSS is the residual sum of squares.
    """
    n = len(targets)
    residuals = targets - predictions
    rss = np.dot(residuals, residuals)
    # Maximum‑likelihood estimate of variance σ² = RSS / n
    sigma2 = rss / n if n > 0 else 1.0
    # Log‑likelihood (ignoring constants that cancel in BIC differences)
    log_likelihood = -0.5 * n * (math.log(2 * math.pi) + 1 + math.log(sigma2))
    return bayesian_information_criterion(log_likelihood, n_params, n_samples=n)


def hybrid_step(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    delta_e: float,
    temperature: float,
    tokens_current: set,
    tokens_ref: set,
    date: datetime,
    mu_base: float = 0.5,
) -> tuple[np.ndarray, float]:
    """
    Execute a full hybrid iteration: update weights, produce a prediction,
    and return the prediction together with the updated weights.

    This helper showcases the interplay of all three core components
    (entropy‑modulated acceptance, calendar activation, NLMS update).
    """
    new_weights = nlms_update_hybrid(
        weights,
        x,
        target,
        delta_e,
        temperature,
        tokens_current,
        tokens_ref,
        date,
        mu_base=mu_base,
    )
    pred = nlms_predict(new_weights, x)
    return new_weights, pred

# ----------------------------------------------------------------------
# Smoke test
if __name__ == "__main__":
    # Deterministic seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Synthetic data
    dim = 8
    weights = np.random.randn(dim)
    x = np.random.randn(dim)
    target = float(np.dot(np.random.randn(dim), x) + np.random.randn() * 0.1)

    # Leader‑tree parameters
    delta_e = random.uniform(-1.0, 2.0)   # could be negative or positive
    temperature = 0.8

    # Token sets for entropy term
    tokens_current = {"alpha", "beta", "gamma", "delta"}
    tokens_ref = {"beta", "delta", "epsilon"}

    # Calendar date
    today = datetime.now()

    # Perform hybrid update
    updated_weights, prediction = hybrid_step(
        weights,
        x,
        target,
        delta_e,
        temperature,
        tokens_current,
        tokens_ref,
        today,
        mu_base=0.5,
    )

    # Evaluate BIC on a tiny batch (here just one sample for demonstration)
    preds = np.array([prediction])
    targets = np.array([target])
    bic_score = evaluate_bic_hybrid(preds, targets, n_params=dim)

    # Simple sanity prints (allowed in smoke test)
    print("Original weights norm :", np.linalg.norm(weights))
    print("Updated weights norm  :", np.linalg.norm(updated_weights))
    print("Prediction            :", prediction)
    print("Target                :", target)
    print("BIC score             :", bic_score)