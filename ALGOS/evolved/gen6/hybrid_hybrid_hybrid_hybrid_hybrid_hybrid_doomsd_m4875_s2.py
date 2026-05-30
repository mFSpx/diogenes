# DARWIN HAMMER — match 4875, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_fold_c_hybrid_ternary_route_m889_s0.py (gen5)
# parent_b: hybrid_hybrid_doomsday_cale_hybrid_bayes_claim_k_m455_s3.py (gen4)
# born: 2026-05-29T23:58:30Z

"""
Hybrid Doomsday‑Bandit Metric
------------------------------

Parents
~~~~~~~
* **Parent A** – `hybrid_hybrid_hybrid_fold_c_hybrid_ternary_route_m889_s0.py`  
  Provides fold‑change detection, pheromone‑infotaxis, and a Shannon‑entropy
  decision‑hygiene metric for bandit‑style action selection.

* **Parent B** – `hybrid_hybrid_doomsday_cale_hybrid_bayes_claim_k_m455_s3.py`  
  Supplies a weekday‑based categorical prior (via the Doomsday calendar),
  a Dirichlet‑multinomial Bayesian update, and a Gini‑coefficient that
  quantifies distributional inequality.

Mathematical Bridge
~~~~~~~~~~~~~~~~~~~
The weekday count vector of a given month is interpreted as a **categorical
prior** (Dirichlet parameters) for the bandit actions.  Observed
weekday frequencies are incorporated through a Dirichlet‑multinomial update,
producing a posterior probability vector *p*.  The **Gini coefficient** of
the posterior, *G(p)*, is used as an **uncertainty‑inflation factor** that
scales the Shannon‑entropy‑based hygiene metric from Parent A.

For each action *i* (weekday index) we compute


fold_i   = log( max( obs_i / eps , 1 ) )                     # fold‑change
logR_i   = log( (obs_i + 1) / (prior_i + 1e-6) )             # log‑count ratio
inf_i    = p_i * logR_i                                      # pheromone infotaxis
ent_i    = -inf_i * log( max(inf_i, 1e-10) )                 # decision‑hygiene entropy
score_i  = ent_i * (1 + G(p)) * fold_i                       # hybrid score


The action with the highest *score_i* is selected.  This formulation fuses
the Bayesian updating and inequality measurement of Parent B with the
fold‑change and entropy machinery of Parent A into a single unified
hybrid system.

The module below implements this hybrid algorithm with three public
functions that demonstrate the combined operation.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import datetime as dt
import numpy as np

# ----------------------------------------------------------------------
# Parent B utilities (weekday distribution, Gini, Bayesian update)
# ----------------------------------------------------------------------


def doomsday(year: int, month: int, day: int) -> int:
    """Return weekday index where Monday=0 … Sunday=6."""
    return (dt.date(year, month, day).weekday() + 1) % 7


def weekday_counts(year: int, month: int) -> np.ndarray:
    """Counts of each weekday (0‑6) for the whole month."""
    # number of days in month
    if month == 12:
        next_month = dt.date(year + 1, 1, 1)
    else:
        next_month = dt.date(year, month + 1, 1)
    first_day = dt.date(year, month, 1)
    ndays = (next_month - first_day).days
    counts = np.zeros(7, dtype=int)
    for d in range(1, ndays + 1):
        wd = doomsday(year, month, d)
        counts[wd] += 1
    return counts


def gini_coefficient(values: np.ndarray) -> float:
    """Gini coefficient for a non‑negative 1‑D array."""
    xs = np.sort(values.astype(float))
    if xs.size == 0 or xs.sum() == 0.0:
        return 0.0
    if xs[0] < 0:
        raise ValueError("values must be non‑negative")
    n = xs.size
    cumulative = np.cumsum(xs)
    return (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n


def dirichlet_posterior(prior: np.ndarray, observations: np.ndarray) -> np.ndarray:
    """
    Dirichlet‑multinomial posterior mean vector.
    prior and observations are shape (7,) arrays of counts.
    Returns posterior probability vector p (sums to 1).
    """
    alpha = prior + observations  # posterior Dirichlet parameters
    return alpha / alpha.sum()


# ----------------------------------------------------------------------
# Parent A utilities (fold‑change, pheromone infotaxis, entropy)
# ----------------------------------------------------------------------


def _fold_change_detection(x: float, eps: float) -> float:
    """Fold‑change detection: log‑ratio with a small floor."""
    return math.log(max(x / eps, 1.0))


def _pheromone_infotaxis(pheromone: float, log_count_ratio: float) -> float:
    """Pheromone infotaxis term."""
    return pheromone * log_count_ratio


def _decision_hygiene_shannon_entropy(pheromone: float, log_count_ratio: float) -> float:
    """Shannon‑entropy‑based hygiene metric."""
    infotaxis = _pheromone_infotaxis(pheromone, log_count_ratio)
    return -infotaxis * math.log(max(infotaxis, 1e-10))


# ----------------------------------------------------------------------
# Hybrid core
# ----------------------------------------------------------------------


def compute_hybrid_prior(year: int, month: int) -> np.ndarray:
    """
    Build Dirichlet prior from the Doomsday weekday counts.
    Laplace smoothing (+1) ensures all categories are >0.
    Returns a (7,) array of prior counts.
    """
    counts = weekday_counts(year, month)
    return counts.astype(float) + 1.0  # smoothing


def generate_random_observations(year: int, month: int, n_samples: int) -> np.ndarray:
    """
    Simulate `n_samples` random weekday observations drawn uniformly from the
    days of the month.  Returns a (7,) count vector.
    """
    counts = weekday_counts(year, month)
    days = []
    for wd, c in enumerate(counts):
        days.extend([wd] * c)
    samples = random.choices(days, k=n_samples)
    obs = np.zeros(7, dtype=int)
    for s in samples:
        obs[s] += 1
    return obs


def hybrid_action_scores(
    year: int,
    month: int,
    observations: np.ndarray,
    eps: float = 1e-3,
) -> Dict[int, float]:
    """
    Compute hybrid scores for each weekday‑action.

    Parameters
    ----------
    year, month : int
        Calendar month defining the prior.
    observations : np.ndarray, shape (7,)
        Observed weekday frequencies.
    eps : float
        Small constant for fold‑change detection.

    Returns
    -------
    dict mapping weekday index (0‑6) → hybrid score (float).
    """
    prior = compute_hybrid_prior(year, month)
    prior_mean = prior / prior.sum()
    posterior = dirichlet_posterior(prior, observations)
    gini = gini_coefficient(posterior)

    scores: Dict[int, float] = {}
    for i in range(7):
        obs_i = float(observations[i])
        prior_i = float(prior[i])

        # Fold‑change detection (Parent A)
        fold_i = _fold_change_detection(obs_i, eps)

        # Log‑count ratio for infotaxis
        logR_i = math.log((obs_i + 1.0) / (prior_i + 1e-6))

        # Pheromone term is the posterior probability for this weekday
        pheromone_i = float(posterior[i])

        # Decision‑hygiene entropy (Parent A)
        ent_i = _decision_hygiene_shannon_entropy(pheromone_i, logR_i)

        # Hybrid score: entropy scaled by (1 + Gini) and by fold‑change
        score_i = ent_i * (1.0 + gini) * fold_i
        scores[i] = score_i
    return scores


def select_best_weekday(scores: Dict[int, float]) -> Tuple[int, float]:
    """
    Return the weekday index with the highest hybrid score and the score itself.
    """
    best_weekday = max(scores, key=scores.get)
    return best_weekday, scores[best_weekday]


def hybrid_step(
    year: int,
    month: int,
    n_observations: int = 30,
    eps: float = 1e-3,
) -> Tuple[int, float, Dict[int, float]]:
    """
    Perform a full hybrid iteration:
    1. Generate random observations.
    2. Compute hybrid scores.
    3. Select the best weekday action.

    Returns
    -------
    best_weekday : int
        Index of the selected weekday (0=Monday … 6=Sunday).
    best_score : float
        Corresponding hybrid score.
    all_scores : dict
        Mapping of all weekday scores.
    """
    observations = generate_random_observations(year, month, n_observations)
    scores = hybrid_action_scores(year, month, observations, eps=eps)
    best_weekday, best_score = select_best_weekday(scores)
    return best_weekday, best_score, scores


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Example: use May 2026, generate 50 observations
    Y, M = 2026, 5
    best_wd, best_sc, all_sc = hybrid_step(Y, M, n_observations=50)

    weekday_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    print(f"Hybrid selection for {Y}-{M:02d}:")
    print(f"  Chosen weekday : {weekday_names[best_wd]} (score={best_sc:.6f})")
    print("  All scores:")
    for wd, sc in sorted(all_sc.items()):
        print(f"    {weekday_names[wd]} : {sc:.6f}")
    sys.exit(0)