# DARWIN HAMMER — match 396, survivor 6
# gen: 3
# parent_a: doomsday_calendar.py (gen0)
# parent_b: hybrid_rlct_grokking_hybrid_nlms_omni_cha_m118_s1.py (gen2)
# born: 2026-05-29T23:28:45Z

"""Hybrid Doomsday-NLMS Module

Parents:
- doomsday_calendar.py (weekday calculator)
- hybrid_rlct_grokking_nlms_omni_chaotic.py (RLCT‑adjusted NLMS)

Mathematical bridge:
We treat the weekday (0‑6) as a regression target and train an
NLMS adaptive filter to map a numeric date feature vector to that target.
The learning rate μ of NLMS is modulated by a lightweight RLCT‑inspired
free‑energy term:
    μ̂ = μ₀ / (1 + log(1 + ||w||₂))
where w are the current weights. This couples the RLCT concept (complexity
penalty via weight norm) to the NLMS update, yielding a unified predictor
for the Doomsday calendar. """

from __future__ import annotations
import datetime as dt
import math
import random
import sys
from pathlib import Path
import numpy as np

# ---------- Parent A ----------
def doomsday(year: int, month: int, day: int) -> int:
    """Return weekday index where 0=Sunday … 6=Saturday."""
    return (dt.date(year, month, day).weekday() + 1) % 7

# ---------- Parent B (NLMS core) ----------
def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Linear NLMS prediction."""
    return float(weights @ x)

def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> tuple[np.ndarray, float]:
    """Perform one NLMS weight update and return new weights and error."""
    y = nlms_predict(weights, x)
    e = target - y
    norm_x = float(x @ x) + eps
    delta = (mu / norm_x) * e * x
    new_weights = weights + delta
    return new_weights, e

# ---------- Hybrid Extensions ----------
def rlct_adjusted_mu(weights: np.ndarray, base_mu: float = 0.5) -> float:
    """
    RLCT‑inspired adjustment of the NLMS learning rate.

    μ̂ = base_mu / (1 + log(1 + ||w||₂))

    The logarithmic term penalises large weight norms, mimicking the
    free‑energy complexity penalty of the Real Log Canonical Threshold.
    """
    norm = np.linalg.norm(weights)
    return base_mu / (1.0 + math.log1p(norm))

def date_features(year: int, month: int, day: int) -> np.ndarray:
    """
    Convert a calendar date to a normalized feature vector.

    Features:
    - year scaled to [0,1] over a reasonable window (1900‑2100)
    - month as sin/cos pair (captures cyclic nature)
    - day as sin/cos pair (captures cyclic nature)
    - bias term (1.0)

    Returns a column‑vector (shape (5,)).
    """
    # Normalise year
    yr_min, yr_max = 1900, 2100
    yr_norm = (year - yr_min) / (yr_max - yr_min)

    # Encode month cyclically
    month_angle = 2.0 * math.pi * (month - 1) / 12.0
    month_sin = math.sin(month_angle)
    month_cos = math.cos(month_angle)

    # Encode day cyclically (max 31)
    day_angle = 2.0 * math.pi * (day - 1) / 31.0
    day_sin = math.sin(day_angle)
    day_cos = math.cos(day_angle)

    # Assemble feature vector (bias at the end)
    return np.array([yr_norm, month_sin, month_cos, day_sin, day_cos, 1.0])

def hybrid_train(
    dates: list[tuple[int, int, int]],
    epochs: int = 5,
    base_mu: float = 0.5,
) -> np.ndarray:
    """
    Train NLMS weights to predict weekday from dates using RLCT‑adjusted μ.

    Parameters
    ----------
    dates : list of (year, month, day) tuples used as training samples.
    epochs : number of passes over the data.
    base_mu : base learning rate before RLCT adjustment.

    Returns
    -------
    np.ndarray
        Learned weight vector.
    """
    # Initialise small random weights (including bias)
    rng = np.random.default_rng(42)
    w = rng.normal(scale=0.01, size=6)

    for _ in range(epochs):
        random.shuffle(dates)
        for y, m, d in dates:
            x = date_features(y, m, d)
            target = float(doomsday(y, m, d))  # regression target 0‑6
            mu_adj = rlct_adjusted_mu(w, base_mu=base_mu)
            w, _ = nlms_update(w, x, target, mu=mu_adj)

    return w

def hybrid_predict_weekday(weights: np.ndarray, year: int, month: int, day: int) -> int:
    """
    Predict the weekday index for a given date using the trained NLMS weights.
    The raw NLMS output is rounded to the nearest integer and wrapped to [0,6].
    """
    x = date_features(year, month, day)
    raw = nlms_predict(weights, x)
    pred = int(round(raw)) % 7
    return pred

def compute_bic(weights: np.ndarray, data: list[tuple[int, int, int]]) -> float:
    """
    Compute a Bayesian Information Criterion score for the hybrid model.

    Uses squared error as a proxy for negative log‑likelihood.
    """
    n = len(data)
    k = weights.size
    sse = 0.0
    for y, m, d in data:
        x = date_features(y, m, d)
        err = doomsday(y, m, d) - nlms_predict(weights, x)
        sse += err * err
    # Assuming Gaussian errors with variance = sse/n
    if sse == 0:
        sse = 1e-12
    log_likelihood = -0.5 * n * (math.log(2 * math.pi) + math.log(sse / n) + 1)
    return bayesian_information_criterion(log_likelihood, k, n)

def bayesian_information_criterion(log_likelihood: float, n_params: int, n_samples: int) -> float:
    """Standard BIC = -2*LL + n_params*log(n_samples)."""
    return -2.0 * log_likelihood + n_params * math.log(n_samples)

# ---------- Smoke test ----------
if __name__ == "__main__":
    # Generate synthetic training data (random dates)
    train_dates = []
    for _ in range(200):
        yr = random.randint(1900, 2100)
        mo = random.randint(1, 12)
        # Choose a valid day for the month
        if mo == 2:
            max_day = 29 if (yr % 4 == 0 and (yr % 100 != 0 or yr % 400 == 0)) else 28
        elif mo in {4, 6, 9, 11}:
            max_day = 30
        else:
            max_day = 31
        dy = random.randint(1, max_day)
        train_dates.append((yr, mo, dy))

    # Train hybrid model
    learned_w = hybrid_train(train_dates, epochs=10, base_mu=0.6)

    # Test on a handful of unseen dates
    test_dates = [
        (2023, 1, 1),
        (1999, 12, 31),
        (2000, 2, 29),
        (2100, 7, 4),
        (1900, 1, 1),
    ]

    print("Hybrid Doomsday Predictor")
    print("-" * 30)
    for y, m, d in test_dates:
        true_wd = doomsday(y, m, d)
        pred_wd = hybrid_predict_weekday(learned_w, y, m, d)
        print(f"{y:04d}-{m:02d}-{d:02d} -> true: {true_wd}, pred: {pred_wd}")

    # BIC score for curiosity
    bic_score = compute_bic(learned_w, train_dates)
    print(f"\nBIC score on training set: {bic_score:.2f}")

    sys.exit(0)