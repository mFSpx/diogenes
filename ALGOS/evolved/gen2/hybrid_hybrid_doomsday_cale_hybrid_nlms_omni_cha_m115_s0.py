# DARWIN HAMMER — match 115, survivor 0
# gen: 2
# parent_a: hybrid_doomsday_calendar_gini_coefficient_m49_s4.py (gen1)
# parent_b: hybrid_nlms_omni_chaotic_sprint_m59_s5.py (gen1)
# born: 2026-05-29T23:25:39Z

"""
Module hybrid_fusion: This module represents the fusion of two parent algorithms: 
hybrid_doomsday_calendar_gini_coefficient_m49_s4 and hybrid_nlms_omni_chaotic_sprint_m59_s5.
The mathematical bridge between the two is the use of vectorized operations and matrix manipulations.
The first parent deals with date calculations and Gini coefficient computation, while the second parent 
involves NLMS (Normalized Least Mean Squares) prediction and batch updates. This fusion integrates 
the date-based calculations with the NLMS algorithm to create a novel hybrid system.

Authors: [Your Name]
Date: [Today's Date]
"""

import datetime as dt
import math
import random
import sys
from pathlib import Path
import numpy as np

def weekday_sakamoto(
    years: np.ndarray,
    months: np.ndarray,
    days: np.ndarray,
) -> np.ndarray:
    """
    Compute weekday indices for vectorised (year, month, day) arrays using
    Tomohiko Sakamoto's algorithm.  The result follows the convention
    0 = Sunday, …, 6 = Saturday, which matches the ``(weekday + 1) % 7`` mapping
    used in the original hybrid.
    """
    y = years.astype(np.int64)
    m = months.astype(np.int64)
    d = days.astype(np.int64)

    m_adj = np.where(m < 3, m + 12, m)
    y_adj = np.where(m < 3, y - 1, y)

    t = np.array([0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4], dtype=np.int64)

    w = (y_adj + y_adj // 4 - y_adj // 100 + y_adj // 400 + t[m_adj - 1] + d) % 7
    return w.astype(np.int8)


def gini_coefficient(values: np.ndarray) -> float:
    """
    Return the Gini coefficient of a 1‑D array of non‑negative numbers.
    The implementation follows the definition

        G = Σ_i (2·i – n – 1)·x_i / (n·Σ x_i),

    where ``x_i`` are the values sorted in non‑decreasing order and ``i`` is
    1‑based.  Edge cases (empty array, all zeros) yield ``0.0``.
    """
    if values.ndim != 1:
        raise ValueError("values must be a 1‑D array")
    x = np.sort(values.astype(np.float64))
    if x.size == 0 or np.isclose(x.sum(), 0.0):
        return 0.0
    if np.any(x < 0):
        raise ValueError("values must be non‑negative")
    n = x.size
    i = np.arange(1, n + 1, dtype=np.float64)
    numerator = np.sum((2 * i - n - 1) * x)
    denominator = n * x.sum()
    return float(numerator / denominator)


def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Return the dot‑product prediction w·x."""
    return float(weights @ x)


def nlms_batch_update(
    weights: np.ndarray,
    X: np.ndarray,
    targets: np.ndarray,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform a *batch* NLMS update.

    Parameters
    ----------
    weights : np.ndarray
        Current weight vector (shape: (d,)).
    X : np.ndarray
        Input matrix where each row is an input vector x_i (shape: (N, d)).
    targets : np.ndarray
        Desired scalar outputs v_i (shape: (N,)).
    mu : float
        Learning rate in (0, 2).
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    new_weights : np.ndarray
        Updated weight vector.
    errors : np.ndarray
        Prediction errors for each sample.
    """
    if not (0.0 < mu < 2.0):
        raise ValueError("mu must be in the interval (0, 2)")

    preds = X @ weights
    errors = targets - preds

    powers = np.sum(X * X, axis=1) + eps  
    steps = (mu * errors / powers)[:, None] * X  

    delta_w = steps.sum(axis=0)
    new_weights = weights + delta_w
    return new_weights, errors


def hybrid_update(
    weights: np.ndarray,
    X: np.ndarray,
    targets: np.ndarray,
    dates: np.ndarray,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform a hybrid update that incorporates the weekday calculation into the NLMS update.

    Parameters
    ----------
    weights : np.ndarray
        Current weight vector (shape: (d,)).
    X : np.ndarray
        Input matrix where each row is an input vector x_i (shape: (N, d)).
    targets : np.ndarray
        Desired scalar outputs v_i (shape: (N,)).
    dates : np.ndarray
        Array of dates (shape: (N, 3)) where each row is a date in the format (year, month, day).
    mu : float
        Learning rate in (0, 2).
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    new_weights : np.ndarray
        Updated weight vector.
    errors : np.ndarray
        Prediction errors for each sample.
    """
    weekdays = weekday_sakamoto(dates[:, 0], dates[:, 1], dates[:, 2])
    X_with_weekday = np.hstack((X, weekdays[:, None]))
    return nlms_batch_update(weights, X_with_weekday, targets, mu, eps)


def gini_nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    """
    Return the dot‑product prediction w·x and the Gini coefficient of the prediction.

    Parameters
    ----------
    weights : np.ndarray
        Current weight vector (shape: (d,)).
    x : np.ndarray
        Input vector (shape: (d,)).

    Returns
    -------
    prediction : float
        The dot‑product prediction w·x.
    gini : float
        The Gini coefficient of the prediction.
    """
    prediction = nlms_predict(weights, x)
    gini = gini_coefficient(np.array([prediction]))
    return prediction, gini


if __name__ == "__main__":
    # Smoke test
    weights = np.array([1.0, 2.0, 3.0])
    X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    targets = np.array([10.0, 20.0])
    dates = np.array([[2022, 1, 1], [2022, 1, 2]])

    new_weights, errors = hybrid_update(weights, X, targets, dates)
    print(new_weights)
    print(errors)

    prediction, gini = gini_nlms_predict(weights, np.array([1.0, 2.0, 3.0]))
    print(prediction)
    print(gini)