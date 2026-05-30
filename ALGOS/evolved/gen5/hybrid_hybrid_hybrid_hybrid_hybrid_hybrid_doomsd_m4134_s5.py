# DARWIN HAMMER — match 4134, survivor 5
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_rbf_su_m1969_s2.py (gen4)
# parent_b: hybrid_hybrid_doomsday_cale_hybrid_fold_change_d_m1433_s5.py (gen4)
# born: 2026-05-29T23:53:41Z

"""Hybrid RBF‑TTT‑FCD Bandit Module
Parents:
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_rbf_su_m1969_s2.py (RBF surrogate + TTT‑Linear)
- hybrid_hybrid_doomsday_cale_hybrid_fold_change_d_m1433_s5.py (Fold‑Change Detection modulating NLMS & bandit)

Mathematical Bridge:
The TTT‑Linear core maintains a weight matrix **W** that maps an input vector **x**
to a prediction **p = W·x**.  Instead of a plain squared error, we evaluate the
prediction quality with a Gaussian radial‑basis function  
ϕ(r)=exp(−(ε·r)²) where *r* is the Euclidean distance between **p** and the target **t**.
The resulting similarity score is turned into a loss L=1−ϕ(r).  

The Fold‑Change Detection (FCD) subsystem produces a slow component *yₜ*.
Its time‑average ⟨y⟩ modulates two independent mechanisms:

1. **Learning‑rate modulation** – the Normalised Least‑Mean‑Squares (NLMS) step
   size μ̂ = μ·(1+⟨y⟩) scales the gradient of the RBF‑loss, thereby coupling the
   free‑energy penalty of the TTT‑Linear surrogate with external dynamics.

2. **Bandit propensity scaling** – action propensities **π** are multiplied by
   exp(⟨y⟩) before the usual reward‑based update, letting the temporal context
   steer exploration.

The unified algorithm therefore:
* computes an RBF‑based loss,
* updates **W** with an NLMS rule whose step size is FCD‑modulated,
* updates a multi‑armed bandit policy whose preferences are also FCD‑scaled.

The three public functions below illustrate the hybrid operation:
`rbf_similarity`, `modulated_nlms_update`, and `fcd_bandit_step`. """

import math
import random
import sys
from pathlib import Path
from typing import Sequence, List, Tuple, Dict

import numpy as np
from datetime import date

Vector = Sequence[float]


def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian radial‑basis function."""
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: Vector, b: Vector) -> float:
    """Euclidean distance between two equal‑length vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def rbf_similarity(pred: np.ndarray, target: np.ndarray, epsilon: float = 1.0) -> float:
    """
    Returns a similarity score in (0,1] based on a Gaussian RBF.
    1 means perfect match, values →0 as distance grows.
    """
    dist = euclidean(pred, target)
    return gaussian(dist, epsilon)


def solve_linear(a: List[List[float]], b: List[float]) -> List[float]:
    """Gaussian elimination with partial pivoting."""
    n = len(b)
    m = [row[:] + [rhs] for row, rhs in zip(a, b)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            raise ValueError("singular surrogate system")
        m[col], m[pivot] = m[pivot], m[col]
        div = m[col][col]
        m[col] = [v / div for v in m[col]]
        for row in range(n):
            if row == col:
                continue
            factor = m[row][col]
            m[row] = [v - factor * p for v, p in zip(m[row], m[col])]
    return [row[-1] for row in m]


def init_ttt(d_in: int, d_out: int = None, scale: float = 0.01, seed: int = 0) -> np.ndarray:
    """Initialize weight matrix W for the TTT‑Linear core."""
    rng = np.random.default_rng(seed)
    if d_out is None:
        d_out = d_in
    return rng.standard_normal((d_out, d_in)) * scale


def nlms_step(
    W: np.ndarray,
    x: np.ndarray,
    grad: np.ndarray,
    mu: float,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Normalised LMS update:
        W ← W - μ * grad / (‖x‖² + ε)
    """
    norm_sq = float(np.dot(x, x)) + eps
    return W - mu * grad / norm_sq


def modulated_nlms_update(
    W: np.ndarray,
    x: np.ndarray,
    target: np.ndarray,
    base_mu: float,
    avg_y: float,
    epsilon: float = 1.0,
) -> Tuple[np.ndarray, float]:
    """
    Perform one NLMS update of the TTT‑Linear weight matrix using an RBF‑derived loss.
    The step size is modulated by the Fold‑Change Detection average ⟨y⟩:
        μ̂ = base_mu * (1 + avg_y)

    Returns the updated weight matrix and the scalar loss (1‑ϕ).
    """
    # forward pass
    pred = W @ x
    # RBF similarity and loss
    sim = rbf_similarity(pred, target, epsilon)
    loss = 1.0 - sim

    # gradient of loss w.r.t. prediction:
    # dL/dp = -(dϕ/dr) * (p - t)/r   with ϕ = exp(-(ε r)^2)
    # dϕ/dr = -2 ε^2 r ϕ
    # => dL/dp = 2 ε^2 ϕ (p - t)
    r = euclidean(pred, target) + 1e-12  # avoid zero
    phi = gaussian(r, epsilon)
    grad_pred = 2 * (epsilon ** 2) * phi * (pred - target)

    # chain rule to weight matrix (W appears linearly)
    grad_W = np.outer(grad_pred, x)  # shape (d_out, d_in)

    mu_hat = base_mu * (1.0 + avg_y)
    W_new = nlms_step(W, x, grad_W, mu_hat)

    return W_new, loss


def generate_fcd_series(length: int = 50, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate a Fold‑Change Detection (FCD) signal.
    Returns two series (x_t, y_t) where y_t is a low‑pass filtered version of x_t.
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(length)
    y = np.zeros_like(x)
    alpha = 0.1  # smoothing factor
    for t in range(length):
        y[t] = (1 - alpha) * (y[t - 1] if t > 0 else 0.0) + alpha * x[t]
    return x, y


def fcd_average(y: np.ndarray, horizon: int = 10) -> float:
    """
    Compute the average of the slow component y over the last `horizon` samples.
    """
    if len(y) == 0:
        return 0.0
    return float(np.mean(y[-horizon:]))


class FCDBandit:
    """
    Multi‑armed bandit whose action propensities are scaled by exp(⟨y⟩).
    Uses a simple epsilon‑greedy policy with reward updates.
    """

    def __init__(self, n_actions: int, epsilon: float = 0.1, seed: int = 0):
        self.n = n_actions
        self.epsilon = epsilon
        rng = np.random.default_rng(seed)
        self.preferences = rng.standard_normal(n_actions)  # raw scores
        self.counts = np.zeros(n_actions, dtype=int)

    def select_action(self, avg_y: float) -> int:
        """Return an action index, applying the FCD scaling."""
        scaled = self.preferences * math.exp(avg_y)
        probs = np.exp(scaled - np.max(scaled))  # softmax for stability
        probs /= probs.sum()
        if random.random() < self.epsilon:
            return random.randrange(self.n)
        return int(np.argmax(probs))

    def update(self, action: int, reward: float, avg_y: float) -> None:
        """Incremental update of preferences using a REINFORCE‑like rule."""
        self.counts[action] += 1
        lr = 1.0 / (self.counts[action] + 1.0)  # diminishing step size
        # Scale reward by the same exp(avg_y) factor used in selection
        scaled_reward = reward * math.exp(avg_y)
        self.preferences[action] += lr * (scaled_reward - self.preferences[action])


def date_features(year: int, month: int, day: int) -> np.ndarray:
    """
    Normalized feature vector for a calendar date.
    Features (order):
        0 : year scaled to [0,1] over [1900,2100]
        1 : sin(2π*month/12)
        2 : cos(2π*month/12)
        3 : sin(2π*day/31)
        4 : cos(2π*day/31)
        5 : constant bias term (1.0)
    """
    yr_min, yr_max = 1900.0, 2100.0
    yr_scaled = (year - yr_min) / (yr_max - yr_min)
    month_rad = 2 * math.pi * month / 12.0
    day_rad = 2 * math.pi * day / 31.0
    return np.array(
        [
            yr_scaled,
            math.sin(month_rad),
            math.cos(month_rad),
            math.sin(day_rad),
            math.cos(day_rad),
            1.0,
        ],
        dtype=float,
    )


def doomsday_weekday(year: int, month: int, day: int) -> int:
    """Return weekday index where 0=Sunday … 6=Saturday."""
    return (date(year, month, day).weekday() + 1) % 7


def hybrid_step(
    W: np.ndarray,
    x: np.ndarray,
    target: np.ndarray,
    base_mu: float,
    avg_y: float,
    epsilon: float = 1.0,
) -> Tuple[np.ndarray, float]:
    """
    One unified hybrid iteration:
    * Update the TTT‑Linear weights with an RBF‑derived loss, NLMS‑modulated by ⟨y⟩.
    * Return the new weights and the scalar loss.
    """
    return modulated_nlms_update(W, x, target, base_mu, avg_y, epsilon)


def demo_hybrid():
    """Simple smoke‑test demonstrating the fused algorithm."""
    # 1. Initialise TTT‑Linear model (predict weekday from date features)
    W = init_ttt(d_in=6, d_out=7, scale=0.05, seed=1)  # 7 output logits (one per weekday)

    # 2. Simulate a Fold‑Change Detection signal
    _, y_series = generate_fcd_series(length=30, seed=7)
    avg_y = fcd_average(y_series, horizon=10)

    # 3. Initialise bandit with 5 actions
    bandit = FCDBandit(n_actions=5, epsilon=0.2, seed=3)

    # 4. Run a few hybrid steps on random dates
    total_loss = 0.0
    for _ in range(12):
        # random date in range 1950‑2000
        yr = random.randint(1950, 2000)
        mo = random.randint(1, 12)
        dy = random.randint(1, 28)  # keep simple
        x = date_features(yr, mo, dy)
        target_idx = doomsday_weekday(yr, mo, dy)
        target = np.zeros(7)
        target[target_idx] = 1.0  # one‑hot target vector

        W, loss = hybrid_step(W, x, target, base_mu=0.1, avg_y=avg_y, epsilon=1.0)
        total_loss += loss

        # Bandit interaction using the same avg_y
        action = bandit.select_action(avg_y)
        reward = 1.0 if action == target_idx % bandit.n else 0.0
        bandit.update(action, reward, avg_y)

    print(f"Average hybrid loss over steps: {total_loss / 12:.4f}")
    print(f"Bandit preferences after training: {bandit.preferences}")


if __name__ == "__main__":
    demo_hybrid()