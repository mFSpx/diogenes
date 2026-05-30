# DARWIN HAMMER — match 3383, survivor 2
# gen: 4
# parent_a: hybrid_hybrid_hybrid_minimu_hybrid_nlms_omni_cha_m1_s1.py (gen3)
# parent_b: hybrid_hybrid_hybrid_distri_kan_m2190_s0.py (gen3)
# born: 2026-05-29T23:49:53Z

import math
import random
import sys
from pathlib import Path
from typing import Tuple

import numpy as np


# ----------------------------------------------------------------------
# Epistemic certainty handling (Parent A)
# ----------------------------------------------------------------------
EPISTEMIC_FLAGS = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")
_CERTAINTY_FACTORS = {
    "FACT": 1.0,
    "PROBABLE": 0.9,
    "POSSIBLE": 0.7,
    "BULLSHIT": 0.4,
    "SURE_MAYBE": 0.2,
}
_CERTAINTY_DELTAS = {
    "FACT": 0.01,
    "PROBABLE": 0.05,
    "POSSIBLE": 0.10,
    "BULLSHIT": 0.20,
    "SURE_MAYBE": 0.30,
}


def certainty_factor(flag: str) -> float:
    """Scaling factor for the NLMS learning rate."""
    if flag not in EPISTEMIC_FLAGS:
        raise ValueError(f"Unknown epistemic flag {flag!r}")
    return _CERTAINTY_FACTORS[flag]


def certainty_delta(flag: str) -> float:
    """Confidence‑adjusted Hoeffding δ parameter."""
    if flag not in EPISTEMIC_FLAGS:
        raise ValueError(f"Unknown epistemic flag {flag!r}")
    return _CERTAINTY_DELTAS[flag]


# ----------------------------------------------------------------------
# NLMS core (Parent A)
# ----------------------------------------------------------------------
def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Linear prediction w·x."""
    return float(weights @ x)


def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    d: float,
    mu: float,
    flag: str,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """
    Single‑sample NLMS update with epistemic scaling.

    Parameters
    ----------
    weights : np.ndarray
        Current weight vector.
    x : np.ndarray
        Feature vector (1‑D).
    d : float
        Desired output.
    mu : float
        Base learning rate.
    flag : str
        Epistemic certainty flag.
    epsilon : float
        Small constant to avoid division by zero.

    Returns
    -------
    np.ndarray
        Updated weight vector.
    """
    factor = certainty_factor(flag)
    mu_adj = mu * factor
    y_pred = nlms_predict(weights, x)
    error = d - y_pred
    norm = float(x @ x) + epsilon
    weights += (mu_adj * error / norm) * x
    return weights


# ----------------------------------------------------------------------
# Multi‑dimensional KAN transformation (Parent B)
# ----------------------------------------------------------------------
def linear_bspline_basis(x: float, knots: np.ndarray) -> np.ndarray:
    """
    Construct a vector of linear B‑spline basis values for a scalar ``x``.
    For N knots we obtain N‑1 basis functions that are piece‑wise linear
    (hat functions).  The returned vector has length ``len(knots) - 1``.
    """
    if knots.ndim != 1:
        raise ValueError("knots must be a 1‑D array")
    n = len(knots) - 1
    basis = np.zeros(n, dtype=float)

    # Clamp to the outermost interval
    if x <= knots[0]:
        basis[0] = 1.0
        return basis
    if x >= knots[-1]:
        basis[-1] = 1.0
        return basis

    # Locate the interval [k_i, k_{i+1}) containing x
    idx = np.searchsorted(knots, x) - 1
    left, right = knots[idx], knots[idx + 1]
    t = (x - left) / (right - left)

    # Linear hat: rising on the left side, falling on the right side
    if idx > 0:
        basis[idx - 1] = 1.0 - t
    basis[idx] = t
    if idx + 1 < n:
        basis[idx + 1] = 0.0  # already zero, kept for clarity
    return basis


def kan_transform(x: float, knots: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """
    Full KAN transformation: basis → linear combination → non‑linear activation.
    The activation used here is a simple tanh to keep the model differentiable.
    """
    if len(knots) - 1 != len(coeffs):
        raise ValueError("coeffs length must be len(knots) - 1")
    basis = linear_bspline_basis(x, knots)          # (n_basis,)
    linear = float(basis @ coeffs)                  # scalar
    return np.tanh(linear)                          # non‑linear scalar feature


# ----------------------------------------------------------------------
# Hoeffding‑type bound with adaptive range (Parent B)
# ----------------------------------------------------------------------
def hoeffding_epsilon(R: float, delta: float, n: int) -> float:
    """Standard Hoeffding bound."""
    if n <= 0:
        raise ValueError("n must be positive")
    return math.sqrt((R ** 2 * math.log(1.0 / delta)) / (2 * n))


def dynamic_hoeffding_gain_test(
    gain: float,
    n_samples: int,
    *,
    delta: float,
    R: float,
) -> bool:
    """Return True if the observed gain exceeds the Hoeffding epsilon."""
    eps = hoeffding_epsilon(R, delta, n_samples)
    return gain > eps


# ----------------------------------------------------------------------
# Simulated‑annealing style acceptance (Parent B)
# ----------------------------------------------------------------------
def acceptance_probability(gain: float, temperature: float) -> float:
    """
    Logistic‑style acceptance that is numerically stable.
    For temperature → 0 the function tends to a step at gain = 0.
    """
    if temperature <= 0:
        return 1.0 if gain > 0 else 0.0
    # Clip the exponent to avoid overflow
    exponent = min(gain / temperature, 50.0)
    prob = 1.0 / (1.0 + math.exp(-exponent))
    return prob


# ----------------------------------------------------------------------
# Hybrid model – deeper mathematical integration
# ----------------------------------------------------------------------
class HybridNLMSKAN:
    """
    Encapsulates the fused NLMS‑KAN algorithm with:
      * multi‑dimensional spline features,
      * epistemic‑aware learning‑rate and Hoeffding confidence,
      * adaptive gain range for tighter statistical tests,
      * stateful tracking of observed gain statistics.
    """

    def __init__(
        self,
        mu: float = 0.1,
        temperature: float = 1.0,
        init_weight: float = 0.0,
        knots: np.ndarray | None = None,
        coeffs: np.ndarray | None = None,
    ):
        self.mu = mu
        self.temperature = temperature
        self.weights = np.array([init_weight], dtype=float)   # single‑feature NLMS
        # Default spline configuration if none supplied
        if knots is None:
            self.knots = np.linspace(0.0, 10.0, num=6)          # 5 basis functions
        else:
            self.knots = np.asarray(knots, dtype=float)
        if coeffs is None:
            self.coeffs = np.ones(len(self.knots) - 1, dtype=float) * 0.5
        else:
            self.coeffs = np.asarray(coeffs, dtype=float)

        # Statistics for adaptive Hoeffding range
        self._gain_min = math.inf
        self._gain_max = -math.inf
        self._observations = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def step(
        self,
        x_raw: float,
        y_true: float,
        flag: str = "FACT",
    ) -> Tuple[np.ndarray, bool, float]:
        """
        Execute one hybrid iteration and return updated weights,
        split decision, and observed gain.
        """
        # 1️⃣  KAN feature (scalar)
        x_feat_scalar = kan_transform(x_raw, self.knots, self.coeffs)
        x_feat = np.array([x_feat_scalar], dtype=float)   # shape (1,)

        # 2️⃣  Prediction before weight update
        y_pred_before = nlms_predict(self.weights, x_feat)
        prev_error = (y_true - y_pred_before) ** 2

        # 3️⃣  NLMS weight update with epistemic scaling
        self.weights = nlms_update(
            self.weights,
            x_feat,
            y_true,
            self.mu,
            flag=flag,
        )

        # 4️⃣  Prediction after update and gain computation
        y_pred_after = nlms_predict(self.weights, x_feat)
        new_error = (y_true - y_pred_after) ** 2
        gain = max(0.0, prev_error - new_error)

        # 5️⃣  Update adaptive gain statistics
        self._observations += 1
        self._gain_min = min(self._gain_min, gain)
        self._gain_max = max(self._gain_max, gain)

        # 6️⃣  Adaptive Hoeffding test
        #   - R is the observed range of gains (with a tiny epsilon to avoid zero)
        R = max(self._gain_max - self._gain_min, 1e-8)
        delta = certainty_delta(flag)                     # more certain → tighter bound
        split_by_hoeffding = dynamic_hoeffding_gain_test(
            gain,
            self._observations,
            delta=delta,
            R=R,
        )

        # 7️⃣  Probabilistic fallback
        split_by_anneal = random.random() < acceptance_probability(gain, self.temperature)

        split_decision = split_by_hoeffding or split_by_anneal
        return self.weights.copy(), split_decision, gain

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset internal statistics without touching the learned parameters."""
        self._gain_min = math.inf
        self._gain_max = -math.inf
        self._observations = 0


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    model = HybridNLMSKAN(mu=0.05, temperature=0.5)

    def generate_point() -> Tuple[float, float]:
        """Linear relationship with Gaussian noise."""
        x = random.uniform(0.0, 10.0)
        noise = random.gauss(0.0, 0.5)
        y = 2.0 * x + noise
        return x, y

    # Run a few iterations with varying epistemic flags
    flags_cycle = ["FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE"]
    for i in range(200):
        x_raw, y_true = generate_point()
        flag = flags_cycle[i % len(flags_cycle)]
        weights, split, gain = model.step(x_raw, y_true, flag=flag)

        if i % 40 == 0:
            print(
                f"Iter {i:3d} | flag={flag:11s} | weight={weights[0]:.4f} "
                f"| gain={gain:.4f} | split={split}"
            )