# DARWIN HAMMER — match 3383, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_hybrid_minimu_hybrid_nlms_omni_cha_m1_s1.py (gen3)
# parent_b: hybrid_hybrid_hybrid_distri_kan_m2190_s0.py (gen3)
# born: 2026-05-29T23:49:53Z

"""Hybrid NLMS‑KAN with Epistemic Certainty and Hoeffding‑Bound Decision

Parents:
- hybrid_hybrid_hybrid_minimu_hybrid_nlms_omni_cha_m1_s1.py (NLMS + epistemic certainty)
- hybrid_hybrid_hybrid_distri_kan_m2190_s0.py (KAN + Hoeffding bound + probabilistic acceptance)

Mathematical bridge:
1. Raw scalar inputs are passed through a univariate Kolmogorov‑Arnold Network
   (implemented as a piece‑wise linear B‑spline transform).  The transformed
   scalar becomes the single feature vector fed to the Normalized Least Mean
   Squares (NLMS) predictor.
2. The epistemic‑certainty flag associated with each observation rescales the
   NLMS learning rate μ, providing a dynamic adaptation of the gradient step.
3. The reduction in squared prediction error after the NLMS update is treated as
   a “gain’’ that feeds a Hoeffding‑bound test.  If the gain exceeds the bound,
   a split would be statistically justified; otherwise a simulated‑annealing
   acceptance probability (temperature‑driven) decides whether to accept the
   gain anyway.
4. The resulting boolean (split decision) demonstrates the fused leader‑election
   / tree‑splitting logic while the NLMS weights continue to learn in the
   transformed feature space.

The module therefore intertwines the core mathematics of both parents
instead of merely concatenating them.
"""

import math
import random
import sys
import pathlib
import numpy as np

# ----------------------------------------------------------------------
# Epistemic certainty handling (Parent A)
# ----------------------------------------------------------------------
EPISTEMIC_FLAGS = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")
# Mapping from flag to a multiplicative factor for the learning rate μ
_CERTAINTY_FACTORS = {
    "FACT": 1.0,
    "PROBABLE": 0.9,
    "POSSIBLE": 0.7,
    "BULLSHIT": 0.4,
    "SURE_MAYBE": 0.2,
}


def certainty_factor(flag: str) -> float:
    """Return the scaling factor associated with an epistemic flag."""
    if flag not in EPISTEMIC_FLAGS:
        raise ValueError(f"Unknown epistemic flag {flag!r}")
    return _CERTAINTY_FACTORS[flag]


# ----------------------------------------------------------------------
# NLMS core (Parent A)
# ----------------------------------------------------------------------
def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Dot‑product prediction w·x."""
    return float(weights @ x)


def nlms_batch_update(
    weights: np.ndarray,
    X: np.ndarray,
    d: np.ndarray,
    mu: float,
    *,
    epsilon: float = 1e-6,
    flag: str = "FACT",
) -> np.ndarray:
    """
    Perform a batch NLMS update.

    Parameters
    ----------
    weights : np.ndarray
        Current weight vector (shape (n_features,)).
    X : np.ndarray
        Input matrix (shape (n_samples, n_features)).
    d : np.ndarray
        Desired output vector (shape (n_samples,)).
    mu : float
        Base learning rate.
    epsilon : float
        Small constant to avoid division by zero.
    flag : str
        Epistemic certainty flag that rescales μ.

    Returns
    -------
    np.ndarray
        Updated weight vector.
    """
    factor = certainty_factor(flag)
    mu_adj = mu * factor
    for xi, di in zip(X, d):
        y_pred = nlms_predict(weights, xi)
        error = di - y_pred
        norm = float(xi @ xi) + epsilon
        weights += (mu_adj * error / norm) * xi
    return weights


# ----------------------------------------------------------------------
# Simple KAN transformation (Parent B)
# ----------------------------------------------------------------------
def bspline_transform(x: float, knots: np.ndarray, coeffs: np.ndarray) -> float:
    """
    Piece‑wise linear “B‑spline’’ transform.

    Parameters
    ----------
    x : float
        Input scalar.
    knots : np.ndarray
        Monotonically increasing knot positions (length N).
    coeffs : np.ndarray
        Coefficients associated with each knot (length N).

    Returns
    -------
    float
        Transformed value.
    """
    if len(knots) != len(coeffs):
        raise ValueError("knots and coeffs must have the same length")
    # Clamp outside the knot range
    if x <= knots[0]:
        return float(coeffs[0])
    if x >= knots[-1]:
        return float(coeffs[-1])
    # Find interval
    idx = np.searchsorted(knots, x) - 1
    t = (x - knots[idx]) / (knots[idx + 1] - knots[idx])
    return float((1 - t) * coeffs[idx] + t * coeffs[idx + 1])


# ----------------------------------------------------------------------
# Hoeffding bound & probabilistic acceptance (Parent B)
# ----------------------------------------------------------------------
def hoeffding_epsilon(R: float, delta: float, n: int) -> float:
    """
    Compute Hoeffding bound epsilon.

    R : range of the random variable (max - min)
    delta : confidence parameter
    n : number of observations
    """
    if n <= 0:
        raise ValueError("n must be positive")
    return math.sqrt((R ** 2 * math.log(1.0 / delta)) / (2 * n))


def should_split(gain: float, n_samples: int, *, delta: float = 0.05, R: float = 1.0) -> bool:
    """
    Decide whether a split is statistically justified using the Hoeffding bound.
    """
    eps = hoeffding_epsilon(R, delta, n_samples)
    return gain > eps


def acceptance_probability(gain: float, temperature: float) -> float:
    """
    Simulated‑annealing style acceptance probability for non‑significant gains.
    """
    if temperature <= 0:
        return 1.0
    # The exponential formulation ensures a probability in (0,1]
    prob = math.exp(gain / temperature)
    return min(1.0, prob)


# ----------------------------------------------------------------------
# Hybrid operation
# ----------------------------------------------------------------------
def compute_gain(prev_error: float, new_error: float) -> float:
    """Positive reduction in squared error."""
    return max(0.0, prev_error - new_error)


def hybrid_step(
    weights: np.ndarray,
    coeffs: np.ndarray,
    knots: np.ndarray,
    x_raw: float,
    y_true: float,
    *,
    flag: str,
    mu: float,
    n_samples: int,
    temperature: float,
) -> tuple[np.ndarray, bool, float]:
    """
    Execute one hybrid iteration:
      1. Transform raw input via KAN.
      2. Predict with NLMS.
      3. Update NLMS weights using epistemic certainty.
      4. Compute error‑reduction gain.
      5. Apply Hoeffding bound and annealing acceptance to obtain a split decision.

    Returns
    -------
    weights : np.ndarray
        Updated NLMS weight vector.
    split_decision : bool
        Result of the combined statistical / probabilistic test.
    gain : float
        Observed reduction in squared error.
    """
    # 1. KAN transform (produces a single scalar feature)
    x_feat = np.array([bspline_transform(x_raw, knots, coeffs)])

    # 2. Prediction before update
    y_pred_before = nlms_predict(weights, x_feat)
    prev_error = (y_true - y_pred_before) ** 2

    # 3. NLMS weight update with epistemic scaling
    weights = nlms_batch_update(
        weights,
        X=x_feat.reshape(1, -1),
        d=np.array([y_true]),
        mu=mu,
        flag=flag,
    )

    # 4. Prediction after update and gain computation
    y_pred_after = nlms_predict(weights, x_feat)
    new_error = (y_true - y_pred_after) ** 2
    gain = compute_gain(prev_error, new_error)

    # 5. Decision logic
    split = should_split(gain, n_samples) or (
        random.random() < acceptance_probability(gain, temperature)
    )
    return weights, split, gain


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    # Synthetic linear data y = 2*x + noise
    def generate_point():
        x = random.uniform(0.0, 10.0)
        noise = random.gauss(0.0, 0.5)
        y = 2.0 * x + noise
        return x, y

    # Initialise NLMS (single feature) and KAN parameters
    init_weight = np.zeros(1)  # shape (1,)
    knots = np.linspace(0.0, 10.0, 6)          # 6 knots
    coeffs = np.random.rand(len(knots))       # random spline coefficients

    mu_base = 0.1
    temperature = 0.5
    n_samples = 1

    # Run a few hybrid steps
    for step in range(1, 11):
        x_raw, y_true = generate_point()
        flag = random.choice(EPISTEMIC_FLAGS)

        init_weight, split, gain = hybrid_step(
            weights=init_weight,
            coeffs=coeffs,
            knots=knots,
            x_raw=x_raw,
            y_true=y_true,
            flag=flag,
            mu=mu_base,
            n_samples=n_samples,
            temperature=temperature,
        )

        print(
            f"Step {step:2d} | x={x_raw:6.3f} y={y_true:6.3f} flag={flag:9s} "
            f"gain={gain:8.5f} split={'YES' if split else 'no '} "
            f"w={init_weight[0]:.4f}"
        )
        n_samples += 1  # emulate accumulating observations
    sys.exit(0)