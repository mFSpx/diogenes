# DARWIN HAMMER — match 4779, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2484_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_label__hybrid_xgboost_objec_m1588_s0.py (gen4)
# born: 2026-05-29T23:58:01Z

"""Hybrid NLMS‑Fisher‑Regret–XGBoost Module
Parent A: hybrid_hybrid_hybrid_hybrid_m2484_s0 – NLMS weight adaptation modulated by Fisher
information and regret‑weighted strategy.
Parent B: hybrid_hybrid_hybrid_label__hybrid_xgboost_objec_m1588_s0 – stylometry‑based
feature extraction and XGBoost binary‑logistic gradient/Hessian computation.

Mathematical bridge
------------------
Tokens are mapped to a *stylometry feature vector*  u ∈ℝⁿ using the
category dictionaries of Parent B.  This vector is the NLMS input of Parent A.
The XGBoost gradient g and Hessian h (Parent B) are interpreted as a *regret
signal*; a per‑component regret weight ρᵢ is built from |g|, h and the magnitude
of uᵢ.  The NLMS update is then performed with Fisher‑information scores
fᵢ = FisherScore(uᵢ; μ, λ) (Parent A) multiplied by the regret weights ρᵢ,
thereby fusing the two governing equations into a single adaptive rule.

The core update for each weight wᵢ is

    e      = d − w·u                                 (prediction error)
    fᵢ    = FisherScore(uᵢ; μ, λ)
    ρᵢ    = regret_weight(i)                        (from XGBoost g, h)
    Δwᵢ   = η·e·uᵢ·(fᵢ·ρᵢ) / (‖u‖² + ε)

where η is the NLMS step size.  The module provides three public functions
illustrating this hybrid operation."""
import math
import random
import sys
from pathlib import Path
import numpy as np

# ----------------------------------------------------------------------
# Parent A – Fisher‑score utilities
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a Gaussian beam."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


# ----------------------------------------------------------------------
# Parent B – Stylometry utilities
# ----------------------------------------------------------------------
FUNCTION_CATS: dict[str, set[str]] = {
    "pronoun": set(
        "i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()
    ),
    "article": set("a an the".split()),
    "preposition": set(
        "about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()
    ),
    "auxiliary": set(
        "am are be been being can could did do does had has have is may might must shall should was were will would".split()
    ),
    "conjunction": set(
        "and but or nor so yet because although whoever that which what how why when where who whom since as until long".split()
    ),
    "adverb": set(
        "how very rather more less just about equally near almost exactly still yet rather".split()
    ),
}


def stylometry_vector(tokens: list[str]) -> np.ndarray:
    """Return a normalized count vector per FUNCTION_CATS category."""
    counts = []
    token_set = {t.lower() for t in tokens}
    total = len(token_set) if token_set else 1
    for cat in FUNCTION_CATS.values():
        counts.append(len(token_set & cat) / total)
    vec = np.array(counts, dtype=float)
    # Ensure non‑zero norm for NLMS denominator
    if np.linalg.norm(vec) == 0.0:
        vec[0] = 1.0
    return vec


# ----------------------------------------------------------------------
# Parent B – XGBoost binary logistic gradient / Hessian
# ----------------------------------------------------------------------
def sigmoid(margin: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-margin))


def binary_logistic_grad_hess(y_true: np.ndarray, margin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return gradient and Hessian of binary logistic loss."""
    p = sigmoid(margin)
    g = p - y_true
    h = p * (1.0 - p)
    return g, h


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def regret_weights_from_grad_hess(
    grad: np.ndarray,
    hess: np.ndarray,
    u: np.ndarray,
    eps: float = 1e-12,
) -> dict[int, float]:
    """
    Build a per‑component regret weight ρᵢ from gradient, Hessian and the
    NLMS input vector u.  The formulation is heuristic but respects the
    intuition that larger |g| or larger curvature (h) increase regret,
    and that features with larger magnitude should be weighted proportionally.
    """
    base = np.abs(grad) + hess  # scalar per‑sample, broadcastable
    # Broadcast to feature dimension
    base_vec = np.full_like(u, base.item())
    rho = np.maximum(base_vec * np.abs(u), eps)
    return {i: rho[i] for i in range(len(u))}


def hybrid_nlms_fisher(
    w: np.ndarray,
    u: np.ndarray,
    d: float,
    mu: float,
    lam: float,
    regret_weights: dict[int, float],
    eps: float = 1e-12,
) -> np.ndarray:
    """
    NLMS weight update modulated by Fisher information and regret weights.

    Parameters
    ----------
    w : np.ndarray
        Current weight vector (n,).
    u : np.ndarray
        Input (stylometry) vector (n,).
    d : float
        Desired scalar output (e.g., target label).
    mu : float
        NLMS step size (η).
    lam : float
        Width parameter for FisherScore (σ).
    regret_weights : dict[int, float]
        Mapping from component index to regret multiplier ρᵢ.
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    np.ndarray
        Updated weight vector.
    """
    # Prediction error
    e = d - float(np.dot(w, u))

    # Fisher scores per component
    f_scores = np.array([fisher_score(u_i, mu, lam, eps) for u_i in u])

    # Apply regret weighting
    rho_vec = np.array([regret_weights.get(i, 1.0) for i in range(len(u))])

    # Modulation term
    modulation = f_scores * rho_vec

    denom = np.dot(u, u) + eps
    delta_w = (mu * e / denom) * (u * modulation)
    return w + delta_w


def hybrid_predict_and_update(
    tokens: list[str],
    w: np.ndarray,
    mu: float,
    lam: float,
    y_true: float,
) -> tuple[float, np.ndarray]:
    """
    End‑to‑end hybrid step:
    1. Convert tokens to stylometry vector u.
    2. Compute current margin m = w·u.
    3. Obtain XGBoost gradient/hessian (regret signal).
    4. Build regret weights from the signal.
    5. Update w via NLMS‑Fisher with regret.
    6. Return new prediction (sigmoid of updated margin) and updated weights.
    """
    u = stylometry_vector(tokens)

    # Current margin and logistic prediction
    margin = float(np.dot(w, u))
    pred_before = sigmoid(margin)

    # XGBoost gradient/Hessian (scalar because binary loss)
    g, h = binary_logistic_grad_hess(np.array([y_true]), np.array([margin]))
    regret_w = regret_weights_from_grad_hess(g, h, u)

    # Desired output for NLMS is the true label (0 or 1)
    w_new = hybrid_nlms_fisher(w, u, d=y_true, mu=mu, lam=lam, regret_weights=regret_w)

    # New margin after weight update
    new_margin = float(np.dot(w_new, u))
    pred_after = sigmoid(new_margin)

    return pred_after, w_new


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple deterministic token list
    sample_tokens = [
        "I", "am", "the", "author", "of", "this", "paper", "and", "I", "write",
        "about", "the", "method", "in", "the", "introduction"
    ]

    # Initialise a weight vector matching the number of stylometry categories
    n_features = len(FUNCTION_CATS)
    rng = np.random.default_rng(42)
    w0 = rng.normal(loc=0.0, scale=0.1, size=n_features)

    # Hyper‑parameters
    mu = 0.5          # NLMS step size
    lam = 0.1         # Fisher width
    y_true = 1.0      # Assume positive class for this example

    # Run a single hybrid update
    pred, w1 = hybrid_predict_and_update(sample_tokens, w0, mu, lam, y_true)

    print(f"Prediction after update: {pred:.4f}")
    print(f"Weight vector change norm: {np.linalg.norm(w1 - w0):.6f}")