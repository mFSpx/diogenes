# DARWIN HAMMER — match 3052, survivor 5
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1219_s6.py (gen5)
# parent_b: hybrid_hybrid_hard_truth_ma_hybrid_hybrid_hybrid_m1853_s0.py (gen6)
# born: 2026-05-29T23:47:31Z

"""Hybrid Algorithm: Stylometry‑TTT‑Bayesian‑DP fused with Temperature‑Scaled NLMS Bandit

Parents:
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1219_s6.py (stylometry → TTT‑Linear → Bayesian cost → DP → SSIM)
- hybrid_hybrid_hard_truth_ma_hybrid_hybrid_hybrid_m1853_s0.py (LSM feature vector → perceptual hash → Schoolfield temperature model → NLMS update with Gini‑scaled step)

Mathematical Bridge:
Both parents generate a categorical frequency vector **f** over the same FUNCTION_CATS.
We treat **f** as the shared feature column **x**.  The vector is linearly mapped
by a TTT‑Linear matrix **W** (parent A) to **z = W·x**.  Parent B supplies a
temperature‑performance scaling **S(T)** (Schoolfield) that modulates the
NLMS adaptive step size, while the Gini coefficient of recent Bayesian costs
acts as an additional dynamic gain.  The updated matrix **W** is thus
simultaneously a Bayesian sufficient‑statistic transformer and a
contextual multi‑armed‑bandit learner.

The module implements:
1. Feature extraction (LSM / stylometry).
2. Linear transformation, Bayesian cost, Laplace DP, and SSIM reconstruction risk.
3. Temperature‑scaled NLMS weight adaptation with Gini‑based step scaling.
"""

import math
import random
import sys
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Shared linguistic categories (identical in both parents)
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
        "and but or nor so yet because although while if when where whereas unless until".split()
    ),
    "negation": set(
        "no not never none neither cannot cant wont dont didnt isnt arent wasnt werent".split()
    ),
    "quantifier": set(
        "all any both each few many more most much none several some such".split()
    ),
    "adverb_common": set(
        "very really just still already also even only then there here now often always sometimes".split()
    ),
}

# ----------------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------------
def words(text: str) -> List[str]:
    """Extract lowercase alphabetic tokens."""
    return re.findall(r"[a-z]+(?:'[a-z]+)?", (text or "").lower())


def lsm_vector(text: str) -> np.ndarray:
    """
    Linguistic Style Metric (LSM) – returns a column vector of category frequencies.
    Mirrors parent B's `lsm_vector` but returns a NumPy array for linear algebra.
    """
    ws = words(text)
    total = max(1, len(ws))
    cnt = {}
    for w in ws:
        cnt[w] = cnt.get(w, 0) + 1
    vec = []
    for cat in FUNCTION_CATS.keys():
        vocab = FUNCTION_CATS[cat]
        cat_count = sum(cnt.get(w, 0) for w in vocab)
        vec.append(cat_count / total)
    return np.array(vec, dtype=float)  # shape (C,)


def pseudo_inverse(mat: np.ndarray) -> np.ndarray:
    """Moore‑Penrose pseudo‑inverse using NumPy's SVD (no SciPy)."""
    u, s, vt = np.linalg.svd(mat, full_matrices=False)
    # Invert non‑zero singular values
    s_inv = np.array([1 / val if val > 1e-12 else 0.0 for val in s])
    return vt.T @ np.diag(s_inv) @ u.T


def laplace_noise(scale: float) -> float:
    """Draw a single Laplace noise sample with given scale."""
    u = random.random() - 0.5
    return scale * math.copysign(1.0, u) * math.log(1 - 2 * abs(u))


def gini_coefficient(values: List[float]) -> float:
    """Compute Gini coefficient of a non‑empty list of non‑negative numbers."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(values)
    cumulative = 0.0
    for i, val in enumerate(sorted_vals, 1):
        cumulative += i * val
    total = sum(sorted_vals)
    if total == 0:
        return 0.0
    gini = (2 * cumulative) / (n * total) - (n + 1) / n
    return gini


def ssim_index(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Simplified Structural Similarity Index (SSIM) for 1‑D vectors.
    Uses luminance, contrast, and structure components with default constants.
    """
    C1 = (0.01 * 1.0) ** 2
    C2 = (0.03 * 1.0) ** 2

    mu1 = img1.mean()
    mu2 = img2.mean()
    sigma1 = img1.var()
    sigma2 = img2.var()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()

    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2)
    return numerator / denominator if denominator != 0 else 0.0


def schoolfield_rate(
    temperature: float,
    A: float = 1.0,
    Ea: float = 0.65,
    Th: float = 303.0,
    Δ: float = 10.0,
    k: float = 8.617e-5,
) -> float:
    """
    Schoolfield temperature‑performance model.
    rate = (A * exp(-Ea/(k*T))) / (1 + exp((Th - T)/Δ))
    """
    if temperature <= 0:
        return 0.0
    num = A * math.exp(-Ea / (k * temperature))
    denom = 1 + math.exp((Th - temperature) / Δ)
    return num / denom


def compute_dhash(values: List[float]) -> int:
    """
    Simple perceptual hash: compare each adjacent pair, set bit to 1 if left > right.
    Returns an integer hash of length len(values)-1 bits.
    """
    h = 0
    for i, (a, b) in enumerate(zip(values, values[1:])):
        if a > b:
            h |= 1 << i
    return h


# ----------------------------------------------------------------------
# Core hybrid operations
# ----------------------------------------------------------------------
def transform_and_cost(
    features: np.ndarray,
    W: np.ndarray,
    mu: np.ndarray,
    Sigma: np.ndarray,
    epsilon: float = 1.0,
) -> Tuple[float, np.ndarray, float]:
    """
    1. Linear transform   z = W·x
    2. Gaussian Bayesian negative log‑likelihood cost
    3. Laplace DP noise addition
    4. Reconstruction risk via SSIM between x and x̂ = W⁺·z
    Returns (noisy_cost, z, ssim_score)
    """
    # 1. Transform
    z = W @ features  # shape (K,)

    # 2. Bayesian cost (negative log‑likelihood up to constant)
    diff = z - mu
    try:
        inv_Sigma = np.linalg.inv(Sigma)
    except np.linalg.LinAlgError:
        inv_Sigma = np.linalg.pinv(Sigma)
    cost = 0.5 * diff.T @ inv_Sigma @ diff

    # 3. Differential privacy
    scale = 1.0 / epsilon
    noisy_cost = cost + laplace_noise(scale)

    # 4. Reconstruction risk
    W_pinv = pseudo_inverse(W)
    recon = W_pinv @ z
    ssim_score = ssim_index(features, recon)

    return float(noisy_cost), z, float(ssim_score)


def adaptive_nlms_update(
    W: np.ndarray,
    features: np.ndarray,
    target: np.ndarray,
    recent_costs: List[float],
    temperature: float,
    base_step: float = 0.1,
    delta: float = 1e-6,
) -> np.ndarray:
    """
    NLMS weight update where the step size η is scaled by:
        η = base_step * Gini(recent_costs) * Schoolfield(temperature)
    The error signal is (target - W·features).
    Returns the updated matrix W_new.
    """
    # Compute dynamic scaling factors
    gini = gini_coefficient(recent_costs)
    temp_scale = schoolfield_rate(temperature)
    eta = base_step * gini * temp_scale

    # Prediction and error
    pred = W @ features
    error = target - pred  # shape (K,)

    # NLMS update
    norm_sq = np.dot(features, features) + delta
    update = (eta / norm_sq) * np.outer(error, features)  # shape (K, C)
    W_new = W + update
    return W_new


def hybrid_pipeline(
    text: str,
    W: np.ndarray,
    mu: np.ndarray,
    Sigma: np.ndarray,
    epsilon: float,
    recent_costs: List[float],
    temperature: float,
) -> Dict[str, Any]:
    """
    Executes the full hybrid workflow:
    - Extract LSM features.
    - Transform, compute DP‑noised Bayesian cost, and SSIM.
    - Produce a perceptual hash of the transformed vector.
    - Adapt the weight matrix via temperature‑scaled NLMS.
    Returns a dictionary with all intermediate results.
    """
    # Feature extraction (bridge vector)
    x = lsm_vector(text)  # shape (C,)

    # Transform & Bayesian cost
    noisy_cost, z, ssim_val = transform_and_cost(x, W, mu, Sigma, epsilon)

    # Perceptual hash of transformed representation
    dhash = compute_dhash(z.tolist())

    # Adaptive weight update (target can be mu for simplicity)
    W_new = adaptive_nlms_update(
        W, x, mu, recent_costs, temperature, base_step=0.2
    )

    return {
        "features": x,
        "transformed": z,
        "noisy_cost": noisy_cost,
        "ssim": ssim_val,
        "dhash": dhash,
        "W_updated": W_new,
    }


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_text = (
        "I think that the quick brown fox jumps over the lazy dog while "
        "you watch it from the hill. It is very common to see such sentences."
    )

    # Dimensionalities
    C = len(FUNCTION_CATS)          # number of categories
    K = 12                          # latent dimension for TTT‑Linear matrix

    # Initialise TTT‑Linear weight matrix (random but reproducible)
    rng = np.random.default_rng(42)
    W_init = rng.normal(loc=0.0, scale=1.0, size=(K, C))

    # Prior for Bayesian model (zero‑mean, identity covariance scaled)
    mu_prior = np.zeros(K)
    Sigma_prior = np.eye(K) * 2.0

    # Differential privacy parameter
    eps = 0.8

    # Simulated recent Bayesian costs (for Gini scaling)
    recent = [0.95, 1.12, 0.88, 1.05, 0.97]

    # Temperature (Kelvin) for Schoolfield scaling
    temp_K = 310.0

    results = hybrid_pipeline(
        text=sample_text,
        W=W_init,
        mu=mu_prior,
        Sigma=Sigma_prior,
        epsilon=eps,
        recent_costs=recent,
        temperature=temp_K,
    )

    # Simple sanity prints (no external dependencies)
    print("Feature vector (LSM):", results["features"])
    print("Transformed vector (z):", results["transformed"])
    print("DP‑noised Bayesian cost:", results["noisy_cost"])
    print("SSIM reconstruction score:", results["ssim"])
    print("Perceptual dhash (int):", results["dhash"])
    print("Updated weight matrix shape:", results["W_updated"].shape)