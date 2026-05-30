# DARWIN HAMMER — match 3999, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_nlms_o_hybrid_hybrid_hybrid_m1997_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2246_s0.py (gen6)
# born: 2026-05-29T23:53:08Z

"""Hybrid Algorithm: Fusion of NLMS‑Krampus‑SHAP (Parent A) with VRAM‑Privacy‑Hoeffding (Parent B)

This module mathematically bridges the two parent algorithms by treating the
SHAP‑kernel weight (Parent A) as a *value function* that is fed to a
Normalized Least‑Mean‑Squares (NLMS) adaptive filter.  The desired NLMS
target is constructed from the privacy‑risk side of Parent B:

* ``reconstruction_risk_score`` provides a scalar risk derived from the
  endpoint health score and the quasi‑identifier statistics.
* ``hoeffding_bound`` supplies a confidence‑scaled factor that incorporates
  the model’s VRAM demand and morphology scaling factor.

The NLMS update therefore continuously adapts the feature‑weight vector so
that the inner product of the weight vector with the fused feature vector
approximates the product

``risk × hoeffding_bound × shapley_kernel_weight``

Thus the hybrid system jointly optimises feature importance (via SHAP), privacy
risk (via reconstruction risk and Hoeffding bound), and adaptive learning
(NLMS).  All core equations from the parents are retained and inter‑linked."""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Feature extraction, SHAP kernel, NLMS core
# ----------------------------------------------------------------------


def _deterministic_features(text: str) -> Dict[str, float]:
    """Deterministic pseudo‑random features based on a hash of the input."""
    rnd = random.Random(hash(text))
    keys = [
        "operator_visceral_ratio",
        "operator_tech_ratio",
        "operator_legal_osint_ratio",
    ]
    return {key: rnd.random() for key in keys}


def _stochastic_features(text: str) -> Dict[str, float]:
    """Purely stochastic features."""
    keys = [
        "operator_visceral_ratio",
        "operator_tech_ratio",
        "operator_legal_osint_ratio",
    ]
    return {key: random.random() for key in keys}


def feature_fusion(text: str, alpha: float = 0.5) -> Dict[str, float]:
    """
    Linear interpolation between deterministic and stochastic feature sets.

    Parameters
    ----------
    text: str
        Input text used as seed for deterministic features.
    alpha: float, default 0.5
        Weight of deterministic component (0 ≤ α ≤ 1).

    Returns
    -------
    dict
        Fused feature dictionary.
    """
    det = _deterministic_features(text)
    sto = _stochastic_features(text)
    fused = {}
    for k in det:
        fused[k] = alpha * det[k] + (1.0 - alpha) * sto[k]
    return fused


def shapley_kernel_weight(subset_size: int, feature_count: int) -> float:
    """
    Exact Shapley kernel weight for a subset of features.

    The formula follows the original Kernel SHAP definition:

        w = (|S|! * (M‑|S|‑1)!) / M!

    where |S| is the subset size and M is the total number of features.
    """
    if subset_size < 0 or subset_size > feature_count:
        raise ValueError("subset_size must be in [0, feature_count]")
    numerator = math.factorial(subset_size) * math.factorial(feature_count - subset_size - 1)
    denominator = math.factorial(feature_count)
    return numerator / denominator


def nlms_update(
    w: np.ndarray,
    x: np.ndarray,
    d: float,
    mu: float = 0.1,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Normalized LMS weight update.

    w_{k+1} = w_k + (mu / (eps + ||x||²)) * x * (d - w_k·x)

    Parameters
    ----------
    w : np.ndarray
        Current weight vector (shape (N,)).
    x : np.ndarray
        Input feature vector (shape (N,)).
    d : float
        Desired scalar response.
    mu : float, default 0.1
        Step‑size (learning rate).
    eps : float, default 1e-8
        Small constant to avoid division by zero.

    Returns
    -------
    np.ndarray
        Updated weight vector.
    """
    x_norm_sq = np.dot(x, x) + eps
    error = d - np.dot(w, x)
    return w + (mu / x_norm_sq) * x * error


# ----------------------------------------------------------------------
# Parent B – Privacy risk, Hoeffding bound, model / endpoint descriptors
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ModelTier:
    """Lightweight descriptor for a model."""
    name: str
    ram_mb: int
    tier: str


@dataclass(frozen=True)
class Morphology:
    """Geometric description of a logical entity."""
    length: float
    width: float
    height: float
    mass: float


@dataclass(frozen=True)
class ModelSpec:
    """Combined specification used by the hybrid scheduler."""
    tier: ModelTier
    morphology: Morphology
    unique_quasi_identifiers: int
    total_records: int
    vram_demand: int  # added for Hoeffding bound


@dataclass
class Endpoint:
    health_score: float          # ∈ [0, 1]
    failure_rate: float          # ∈ [0, 1]
    recovery_priority: float    # arbitrary scale


def reconstruction_risk_score(
    unique_quasi_identifiers: int,
    total_records: int,
    health_score: float,
) -> float:
    """
    Probability that a record can be re‑identified.

    risk = (U / N) * health_score
    """
    if total_records == 0:
        raise ValueError("total_records must be > 0")
    return (unique_quasi_identifiers / total_records) * health_score


def hoeffding_bound(
    r: float,
    delta: float,
    n: int,
    vram_demand: int,
    morphology_scaling_factor: float,
) -> float:
    """
    Hoeffding‑type confidence bound with VRAM and morphology scaling.

    Classic Hoeffding epsilon:
        ε = sqrt( (1/(2n)) * ln(2/δ) )

    The bound is inflated by a factor that grows with VRAM demand and the
    morphology scaling factor, reflecting the extra uncertainty introduced by
    larger memory footprints and more complex shapes.

    Returns r + scaling_factor * ε.
    """
    if n <= 0:
        raise ValueError("sample size n must be > 0")
    if not (0 < delta < 1):
        raise ValueError("delta must be in (0,1)")
    epsilon = math.sqrt((1.0 / (2 * n)) * math.log(2.0 / delta))
    scaling = morphology_scaling_factor * vram_demand
    return r + scaling * epsilon


# ----------------------------------------------------------------------
# Hybrid core – mathematically fusing both parents
# ----------------------------------------------------------------------


def hybrid_target_scalar(
    text: str,
    endpoint: Endpoint,
    model_spec: ModelSpec,
    delta: float = 0.05,
    sample_size: int = 1000,
) -> Tuple[float, float]:
    """
    Compute the scalar that the NLMS filter should track.

    The scalar is:

        d = risk × hoeffding_bound × shapley_weight

    where
        risk = reconstruction_risk_score(...)
        hoeffding_bound = hoeffding_bound(risk, delta, sample_size, ...)
        shapley_weight = shapley_kernel_weight(|S|, M)

    |S| is taken as the number of features whose fused value exceeds the mean
    of the fused vector; M is the total number of features.

    Returns
    -------
    d : float
        Desired NLMS response.
    shapley_w : float
        The raw Shapley kernel weight (useful for diagnostics).
    """
    # 1. privacy risk
    risk = reconstruction_risk_score(
        model_spec.unique_quasi_identifiers,
        model_spec.total_records,
        endpoint.health_score,
    )

    # 2. confidence‑scaled bound
    bound = hoeffding_bound(
        r=risk,
        delta=delta,
        n=sample_size,
        vram_demand=model_spec.vram_demand,
        morphology_scaling_factor=model_spec.morphology.mass,
    )

    # 3. SHAP kernel weight
    fused = feature_fusion(text)
    feature_vals = np.fromiter(fused.values(), dtype=float)
    mean_val = feature_vals.mean()
    subset_size = int((feature_vals > mean_val).sum())
    feature_count = len(fused)
    shapley_w = shapley_kernel_weight(subset_size, feature_count)

    # 4. combine
    d = risk * bound * shapley_w
    return d, shapley_w


def hybrid_nlms_step(
    text: str,
    endpoint: Endpoint,
    model_spec: ModelSpec,
    w_prev: np.ndarray,
    mu: float = 0.1,
    alpha: float = 0.5,
) -> Tuple[np.ndarray, float]:
    """
    Perform one hybrid NLMS adaptation step.

    Parameters
    ----------
    text : str
        Input data used for feature extraction.
    endpoint : Endpoint
        Current endpoint descriptor (provides health_score).
    model_spec : ModelSpec
        Model specification (provides privacy statistics and VRAM demand).
    w_prev : np.ndarray
        Previous weight vector (shape (M,)).
    mu : float, default 0.1
        NLMS learning rate.
    alpha : float, default 0.5
        Fusion coefficient for deterministic vs stochastic features.

    Returns
    -------
    w_new : np.ndarray
        Updated weight vector.
    d : float
        Desired scalar used in the update (for inspection).
    """
    # Feature vector
    fused_dict = feature_fusion(text, alpha=alpha)
    x = np.fromiter(fused_dict.values(), dtype=float)

    # Desired response
    d, _ = hybrid_target_scalar(text, endpoint, model_spec)

    # NLMS update
    w_new = nlms_update(w_prev, x, d, mu=mu)

    return w_new, d


def evaluate_hybrid(
    text: str,
    endpoint: Endpoint,
    model_spec: ModelSpec,
    w: np.ndarray,
    alpha: float = 0.5,
) -> Dict[str, float]:
    """
    Evaluate the current hybrid state without performing a weight update.

    Returns a dictionary with:
        - "prediction": w·x
        - "desired":   d (as defined in :func:`hybrid_target_scalar`)
        - "error":     |d - prediction|
        - "shapley_weight": raw SHAP kernel weight
    """
    fused = feature_fusion(text, alpha=alpha)
    x = np.fromiter(fused.values(), dtype=float)
    pred = float(np.dot(w, x))
    d, shapley_w = hybrid_target_scalar(text, endpoint, model_spec)
    err = abs(d - pred)
    return {
        "prediction": pred,
        "desired": d,
        "error": err,
        "shapley_weight": shapley_w,
    }


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # deterministic seed for reproducibility
    random.seed(0)
    np.random.seed(0)

    # Dummy inputs
    sample_text = "The quick brown fox jumps over the lazy dog."
    endpoint = Endpoint(health_score=0.8, failure_rate=0.1, recovery_priority=0.5)

    tier = ModelTier(name="gpt‑mini", ram_mb=2048, tier="small")
    morph = Morphology(length=1.0, width=0.5, height=0.2, mass=0.8)
    model_spec = ModelSpec(
        tier=tier,
        morphology=morph,
        unique_quasi_identifiers=150,
        total_records=10000,
        vram_demand=4096,
    )

    # Initialise weight vector (3 features → length 3)
    w = np.zeros(3)

    # Run a few adaptation steps
    for step in range(5):
        w, d = hybrid_nlms_step(
            text=sample_text,
            endpoint=endpoint,
            model_spec=model_spec,
            w_prev=w,
            mu=0.2,
            alpha=0.6,
        )
        eval_res = evaluate_hybrid(sample_text, endpoint, model_spec, w, alpha=0.6)
        print(f"Step {step+1}: prediction={eval_res['prediction']:.6f}, "
              f"desired={eval_res['desired']:.6f}, error={eval_res['error']:.6f}")

    print("Final weight vector:", w)