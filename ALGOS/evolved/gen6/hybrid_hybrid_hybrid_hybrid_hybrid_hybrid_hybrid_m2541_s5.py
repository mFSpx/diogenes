# DARWIN HAMMER — match 2541, survivor 5
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hard_t_hybrid_hybrid_bayes__m19_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_nlms_h_m901_s0.py (gen5)
# born: 2026-05-29T23:42:46Z

"""Hybrid Algorithm: Fusion of Stylometry Feature Generation and Temperature‑Dependent NLMS Adaptation

Parents:
- hybrid_hybrid_hard_t_hybrid_hybrid_bayes__m19_s0.py (Parent A)
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_nlms_h_m901_s0.py (Parent B)

Mathematical Bridge:
Parent A extracts stylometric features by seeding a pseudo‑random generator with a
deterministic SHA‑256 hash of the input text.  
Parent B adapts NLMS (Normalized Least‑Mean‑Squares) weights using a temperature‑
dependent developmental rate derived from the Schoolfield poikilotherm model.

The hybrid algorithm unifies these structures by:
1. Computing a deterministic hash of the raw text.
2. Using that hash as the seed for a random generator that produces a stylometric
   feature vector.
3. Treating the feature vector as the input signal to an NLMS filter.
4. Scaling the NLMS step‑size with the developmental rate computed from an
   ambient temperature (°C) via the Schoolfield equation.

Thus the stochastic feature representation becomes temperature‑aware, yielding a
single, mathematically coherent system.

"""

import math
import random
import hashlib
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

# ----------------------------------------------------------------------
# Parent A – Feature extraction utilities
# ----------------------------------------------------------------------
def deterministic_hash(text: str) -> int:
    """Return an integer hash derived deterministically from *text* using SHA‑256."""
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    # Use first 16 hex digits to stay within Python's int range for seeding
    return int(h[:16], 16)


def generate_stylometry_features(text: str, dim: int = 32) -> np.ndarray:
    """
    Produce a reproducible pseudo‑random stylometric feature vector.

    The deterministic hash of *text* seeds a ``random.Random`` instance,
    guaranteeing that the same text always yields the same feature vector.
    """
    seed = deterministic_hash(text)
    rng = random.Random(seed)
    # Generate values in [0, 1)
    return np.array([rng.random() for _ in range(dim)], dtype=np.float64)


# ----------------------------------------------------------------------
# Parent B – Schoolfield developmental rate and NLMS adaptation
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class SchoolfieldParams:
    """Parameters of the Schoolfield temperature‑dependence model."""
    rho_25: float = 1.0          # Developmental rate at 25 °C (298.15 K)
    delta_h_activation: float = 12_000.0   # Activation enthalpy (cal mol⁻¹)
    t_low: float = 283.15        # Low‑temperature threshold (K)
    t_high: float = 307.15       # High‑temperature threshold (K)
    delta_h_low: float = -45_000.0  # Low‑temperature enthalpy (cal mol⁻¹)
    delta_h_high: float = 65_000.0   # High‑temperature enthalpy (cal mol⁻¹)
    r_cal: float = 1.987         # Gas constant (cal mol⁻¹ K⁻¹)


def celsius_to_kelvin(c: float) -> float:
    """Convert Celsius to Kelvin."""
    return c + 273.15


def developmental_rate(temp_k: float, params: SchoolfieldParams = SchoolfieldParams()) -> float:
    """
    Compute the temperature‑dependent developmental rate using the
    Schoolfield model.

    rho(T) = rho_25 * exp( -ΔH_a/(R) * (1/T - 1/298.15) )
             / ( 1 + exp( ΔH_l/(R) * (1/T_l - 1/T) )
                 + exp( ΔH_h/(R) * (1/T - 1/T_h) ) )
    """
    if temp_k <= 0:
        raise ValueError("Temperature in Kelvin must be positive")
    # Exponential terms
    exp_activation = math.exp(
        -(params.delta_h_activation / params.r_cal) *
        (1.0 / temp_k - 1.0 / 298.15)
    )
    exp_low = math.exp(
        (params.delta_h_low / params.r_cal) *
        (1.0 / params.t_low - 1.0 / temp_k)
    )
    exp_high = math.exp(
        (params.delta_h_high / params.r_cal) *
        (1.0 / temp_k - 1.0 / params.t_high)
    )
    denominator = 1.0 + exp_low + exp_high
    return params.rho_25 * exp_activation / denominator


def nlms_update(
    w: np.ndarray,
    x: np.ndarray,
    error: float,
    mu: float,
    temp_c: float,
    epsilon: float = 1e-8
) -> np.ndarray:
    """
    Perform a single Normalized LMS weight update, scaling the step size
    with the temperature‑dependent developmental rate.

    w_{new} = w + (mu * rate / (epsilon + ||x||²)) * error * x
    """
    if w.shape != x.shape:
        raise ValueError("Weight and input vectors must have the same shape")
    temp_k = celsius_to_kelvin(temp_c)
    rate = developmental_rate(temp_k)
    norm_factor = epsilon + np.dot(x, x)
    step = (mu * rate / norm_factor) * error
    return w + step * x


# ----------------------------------------------------------------------
# Hybrid Operations
# ----------------------------------------------------------------------
def hybrid_feature_nlms(
    text: str,
    temperature_c: float,
    nlms_mu: float = 0.1,
    feature_dim: int = 32
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate stylometric features from *text* and adapt an NLMS filter
    using the temperature *temperature_c*.

    Returns a tuple ``(features, updated_weights)`` where ``updated_weights``
    are the NLMS weights after one adaptation step with a synthetic error.
    """
    # 1. Feature extraction (Parent A)
    features = generate_stylometry_features(text, dim=feature_dim)

    # 2. Initialise NLMS weights (zero vector)
    w = np.zeros_like(features)

    # 3. Synthetic desired signal – for demonstration we use the mean of features
    d = features.mean()

    # 4. Compute error (desired - current output)
    y = np.dot(w, features)  # current output (initially zero)
    error = d - y

    # 5. Update weights using temperature‑scaled NLMS (Parent B)
    w_updated = nlms_update(w, features, error, nlms_mu, temperature_c)

    return features, w_updated


def hybrid_predict(
    text: str,
    temperature_c: float,
    nlms_mu: float = 0.1,
    feature_dim: int = 32
) -> float:
    """
    Produce a scalar prediction by feeding stylometric features through an
    NLMS filter whose adaptation is temperature‑aware.

    The prediction is the dot product of the updated weights with the feature
    vector, representing a temperature‑modulated stylometric score.
    """
    features, w_updated = hybrid_feature_nlms(
        text, temperature_c, nlms_mu, feature_dim
    )
    return float(np.dot(w_updated, features))


def hybrid_batch_process(
    texts: List[str],
    temperatures_c: List[float],
    nlms_mu: float = 0.1,
    feature_dim: int = 32
) -> List[float]:
    """
    Process a batch of (text, temperature) pairs, returning a list of predictions.
    Demonstrates vectorised handling while preserving the hybrid dynamics.
    """
    if len(texts) != len(temperatures_c):
        raise ValueError("texts and temperatures_c must have the same length")
    predictions = []
    for txt, temp in zip(texts, temperatures_c):
        pred = hybrid_predict(txt, temp, nlms_mu, feature_dim)
        predictions.append(pred)
    return predictions


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_text = "The quick brown fox jumps over the lazy dog."
    sample_temp_c = 22.0  # ambient temperature in Celsius

    # Single hybrid prediction
    pred = hybrid_predict(sample_text, sample_temp_c)
    print(f"Hybrid prediction (single): {pred:.6f}")

    # Batch processing demo
    texts = [
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
        "Data science blends statistics with computer science.",
        "Quantum mechanics reveals the probabilistic nature of reality."
    ]
    temps = [18.5, 25.0, 30.2]
    batch_preds = hybrid_batch_process(texts, temps)
    print("Batch predictions:", ["{:.6f}".format(p) for p in batch_preds])