# DARWIN HAMMER — match 1599, survivor 4
# gen: 5
# parent_a: hybrid_hybrid_physarum_netw_hybrid_hybrid_infota_m875_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_krampu_hybrid_hybrid_hard_t_m1107_s0.py (gen4)
# born: 2026-05-29T23:37:50Z

"""
Hybrid Physarum‑Infotaxis ↔ Krampus‑Ollivier‑Ricci Fusion

Parent A: hybrid_hybrid_physarum_netw_hybrid_hybrid_infota_m875_s0.py  
Parent B: hybrid_hybrid_hybrid_krampu_hybrid_hybrid_hard_t_m1107_s0.py  

Mathematical bridge:
Both parents manipulate *densities* that modulate a high‑dimensional feature space.

* In A the pressure difference on an edge yields an **information density**
  `I = log(pressure + 1)`.  
* In B the Ollivier‑Ricci curvature calculation produces a **feature scaling**
  vector `c_f = κ * f` (κ – curvature factor, f – raw feature).

The fusion treats the information density `I` as a *scalar curvature weight* that
multiplies the curvature‑scaled feature vector.  Consequently the conductance
update, entropy estimate and decision metric are all coupled through the
product `I·c_f`.  This creates a single unified dynamical system where
physarum‑style flux drives the evolution of a curvature‑aware belief vector,
while infotaxis‑style entropy guides action selection.
"""

import math
import random
import sys
import pathlib
import hashlib
import numpy as np
from typing import List, Dict, Tuple

# ----------------------------------------------------------------------
# Utilities from Parent A
# ----------------------------------------------------------------------
def flux(conductance: float, edge_length: float,
         pressure_a: float, pressure_b: float,
         eps: float = 1e-12) -> float:
    """Physarum flux on an edge."""
    if edge_length <= 0:
        raise ValueError('edge_length must be positive')
    return conductance / max(edge_length, eps) * (pressure_a - pressure_b)


def update_conductance(conductance: float, q: float,
                       dt: float = 1.0, gain: float = 1.0,
                       decay: float = 0.05) -> float:
    """Physarum conductance dynamics."""
    if dt < 0 or decay < 0:
        raise ValueError('dt and decay must be non‑negative')
    return max(0.0, conductance + dt * (gain * abs(q) - decay * conductance))


def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def signature(tokens: List[str], k: int = 128) -> List[int]:
    """Min‑hash style signature of a token list."""
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [max(0, (1 << 64) - 1)] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]


def similarity(sig_a: List[int], sig_b: List[int]) -> float:
    """Jaccard‑like similarity of two signatures."""
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have equal length")
    if not sig_a:
        raise ValueError("signatures must not be empty")
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)


def calculate_pressure(conductance: float, edge_length: float, q: float) -> float:
    """Pressure induced by a flow q on an edge."""
    return conductance * q / edge_length


def information_density(pressure: float) -> float:
    """Entropy‑style information density derived from pressure."""
    return math.log(pressure + 1.0)


# ----------------------------------------------------------------------
# Utilities from Parent B (cleaned and extended)
# ----------------------------------------------------------------------
def extract_full_features(text: str) -> Dict[str, float]:
    """Randomly generate a high‑dimensional feature dict from raw text."""
    # In a real system this would be a NLP pipeline; here we use placeholders.
    features: Dict[str, float] = {}
    rng = random.Random(hash(text))
    # Groups of correlated features
    for prefix in ("operator", "psyche", "resilience",
                   "rainmaker", "telemetry"):
        for i in range(3):
            key = f"{prefix}_{i}_ratio" if prefix != "psyche" else f"{prefix}_{i}_entropy"
            features[key] = rng.random()
    return features


def calculate_ricci_curvature(features: Dict[str, float]) -> Dict[str, float]:
    """
    Simplified Ollivier‑Ricci curvature: each feature is scaled by a
    group‑specific factor (0.1, 0.2, 0.3) mimicking curvature weighting.
    """
    curvature: Dict[str, float] = {}
    for name, val in features.items():
        if name.startswith("operator"):
            curvature[name] = val * 0.1
        elif name.startswith("psyche"):
            curvature[name] = val * 0.2
        elif name.startswith("resilience"):
            curvature[name] = val * 0.3
        else:
            curvature[name] = val * 0.05  # fallback weight
    return curvature


def curvature_vector(curv: Dict[str, float]) -> np.ndarray:
    """Convert curvature dict to a dense vector (sorted by key)."""
    keys = sorted(curv.keys())
    return np.array([curv[k] for k in keys], dtype=np.float64)


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def hybrid_flux_curvature(conductance: float, edge_length: float,
                          pressure_a: float, pressure_b: float,
                          feature_text: str) -> Tuple[float, np.ndarray]:
    """
    Compute physarum flux and a curvature‑scaled feature vector that is
    weighted by the information density derived from the pressure on the edge.

    Returns
    -------
    q : float
        Net flux on the edge.
    weighted_vec : np.ndarray
        Information‑density‑weighted curvature vector.
    """
    q = flux(conductance, edge_length, pressure_a, pressure_b)
    # Pressure at the midpoint (simple average) for density estimation
    mid_pressure = (pressure_a + pressure_b) / 2.0
    I = information_density(mid_pressure)

    # Feature extraction + curvature
    raw_features = extract_full_features(feature_text)
    curv = calculate_ricci_curvature(raw_features)
    vec = curvature_vector(curv)

    weighted_vec = I * vec
    return q, weighted_vec


def hybrid_conductance_update(conductance: float, q: float,
                              curvature_vec: np.ndarray,
                              dt: float = 1.0,
                              gain: float = 1.0,
                              decay: float = 0.05,
                              curvature_gain: float = 0.2) -> float:
    """
    Update conductance using physarum flux *and* the L2‑norm of the curvature
    vector.  The curvature term acts as an additional “resource” that can
    accelerate growth when the belief (curvature) is strong.

    Parameters
    ----------
    curvature_vec : np.ndarray
        Vector returned by ``hybrid_flux_curvature``.
    curvature_gain : float
        Scaling of the curvature contribution.

    Returns
    -------
    new_conductance : float
    """
    curvature_norm = np.linalg.norm(curvature_vec)
    effective_gain = gain + curvature_gain * curvature_norm
    return update_conductance(conductance, q, dt=dt,
                              gain=effective_gain, decay=decay)


def hybrid_expected_entropy(sig_a: List[int], sig_b: List[int],
                            curvature_vec: np.ndarray,
                            p_hit: float = 0.5) -> float:
    """
    Combine infotaxis expected entropy with a curvature‑aware correction.
    The correction penalises high curvature magnitude (i.e. strong belief)
    because it reduces uncertainty.

    Returns
    -------
    expected_entropy : float
    """
    base_entropy = -p_hit * math.log(p_hit + 1e-12) - (1 - p_hit) * math.log(1 - p_hit + 1e-12)
    sim = similarity(sig_a, sig_b)
    curvature_penalty = 0.1 * np.linalg.norm(curvature_vec)  # tunable factor
    return base_entropy - sim + curvature_penalty


def hybrid_decision(sig_current: List[int], sig_candidate: List[int],
                    curvature_vec: np.ndarray) -> bool:
    """
    Decide whether to move to a candidate state.
    Returns True if the candidate reduces the hybrid expected entropy.
    """
    entropy_current = hybrid_expected_entropy(sig_current, sig_current,
                                              curvature_vec)
    entropy_candidate = hybrid_expected_entropy(sig_current, sig_candidate,
                                                curvature_vec)
    return entropy_candidate < entropy_current


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple synthetic edge
    conduct = 0.8
    length = 2.5
    p_a = 1.2
    p_b = 0.4
    text = "sample observation for curvature extraction"

    # Compute hybrid flux and curvature vector
    q, curv_vec = hybrid_flux_curvature(conduct, length, p_a, p_b, text)

    # Update conductance
    new_conduct = hybrid_conductance_update(conduct, q, curv_vec)

    # Signatures for two dummy token sets
    tokens_a = ["alpha", "beta", "gamma"]
    tokens_b = ["beta", "delta", "epsilon"]
    sig_a = signature(tokens_a)
    sig_b = signature(tokens_b)

    # Decision test
    move = hybrid_decision(sig_a, sig_b, curv_vec)

    print(f"Flux q = {q:.4f}")
    print(f"Curvature vector norm = {np.linalg.norm(curv_vec):.4f}")
    print(f"Updated conductance = {new_conduct:.4f}")
    print(f"Decision to move: {move}")