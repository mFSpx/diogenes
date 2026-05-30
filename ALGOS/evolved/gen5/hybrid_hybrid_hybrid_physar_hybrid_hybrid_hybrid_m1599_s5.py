# DARWIN HAMMER — match 1599, survivor 5
# gen: 5
# parent_a: hybrid_hybrid_physarum_netw_hybrid_hybrid_infota_m875_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_krampu_hybrid_hybrid_hard_t_m1107_s0.py (gen4)
# born: 2026-05-29T23:37:50Z

import math
import hashlib
import numpy as np
from typing import List, Dict, Tuple

# ----------------------------------------------------------------------
# Core Physarum utilities (Parent A)
# ----------------------------------------------------------------------
def flux(conductance: float, edge_length: float,
         pressure_a: float, pressure_b: float,
         eps: float = 1e-12) -> float:
    """
    Physarum flux on an edge.

    Parameters
    ----------
    conductance : float
        Edge conductance (non‑negative).
    edge_length : float
        Positive edge length.
    pressure_a, pressure_b : float
        Pressures at the two endpoints.
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    float
        Net flux from a to b.
    """
    if edge_length <= 0:
        raise ValueError('edge_length must be positive')
    return conductance / max(edge_length, eps) * (pressure_a - pressure_b)


def update_conductance(conductance: float, q: float,
                       dt: float = 1.0, gain: float = 1.0,
                       decay: float = 0.05,
                       gain_cap: float = 10.0) -> float:
    """
    Physarum conductance dynamics with a hard cap on the effective gain
    to avoid blow‑up when curvature contributions become large.

    Parameters
    ----------
    conductance : float
        Current conductance.
    q : float
        Flux magnitude.
    dt : float
        Time step.
    gain : float
        Base gain factor.
    decay : float
        Linear decay coefficient.
    gain_cap : float
        Upper bound for the effective gain.

    Returns
    -------
    float
        Updated conductance (non‑negative).
    """
    if dt < 0 or decay < 0:
        raise ValueError('dt and decay must be non‑negative')
    effective_gain = min(gain, gain_cap)
    new_val = conductance + dt * (effective_gain * abs(q) - decay * conductance)
    return max(0.0, new_val)


def information_density(pressure: float) -> float:
    """
    Entropy‑style information density derived from pressure.
    The ``+1`` ensures the argument of the log stays >0.
    """
    return math.log(pressure + 1.0)


# ----------------------------------------------------------------------
# Feature & curvature utilities (Parent B) – deeper integration
# ----------------------------------------------------------------------
def _deterministic_rng(seed_str: str) -> np.random.Generator:
    """
    Produce a reproducible NumPy random generator from an arbitrary string.
    Uses SHA‑256 to obtain a 256‑bit seed, then folds it into a 64‑bit integer.
    """
    digest = hashlib.sha256(seed_str.encode('utf-8')).digest()
    # Take the first 8 bytes as a uint64 seed
    seed = int.from_bytes(digest[:8], 'big')
    return np.random.default_rng(seed)


def extract_full_features(text: str) -> Dict[str, float]:
    """
    Deterministically generate a high‑dimensional feature dictionary from raw text.
    The distribution mimics correlated groups but is fully reproducible.
    """
    rng = _deterministic_rng(text)
    features: Dict[str, float] = {}
    groups = {
        "operator": 0.1,
        "psyche": 0.2,
        "resilience": 0.3,
        "rainmaker": 0.05,
        "telemetry": 0.07,
    }
    for prefix, base in groups.items():
        for i in range(3):
            key = f"{prefix}_{i}_ratio" if prefix != "psyche" else f"{prefix}_{i}_entropy"
            # Correlated values: base + small random perturbation
            features[key] = base + 0.02 * rng.random()
    return features


def calculate_ricci_curvature(features: Dict[str, float]) -> Dict[str, float]:
    """
    Simplified Ollivier‑Ricci curvature: each feature is scaled by a
    group‑specific factor. The factors are stored in a lookup table.
    """
    group_factors = {
        "operator": 0.12,
        "psyche": 0.22,
        "resilience": 0.33,
        "rainmaker": 0.08,
        "telemetry": 0.09,
    }
    curvature: Dict[str, float] = {}
    for name, val in features.items():
        group = name.split('_')[0]
        factor = group_factors.get(group, 0.05)
        curvature[name] = val * factor
    return curvature


def curvature_vector(curv: Dict[str, float]) -> np.ndarray:
    """
    Convert curvature dict to a dense, L2‑normalized vector.
    Normalization prevents the curvature norm from dominating the dynamics.
    """
    if not curv:
        return np.zeros(0, dtype=np.float64)
    keys = sorted(curv.keys())
    vec = np.array([curv[k] for k in keys], dtype=np.float64)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


# ----------------------------------------------------------------------
# Min‑hash utilities (shared)
# ----------------------------------------------------------------------
def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def signature(tokens: List[str], k: int = 128) -> List[int]:
    """
    Min‑hash style signature of a token list.
    Returns a list of length ``k`` where each entry is the minimal hash
    of the token set under a distinct seed.
    """
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [2**64 - 1] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]


def similarity(sig_a: List[int], sig_b: List[int]) -> float:
    """
    Jaccard‑like similarity of two signatures.
    """
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have equal length")
    if not sig_a:
        raise ValueError("signatures must not be empty")
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)


# ----------------------------------------------------------------------
# Deeply fused core functions
# ----------------------------------------------------------------------
def hybrid_flux_curvature(conductance: float, edge_length: float,
                          pressure_a: float, pressure_b: float,
                          feature_text: str) -> Tuple[float, np.ndarray]:
    """
    Compute physarum flux and a curvature‑scaled feature vector that is
    *modulated per‑feature* by the information density.

    Returns
    -------
    q : float
        Net flux on the edge.
    weighted_vec : np.ndarray
        Vector where each component = I * curvature_i.
    """
    q = flux(conductance, edge_length, pressure_a, pressure_b)

    # Mid‑edge pressure as a proxy for local information density
    mid_pressure = (pressure_a + pressure_b) / 2.0
    I = information_density(mid_pressure)

    raw_features = extract_full_features(feature_text)
    curv = calculate_ricci_curvature(raw_features)

    # Apply information density *per component* before normalisation
    vec = np.array([I * curv[k] for k in sorted(curv.keys())], dtype=np.float64)

    # Normalise to keep magnitudes comparable across edges
    norm = np.linalg.norm(vec)
    weighted_vec = vec / norm if norm > 0 else vec
    return q, weighted_vec


def hybrid_conductance_update(conductance: float, q: float,
                              curvature_vec: np.ndarray,
                              dt: float = 1.0,
                              gain: float = 1.0,
                              decay: float = 0.05,
                              curvature_gain: float = 0.3,
                              gain_cap: float = 10.0) -> float:
    """
    Update conductance using physarum flux *and* the L2‑norm of the
    curvature vector. The curvature contribution is bounded and blended
    with the base gain.

    Parameters
    ----------
    curvature_vec : np.ndarray
        Normalised curvature vector from ``hybrid_flux_curvature``.
    curvature_gain : float
        Scaling factor for the curvature contribution.
    gain_cap : float
        Upper bound for the effective gain (prevents runaway growth).

    Returns
    -------
    float
        Updated conductance.
    """
    curvature_norm = np.linalg.norm(curvature_vec)  # ≤ 1 due to normalisation
    effective_gain = min(gain + curvature_gain * curvature_norm, gain_cap)
    return update_conductance(conductance, q, dt=dt,
                              gain=effective_gain, decay=decay,
                              gain_cap=gain_cap)


def hybrid_expected_entropy(sig_a: List[int], sig_b: List[int],
                            curvature_vec: np.ndarray,
                            p_hit: float = 0.5,
                            curvature_weight: float = 0.05) -> float:
    """
    Combine infotaxis expected entropy with a curvature‑aware correction.
    The correction *reduces* entropy proportionally to the alignment of the
    curvature vector with the similarity direction, encouraging exploration
    when belief is weak.

    Parameters
    ----------
    curvature_weight : float
        Scaling of the curvature‑based penalty/bonus.

    Returns
    -------
    float
        Expected entropy (lower is better).
    """
    # Binary entropy of a Bernoulli trial with success probability p_hit
    base_entropy = -p_hit * math.log(p_hit + 1e-12) - (1 - p_hit) * math.log(1 - p_hit + 1e-12)

    sim = similarity(sig_a, sig_b)

    # Curvature term: we penalise high norm *and* reward alignment with similarity
    curvature_term = curvature_weight * np.linalg.norm(curvature_vec) * (1 - sim)

    return base_entropy - sim + curvature_term


def hybrid_decision(sig_current: List[int], sig_candidate: List[int],
                    curvature_vec: np.ndarray,
                    threshold: float = 0.0,
                    curvature_bias: float = 0.1) -> bool:
    """
    Decide whether to move to a candidate state.

    Decision rule:
        move if (similarity gain) - (curvature penalty) > threshold

    Parameters
    ----------
    curvature_bias : float
        Weight of the curvature penalty in the decision.
    threshold : float
        Decision threshold; default 0 means any positive net gain triggers a move.

    Returns
    -------
    bool
        True if the candidate should be adopted.
    """
    sim = similarity(sig_current, sig_candidate)
    curvature_penalty = curvature_bias * np.linalg.norm(curvature_vec)
    return (sim - curvature_penalty) > threshold


# ----------------------------------------------------------------------
# Optional: lightweight state holder for iterative simulations
# ----------------------------------------------------------------------
class HybridPhysarumInfotaxis:
    """
    Encapsulates the fused dynamics, exposing a simple step interface.
    """

    def __init__(self,
                 conductance: float,
                 edge_length: float,
                 init_pressure_a: float,
                 init_pressure_b: float,
                 feature_text: str,
                 dt: float = 1.0):
        self.conductance = conductance
        self.edge_length = edge_length
        self.pressure_a = init_pressure_a
        self.pressure_b = init_pressure_b
        self.feature_text = feature_text
        self.dt = dt

    def step(self,
             sig_current: List[int],
             sig_candidate: List[int],
             p_hit: float = 0.5) -> Tuple[bool, float]:
        """
        Perform one simulation step:
          1. Compute flux and weighted curvature.
          2. Update conductance.
          3. Evaluate expected entropy.
          4. Decide whether to adopt the candidate signature.

        Returns
        -------
        moved : bool
            Whether the candidate was accepted.
        entropy : float
            Expected entropy after the step.
        """
        q, curv_vec = hybrid_flux_curvature(
            self.conductance,
            self.edge_length,
            self.pressure_a,
            self.pressure_b,
            self.feature_text,
        )

        self.conductance = hybrid_conductance_update(
            self.conductance,
            q,
            curv_vec,
            dt=self.dt,
        )

        entropy = hybrid_expected_entropy(
            sig_current,
            sig_candidate,
            curv_vec,
            p_hit=p_hit,
        )

        moved = hybrid_decision(sig_current, sig_candidate, curv_vec)

        # If we move, we simulate a pressure update (placeholder dynamics)
        if moved:
            delta = 0.1 * q  # simple heuristic: shift pressures proportionally
            self.pressure_a += delta
            self.pressure_b -= delta

        return moved, entropy

# End of improved hybrid implementation.