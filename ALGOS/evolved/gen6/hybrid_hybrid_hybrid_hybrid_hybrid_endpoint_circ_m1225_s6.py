# DARWIN HAMMER — match 1225, survivor 6
# gen: 6
# parent_a: hybrid_hybrid_hybrid_minimu_hybrid_serpentina_se_m872_s3.py (gen5)
# parent_b: hybrid_endpoint_circuit_bre_hybrid_hybrid_liquid_m147_s0.py (gen3)
# born: 2026-05-29T23:34:47Z

import math
import random
import sys
import pathlib
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Tuple, List, Dict, Any

import numpy as np

# ----------------------------------------------------------------------
# Parent A – geometric & similarity utilities
# ----------------------------------------------------------------------

Point = Tuple[float, float]
Edge = Tuple[str, str]


def euclidean(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / max(length, width, height)


def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


def gaussian_beam(theta: float, center: float, width: float, sphericity: float) -> float:
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return sphericity * math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, sphericity: float,
                 eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width, sphericity), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def ssim(x: np.ndarray, y: np.ndarray,
         dynamic_range: float = 255.0,
         k1: float = 0.01,
         k2: float = 0.03,
         morphology: Morphology = None) -> float:
    """Structural Similarity Index (SSIM) with optional morphology scaling."""
    if x.shape != y.shape:
        raise ValueError("Input images must have the same dimensions")
    C1 = (k1 * dynamic_range) ** 2
    C2 = (k2 * dynamic_range) ** 2

    if morphology:
        scale = sphericity_index(morphology.length,
                                 morphology.width,
                                 morphology.height)
        C1 *= scale
        C2 *= scale

    mu_x = x.mean()
    mu_y = y.mean()
    sigma_x = x.var()
    sigma_y = y.var()
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean()

    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    return float(numerator / denominator)


# ----------------------------------------------------------------------
# Parent B – endpoint circuit‑breaker & LTC diffusion
# ----------------------------------------------------------------------

def now_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class EndpointCircuitBreaker:
    """Simple circuit‑breaker that trips after `failure_threshold` failures."""
    def __init__(self, failure_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.last_state = "CLOSED"   # CLOSED = allow traffic, OPEN = block

    def record_success(self):
        self.failures = max(0, self.failures - 1)
        self.last_state = "CLOSED"

    def record_failure(self):
        self.failures += 1
        if self.failures >= self.failure_threshold:
            self.last_state = "OPEN"

    def is_closed(self) -> bool:
        return self.last_state == "CLOSED"


def minhash_signature(tokens: List[str], num_perm: int = 64) -> List[int]:
    """
    Very lightweight MinHash approximation: hash each token with a different
    seed and keep the minimum per permutation.
    """
    max_hash = (1 << 32) - 1
    signatures = [max_hash] * num_perm
    for token in tokens:
        token_bytes = token.encode('utf-8')
        for i in range(num_perm):
            h = hashlib.blake2b(token_bytes, digest_size=4, person=bytes([i])).digest()
            hv = int.from_bytes(h, 'big')
            if hv < signatures[i]:
                signatures[i] = hv
    return signatures


def jaccard_estimate(sig_a: List[int], sig_b: List[int]) -> float:
    """Estimate Jaccard similarity from two MinHash signatures."""
    if len(sig_a) != len(sig_b):
        raise ValueError("Signatures must be of equal length")
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / len(sig_a)


def ltc_diffusion_step(x: np.ndarray,
                       I: np.ndarray,
                       similarity: float,
                       T: int = 10,
                       alpha_curve: List[float] = None) -> Tuple[np.ndarray, int]:
    """
    Perform one LTC diffusion step.

    Parameters
    ----------
    x : np.ndarray
        Current state vector.
    I : np.ndarray
        Input vector (same shape as x).
    similarity : float
        Similarity in [0,1] that drives the diffusion timestep.
    T : int
        Maximum diffusion timesteps.
    alpha_curve : List[float] | None
        Pre‑computed α values for each possible timestep. If None, a simple
        linear schedule α[t] = t / T is used.

    Returns
    -------
    x_new : np.ndarray
        Updated state after diffusion and noise injection.
    t_i : int
        Chosen diffusion timestep.
    """
    if not (0.0 <= similarity <= 1.0):
        raise ValueError("similarity must be within [0, 1]")

    t_i = int(round((1.0 - similarity) * T))
    t_i = max(0, min(T, t_i))

    if alpha_curve is None:
        alpha_curve = [(i / T) for i in range(T + 1)]

    alpha = alpha_curve[t_i]

    # Noise injection
    epsilon = np.random.normal(size=I.shape)
    x_noisy = np.sqrt(alpha) * I + np.sqrt(1.0 - alpha) * epsilon

    # Simple LTC dynamics: dx/dt = -(1/τ) * x + (1/τ) * x_noisy
    # Choose τ = 1 for convenience; discretise with Euler step Δt = 1.
    x_new = x + (x_noisy - x) / (1 + t_i)

    return x_new, t_i


# ----------------------------------------------------------------------
# Hybrid structures & operations
# ----------------------------------------------------------------------

@dataclass
class EndpointState:
    """Container for an endpoint's mutable state."""
    name: str
    x: np.ndarray                     # latent state vector
    breaker: EndpointCircuitBreaker = field(default_factory=EndpointCircuitBreaker)
    signature_history: List[int] = field(default_factory=list)  # accumulated MinHash


def compute_similarity(reference_img: np.ndarray,
                       current_img: np.ndarray,
                       morphology: Morphology = None) -> float:
    """
    Use SSIM (Parent A) as the similarity metric required by Parent B.
    """
    return ssim(reference_img, current_img, morphology=morphology)


def hybrid_update(endpoint_state: EndpointState,
                   reference_img: np.ndarray,
                   current_img: np.ndarray,
                   T: int = 10) -> Tuple[EndpointState, np.ndarray, int]:
    """
    Perform a hybrid update.

    Parameters
    ----------
    endpoint_state : EndpointState
        Current endpoint state.
    reference_img : np.ndarray
        Reference image.
    current_img : np.ndarray
        Current image.
    T : int
        Maximum diffusion timesteps.

    Returns
    -------
    endpoint_state_new : EndpointState
        Updated endpoint state.
    x_new : np.ndarray
        Updated state after diffusion and noise injection.
    t_i : int
        Chosen diffusion timestep.
    """
    similarity = compute_similarity(reference_img, current_img)
    t_i = int(round((1.0 - similarity) * T))
    t_i = max(0, min(T, t_i))

    alpha_curve = [(i / T) for i in range(T + 1)]
    alpha = alpha_curve[t_i]

    # Noise injection
    epsilon = np.random.normal(size=current_img.shape)
    x_noisy = np.sqrt(alpha) * current_img + np.sqrt(1.0 - alpha) * epsilon

    # Simple LTC dynamics: dx/dt = -(1/τ) * x + (1/τ) * x_noisy
    # Choose τ = 1 for convenience; discretise with Euler step Δt = 1.
    x_new = endpoint_state.x + (x_noisy - endpoint_state.x) / (1 + t_i)

    # Update circuit-breaker state
    if np.allclose(x_new, endpoint_state.x):
        endpoint_state.breaker.record_success()
    else:
        endpoint_state.breaker.record_failure()

    endpoint_state_new = EndpointState(name=endpoint_state.name,
                                        x=x_new,
                                        breaker=endpoint_state.breaker)

    return endpoint_state_new, x_new, t_i