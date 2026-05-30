# DARWIN HAMMER — match 1225, survivor 5
# gen: 6
# parent_a: hybrid_hybrid_hybrid_minimu_hybrid_serpentina_se_m872_s3.py (gen5)
# parent_b: hybrid_endpoint_circuit_bre_hybrid_hybrid_liquid_m147_s0.py (gen3)
# born: 2026-05-29T23:34:47Z

"""Hybrid Image‑Similarity‑Driven Liquid‑Time‑Constant Diffusion with Circuit‑Breaker

Parents
-------
- **Parent A** (`hybrid_hybrid_hybrid_minimu_hybrid_serpentina_se_m872_s3.py`):
  Provides geometric utilities, a Gaussian‑beam based Fisher score and an
  SSIM implementation that can incorporate a `Morphology` object.

- **Parent B** (`hybrid_endpoint_circuit_bre_hybrid_hybrid_liquid_m147_s0.py`):
  Implements per‑endpoint circuit‑breaker logic and a Liquid‑Time‑Constant
  (LTC) recurrent cell whose diffusion timestep is driven by a similarity
  measure `s`.

Mathematical Bridge
-------------------
For each endpoint we treat an incoming *image* as the observable state.
The Structural Similarity Index (SSIM) from Parent A is used as the similarity
` s_e ` required by Parent B.  This similarity controls:

1. The diffusion timestep `t_i = round((1‑s_e) * T)`.
2. The noisy injection `x_noisy = √α[t_i]·I + √(1‑α[t_i])·ε`.
3. The loss weighting `λ(t_i)=1/(1+t_i)` that feeds back into the
   circuit‑breaker gate `g_e`.

Thus the SSIM metric becomes the *bridge* that fuses the two topologies into a
single closed‑loop system: image similarity → diffusion dynamics → gated update
→ circuit‑breaker state.
"""

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
    x_noisy = math.sqrt(alpha) * I + math.sqrt(1.0 - alpha) * epsilon

    # Simple LTC dynamics: dx/dt = -(1/τ) * x + (1/τ) * x_noisy
    # Choose τ = 1 for convenience; discretise with Euler step Δt = 1.
    x_new = x - (x - x_noisy)

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
    Use SSIM (Parent A) as the similarity metric required by the LTC diffusion.
    """
    return ssim(reference_img, current_img, morphology=morphology)


def hybrid_update(endpoint: EndpointState,
                  input_vec: np.ndarray,
                  reference_img: np.ndarray,
                  current_img: np.ndarray,
                  morphology: Morphology,
                  tokens: List[str],
                  target_vec: np.ndarray,
                  T: int = 10) -> None:
    """
    One hybrid iteration for a single endpoint.

    1. Compute SSIM similarity between reference and current images.
    2. Update MinHash signature history with `tokens`.
    3. Derive diffusion timestep from similarity and perform LTC step.
    4. Compute a weighted loss; feed it back to the circuit‑breaker.
    """
    # 1. Image similarity (bridge)
    s = compute_similarity(reference_img, current_img, morphology=morphology)

    # 2. MinHash update (keeps a running signature)
    new_sig = minhash_signature(tokens)
    # For simplicity we keep only the most recent signature
    endpoint.signature_history = new_sig

    # 3. LTC diffusion driven by similarity
    endpoint.x, t_i = ltc_diffusion_step(endpoint.x, input_vec, similarity=s, T=T)

    # 4. Loss weighting and circuit‑breaker feedback
    lambda_t = 1.0 / (1 + t_i)
    loss = lambda_t * np.mean((endpoint.x - target_vec) ** 2)

    # Threshold for success/failure – arbitrary but deterministic
    if loss < 0.01:
        endpoint.breaker.record_success()
    else:
        endpoint.breaker.record_failure()


def process_pool(endpoints: List[EndpointState],
                 input_vectors: List[np.ndarray],
                 reference_imgs: List[np.ndarray],
                 current_imgs: List[np.ndarray],
                 morphologies: List[Morphology],
                 token_batches: List[List[str]],
                 target_vectors: List[np.ndarray]) -> Dict[str, Any]:
    """
    Iterate over a pool of endpoints, applying `hybrid_update` to each.
    Returns a summary dictionary with breaker states and final latent vectors.
    """
    summary = {}
    for i, ep in enumerate(endpoints):
        hybrid_update(
            endpoint=ep,
            input_vec=input_vectors[i],
            reference_img=reference_imgs[i],
            current_img=current_imgs[i],
            morphology=morphologies[i],
            tokens=token_batches[i],
            target_vec=target_vectors[i]
        )
        summary[ep.name] = {
            "breaker_state": ep.breaker.last_state,
            "failures": ep.breaker.failures,
            "latent_norm": float(np.linalg.norm(ep.x))
        }
    return summary


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # deterministic seed for reproducibility
    np.random.seed(42)
    random.seed(42)

    # Create a tiny pool of 2 endpoints
    dim = 8  # latent dimension
    pool = [
        EndpointState(name="engine_A", x=np.zeros(dim)),
        EndpointState(name="engine_B", x=np.ones(dim) * 0.5)
    ]

    # Dummy inputs
    input_vecs = [np.random.rand(dim) for _ in pool]
    target_vecs = [np.zeros(dim) for _ in pool]

    # Generate synthetic grayscale images (64x64) with slight variations
    base_image = (np.random.rand(64, 64) * 255).astype(np.uint8)
    reference_imgs = [base_image for _ in pool]
    current_imgs = [
        base_image + np.random.randint(-5, 6, size=base_image.shape, dtype=np.int16)
        for _ in pool
    ]
    # Clip to valid range
    current_imgs = [np.clip(img, 0, 255).astype(np.uint8) for img in current_imgs]

    morphologies = [
        Morphology(length=10.0, width=5.0, height=2.0, mass=1.2),
        Morphology(length=8.0, width=4.0, height=3.0, mass=0.9)
    ]

    token_batches = [
        ["alpha", "beta", "gamma"],
        ["delta", "epsilon", "zeta"]
    ]

    summary = process_pool(
        endpoints=pool,
        input_vectors=input_vecs,
        reference_imgs=reference_imgs,
        current_imgs=current_imgs,
        morphologies=morphologies,
        token_batches=token_batches,
        target_vectors=target_vecs
    )

    for name, info in summary.items():
        print(f"{name}: breaker={info['breaker_state']}, "
              f"failures={info['failures']}, "
              f"latent_norm={info['latent_norm']:.4f}")

    sys.exit(0)