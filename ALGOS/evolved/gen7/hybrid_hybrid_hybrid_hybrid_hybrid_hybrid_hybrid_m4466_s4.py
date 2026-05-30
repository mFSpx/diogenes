# DARWIN HAMMER — match 4466, survivor 4
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_endpoint_circ_m1225_s7.py (gen6)
# parent_b: hybrid_hybrid_hybrid_bandit_hybrid_hybrid_korpus_m1956_s2.py (gen6)
# born: 2026-05-29T23:55:56Z

"""Hybrid algorithm merging:
- Parent A: Morphology‑aware SSIM for image similarity, Gaussian beam and Fisher score.
- Parent B: Schoolfield temperature‑dependent developmental rate used to scale the width ε of an
  RBF kernel that operates on MinHash‑based Jaccard distances between text data.

Mathematical bridge:
    ε(T) = ε₀ · ρ(T)   where ρ(T) is the Schoolfield developmental rate.
    k_text = exp( - (ε(T)·d_J)^2 )   with d_J = 1 – Jaccard(minhash_i, minhash_j).
    The final hybrid similarity combines the image SSIM (optionally scaled by morphology)
    and the text‑RBF similarity via a weighted sum.
"""

import math
import random
import sys
import pathlib
from dataclasses import dataclass, field
from typing import Tuple, List, Set, Sequence, Any

import numpy as np

# ==== Parent A core definitions =================================================

Point = Tuple[float, float]

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

def fisher_score(theta: float, center: float, width: float,
                 sphericity: float, eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width, sphericity), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

def ssim(x: np.ndarray, y: np.ndarray,
         dynamic_range: float = 255.0,
         k1: float = 0.01, k2: float = 0.03,
         morphology: Morphology = None) -> float:
    """Structural Similarity Index, optionally scaled by morphology."""
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

# ==== Parent B core definitions =================================================

R_CAL = 1.987  # cal mol^-1 K^-1
K25 = 298.15   # reference temperature (Kelvin)

@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0          # rate at 25 °C (K25)
    delta_h_activation: float = 12_000.0   # J mol⁻¹
    t_low: float = 283.15        # lower temperature bound (K)
    t_high: float = 307.15       # upper temperature bound (K)
    delta_h_low: float = -45_000.0
    delta_h_high: float = 65_000.0
    r_cal: float = R_CAL

def c_to_k(celsius: float) -> float:
    """Convert Celsius to Kelvin."""
    return celsius + 273.15

def developmental_rate(temp_k: float,
                       params: SchoolfieldParams = SchoolfieldParams()) -> float:
    """Schoolfield temperature‑dependent developmental rate ρ(T)."""
    if temp_k <= 0:
        raise ValueError("temperature must be Kelvin‑positive")

    def exp_term(delta_h: float, t: float) -> float:
        return math.exp(-delta_h / (params.r_cal * t))

    num = params.rho_25 * exp_term(params.delta_h_activation, temp_k)
    denom = (1.0
             + exp_term(params.delta_h_low, params.t_low)
             + exp_term(params.delta_h_high, params.t_high))
    return num / denom

# Simple MinHash implementation (deterministic for reproducibility)

def _hash_factory(seed: int):
    rnd = random.Random(seed)
    a = rnd.randint(1, sys.maxsize - 1)
    b = rnd.randint(0, sys.maxsize - 1)
    prime = 4294967311  # a large prime > 2**32
    return lambda x: (a * hash(x) + b) % prime

def minhash_signature(tokens: Sequence[Any], num_perm: int = 128) -> List[int]:
    """Compute a MinHash signature for a token sequence."""
    if not tokens:
        raise ValueError("token sequence must be non‑empty")
    sig = [sys.maxsize] * num_perm
    for i in range(num_perm):
        h = _hash_factory(i)
        for token in tokens:
            hv = h(token)
            if hv < sig[i]:
                sig[i] = hv
    return sig

def jaccard_estimate(sig1: List[int], sig2: List[int]) -> float:
    """Estimate Jaccard similarity from two MinHash signatures."""
    if len(sig1) != len(sig2):
        raise ValueError("signatures must have equal length")
    matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
    return matches / len(sig1)

def rbf_kernel(distance: float, epsilon: float) -> float:
    """Gaussian RBF kernel."""
    return math.exp(-(epsilon * distance) ** 2)

# ==== Hybrid functions =========================================================

def temperature_scaled_epsilon(base_epsilon: float, temp_celsius: float) -> float:
    """Scale the RBF width ε by the developmental rate at a given temperature."""
    temp_k = c_to_k(temp_celsius)
    rho = developmental_rate(temp_k)
    return base_epsilon * rho

def text_similarity_rbf(text_a: str, text_b: str,
                       base_epsilon: float,
                       temp_celsius: float,
                       num_perm: int = 128) -> float:
    """Compute RBF similarity between two texts using MinHash Jaccard distance."""
    tokens_a = text_a.lower().split()
    tokens_b = text_b.lower().split()
    sig_a = minhash_signature(tokens_a, num_perm)
    sig_b = minhash_signature(tokens_b, num_perm)
    jacc = jaccard_estimate(sig_a, sig_b)
    distance = 1.0 - jacc
    eps = temperature_scaled_epsilon(base_epsilon, temp_celsius)
    return rbf_kernel(distance, eps)

def hybrid_similarity(image_x: np.ndarray, image_y: np.ndarray,
                      text_x: str, text_y: str,
                      temp_celsius: float,
                      morphology: Morphology = None,
                      weight_image: float = 0.5,
                      weight_text: float = 0.5,
                      base_epsilon: float = 1.0) -> float:
    """
    Unified similarity measure:
        S = w_img * SSIM_image + w_txt * RBF_text
    where SSIM may be scaled by morphology and the RBF width ε depends on temperature.
    """
    img_sim = ssim(image_x, image_y, morphology=morphology)
    txt_sim = text_similarity_rbf(text_x, text_y,
                                  base_epsilon=base_epsilon,
                                  temp_celsius=temp_celsius)
    # Normalise weights
    total = weight_image + weight_text
    w_img = weight_image / total
    w_txt = weight_text / total
    return w_img * img_sim + w_txt * txt_sim

def fisher_image_score(image: np.ndarray,
                       theta_center: float,
                       width: float,
                       morphology: Morphology) -> float:
    """
    Apply the Fisher information score across an image interpreted as a
    one‑dimensional signal of intensity values.
    """
    spher = sphericity_index(morphology.length,
                             morphology.width,
                             morphology.height)
    # Flatten image to 1‑D for demonstration
    flattened = image.ravel()
    scores = [fisher_score(float(val), theta_center, width, spher) for val in flattened]
    return float(np.mean(scores))

# ==== Smoke test ===============================================================

if __name__ == "__main__":
    # Create two simple 8‑bit grayscale images
    img1 = np.full((64, 64), 120, dtype=np.uint8)
    img2 = np.full((64, 64), 125, dtype=np.uint8)

    # Slightly modify img2 to give a non‑trivial SSIM
    img2[32, 32] = 200

    # Text samples
    txt1 = "The quick brown fox jumps over the lazy dog"
    txt2 = "A fast dark-colored fox leaps above a sleepy canine"

    # Morphology example
    morph = Morphology(length=5.0, width=3.0, height=2.0, mass=0.8)

    # Temperature in Celsius
    temperature = 22.0

    # Compute hybrid similarity
    sim = hybrid_similarity(img1, img2, txt1, txt2,
                            temp_celsius=temperature,
                            morphology=morph,
                            weight_image=0.6,
                            weight_text=0.4,
                            base_epsilon=0.8)

    print(f"Hybrid similarity: {sim:.4f}")

    # Demonstrate Fisher image score
    fisher_score_val = fisher_image_score(img1, theta_center=100.0,
                                          width=15.0, morphology=morph)
    print(f"Fisher image score: {fisher_score_val:.6f}")