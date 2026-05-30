# DARWIN HAMMER — match 1636, survivor 4
# gen: 5
# parent_a: hybrid_hybrid_hybrid_label__hybrid_ternary_route_m910_s3.py (gen4)
# parent_b: hybrid_shannon_entropy_rsa_cipher_m51_s2.py (gen1)
# born: 2026-05-29T23:37:56Z

"""Hybrid algorithm merging Morphology‑based KAN confidence (Parent A) with
Shannon‑entropy‑preserving RSA transformation (Parent B).

Mathematical bridge:
1. A Morphology instance is turned into a normalized feature vector **v**.
2. A lightweight KAN surrogate yields a confidence c∈[0,1] via a sigmoid
   σ(w·v+b).  This confidence is interpreted as a probability for a label.
3. A set of probabilities **p** is scaled by a common integer factor Q,
   rounded to integer masses **m**, and each mass is encrypted with RSA:
   m′=m^e mod n.  Because RSA is a permutation on ℤₙ⁎, the multiset
   {m′} has the same cardinality as {m}.  Renormalising m′ yields a new
   probability distribution **p̃**.
4. The Shannon entropy H(p) and H(p̃) are computed; their difference ΔH
   quantifies the information‑theoretic distortion introduced by the
   number‑theoretic RSA map.
5. The final hybrid score combines the original KAN confidence with the
   entropy change, e.g.  s = α·c + (1‑α)·(1‑ΔH/H_max).

The module implements this pipeline and provides three public functions
demonstrating the hybrid operation.
"""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Shared data structures (from Parent A)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Morphology:
    """Geometric description of an endpoint/document."""
    length: float
    width: float
    height: float
    mass: float


# ----------------------------------------------------------------------
# Parent A core – KAN surrogate
# ----------------------------------------------------------------------


def morphology_vector(morph: Morphology) -> np.ndarray:
    """L2‑normalised feature vector derived from a Morphology."""
    vec = np.array([morph.length, morph.width, morph.height, morph.mass], dtype=float)
    norm = np.linalg.norm(vec)
    if norm == 0:
        raise ValueError("Morphology vector cannot be all zeros")
    return vec / norm


# deterministic weight & bias for reproducibility
_RNG = np.random.default_rng(42)
_KAN_WEIGHT = _RNG.normal(loc=0.0, scale=1.0, size=4)
_KAN_BIAS = _RNG.normal(loc=0.0, scale=0.5)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Smooth exponential sigmoid σ(z)=exp(z)/(1+exp(z))."""
    return np.exp(z) / (1.0 + np.exp(z))


def kan_confidence(vec: np.ndarray) -> float:
    """
    Lightweight KAN surrogate.
    Returns a confidence in [0,1] for the given feature vector.
    """
    z = np.dot(_KAN_WEIGHT, vec) + _KAN_BIAS
    return float(_sigmoid(z))


# ----------------------------------------------------------------------
# Parent B core – RSA utilities and Shannon entropy
# ----------------------------------------------------------------------


def _egcd(a: int, b: int) -> Tuple[int, int, int]:
    """Extended Euclidean algorithm."""
    if a == 0:
        return b, 0, 1
    g, y, x = _egcd(b % a, a)
    return g, x - (b // a) * y, y


def _modinv(a: int, m: int) -> int:
    """Modular inverse of a modulo m."""
    g, x, _ = _egcd(a, m)
    if g != 1:
        raise ValueError("inverse does not exist")
    return x % m


def generate_rsa_keypair(prime_bits: int = 16) -> Tuple[int, int, int]:
    """Generate a tiny RSA keypair (e, d, n) for demonstration purposes."""
    def _rand_prime(bits: int) -> int:
        while True:
            p = random.getrandbits(bits) | 1
            if all(p % q for q in range(3, int(math.isqrt(p)) + 1, 2)):
                return p

    p = _rand_prime(prime_bits)
    q = _rand_prime(prime_bits)
    n = p * q
    phi = (p - 1) * (q - 1)

    e = 65537
    while math.gcd(e, phi) != 1:
        e += 2
    d = _modinv(e, phi)
    return e, d, n


def rsa_encrypt_int(message: int, e: int, n: int) -> int:
    """RSA encryption of a single integer."""
    if not 0 <= message < n:
        raise ValueError("message must be in [0, n)")
    return pow(message, e, n)


def rsa_decrypt_int(ciphertext: int, d: int, n: int) -> int:
    """RSA decryption of a single integer."""
    if not 0 <= ciphertext < n:
        raise ValueError("ciphertext must be in [0, n)")
    return pow(ciphertext, d, n)


def shannon_entropy(probs: np.ndarray) -> float:
    """Compute Shannon entropy H = -∑ p·log₂(p) (0·log0 treated as 0)."""
    probs = probs[probs > 0]  # filter zeros to avoid log(0)
    return -float(np.sum(probs * np.log2(probs)))


# ----------------------------------------------------------------------
# Hybrid pipeline
# ----------------------------------------------------------------------


def probabilities_from_morphologies(
    morphs: List[Morphology],
) -> Tuple[np.ndarray, List[float]]:
    """
    Convert a list of Morphology objects into a probability distribution.
    Returns the array of probabilities and the list of raw KAN confidences.
    """
    confidences = [kan_confidence(morphology_vector(m)) for m in morphs]
    total = sum(confidences)
    if total == 0:
        raise ValueError("All KAN confidences are zero")
    probs = np.array([c / total for c in confidences], dtype=float)
    return probs, confidences


def rsa_transform_distribution(
    probs: np.ndarray,
    e: int,
    n: int,
    scaling_factor: int = 10_000,
) -> np.ndarray:
    """
    Apply the RSA‑preserving transformation to a probability distribution.

    Steps:
    1. Scale by Q and round to integer masses m_i.
    2. Encrypt each m_i → m'_i = m_i^e mod n.
    3. Renormalise the encrypted masses to a probability vector.
    """
    if scaling_factor <= 0:
        raise ValueError("scaling_factor must be positive")
    masses = np.rint(probs * scaling_factor).astype(int)

    # Ensure every mass is non‑zero and < n (RSA domain)
    masses = np.where(masses == 0, 1, masses)
    masses = np.mod(masses, n - 1) + 1  # map to [1, n-1]

    encrypted = np.array([rsa_encrypt_int(int(m), e, n) for m in masses], dtype=int)

    # Renormalise
    total_enc = encrypted.sum()
    if total_enc == 0:
        raise ValueError("Encrypted masses summed to zero")
    return encrypted.astype(float) / total_enc


def hybrid_score(
    morphs: List[Morphology],
    e: int,
    n: int,
    alpha: float = 0.6,
    scaling_factor: int = 10_000,
) -> List[float]:
    """
    Compute a hybrid confidence score for each Morphology.

    For each item i:
        c_i   = KAN confidence from morphology i
        p_i   = normalized confidence vector over all items
        p̃_i  = RSA‑transformed probability for item i
        ΔH    = |H(p) - H(p̃)|
        s_i   = α·c_i + (1‑α)·(1‑ΔH/H_max)

    Returns the list of hybrid scores s_i in the original order.
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in [0,1]")

    probs, raw_confidences = probabilities_from_morphologies(morphs)
    transformed = rsa_transform_distribution(probs, e, n, scaling_factor)

    H_original = shannon_entropy(probs)
    H_transformed = shannon_entropy(transformed)
    delta_H = abs(H_original - H_transformed)

    # Upper bound for entropy with |morphs| outcomes is log2(|morphs|)
    H_max = math.log2(len(morphs)) if len(morphs) > 1 else 1.0

    entropy_factor = 1.0 - (delta_H / H_max)  # in [0,1]

    hybrid = [
        alpha * c + (1 - alpha) * entropy_factor
        for c in raw_confidences
    ]
    return hybrid


# ----------------------------------------------------------------------
# Demonstration functions
# ----------------------------------------------------------------------


def demo_confidences():
    """Show raw KAN confidences for a synthetic set of Morphologies."""
    sample = [
        Morphology(1.2, 0.5, 0.3, 2.0),
        Morphology(0.8, 0.7, 0.4, 1.5),
        Morphology(1.5, 0.6, 0.2, 2.5),
    ]
    confidences = [kan_confidence(morphology_vector(m)) for m in sample]
    print("Raw KAN confidences:", confidences)


def demo_rsa_entropy():
    """Compute entropy before and after RSA transformation for a probability vector."""
    probs = np.array([0.2, 0.5, 0.3])
    e, d, n = generate_rsa_keypair()
    transformed = rsa_transform_distribution(probs, e, n)
    print("Original probs :", probs)
    print("Transformed probs:", transformed)
    print("Entropy original :", shannon_entropy(probs))
    print("Entropy transformed :", shannon_entropy(transformed))


def demo_hybrid():
    """Run the full hybrid pipeline on a small random dataset."""
    random.seed(0)
    np.random.seed(0)

    # generate random morphologies
    morphs = [
        Morphology(
            length=random.uniform(0.5, 2.0),
            width=random.uniform(0.3, 1.5),
            height=random.uniform(0.1, 0.8),
            mass=random.uniform(1.0, 3.0),
        )
        for _ in range(5)
    ]

    e, d, n = generate_rsa_keypair()
    scores = hybrid_score(morphs, e, n, alpha=0.7)
    print("Hybrid scores:", scores)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    print("=== Demo: Raw KAN confidences ===")
    demo_confidences()
    print("\n=== Demo: RSA entropy transformation ===")
    demo_rsa_entropy()
    print("\n=== Demo: Full hybrid pipeline ===")
    demo_hybrid()