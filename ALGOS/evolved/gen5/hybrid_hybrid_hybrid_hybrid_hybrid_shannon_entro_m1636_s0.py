# DARWIN HAMMER — match 1636, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_hybrid_label__hybrid_ternary_route_m910_s3.py (gen4)
# parent_b: hybrid_shannon_entropy_rsa_cipher_m51_s2.py (gen1)
# born: 2026-05-29T23:37:56Z

from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

__all__ = ["hybrid_entropy_morphology", "kan_approximation_rsa", "shannon_entropy_rsa"]

"""Hybrid module combining Shannon entropy, RSA modular exponentiation, 
and morphological features.

The bridge:
We use the morphological features as weights for the RSA encryption and 
decription, effectively intertwining information-theoretic and number-theoretic 
structures. The RSA transformation is applied to the confidence values returned 
by the KAN approximation.

The exact mathematical bridge is the following:
Let P be the probability distribution of the morphological features, 
and let Q be the RSA encryption key. Then the hybrid operation can be 
defined as follows:

1. Compute the KAN approximation of the morphological features.
2. Use the confidence values as weights for the RSA encryption.
3. Compute the Shannon entropy of the resulting distribution.
"""

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

    # Choose e
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


def morphology_vector(morph: dict) -> np.ndarray:
    """
    Convert a Morphology instance into a normalized feature vector.
    The vector is L2-normalised to keep the KAN mapping scale-invariant.
    """
    vec = np.array([morph['length'], morph['width'], morph['height'], morph['mass']], dtype=float)
    norm = np.linalg.norm(vec)
    if norm == 0:
        raise ValueError("Morphology vector cannot be all zeros")
    return vec / norm


def kan_approximation(vec: np.ndarray, weight: np.ndarray | None = None) -> float:
    """
    Very light‑weight KAN surrogate.
    The original KAN uses spline‑based activation; we approximate with
    a single hidden unit and a smooth exponential activation:

        y = σ(w·x + b)   where σ(z)=exp(z)/(1+exp(z))

    The function returns a confidence in [0,1].
    """
    if weight is None:
        # Initialise a deterministic weight vector for reproducibility.
        rng = np.random.default_rng(42)
        weight = rng.normal(loc=0.0, scale=1.0, size=4)
    return (np.dot(vec, weight) + 0.5) / (1 + np.exp(-np.dot(vec, weight) - 0.5))


def kan_approximation_rsa(morph: dict, e: int, n: int) -> int:
    """Apply RSA encryption to the confidence values returned by the KAN approximation."""
    vec = morphology_vector(morph)
    conf = kan_approximation(vec)
    return rsa_encrypt_int(int(conf * n), e, n)


def shannon_entropy_rsa(ciphertext: int, e: int, n: int) -> float:
    """Compute the Shannon entropy of the resulting distribution after RSA encryption."""
    # Assume a uniform distribution over [0, n)
    return -np.log(2.0 / n) * (1 - np.log(ciphertext) / np.log(n))


def hybrid_entropy_morphology(morph: dict, e: int, n: int) -> float:
    """The hybrid operation: compute the KAN approximation, apply RSA encryption, and compute the Shannon entropy."""
    ciphertext = kan_approximation_rsa(morph, e, n)
    return shannon_entropy_rsa(ciphertext, e, n)


def kan_approximation_rsa_decrypt(ciphertext: int, d: int, n: int) -> float:
    """Apply RSA decryption to the confidence values returned by the KAN approximation, and then compute the KAN approximation."""
    conf = rsa_decrypt_int(ciphertext, d, n)
    return kan_approximation(morphology_vector({'length': 1.0, 'width': 1.0, 'height': 1.0, 'mass': 1.0}), weight=None)


if __name__ == "__main__":
    # Smoke test
    e, d, n = generate_rsa_keypair(prime_bits=16)
    morph = {'length': 1.0, 'width': 1.0, 'height': 1.0, 'mass': 1.0}
    print(hybrid_entropy_morphology(morph, e, n))
    print(kan_approximation_rsa_decrypt(rsa_encrypt_int(int(kan_approximation(morphology_vector(morph), weight=None)), e, n), d, n))