# DARWIN HAMMER — match 36, survivor 0
# gen: 2
# parent_a: hybrid_shannon_entropy_rsa_cipher_m51_s2.py (gen1)
# parent_b: sparse_wta.py (gen0)
# born: 2026-05-29T23:26:23Z

"""Hybrid module fusing Sparse Winner-Take-All (WTA) and Hybrid Shannon Entropy RSA Cipher.

The bridge:
Sparse WTA uses hashing to project high-dimensional vectors onto a lower-dimensional space.
Hybrid Shannon Entropy RSA Cipher applies RSA modular exponentiation to encoded probability distributions,
preserving their total mass modulo n. By representing the WTA output as a probability distribution
and applying the RSA transformation, we intertwine the information-theoretic and number-theoretic structures.

The mathematical interface:
The output of the Sparse WTA algorithm (a binary mask) can be interpreted as a probability distribution
by normalizing it. This distribution can then be encoded as integer masses and transformed using RSA.
The resulting distribution can be renormalized and its Shannon entropy computed.

"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# RSA utilities
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


# ----------------------------------------------------------------------
# Sparse WTA utilities
# ----------------------------------------------------------------------
def expand(values: list[float], m: int, salt: str = '') -> list[float]:
    if m <= 0:
        raise ValueError('m must be positive')
    out = [0.0] * m
    for i, v in enumerate(values):
        for r in range(3):
            h = hash(f'{salt}:{i}:{r}')
            j = h % m
            sign = 1.0 if h & 1 else -1.0
            out[j] += sign * v
    return out


def top_k_mask(values: list[float], k: int) -> list[int]:
    k = max(0, min(k, len(values)))
    winners = {i for i, _ in sorted(enumerate(values), key=lambda x: (-x[1], x[0]))[:k]}
    return [1 if i in winners else 0 for i in range(len(values))]


def hamming(a: list[int], b: list[int]) -> int:
    if len(a) != len(b):
        raise ValueError('vectors must be same length')
    return sum(x != y for x, y in zip(a, b))


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def wta_to_probability_distribution(wta_output: list[int]) -> list[float]:
    total = sum(wta_output)
    return [x / total for x in wta_output]


def encode_probability_distribution(distribution: list[float], Q: int) -> list[int]:
    return [round(p * Q) for p in distribution]


def hybrid_wta_rsa(e: int, n: int, values: list[float], m: int, k: int, Q: int) -> list[float]:
    wta_output = top_k_mask(values, k)
    distribution = wta_to_probability_distribution(wta_output)
    encoded_distribution = encode_probability_distribution(distribution, Q)
    encrypted_distribution = [rsa_encrypt_int(m, e, n) for m in encoded_distribution]
    total = sum(encrypted_distribution)
    return [x / total for x in encrypted_distribution]


def shannon_entropy(distribution: list[float]) -> float:
    return -sum(p * math.log(p, 2) for p in distribution if p > 0)


def hybrid_shannon_entropy(e: int, n: int, values: list[float], m: int, k: int, Q: int) -> Tuple[float, float]:
    encrypted_distribution = hybrid_wta_rsa(e, n, values, m, k, Q)
    original_distribution = wta_to_probability_distribution(top_k_mask(values, k))
    original_entropy = shannon_entropy(original_distribution)
    encrypted_entropy = shannon_entropy(encrypted_distribution)
    return original_entropy, encrypted_entropy


if __name__ == "__main__":
    e, d, n = generate_rsa_keypair()
    values = [random.random() for _ in range(100)]
    m = 10
    k = 5
    Q = 100
    original_entropy, encrypted_entropy = hybrid_shannon_entropy(e, n, values, m, k, Q)
    print(f"Original entropy: {original_entropy:.4f}")
    print(f"Encrypted entropy: {encrypted_entropy:.4f}")