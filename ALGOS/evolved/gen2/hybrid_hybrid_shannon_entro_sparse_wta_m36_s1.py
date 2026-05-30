# DARWIN HAMMER — match 36, survivor 1
# gen: 2
# parent_a: hybrid_shannon_entropy_rsa_cipher_m51_s2.py (gen1)
# parent_b: sparse_wta.py (gen0)
# born: 2026-05-29T23:26:23Z

"""Hybrid module combining Sparse Winner-Take-All (WTA) and Shannon Entropy with RSA.

The bridge:
Sparse WTA operates on high-dimensional similarity vectors, producing sparse binary tags.
Shannon entropy measures information content in probability distributions.
By encoding WTA tags as probability distributions and applying Shannon entropy,
we can quantify the information content of WTA outputs.
RSA encryption can be used to protect these WTA tags, and Shannon entropy can be computed
before and after encryption to analyze the effect of encryption on information content.

The hybrid operation:
1. Apply Sparse WTA to a set of input vectors, producing sparse binary tags.
2. Encode these tags as probability distributions.
3. Apply Shannon entropy to these distributions.
4. Encrypt the probability distributions using RSA.
5. Compute Shannon entropy on the encrypted distributions.

This hybrid system integrates the governing equations of both parents,
using Sparse WTA to produce high-dimensional similarity vectors,
Shannon entropy to quantify information content,
and RSA to protect the WTA tags.
"""

from __future__ import annotations
import math
import random
import sys
from pathlib import Path
import numpy as np

# ----------------------------------------------------------------------
# RSA utilities
# ----------------------------------------------------------------------
def _egcd(a: int, b: int) -> tuple[int, int, int]:
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


def generate_rsa_keypair(prime_bits: int = 16) -> tuple[int, int, int]:
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
            h = hashlib.sha256(f'{salt}:{i}:{r}'.encode()).digest()
            j = int.from_bytes(h[:8], 'big') % m
            sign = 1.0 if h[8] & 1 else -1.0
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
# Shannon Entropy utilities
# ----------------------------------------------------------------------
def shannon_entropy(probabilities: list[float]) -> float:
    """Compute Shannon entropy of a probability distribution."""
    return -sum(p * math.log(p, 2) for p in probabilities if p > 0)


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def hybrid_wta_rsa_shannon(values: list[float], m: int, e: int, n: int) -> tuple[float, float]:
    """Apply Sparse WTA, encode as probability distribution, apply Shannon entropy,
    encrypt with RSA, and compute Shannon entropy on encrypted distribution."""
    expanded = expand(values, m)
    wta_tags = top_k_mask(expanded, 3)
    probabilities = [t / sum(wta_tags) for t in wta_tags]
    shannon_before = shannon_entropy(probabilities)
    encrypted_probabilities = [rsa_encrypt_int(int(p * n), e, n) / n for p in probabilities]
    shannon_after = shannon_entropy(encrypted_probabilities)
    return shannon_before, shannon_after


def hybrid_wta_shannon_rsa_decrypt(values: list[float], m: int, e: int, d: int, n: int) -> tuple[float, float]:
    """Apply Sparse WTA, encode as probability distribution, apply Shannon entropy,
    encrypt with RSA, decrypt, and compute Shannon entropy on decrypted distribution."""
    expanded = expand(values, m)
    wta_tags = top_k_mask(expanded, 3)
    probabilities = [t / sum(wta_tags) for t in wta_tags]
    shannon_before = shannon_entropy(probabilities)
    encrypted_probabilities = [rsa_encrypt_int(int(p * n), e, n) for p in probabilities]
    decrypted_probabilities = [rsa_decrypt_int(c, d, n) / n for c in encrypted_probabilities]
    shannon_after = shannon_entropy(decrypted_probabilities)
    return shannon_before, shannon_after


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    e, d, n = generate_rsa_keypair()
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    m = 10
    shannon_before, shannon_after = hybrid_wta_rsa_shannon(values, m, e, n)
    print(f"Shannon entropy before RSA: {shannon_before}")
    print(f"Shannon entropy after RSA: {shannon_after}")
    shannon_before, shannon_after = hybrid_wta_shannon_rsa_decrypt(values, m, e, d, n)
    print(f"Shannon entropy before RSA (decrypt): {shannon_before}")
    print(f"Shannon entropy after RSA (decrypt): {shannon_after}")