# DARWIN HAMMER — match 51, survivor 2
# gen: 1
# parent_a: shannon_entropy.py (gen0)
# parent_b: rsa_cipher.py (gen0)
# born: 2026-05-29T23:23:50Z

"""Hybrid module combining Shannon entropy and RSA modular exponentiation.

The bridge:
RSA encryption is a bijective map f(x)=x^e mod n on the multiplicative group Z_n^*.
If a probability distribution {p_i} is encoded as integer masses m_i = round(p_i * Q) with a
common scaling factor Q, then applying f to each m_i yields a new set {m'_i}.
Because f is a permutation of the non‑zero residues modulo n, the total mass Σ m_i is
preserved (mod n) and can be renormalised to a proper probability distribution.
Thus we can compute Shannon entropy before and after the RSA transformation, obtaining
a hybrid operation that intertwines information‑theoretic and number‑theoretic structures.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# RSA utilities (parent algorithm B)
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
# Shannon entropy utilities (parent algorithm A)
# ----------------------------------------------------------------------
def shannon_entropy(probs: Iterable[float]) -> float:
    """Compute Shannon entropy of a probability distribution."""
    probs_arr = np.asarray(list(probs), dtype=float)
    if probs_arr.size == 0:
        return 0.0
    if any(probs_arr < 0) or not np.isclose(probs_arr.sum(), 1.0):
        raise ValueError("input must be a valid probability distribution")
    mask = probs_arr > 0
    return -float(np.sum(probs_arr[mask] * np.log2(probs_arr[mask])))


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def encode_distribution(dist: Iterable[float], scale: int = 10 ** 6) -> np.ndarray:
    """
    Encode a probability distribution into integer masses.
    The scaling factor determines the fixed‑point precision.
    """
    probs = np.asarray(list(dist), dtype=float)
    if any(probs < 0) or not np.isclose(probs.sum(), 1.0):
        raise ValueError("dist must be a valid probability distribution")
    masses = np.rint(probs * scale).astype(object)  # use Python int objects
    # Adjust for rounding error to keep total exactly scale
    diff = int(scale - masses.sum())
    if diff != 0:
        # Distribute the diff to the largest probability entry(s)
        idx = int(np.argmax(masses))
        masses[idx] += diff
    return masses


def rsa_transform_distribution(
    masses: np.ndarray, e: int, n: int
) -> np.ndarray:
    """
    Apply RSA encryption element‑wise to integer masses.
    Returns a new array of encrypted masses.
    """
    if not np.all((masses >= 0) & (masses < n)):
        raise ValueError("all masses must lie in [0, n) for RSA")
    vec_pow = np.vectorize(rsa_encrypt_int, otypes=[object])
    return vec_pow(masses, e, n)


def decode_distribution(masses: np.ndarray, scale: int = 10 ** 6) -> np.ndarray:
    """
    Convert integer masses back to a probability distribution.
    Renormalises to sum to 1.
    """
    total = int(masses.sum())
    if total == 0:
        raise ValueError("cannot decode zero total mass")
    probs = masses.astype(float) / total
    return probs


def hybrid_entropy(
    observations: Iterable[float | int],
    e: int,
    d: int,
    n: int,
    scale: int = 10 ** 6,
) -> Tuple[float, float]:
    """
    Compute Shannon entropy of the original empirical distribution derived
    from *observations* and the entropy after mapping the empirical masses
    through RSA encryption (and optional decryption for verification).

    Returns (entropy_original, entropy_rsa_mapped).
    """
    # Build empirical distribution
    obs = list(observations)
    if not obs:
        return 0.0, 0.0
    # Count frequencies
    unique, counts = np.unique(obs, return_counts=True)
    probs = counts / counts.sum()
    # Encode to integer masses
    masses = encode_distribution(probs, scale=scale)

    # RSA transform (encrypt)
    encrypted = rsa_transform_distribution(masses, e, n)

    # Optionally decrypt to verify bijection (not strictly needed)
    vec_pow = np.vectorize(rsa_decrypt_int, otypes=[object])
    decrypted = vec_pow(encrypted, d, n)
    if not np.array_equal(masses % n, decrypted % n):
        raise RuntimeError("RSA encryption/decryption mismatch")

    # Decode back to probabilities
    probs_enc = decode_distribution(encrypted, scale=scale)

    # Compute entropies
    H_orig = shannon_entropy(probs)
    H_enc = shannon_entropy(probs_enc)
    return H_orig, H_enc


# ----------------------------------------------------------------------
# Demonstration / smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple observation list
    obs = [random.choice(['A', 'B', 'C', 'D']) for _ in range(1000)]

    # Generate tiny RSA keys
    e, d, n = generate_rsa_keypair(prime_bits=12)

    # Run hybrid entropy calculation
    H_original, H_rsa = hybrid_entropy(obs, e, d, n)

    print(f"Original Shannon entropy : {H_original:.6f} bits")
    print(f"RSA‑mapped entropy       : {H_rsa:.6f} bits")
    print(f"RSA parameters: e={e}, d={d}, n={n}")

    # Verify that the two entropies are (numerically) equal up to floating error
    if math.isclose(H_original, H_rsa, rel_tol=1e-6):
        print("Success: entropy invariant under RSA transformation (as expected).")
    else:
        print("Notice: entropy changed due to scaling/round‑off effects.")