# DARWIN HAMMER — match 1132, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_decision_hygi_m338_s1.py (gen4)
# parent_b: hybrid_hybrid_shannon_entro_sparse_wta_m36_s1.py (gen2)
# born: 2026-05-29T23:32:57Z

"""Hybrid algorithm merging:

- **Parent A**: weekday‑dependent stochastic weight vector, sheaf allocation over a chain graph,
  coboundary (edge‑wise difference) L²‑norm and Shannon entropy of the allocation.
- **Parent B**: Sparse Winner‑Take‑All (WTA) producing binary tags, encoding tags as a
  probability distribution, Shannon entropy of that distribution, RSA encryption of the
  encoded distribution and entropy of the ciphertext.

**Mathematical bridge**

1. A scalar cue *s* (e.g. extracted from text) is linearly mapped to a group‑wise
   allocation **a = s·w**, where **w∈ℝⁿ** is the row‑stochastic weekday weight vector
   (Parent A).
2. The allocation vector **a** is interpreted as a high‑dimensional similarity vector.
   Sparse WTA keeps the *k* largest components, yielding a binary tag vector **t**∈{0,1}ⁿ
   (Parent B).
3. **t** is normalised to a probability distribution **p = t / ‖t‖₁**; its Shannon
   entropy **H(p)** quantifies information content before encryption.
4. The distribution is encoded as integers (by scaling) and RSA‑encrypted component‑wise,
   producing ciphertext **c**. Normalising **c** yields a second distribution **q**
   whose entropy **H(q)** measures information loss/gain through encryption.
5. Independently, the original allocation **a** is viewed as a sheaf section over the
   chain graph; the coboundary operator **δ** computes edge‑wise differences
   **δₑ(a)=a_i−a_j**, and the L²‑norm **‖δ(a)‖₂** assesses allocation coherence.

The functions below implement this fused pipeline."""


from __future__ import annotations

import datetime as dt
import math
import random
import sys
from pathlib import Path
from typing import Tuple, List

import numpy as np

# ----------------------------------------------------------------------
# Shared structural constants (Parent A)
# ----------------------------------------------------------------------
GROUPS = ("codex", "groq", "cohere", "local_models")
EDGES: List[Tuple[str, str]] = [
    ("codex", "groq"),
    ("groq", "cohere"),
    ("cohere", "local_models"),
]

# ----------------------------------------------------------------------
# Parent A utilities
# ----------------------------------------------------------------------
def _pct(value: float) -> float:
    """Round a float to six decimal places."""
    return round(float(value), 6)


def weekday_weight_vector(groups: Tuple[str, ...], date: dt.date) -> np.ndarray:
    """
    Produce a row‑stochastic weight vector that depends on the weekday.

    The vector is constructed by giving the current weekday a higher base
    weight and distributing the remainder uniformly.
    """
    base = np.full(len(groups), 1.0 / len(groups))
    weekday = date.weekday()  # Monday == 0
    boost = 0.2  # extra weight for the “active” group
    base[weekday % len(groups)] += boost
    base = base / base.sum()  # row‑stochastic
    return np.array([_pct(v) for v in base])


def coboundary_l2_norm(allocation: np.ndarray) -> float:
    """
    Compute the L²‑norm of the coboundary δ(allocation) over the chain graph.
    δₑ(a) = a_i – a_j for each edge (i,j).
    """
    diffs = []
    name_to_idx = {name: i for i, name in enumerate(GROUPS)}
    for i_name, j_name in EDGES:
        i = name_to_idx[i_name]
        j = name_to_idx[j_name]
        diffs.append(allocation[i] - allocation[j])
    diffs = np.array(diffs, dtype=float)
    norm = np.linalg.norm(diffs, ord=2)
    return _pct(norm)


def shannon_entropy(probs: np.ndarray) -> float:
    """Standard Shannon entropy H(p) = -∑ p·log₂(p) (0·log2(0) treated as 0)."""
    probs = probs[probs > 0]
    if probs.size == 0:
        return 0.0
    ent = -np.sum(probs * np.log2(probs))
    return _pct(ent)


# ----------------------------------------------------------------------
# Parent B utilities
# ----------------------------------------------------------------------
def sparse_wta(vector: np.ndarray, k: int) -> np.ndarray:
    """
    Sparse Winner‑Take‑All: keep the k largest entries, set the rest to 0,
    then binarise (1 for kept entries, 0 otherwise).
    """
    if k <= 0:
        return np.zeros_like(vector, dtype=int)
    if k >= len(vector):
        return np.ones_like(vector, dtype=int)

    # argsort descending, pick top‑k indices
    top_idx = np.argpartition(-vector, k - 1)[:k]
    tags = np.zeros_like(vector, dtype=int)
    tags[top_idx] = 1
    return tags


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
    return pow(message, e, n)


def rsa_decrypt_int(cipher: int, d: int, n: int) -> int:
    """RSA decryption of a single integer."""
    return pow(cipher, d, n)


# ----------------------------------------------------------------------
# Hybrid core functions (fusion of A & B)
# ----------------------------------------------------------------------
def allocate_and_tag(
    date: dt.date,
    scalar: float,
    top_k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    1. Build weekday weight vector **w**.
    2. Allocate resources **a = scalar·w** (sheaf section).
    3. Apply Sparse WTA to **a**, yielding binary tags **t**.
    Returns (allocation, tags).
    """
    w = weekday_weight_vector(GROUPS, date)          # row‑stochastic
    allocation = scalar * w                           # section a ∈ ℝⁿ
    tags = sparse_wta(allocation, top_k)              # t ∈ {0,1}ⁿ
    return allocation, tags


def entropy_and_rsa(
    tags: np.ndarray,
    rsa_key: Tuple[int, int, int],
    scale: int = 1_000_000,
) -> Tuple[float, float, List[int], List[int]]:
    """
    Encode binary **tags** as a probability distribution, compute its Shannon entropy,
    RSA‑encrypt the scaled integer representation, and compute the entropy of the
    resulting ciphertext distribution.

    Returns (entropy_before, entropy_after, plaintext_ints, ciphertext_ints).
    """
    # Normalise tags to a probability distribution (avoid division by zero)
    if tags.sum() == 0:
        probs = np.full_like(tags, 1.0 / len(tags))
    else:
        probs = tags.astype(float) / tags.sum()

    entropy_before = shannon_entropy(probs)

    # Encode as integers for RSA
    plain_ints = (probs * scale).astype(int).tolist()

    e, _, n = rsa_key
    cipher_ints = [rsa_encrypt_int(m, e, n) for m in plain_ints]

    # Normalise ciphertext to a probability distribution (still sum > 0)
    cipher_arr = np.array(cipher_ints, dtype=float)
    cipher_probs = cipher_arr / cipher_arr.sum()
    entropy_after = shannon_entropy(cipher_probs)

    return entropy_before, entropy_after, plain_ints, cipher_ints


def hybrid_process(
    date: dt.date,
    scalar: float,
    top_k: int,
) -> dict:
    """
    Executes the full hybrid pipeline:

    * allocation → coboundary L² norm
    * allocation → tags via Sparse WTA
    * tags → entropy before/after RSA encryption

    Returns a dictionary with all intermediate results.
    """
    allocation, tags = allocate_and_tag(date, scalar, top_k)

    coboundary_norm = coboundary_l2_norm(allocation)

    rsa_key = generate_rsa_keypair()
    entropy_before, entropy_after, plain_ints, cipher_ints = entropy_and_rsa(
        tags, rsa_key
    )

    return {
        "date": date.isoformat(),
        "scalar": scalar,
        "weight_vector": weekday_weight_vector(GROUPS, date).tolist(),
        "allocation": allocation.tolist(),
        "coboundary_l2_norm": coboundary_norm,
        "tags": tags.tolist(),
        "entropy_before_rsa": entropy_before,
        "entropy_after_rsa": entropy_after,
        "rsa_plain_ints": plain_ints,
        "rsa_cipher_ints": cipher_ints,
        "rsa_key": {"e": rsa_key[0], "d": rsa_key[1], "n": rsa_key[2]},
    }


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    today = dt.date.today()
    cue = random.uniform(0.5, 5.0)          # example scalar cue
    k = 2                                  # keep top‑2 entries in WTA

    result = hybrid_process(today, cue, k)

    # Print a concise summary
    print("Hybrid Run Summary")
    print("------------------")
    print(f"Date                : {result['date']}")
    print(f"Scalar cue (s)      : {result['scalar']:.4f}")
    print(f"Weight vector (w)   : {result['weight_vector']}")
    print(f"Allocation (a)      : {result['allocation']}")
    print(f"Coboundary L2 norm  : {result['coboundary_l2_norm']}")
    print(f"WTA tags (t)        : {result['tags']}")
    print(f"Entropy before RSA  : {result['entropy_before_rsa']}")
    print(f"Entropy after RSA   : {result['entropy_after_rsa']}")
    print(f"RSA public e, n     : ({result['rsa_key']['e']}, {result['rsa_key']['n']})")
    print("Done.")