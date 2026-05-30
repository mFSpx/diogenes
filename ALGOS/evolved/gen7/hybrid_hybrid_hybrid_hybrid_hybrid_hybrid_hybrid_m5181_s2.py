# DARWIN HAMMER — match 5181, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_rsa_ci_hybrid_hybrid_percyp_m2466_s0.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2204_s1.py (gen6)
# born: 2026-05-30T00:00:27Z

"""Hybrid RSA‑MinHash Decision Engine
Parents:
* hybrid_hybrid_hybrid_rsa_ci_hybrid_hybrid_percyp_m2466_s0 (RSA‑Metric + Pheromone Advection)
* hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2204_s1 (MinHash similarity + Clifford‑style weight matrix)

Mathematical Bridge:
The sphericity index derived from a Morphology object is quantised and fed as a vector of
integers to the RSA primitive (Parent A).  The resulting ciphertext vector is transformed
into a scalar “hygiene‑entropy” metric.  This metric is encoded as a token list
(e.g. hexadecimal fragments) and processed by the MinHash pipeline of Parent B.
The Jaccard‑like similarity between two such token sets yields a single scalar *s*,
which is promoted to a weight matrix **W = s·𝟙**.  **W** is then used to bias a
regret‑style action selection, completing the fusion of the two topologies.

The module provides three core hybrid functions:
1. `rsa_hygiene_metric` – RSA encryption of a feature vector and entropy extraction.
2. `minhash_weight_matrix` – MinHash similarity turned into a weight matrix.
3. `select_hybrid_action` – Regret‑inspired action scoring weighted by **W**.
"""

import math
import random
import sys
import pathlib
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
Vector = Sequence[float]

# ----------------------------------------------------------------------
# Parent A fragments (RSA primitive & Morphology)
# ----------------------------------------------------------------------
def rsa_encrypt(message: int, e: int, n: int) -> int:
    """RSA encryption: c = m^e mod n."""
    if not 0 <= message < n:
        raise ValueError("message must be in [0, n)")
    return pow(message, e, n)

def rsa_decrypt(ciphertext: int, d: int, n: int) -> int:
    """RSA decryption: m = c^d mod n."""
    if not 0 <= ciphertext < n:
        raise ValueError("ciphertext must be in [0, n)")
    return pow(ciphertext, d, n)

@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

def sphericity_index(morph: Morphology) -> float:
    """Simple sphericity proxy: (volume)/(mass)."""
    volume = morph.length * morph.width * morph.height
    return volume / morph.mass if morph.mass != 0 else 0.0

# ----------------------------------------------------------------------
# Parent B fragments (MinHash & Action model)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathAction:
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0

def _hash(seed: int, token: str) -> int:
    """Deterministic 64‑bit integer hash."""
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")

def signature(tokens: List[str], k: int = 128) -> List[int]:
    """MinHash signature of a token set."""
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [((1 << 64) - 1)] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]

def similarity(sig_a: List[int], sig_b: List[int]) -> float:
    """Jaccard‑like similarity between two MinHash signatures."""
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have equal length")
    if not sig_a:
        raise ValueError("signatures must not be empty")
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)

# ----------------------------------------------------------------------
# Hybrid Functions
# ----------------------------------------------------------------------
def rsa_hygiene_metric(feature_vec: Vector, e: int, d: int, n: int) -> Tuple[np.ndarray, float]:
    """
    Encrypt each component of `feature_vec` with RSA, then compute an entropy‑like
    metric from the ciphertext distribution.

    Returns
    -------
    ciphertexts : np.ndarray
        Encrypted integer vector.
    metric : float
        -∑ p_i log(p_i) where p_i are normalized ciphertext frequencies.
    """
    # Quantise feature vector to non‑negative integers < n
    scaled = np.floor(np.abs(np.array(feature_vec)) * (n - 1) / np.max(np.abs(feature_vec) + 1e-12)).astype(int)
    ciphertexts = np.array([rsa_encrypt(int(m), e, n) for m in scaled])

    # Frequency histogram of ciphertext values
    unique, counts = np.unique(ciphertexts, return_counts=True)
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log(probs + 1e-12))

    return ciphertexts, entropy

def minhash_weight_matrix(tokens_a: List[str], tokens_b: List[str], k: int = 64) -> np.ndarray:
    """
    Build a weight matrix from MinHash similarity.
    The scalar similarity s is lifted to a full matrix W = s * I,
    where I has shape (len(tokens_a), len(tokens_b)).
    """
    sig_a = signature(tokens_a, k)
    sig_b = signature(tokens_b, k)
    s = similarity(sig_a, sig_b)
    # Create a matrix that can broadcast onto action‑score vectors
    W = s * np.ones((len(tokens_a), len(tokens_b)))
    return W

def select_hybrid_action(actions: List[MathAction],
                         weight_matrix: np.ndarray,
                         hygiene_metric: float) -> MathAction:
    """
    Regret‑style scoring:
        score = expected_value - cost - risk
    The score is modulated by the weight matrix (average of its entries) and the
    RSA‑derived hygiene metric.

    The action with the highest adjusted score is returned.
    """
    if not actions:
        raise ValueError("actions list cannot be empty")
    # Reduce weight matrix to a scalar influence factor
    w_factor = weight_matrix.mean() if weight_matrix.size else 1.0
    adjusted_scores = []
    for act in actions:
        base = act.expected_value - act.cost - act.risk
        adjusted = base * (1.0 + w_factor) - hygiene_metric
        adjusted_scores.append(adjusted)
    best_idx = int(np.argmax(adjusted_scores))
    return actions[best_idx]

# ----------------------------------------------------------------------
# Helper to turn ciphertexts into token strings for MinHash
# ----------------------------------------------------------------------
def ciphertexts_to_tokens(ciphertexts: np.ndarray) -> List[str]:
    """
    Convert each ciphertext integer to a short hex token.
    """
    return [format(int(c), 'x') for c in ciphertexts]

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Morphology → RSA → hygiene metric
    morph = Morphology(length=1.2, width=0.8, height=0.5, mass=0.6)
    sph = sphericity_index(morph)

    # RSA small key (for demonstration only)
    p, q = 61, 53
    n = p * q
    phi = (p - 1) * (q - 1)
    e = 17
    # modular inverse for d
    d = pow(e, -1, phi)

    # Feature vector: use the sphericity index plus some dummy features
    feature_vec = np.array([sph, math.sin(sph), math.cos(sph)])

    ciphertexts, hygiene = rsa_hygiene_metric(feature_vec, e, d, n)

    # 2. Tokens for MinHash (compare with a slightly perturbed vector)
    tokens_a = ciphertexts_to_tokens(ciphertexts)

    # Create a second token list by adding a small noise to the original ciphertexts
    noisy = (ciphertexts + np.random.randint(-2, 3, size=ciphertexts.shape)) % n
    tokens_b = ciphertexts_to_tokens(noisy)

    # 3. Build weight matrix from MinHash similarity
    W = minhash_weight_matrix(tokens_a, tokens_b, k=64)

    # 4. Define a few actions
    actions = [
        MathAction(id="A", expected_value=10.0, cost=2.0, risk=1.0),
        MathAction(id="B", expected_value=8.0, cost=1.5, risk=0.5),
        MathAction(id="C", expected_value=12.0, cost=4.0, risk=2.0),
    ]

    # 5. Select hybrid action
    chosen = select_hybrid_action(actions, W, hygiene)

    print("Sphericity index :", sph)
    print("RSA ciphertexts :", ciphertexts.tolist())
    print("Hygiene metric   :", hygiene)
    print("MinHash similarity matrix mean :", W.mean())
    print("Chosen action    :", chosen)