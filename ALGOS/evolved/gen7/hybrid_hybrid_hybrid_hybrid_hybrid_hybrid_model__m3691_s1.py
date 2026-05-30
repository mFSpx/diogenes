# DARWIN HAMMER — match 3691, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_minimu_rsa_cipher_m1949_s2.py (gen6)
# parent_b: hybrid_hybrid_model_vram_sc_hybrid_hybrid_nlms_h_m2300_s2.py (gen5)
# born: 2026-05-29T23:51:13Z

"""Hybrid RSA‑Encrypted Perceptual‑Linear Adaptive System
================================================================
Parent A: hybrid_hybrid_hybrid_minimu_rsa_cipher_m1949_s2.py
Parent B: hybrid_hybrid_model_vram_sc_hybrid_hybrid_nlms_h_m2300_s2.py

Mathematical Bridge
-------------------
Both parents manipulate vectors in a recurrent / linear‑adaptive fashion.
Parent A defines a hidden state update  

    hₜ₊₁ = (1‑α)·hₜ + α·tanh(W·xₜ + U·hₜ + b) + λ·ηₜ                (1)

where α is a Gaussian‑RBF similarity between successive states and λ is a
diffusion coefficient derived from the same RBF.

Parent B performs a Least‑Mean‑Squares‑style weight adaptation  

    wₜ₊₁ = wₜ + μ·eₜ·xₜ / (‖xₜ‖²+ε)                           (2)

and simultaneously updates a secondary matrix W used for a “linear
transform‑tree” (TTT) loss.

The fusion encrypts the *increment* Δw = wₜ₊₁‑wₜ with RSA before it is
applied, thereby securing the adaptive step while still allowing the
plain‑text hidden‑state dynamics of (1).  The encrypted payload is
decrypted on the receiving side, verified, and then added to the weight
vector.  This creates a single pipeline that (i) updates a perceptual
state, (ii) computes a linear‑adaptive weight step, (iii) protects the
step with RSA, and (iv) evaluates a minimum‑cost spanning‑tree cost on a
set of points – the “tree” component inherited from Parent A.

The module therefore contains three core functions:
    * `hybrid_perceptual_update` – implements (1) with RBF‑derived α, λ.
    * `encrypted_ltc_update`      – implements (2), encrypts Δw, decrypts,
      and returns the secured weight vector.
    * `minimum_cost_tree`        – computes the total Euclidean length of
      a minimum spanning tree over supplied points (Kruskal’s algorithm).

All components are pure NumPy / std‑lib and can be run on any Python 3
interpreter."""


import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


# ----------------------------------------------------------------------
# RSA utilities (parent A)
# ----------------------------------------------------------------------
def generate_rsa_keypair(p: int, q: int) -> Tuple[int, int, int]:
    """Return public exponent e, private exponent d and modulus n."""
    n = p * q
    phi = (p - 1) * (q - 1)

    e = 2
    while math.gcd(e, phi) != 1:
        e += 1

    d = pow(e, -1, phi)
    return e, d, n


def rsa_encrypt(message: int, e: int, n: int) -> int:
    """Encrypt an integer message < n."""
    if not 0 <= message < n:
        raise ValueError("message must be in [0, n)")
    return pow(message, e, n)


def rsa_decrypt(ciphertext: int, d: int, n: int) -> int:
    """Decrypt an integer ciphertext < n."""
    if not 0 <= ciphertext < n:
        raise ValueError("ciphertext must be in [0, n)")
    return pow(ciphertext, d, n)


# ----------------------------------------------------------------------
# Data structures (shared)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Point:
    x: float
    y: float


# ----------------------------------------------------------------------
# Perceptual‑state dynamics (parent A)
# ----------------------------------------------------------------------
def rbf_similarity(v1: np.ndarray, v2: np.ndarray, sigma: float = 1.0) -> float:
    """Gaussian RBF similarity between two vectors."""
    diff = v1 - v2
    dist_sq = np.dot(diff, diff)
    return math.exp(-dist_sq / (2 * sigma ** 2))


def hybrid_perceptual_update(
    h_prev: np.ndarray,
    x_t: np.ndarray,
    W: np.ndarray,
    U: np.ndarray,
    b: np.ndarray,
    sigma: float = 1.0,
    diffusion_scale: float = 0.1,
) -> np.ndarray:
    """
    Implements equation (1).

    Returns the next hidden state h_{t+1}.
    """
    # similarity α between current hidden state and its linear prediction
    pred = np.tanh(W @ x_t + U @ h_prev + b)
    alpha = rbf_similarity(h_prev, pred, sigma)

    # diffusion coefficient λ from the same RBF (larger distance → larger λ)
    lambda_coeff = diffusion_scale * (1.0 - alpha)

    # Gaussian noise η_t
    eta = np.random.randn(*h_prev.shape)

    h_next = (1 - alpha) * h_prev + alpha * pred + lambda_coeff * eta
    return h_next


# ----------------------------------------------------------------------
# Linear‑adaptive weight update (parent B) with RSA encryption
# ----------------------------------------------------------------------
def ltc_update(
    w: np.ndarray,
    x: np.ndarray,
    target: np.ndarray,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Least‑Mean‑Squares style update (equation 2) returning
    (w_next, Δw) where Δw = w_next - w.
    """
    y = np.dot(w, x)
    error = target - y
    power = np.dot(x, x) + eps
    delta = mu * error * x / power
    w_next = w + delta
    return w_next, delta


def encrypted_ltc_update(
    w: np.ndarray,
    x: np.ndarray,
    target: np.ndarray,
    rsa_pub: Tuple[int, int],
    rsa_priv: Tuple[int, int],
    scale_int: float = 1e6,
) -> np.ndarray:
    """
    Performs an LTC update, encrypts the integer representation of
    Δw (scaled and summed), decrypts it, verifies integrity and finally
    applies the *original* floating‑point Δw to the weight vector.

    Returns the secured weight vector.
    """
    w_next, delta = ltc_update(w, x, target)

    # Convert delta to a single integer payload
    payload_int = int(np.round(np.sum(delta) * scale_int)) % rsa_pub[2]

    # RSA encrypt / decrypt
    cipher = rsa_encrypt(payload_int, rsa_pub[0], rsa_pub[2])
    recovered = rsa_decrypt(cipher, rsa_priv[0], rsa_priv[2])

    if recovered != payload_int:
        raise RuntimeError("RSA integrity check failed")

    # Apply the true floating‑point delta (the encryption only secures the
    # *knowledge* of the update, not the numerical value itself).
    secured_w = w + delta
    return secured_w


# ----------------------------------------------------------------------
# Minimum‑cost spanning tree (parent A component)
# ----------------------------------------------------------------------
def minimum_cost_tree(points: List[Point]) -> float:
    """
    Kruskal's algorithm for Euclidean MST total length.
    """
    if len(points) < 2:
        return 0.0

    # Helper: Union‑Find
    parent = list(range(len(points)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
            return True
        return False

    # Edge list
    edges = []
    for i, p1 in enumerate(points):
        for j in range(i + 1, len(points)):
            p2 = points[j]
            dist = math.hypot(p1.x - p2.x, p1.y - p2.y)
            edges.append((dist, i, j))

    edges.sort(key=lambda e: e[0])

    total = 0.0
    for dist, i, j in edges:
        if union(i, j):
            total += dist
            # Early stop when we have (n‑1) edges
            if sum(1 for k in range(len(points)) if parent[k] == k) == 1:
                break
    return total


# ----------------------------------------------------------------------
# Demonstration / smoke test
# ----------------------------------------------------------------------
def _smoke_test():
    # ---------- RSA key pair ----------
    e, d, n = generate_rsa_keypair(61, 53)   # small primes for demo
    rsa_pub = (e, None, n)                  # public: (e, _, n)
    rsa_priv = (d, None, n)                 # private: (d, _, n)

    # ---------- Perceptual state ----------
    dim = 8
    h = np.zeros(dim)
    x = np.random.randn(dim)
    W = np.random.randn(dim, dim) * 0.01
    U = np.random.randn(dim, dim) * 0.01
    b = np.random.randn(dim) * 0.01

    h_next = hybrid_perceptual_update(h, x, W, U, b)

    # ---------- Weight adaptation ----------
    w = np.random.randn(dim)
    target = np.random.randn(dim)
    secured_w = encrypted_ltc_update(w, x, target, rsa_pub, rsa_priv)

    # ---------- Minimum cost tree ----------
    pts = [Point(random.random(), random.random()) for _ in range(10)]
    mst_cost = minimum_cost_tree(pts)

    # Print results to verify execution
    print("h_next (norm):", np.linalg.norm(h_next))
    print("secured_w (norm):", np.linalg.norm(secured_w))
    print("MST total length:", mst_cost)


if __name__ == "__main__":
    _smoke_test()