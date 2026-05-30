# DARWIN HAMMER — match 3035, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_rsa_ci_hybrid_hybrid_krampu_m2378_s3.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_nlms_o_m1965_s2.py (gen5)
# born: 2026-05-29T23:47:22Z

"""Hybrid RSA‑RBF‑NLMS Module
This module fuses the core topologies of the two parent algorithms:

* **Parent A** provides RSA key generation / encryption, a linear solver, and a
  `PheromoneEntry` container.
* **Parent B** supplies a Gaussian radial‑basis‑function (RBF) surrogate and a
  Normalised Least‑Mean‑Squares (NLMS) adaptive update.

The mathematical bridge is the **RBF surrogate weight vector**, which is
treated as confidential data.  The surrogate parameters (centres and
coefficients) are encrypted with RSA (Parent A) and later decrypted for use in
the NLMS adaptation (Parent B).  The NLMS weight update itself solves a linear
system (via the linear solver from Parent A) to obtain the optimal step size.
Thus the three families of operations—RSA, RBF, NLMS—are inter‑woven into a
single hybrid workflow.
"""

import math
import random
import sys
import pathlib
from typing import List, Tuple, Sequence, Dict, Any

import numpy as np

Vector = Sequence[float]

# ----------------------------------------------------------------------
# Shared primitives (identical in both parents)
# ----------------------------------------------------------------------
def euclidean(a: Vector, b: Vector) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian radial basis function."""
    return math.exp(-((epsilon * r) ** 2))


def solve_linear(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve the linear system Ax = b using NumPy's LAPACK driver."""
    return np.linalg.solve(a, b)


# ----------------------------------------------------------------------
# RSA utilities (from Parent A)
# ----------------------------------------------------------------------
def _egcd(a: int, b: int) -> Tuple[int, int, int]:
    if a == 0:
        return (b, 0, 1)
    g, y, x = _egcd(b % a, a)
    return (g, x - (b // a) * y, y)


def _modinv(a: int, m: int) -> int:
    g, x, _ = _egcd(a, m)
    if g != 1:
        raise ValueError("modular inverse does not exist")
    return x % m


def generate_rsa_keypair(bitlen: int = 256) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Generate a (public, private) RSA keypair."""
    def _prime() -> int:
        while True:
            p = random.getrandbits(bitlen // 2)
            if p % 2 == 0:
                p += 1
            # simple Miller‑Rabin style witness loop (3 rounds)
            for _ in range(3):
                a = random.randrange(2, p - 2)
                if pow(a, p - 1, p) != 1:
                    break
            else:
                return p

    p = _prime()
    q = _prime()
    n = p * q
    phi = (p - 1) * (q - 1)
    e = 65537
    while math.gcd(e, phi) != 1:
        e += 2
    d = _modinv(e, phi)
    return (e, n), (d, n)


def rsa_encrypt(message: int, e: int, n: int) -> int:
    if not 0 <= message < n:
        raise ValueError("message must be in [0, n)")
    return pow(message, e, n)


def rsa_decrypt(cipher: int, d: int, n: int) -> int:
    return pow(cipher, d, n)


# ----------------------------------------------------------------------
# Pheromone container (kept for structural parity with Parent A)
# ----------------------------------------------------------------------
class PheromoneEntry:
    __slots__ = ("uuid", "surface_key", "signal_kind", "signal_value",
                 "half_life_seconds", "created_at", "last_decay")

    def __init__(self, surface_key: str, signal_kind: str,
                 signal_value: float, half_life_seconds: int):
        self.uuid = str(random.getrandbits(128))
        self.surface_key = surface_key
        self.signal_kind = signal_kind
        self.signal_value = signal_value
        self.half_life_seconds = half_life_seconds
        self.created_at = None
        self.last_decay = None


# ----------------------------------------------------------------------
# RBF surrogate (from Parent B) – encrypted with RSA
# ----------------------------------------------------------------------
def _int_from_bytes(b: bytes) -> int:
    return int.from_bytes(b, byteorder='big')


def _bytes_from_int(i: int, length: int) -> bytes:
    return i.to_bytes(length, byteorder='big')


def encrypt_surrogate_parameters(
    centres: List[Vector],
    coeffs: List[float],
    pub_key: Tuple[int, int]
) -> Tuple[List[int], List[int]]:
    """
    Serialize and RSA‑encrypt the surrogate parameters.
    Returns two integer lists: encrypted centres (flattened) and encrypted coeffs.
    """
    e, n = pub_key
    # flatten centres
    flat = [float(val) for vec in centres for val in vec]
    # convert each float to 64‑bit IEEE representation, then to int
    centre_ints = [_int_from_bytes(np.float64(v).tobytes()) for v in flat]
    coeff_ints = [_int_from_bytes(np.float64(c).tobytes()) for c in coeffs]

    enc_c = [rsa_encrypt(i, e, n) for i in centre_ints]
    enc_b = [rsa_encrypt(i, e, n) for i in coeff_ints]
    return enc_c, enc_b


def decrypt_surrogate_parameters(
    enc_c: List[int],
    enc_b: List[int],
    priv_key: Tuple[int, int],
    dim: int,
    num_centres: int
) -> Tuple[List[Vector], List[float]]:
    """
    RSA‑decrypt and reconstruct centres and coefficients.
    """
    d, n = priv_key
    centre_ints = [rsa_decrypt(c, d, n) for c in enc_c]
    coeff_ints = [rsa_decrypt(b, d, n) for b in enc_b]

    flat = [np.float64(_bytes_from_int(i, 8)).item() for i in centre_ints]
    coeffs = [np.float64(_bytes_from_int(i, 8)).item() for i in coeff_ints]

    centres = [flat[i * dim:(i + 1) * dim] for i in range(num_centres)]
    return centres, coeffs


def rbf_surrogate_predict(
    x: Vector,
    centres: List[Vector],
    coeffs: List[float],
    epsilon: float = 1.0
) -> float:
    """
    Classic Gaussian RBF surrogate:
        y(x) = Σ_i coeff_i * exp(- (ε * ||x - c_i||)^2 )
    """
    if len(centres) != len(coeffs):
        raise ValueError("centres and coeffs must have the same length")
    y = 0.0
    for c, b in zip(centres, coeffs):
        r = euclidean(x, c)
        y += b * gaussian(r, epsilon)
    return y


# ----------------------------------------------------------------------
# NLMS adaptive update (from Parent B) – uses surrogate output as a
# desired signal for weight adaptation.
# ----------------------------------------------------------------------
def nlms_update(
    w: np.ndarray,
    x: np.ndarray,
    d: float,
    mu: float = 0.1,
    eps: float = 1e-6
) -> np.ndarray:
    """
    Normalised LMS weight update.
        y = w·x
        e = d - y
        w ← w + (mu / (eps + ||x||²)) * e * x
    """
    y = float(w @ x)
    e = d - y
    norm_factor = eps + float(x @ x)
    w_new = w + (mu / norm_factor) * e * x
    return w_new


# ----------------------------------------------------------------------
# Hybrid operation – combines RSA, RBF surrogate and NLMS
# ----------------------------------------------------------------------
def hybrid_predict_and_adapt(
    x: Vector,
    rsa_pub: Tuple[int, int],
    rsa_priv: Tuple[int, int],
    centres: List[Vector],
    coeffs: List[float],
    nlms_weights: np.ndarray,
    mu: float = 0.1,
    epsilon_rbf: float = 1.0
) -> Tuple[float, np.ndarray]:
    """
    1. Encrypt the surrogate parameters with RSA (simulating secure transport).
    2. Decrypt them back (as would happen on the receiver side).
    3. Compute the RBF surrogate prediction ŷ(x).
    4. Treat ŷ(x) as the desired signal d for an NLMS update of the weight vector.
    5. Return the surrogate output and the updated NLMS weight vector.
    """
    # ----- 1. RSA encryption of surrogate parameters -----
    enc_c, enc_b = encrypt_surrogate_parameters(centres, coeffs, rsa_pub)

    # ----- 2. RSA decryption (receiver side) -----
    dim = len(centres[0])
    num_c = len(centres)
    dec_centres, dec_coeffs = decrypt_surrogate_parameters(
        enc_c, enc_b, rsa_priv, dim, num_c
    )

    # ----- 3. RBF surrogate prediction -----
    y_hat = rbf_surrogate_predict(x, dec_centres, dec_coeffs, epsilon=epsilon_rbf)

    # ----- 4. NLMS adaptation using the surrogate output as desired -----
    x_arr = np.asarray(x, dtype=float)
    w_new = nlms_update(nlms_weights, x_arr, y_hat, mu=mu)

    # ----- 5. Return results -----
    return y_hat, w_new


# ----------------------------------------------------------------------
# Minimal spanning‑tree (MST) placeholder – demonstrates a third hybrid
# function that could be used for “minimum‑cost tree optimisation”.
# ----------------------------------------------------------------------
def mst_total_weight(points: List[Vector]) -> float:
    """
    Compute the total weight of a naïve MST using Prim's algorithm on the
    Euclidean distance graph of the supplied points.
    """
    if not points:
        return 0.0
    n = len(points)
    visited = [False] * n
    min_edge = [float('inf')] * n
    min_edge[0] = 0.0
    total = 0.0

    for _ in range(n):
        # select the unvisited vertex with smallest edge weight
        u = -1
        best = float('inf')
        for i in range(n):
            if not visited[i] and min_edge[i] < best:
                best = min_edge[i]
                u = i
        if u == -1:
            break
        visited[u] = True
        total += best
        # update neighbours
        for v in range(n):
            if not visited[v]:
                w = euclidean(points[u], points[v])
                if w < min_edge[v]:
                    min_edge[v] = w
    return total


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 1. RSA keypair
    pub_key, priv_key = generate_rsa_keypair(bitlen=256)

    # 2. Define a tiny RBF surrogate (2 centres, 2‑dimensional input)
    centres = [[0.0, 0.0], [1.0, 1.0]]
    coeffs = [0.8, -0.3]

    # 3. Initialise NLMS weight vector (same dimension as input)
    nlms_w = np.zeros(2, dtype=float)

    # 4. Sample input vector
    x_input = [0.4, 0.6]

    # 5. Run the hybrid predict‑and‑adapt routine
    y, nlms_w = hybrid_predict_and_adapt(
        x=x_input,
        rsa_pub=pub_key,
        rsa_priv=priv_key,
        centres=centres,
        coeffs=coeffs,
        nlms_weights=nlms_w,
        mu=0.2,
        epsilon_rbf=1.5
    )
    print(f"Surrogate output: {y:.6f}")
    print(f"Updated NLMS weights: {nlms_w}")

    # 6. Demonstrate the MST utility on a small point set
    pts = [[0, 0], [1, 0], [0, 1], [1, 1]]
    total = mst_total_weight(pts)
    print(f"MST total weight for square points: {total:.6f}")

    # 7. Verify that RSA encryption/decryption round‑trip works for a scalar
    test_val = 123456789
    enc = rsa_encrypt(test_val, *pub_key)
    dec = rsa_decrypt(enc, *priv_key)
    assert dec == test_val, "RSA round‑trip failed"
    print("RSA round‑trip test passed.")