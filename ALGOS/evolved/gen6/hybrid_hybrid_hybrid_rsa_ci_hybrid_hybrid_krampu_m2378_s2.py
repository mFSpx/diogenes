# DARWIN HAMMER — match 2378, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_rsa_cipher_hy_hybrid_hybrid_hybrid_m827_s1.py (gen5)
# parent_b: hybrid_hybrid_krampus_stick_hybrid_rbf_surrogate_m556_s0.py (gen4)
# born: 2026-05-29T23:42:12Z

"""Hybrid RSA‑RBF‑Pheromone Model
================================

This module fuses the two parent algorithms:

* **Parent A** – a hybrid RSA‑RBF‑Surrogate scheme where the scalar output of an
  RBF surrogate model is encrypted with RSA.
* **Parent B** – a pheromone‑based temporal decay system whose signal values are
  used to modulate the Gaussian kernel of an RBF model.

**Mathematical bridge**

The Gaussian radial‑basis function `K(i,j) = exp(- (ε·‖x_i‑x_j‖)² )` is the core of
the surrogate in Parent A.  In Parent B each *surface* carries a collection of
`PheromoneEntry` objects whose decayed `signal_value` represents a dynamic
weight.  We let the total decayed signal for a surface act as a multiplicative
modifier of the kernel shape parameter `ε`:


ε_eff = ε * (1 + Σ_decay(signal_value))
K(i,j) = exp( - (ε_eff·‖x_i‑x_j‖)² )


Thus the pheromone dynamics directly influence the surrogate matrix, while the
surrogate’s scalar output can still be encrypted/decrypted with RSA, producing a
single unified hybrid system.

The public API provides three demonstration functions:

* ``hybrid_fit_encrypt`` – fit an RBF surrogate using pheromone‑adjusted kernels
  and encrypt the resulting surrogate output.
* ``hybrid_predict_decrypt`` – decrypt the ciphertext and recompute the surrogate
  prediction for a new payload.
* ``region_blade_product`` – a lightweight Clifford‑algebra‑inspired product on
  text‑derived vectors (retained from Parent A for completeness).
"""

import math
import random
import sys
import pathlib
import datetime
from typing import List, Tuple, Dict, Any, Sequence

import numpy as np

# ----------------------------------------------------------------------
# Basic linear algebra utilities (shared)
# ----------------------------------------------------------------------
Vector = Sequence[float]

def euclidean(a: Vector, b: Vector) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Standard Gaussian RBF."""
    return math.exp(-((epsilon * r) ** 2))

def solve_linear(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve Ax = b using NumPy."""
    return np.linalg.solve(a, b)

# ----------------------------------------------------------------------
# RSA primitives (Parent A)
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
    """Generate a tiny RSA key pair for demonstration (not cryptographically safe)."""
    # Very small primes for speed; in real use use proper libraries.
    def _prime():
        while True:
            p = random.getrandbits(bitlen // 2)
            if p % 2 == 0:
                p += 1
            # Miller‑Rabin style primality test (3 rounds)
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
    """RSA encryption c = m^e mod n."""
    if not 0 <= message < n:
        raise ValueError("message must be in [0, n)")
    return pow(message, e, n)

def rsa_decrypt(cipher: int, d: int, n: int) -> int:
    """RSA decryption m = c^d mod n."""
    return pow(cipher, d, n)

# ----------------------------------------------------------------------
# Pheromone dynamics (Parent B)
# ----------------------------------------------------------------------
from collections import defaultdict

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
        self.created_at = datetime.datetime.utcnow()
        self.last_decay = self.created_at

    def age_seconds(self) -> float:
        return (datetime.datetime.utcnow() - self.last_decay).total_seconds()

    def decay_factor(self) -> float:
        if self.half_life_seconds <= 0:
            return 0.0
        return 0.5 ** (self.age_seconds() / self.half_life_seconds)

    def apply_decay(self) -> None:
        factor = self.decay_factor()
        self.signal_value *= factor
        self.last_decay = datetime.datetime.utcnow()

class PheromoneStore:
    """In‑memory singleton store for pheromone entries."""
    _entries: Dict[str, PheromoneEntry] = {}

    @classmethod
    def add(cls, entry: PheromoneEntry) -> None:
        cls._entries[entry.uuid] = entry

    @classmethod
    def get_by_surface(cls, surface_key: str) -> List[PheromoneEntry]:
        return [e for e in cls._entries.values() if e.surface_key == surface_key]

    @classmethod
    def total_signal(cls, surface_key: str) -> float:
        total = 0.0
        for e in cls.get_by_surface(surface_key):
            e.apply_decay()
            total += e.signal_value
        return total

# ----------------------------------------------------------------------
# RBF Surrogate with pheromone‑modulated kernel
# ----------------------------------------------------------------------
def build_kernel_matrix(X: np.ndarray, epsilon: float, surface_key: str) -> np.ndarray:
    """Compute the Gaussian kernel matrix with ε scaled by pheromone signal."""
    # Effective epsilon incorporates pheromone signal
    pheromone_factor = PheromoneStore.total_signal(surface_key)
    eps_eff = epsilon * (1.0 + pheromone_factor)
    n = X.shape[0]
    K = np.empty((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            r = euclidean(X[i], X[j])
            K[i, j] = gaussian(r, eps_eff)
    return K

def fit_rbf_surrogate(X: np.ndarray, y: np.ndarray,
                     epsilon: float, surface_key: str) -> np.ndarray:
    """Solve K·w = y for the surrogate weights."""
    K = build_kernel_matrix(X, epsilon, surface_key)
    w = solve_linear(K, y)
    return w

def rbf_predict(x_new: np.ndarray, X_train: np.ndarray,
                w: np.ndarray, epsilon: float, surface_key: str) -> float:
    """Predict a scalar using the trained surrogate."""
    pheromone_factor = PheromoneStore.total_signal(surface_key)
    eps_eff = epsilon * (1.0 + pheromone_factor)
    k_vec = np.array([gaussian(euclidean(x_new, xi), eps_eff) for xi in X_train])
    return float(np.dot(w, k_vec))

# ----------------------------------------------------------------------
# Hybrid API (demonstration functions)
# ----------------------------------------------------------------------
def hybrid_fit_encrypt(X: np.ndarray, y: np.ndarray,
                       epsilon: float, surface_key: str,
                       rsa_pub: Tuple[int, int]) -> Dict[str, Any]:
    """
    Fit an RBF surrogate using pheromone‑adjusted kernels, evaluate the surrogate
    on the training set (mean value), convert the scalar to an integer message,
    and encrypt it with RSA.

    Returns a dictionary containing the ciphertext and all data required for
    decryption/prediction.
    """
    w = fit_rbf_surrogate(X, y, epsilon, surface_key)

    # Use the surrogate to produce a scalar (mean of predictions on training data)
    preds = np.array([rbf_predict(xi, X, w, epsilon, surface_key) for xi in X])
    surrogate_scalar = float(np.mean(preds))

    # Map float to a non‑negative integer suitable for RSA (simple scaling)
    message_int = int(abs(surrogate_scalar) * 1e6) % rsa_pub[1]

    cipher = rsa_encrypt(message_int, *rsa_pub)

    model = {
        "X_train": X.tolist(),
        "y_train": y.tolist(),
        "weights": w.tolist(),
        "epsilon": epsilon,
        "surface_key": surface_key,
        "rsa_pub": rsa_pub,
        "rsa_cipher": cipher,
    }
    return model

def hybrid_predict_decrypt(model: Dict[str, Any],
                           rsa_priv: Tuple[int, int],
                           x_new: np.ndarray) -> Tuple[float, int]:
    """
    Decrypt the RSA ciphertext to recover the surrogate scalar, then recompute
    the surrogate prediction for ``x_new`` using the stored model.

    Returns a tuple ``(prediction, decrypted_message)``.
    """
    cipher = model["rsa_cipher"]
    decrypted_int = rsa_decrypt(cipher, *rsa_priv)

    # Recover the original float approximation (inverse of scaling used in fit)
    recovered_scalar = decrypted_int / 1e6

    # Re‑run the surrogate prediction for the new point
    X_train = np.array(model["X_train"])
    w = np.array(model["weights"])
    epsilon = model["epsilon"]
    surface_key = model["surface_key"]
    prediction = rbf_predict(x_new, X_train, w, epsilon, surface_key)

    return prediction, decrypted_int

def region_blade_product(texts: List[str]) -> float:
    """
    Map each text to a simple vector (character code sums) and compute a
    Clifford‑algebra‑style product: the sum over pairwise dot products.
    This mirrors the geometric‑blade operation from Parent A.
    """
    vectors = []
    for txt in texts:
        # Very naive embedding: each character contributes its ordinal value
        vec = [float(ord(ch)) for ch in txt[:10]]  # limit length for stability
        vectors.append(vec)

    # Pad vectors to equal length
    max_len = max(len(v) for v in vectors)
    padded = [v + [0.0] * (max_len - len(v)) for v in vectors]

    total = 0.0
    for i in range(len(padded)):
        for j in range(i, len(padded)):
            dot = sum(a * b for a, b in zip(padded[i], padded[j]))
            total += dot
    return total

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Create pheromone entries for a surface
    surface = "demo_surface"
    for _ in range(3):
        entry = PheromoneEntry(
            surface_key=surface,
            signal_kind="signal",
            signal_value=random.uniform(0.5, 1.5),
            half_life_seconds=60  # 1 minute half‑life
        )
        PheromoneStore.add(entry)

    # 2. Generate synthetic training data
    rng = np.random.default_rng(42)
    X_train = rng.normal(size=(8, 3))
    y_train = np.sin(X_train[:, 0]) + np.cos(X_train[:, 1])  # some smooth target

    # 3. RSA key pair
    pub, priv = generate_rsa_keypair(bitlen=128)  # tiny keys for demo

    # 4. Fit surrogate and encrypt
    model = hybrid_fit_encrypt(
        X=X_train,
        y=y_train,
        epsilon=0.8,
        surface_key=surface,
        rsa_pub=pub
    )
    print("Ciphertext:", model["rsa_cipher"])

    # 5. Predict on a new point and decrypt
    x_new = rng.normal(size=(3,))
    pred, decrypted = hybrid_predict_decrypt(model, priv, x_new)
    print("Decrypted scalar (scaled int):", decrypted)
    print("Recovered surrogate scalar (float approx):", decrypted / 1e6)
    print("RBF prediction for new point:", pred)

    # 6. Demonstrate region_blade_product
    texts = ["hello world", "cryptic", "fusion"]
    blade_val = region_blade_product(texts)
    print("Region blade product:", blade_val)