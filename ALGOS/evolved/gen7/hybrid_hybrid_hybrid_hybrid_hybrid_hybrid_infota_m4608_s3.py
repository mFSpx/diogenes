# DARWIN HAMMER — match 4608, survivor 3
# gen: 7
# parent_a: hybrid_hybrid_hybrid_rsa_ci_diffusion_forcing_m1607_s0.py (gen6)
# parent_b: hybrid_hybrid_infotaxis_min_hybrid_hybrid_hybrid_m2692_s3.py (gen6)
# born: 2026-05-29T23:56:57Z

import math
import random
import sys
import pathlib
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

Vector = List[float]

def euclidean(a: Vector, b: Vector) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def gaussian_kernel(r: float, epsilon: float = 1.0) -> float:
    return math.exp(-((epsilon * r) ** 2))

def solve_linear(a: List[List[float]], b: List[float]) -> List[float]:
    return np.linalg.solve(np.array(a, dtype=float), np.array(b, dtype=float)).tolist()

def rsa_encrypt(message: int, e: int, n: int) -> int:
    if not 0 <= message < n:
        raise ValueError("message must be in [0, n)")
    return pow(message, e, n)

def rsa_decrypt(ciphertext: int, d: int, n: int) -> int:
    if not 0 <= ciphertext < n:
        raise ValueError("ciphertext must be in [0, n)")
    return pow(ciphertext, d, n)

def diffuse(data: np.ndarray, sigma: float) -> np.ndarray:
    noise = np.random.normal(loc=0.0, scale=sigma, size=data.shape)
    return data + noise

class RBFSurrogate:
    def __init__(self, epsilon: float = 1.0, reg: float = 1e-6):
        self.epsilon = epsilon
        self.reg = reg
        self.weights: np.ndarray | None = None
        self.train_X: np.ndarray | None = None

    def _kernel_matrix(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        dists = np.linalg.norm(X1[:, None, :] - X2[None, :, :], axis=2)
        return np.exp(-((self.epsilon * dists) ** 2))

    def fit(self, X_noisy: np.ndarray, Y_clean: np.ndarray) -> None:
        K = self._kernel_matrix(X_noisy, X_noisy)
        K_reg = K + self.reg * np.eye(K.shape[0])
        w = np.linalg.solve(K_reg, Y_clean)
        self.weights = w
        self.train_X = X_noisy

    def predict(self, X_new: np.ndarray) -> np.ndarray:
        if self.weights is None or self.train_X is None:
            raise RuntimeError("Model has not been fitted yet.")
        K_new = self._kernel_matrix(X_new, self.train_X)
        return K_new @ self.weights

def minhash_signature(tokens: List[str], num_hashes: int = 8, seed: int = 0) -> List[int]:
    signatures = []
    for i in range(num_hashes):
        combined_seed = seed + i * 0x9e3779b9
        min_hash = sys.maxsize
        for token in tokens:
            h = hash((combined_seed, token))
            if h < min_hash:
                min_hash = h
        signatures.append(min_hash & ((1 << 64) - 1))  
    return signatures

@dataclass
class PheromoneEntry:
    uuid: str
    surface_key: str
    signal_kind: str
    signal_value: float
    half_life_seconds: int
    created_at: pathlib.Path
    last_decay: pathlib.Path

    def age_seconds(self) -> float:
        return random.uniform(0, 200)

    def decay_factor(self) -> float:
        if self.half_life_seconds <= 0:
            return 0.0
        return 0.5 ** (self.age_seconds() / self.half_life_seconds)

    def apply_decay(self) -> None:
        factor = self.decay_factor()
        self.signal_value *= factor
        self.last_decay = pathlib.Path.cwd()

def hybrid_encrypt_vector(vec: np.ndarray, e: int, n: int, scale: float = 1e6) -> List[int]:
    encrypted = []
    for val in vec:
        q = int(math.floor(val * scale)) % n
        encrypted.append(rsa_encrypt(q, e, n))
    return encrypted

def hybrid_process(
    raw_data: np.ndarray,
    tokens: List[str],
    rsa_params: Tuple[int, int, int],
    sigma: float = 0.1,
    epsilon: float = 1.0,
    half_life: int = 60,
) -> Tuple[List[int], PheromoneEntry]:
    noisy = diffuse(raw_data, sigma)
    surrogate = RBFSurrogate(epsilon=epsilon, reg=1e-6)
    surrogate.fit(noisy, raw_data)
    fresh_noisy = diffuse(raw_data, sigma)
    pred_clean = surrogate.predict(fresh_noisy).flatten()
    e, d, n = rsa_params
    encrypted_signal = hybrid_encrypt_vector(pred_clean, e, n)
    signature = minhash_signature(tokens, num_hashes=4, seed=42)
    chosen_uuid = f"ph-{signature[0] % 1000:03d}"
    entry = PheromoneEntry(
        uuid=chosen_uuid,
        surface_key="hybrid",
        signal_kind="rsa_rbf",
        signal_value=sum(encrypted_signal),
        half_life_seconds=half_life,
        created_at=pathlib.Path.cwd(),
        last_decay=pathlib.Path.cwd()
    )
    return encrypted_signal, entry