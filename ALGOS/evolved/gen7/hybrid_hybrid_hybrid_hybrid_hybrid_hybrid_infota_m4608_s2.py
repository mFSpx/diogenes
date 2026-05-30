# DARWIN HAMMER — match 4608, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_rsa_ci_diffusion_forcing_m1607_s0.py (gen6)
# parent_b: hybrid_hybrid_infotaxis_min_hybrid_hybrid_hybrid_m2692_s3.py (gen6)
# born: 2026-05-29T23:56:57Z

"""Hybrid RSA‑RBF‑Diffusion‑Infotaxis Model
Combines:
- Parent A: Hybrid RSA‑RBF‑Surrogate with Diffusion Forcing
- Parent B: Hybrid Infotaxis with MinHash‑based feature labeling and pheromone decay

Mathematical bridge:
1. Diffusion forcing creates a stochastic perturbation  X̃ = X + η , η∼𝒩(0,σ²I).
2. An RBF surrogate learns a linear mapping **w** solving K w = Y where K_ij = exp(−ε²‖X̃_i−X̃_j‖²).
   The surrogate predicts clean features Ŷ = K̂ w for new noisy inputs.
3. Each predicted component ŷ_k is quantised, scaled to an integer and RSA‑encrypted:
   c_k = (⌊ŷ_k·scale⌋)^e mod n.
   The encrypted vector C acts as a secure signal.
4. Tokens are hashed by a simple MinHash (multiple seed‑based hashes) producing a low‑dim signature S.
5. A pheromone entry stores the RSA‑encrypted signal together with a decay governed by half‑life τ:
   signal(t) = C·0.5^{t/τ}.
   The signature S selects which pheromone entry to update, linking information entropy (via MinHash) to the secure diffusion‑RBF signal.

Thus diffusion, RBF approximation, RSA encryption, MinHash hashing and pheromone decay are fused into a single pipeline.
"""

import math
import random
import sys
import pathlib
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Shared utilities
# ----------------------------------------------------------------------
Vector = List[float]


def euclidean(a: Vector, b: Vector) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def gaussian_kernel(r: float, epsilon: float = 1.0) -> float:
    """Radial basis function kernel."""
    return math.exp(-((epsilon * r) ** 2))


def solve_linear(a: List[List[float]], b: List[float]) -> List[float]:
    """Solve Ax = b with numpy (wrapper for readability)."""
    return np.linalg.solve(np.array(a, dtype=float), np.array(b, dtype=float)).tolist()


# ----------------------------------------------------------------------
# RSA primitive (Parent A)
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


# ----------------------------------------------------------------------
# Diffusion forcing (Parent A)
# ----------------------------------------------------------------------
def diffuse(data: np.ndarray, sigma: float) -> np.ndarray:
    """Add isotropic Gaussian noise (diffusion) to each row of data."""
    noise = np.random.normal(loc=0.0, scale=sigma, size=data.shape)
    return data + noise


# ----------------------------------------------------------------------
# RBF Surrogate model (Parent A)
# ----------------------------------------------------------------------
class RBFSurrogate:
    """Train a linear surrogate in the RBF kernel space."""

    def __init__(self, epsilon: float = 1.0):
        self.epsilon = epsilon
        self.weights: np.ndarray | None = None
        self.train_X: np.ndarray | None = None

    def _kernel_matrix(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute K_ij = exp(-ε²‖x_i - x_j‖²)."""
        dists = np.linalg.norm(X1[:, None, :] - X2[None, :, :], axis=2)
        return np.exp(-((self.epsilon * dists) ** 2))

    def fit(self, X_noisy: np.ndarray, Y_clean: np.ndarray) -> None:
        """Solve K w = Y for each output dimension."""
        K = self._kernel_matrix(X_noisy, X_noisy)
        # Solve for each column of Y_clean independently
        w = np.linalg.solve(K, Y_clean)
        self.weights = w
        self.train_X = X_noisy

    def predict(self, X_new: np.ndarray) -> np.ndarray:
        """Predict clean output for new (possibly noisy) inputs."""
        if self.weights is None or self.train_X is None:
            raise RuntimeError("Model has not been fitted yet.")
        K_new = self._kernel_matrix(X_new, self.train_X)
        return K_new @ self.weights


# ----------------------------------------------------------------------
# MinHash signature (Parent B)
# ----------------------------------------------------------------------
def minhash_signature(tokens: List[str], num_hashes: int = 8, seed: int = 0) -> List[int]:
    """Simple MinHash: for each hash function, take the minimum hash value of the token set."""
    signatures = []
    for i in range(num_hashes):
        combined_seed = seed + i * 0x9e3779b9
        min_hash = sys.maxsize
        for token in tokens:
            h = hash((combined_seed, token))
            if h < min_hash:
                min_hash = h
        signatures.append(min_hash & ((1 << 64) - 1))  # force 64‑bit unsigned
    return signatures


# ----------------------------------------------------------------------
# Pheromone entry (Parent B)
# ----------------------------------------------------------------------
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
        """Mock age – in a real system this would be a timestamp diff."""
        return random.uniform(0, 200)

    def decay_factor(self) -> float:
        if self.half_life_seconds <= 0:
            return 0.0
        return 0.5 ** (self.age_seconds() / self.half_life_seconds)

    def apply_decay(self) -> None:
        factor = self.decay_factor()
        self.signal_value *= factor
        self.last_decay = pathlib.Path.cwd()


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def hybrid_encrypt_vector(vec: np.ndarray, e: int, n: int, scale: float = 1e6) -> List[int]:
    """Quantise a float vector, then RSA‑encrypt each component."""
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
    """
    End‑to‑end hybrid pipeline:
    1. Diffuse raw data.
    2. Train RBF surrogate on (noisy, clean) pair.
    3. Predict clean data for a fresh noisy sample.
    4. RSA‑encrypt the prediction.
    5. Build a pheromone entry whose signal is the sum of encrypted components,
       selected by a MinHash signature of the token list.
    """
    # 1. Diffusion forcing
    noisy = diffuse(raw_data, sigma)

    # 2. Train surrogate (using the same data as both noisy and clean for demo)
    surrogate = RBFSurrogate(epsilon=epsilon)
    surrogate.fit(noisy, raw_data)  # clean targets are the original data

    # 3. Predict on a fresh noisy copy
    fresh_noisy = diffuse(raw_data, sigma)
    pred_clean = surrogate.predict(fresh_noisy).flatten()

    # 4. RSA encryption of the prediction vector
    e, d, n = rsa_params
    encrypted_signal = hybrid_encrypt_vector(pred_clean, e, n)

    # 5. MinHash signature drives which pheromone entry we touch
    signature = minhash_signature(tokens, num_hashes=4, seed=42)
    chosen_uuid = f"ph-{signature[0] % 1000:03d}"

    entry = PheromoneEntry(
        uuid=chosen_uuid,
        surface_key="hybrid",
        signal_kind="rsa_rbf",
        signal_value=sum(encrypted_signal),
        half_life_seconds=half_life,
        created_at=pathlib.Path.cwd(),
        last_decay=pathlib.Path.cwd(),
    )
    entry.apply_decay()  # apply one decay step

    return encrypted_signal, entry


def compute_rbf_weights(
    X: np.ndarray,
    Y: np.ndarray,
    epsilon: float = 1.0,
) -> np.ndarray:
    """
    Direct computation of RBF surrogate weights without a class wrapper.
    Returns the weight matrix W solving K W = Y.
    """
    dists = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    K = np.exp(-((epsilon * dists) ** 2))
    W = np.linalg.solve(K, Y)
    return W


def update_pheromones_with_entropy(
    entries: List[PheromoneEntry],
    tokens: List[str],
    entropy_scale: float = 1.0,
) -> None:
    """
    Uses the Shannon entropy of token frequencies to modulate the
    signal_value of each pheromone entry after decay.
    """
    # Token frequency entropy
    freq = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    total = len(tokens)
    entropy = -sum((c / total) * math.log(c / total + 1e-12) for c in freq.values())
    factor = 1.0 + entropy_scale * entropy

    for entry in entries:
        entry.apply_decay()
        entry.signal_value *= factor


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic data: 10 samples, 3‑dimensional feature space
    np.random.seed(0)
    raw = np.random.rand(10, 3)

    # Token list simulating a short text
    token_list = ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]

    # Small RSA key (insecure, for demonstration only)
    p, q = 61, 53
    n = p * q
    phi = (p - 1) * (q - 1)
    e = 17
    # Compute d = e⁻¹ mod φ
    d = pow(e, -1, phi)

    enc_signal, pheromone = hybrid_process(
        raw_data=raw,
        tokens=token_list,
        rsa_params=(e, d, n),
        sigma=0.05,
        epsilon=0.8,
        half_life=120,
    )
    print("RSA‑encrypted signal vector (first 5 entries):", enc_signal[:5])
    print("Created pheromone entry:", pheromone)

    # Demonstrate direct RBF weight computation
    W = compute_rbf_weights(raw, raw, epsilon=0.8)
    print("RBF weight matrix shape:", W.shape)

    # Update a list of pheromones with entropy‑driven scaling
    entries = [pheromone]
    update_pheromones_with_entropy(entries, token_list, entropy_scale=0.5)
    print("Pheromone after entropy update:", entries[0])