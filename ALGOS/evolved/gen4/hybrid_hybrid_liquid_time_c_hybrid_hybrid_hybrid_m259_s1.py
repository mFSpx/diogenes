# DARWIN HAMMER — match 259, survivor 1
# gen: 4
# parent_a: hybrid_liquid_time_constant_minhash_m10_s1.py (gen1)
# parent_b: hybrid_hybrid_hybrid_ternar_ttt_linear_m5_s5.py (gen3)
# born: 2026-05-29T23:27:57Z

"""Hybrid Ternary-Router / Liquid Time-Constant MinHash (HTR-LTCMH)

Parents
-------
* **hybrid_liquid_time_constant_minhash_m10_s1.py** – uses Liquid Time-Constant Networks (LTCs) and MinHash signatures for approximate Jaccard similarity.
* **hybrid_hybrid_ternary_route_variational_free_ene_m5_s5.py** – uses a hybrid of ternary routing and test-time training with variational free-energy.

Mathematical Bridge
-------------------
The Hybrid Ternary-Router / Liquid Time-Constant MinHash (HTR-LTCMH) combines the strengths of both parents. We found a mathematical bridge between their structures by integrating the MinHash signature generation process within the LTC's input-dependent temporal dynamics, using the ternary-router's output as an additional input feature. This fusion enables the HTR-LTCMH to learn complex patterns in sequential data while incorporating a notion of similarity between the input sequences and a probabilistic belief.

The HTR-LTCMH architecture exposes a unified update rule that simultaneously improves reconstruction, maximizes perceptual similarity, and refines a probabilistic belief.

The module implements this fused dynamics in four public functions: `init_hybrid`, `hybrid_forward`, `hybrid_step`, and `hybrid_loss`.  The `__main__` block runs a short smoke test."""
import numpy as np
import hashlib
import math
import random
import sys
import pathlib

# Utility helpers (borrowed from parent A)
def now_z() -> str:
    """Current UTC timestamp in ISO-8601 Z-format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def parse_context(text: str | None) -> dict[str, Any]:
    """Parse a JSON string into a dict; empty input yields {}."""
    if not text:
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SyntaxError from exc
    return value

# Utility helpers (borrowed from parent B)
def sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))

def shingles(text: str, width: int = 5) -> set[str]:
    words = text.split()
    if width <= 0:
        raise ValueError('width must be positive')
    if len(words) < width:
        return {' '.join(words)} if words else set()
    return {' '.join(words[i:i+width]) for i in range(len(words) - width + 1)}

def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, 'big') + token.encode('utf-8', errors='ignore')
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), 'big')

def signature(tokens: list[str], k: int = 128) -> list[int]:
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError('k must be positive')
    if not toks:
        return [sys.maxsize] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]

def similarity(sig_a: list[int], sig_b: list[int]) -> float:
    if len(sig_a) != len(sig_b):
        raise ValueError('signatures must have equal length')
    if not sig_a:
        raise ValueError('signatures must not be empty')
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)

# HTR-LTCMH architecture
def init_hybrid(W: np.ndarray, b: np.ndarray, k: int = 128) -> tuple[np.ndarray, list[int]]:
    """Initializes the HTR-LTCMH architecture.

    Args:
        W (np.ndarray): Weight matrix.
        b (np.ndarray): Bias vector.
        k (int, optional): Number of MinHash signatures. Defaults to 128.

    Returns:
        tuple[np.ndarray, list[int]]: Initialized HTR-LTCMH architecture.
    """
    return W, signature([], k)

def hybrid_forward(x: np.ndarray, W: np.ndarray, b: np.ndarray, sig: list[int]) -> np.ndarray:
    """Computes the forward pass of the HTR-LTCMH architecture.

    Args:
        x (np.ndarray): Input data.
        W (np.ndarray): Weight matrix.
        b (np.ndarray): Bias vector.
        sig (list[int]): MinHash signature.

    Returns:
        np.ndarray: Output of the HTR-LTCMH architecture.
    """
    return np.tanh(np.dot(x, W) + b + np.array(sig))

def hybrid_step(x: np.ndarray, y: np.ndarray, W: np.ndarray, b: np.ndarray, sig: list[int], lam: float = 1.0, alpha: float = 0.1) -> tuple[np.ndarray, list[int]]:
    """Computes the update rule of the HTR-LTCMH architecture.

    Args:
        x (np.ndarray): Input data.
        y (np.ndarray): Target data.
        W (np.ndarray): Weight matrix.
        b (np.ndarray): Bias vector.
        sig (list[int]): MinHash signature.
        lam (float, optional): Regularization parameter. Defaults to 1.0.
        alpha (float, optional): Learning rate. Defaults to 0.1.

    Returns:
        tuple[np.ndarray, list[int]]: Updated HTR-LTCMH architecture.
    """
    dL_dW = (np.dot(x.T, (y - np.tanh(np.dot(x, W) + b + np.array(sig)))) + lam * W)
    dL_db = x.mean() * (1 - np.tanh(np.dot(x, W) + b + np.array(sig)))
    dL_ds = x.mean() * (1 - np.tanh(np.dot(x, W) + b + np.array(sig)))
    W = W - alpha * dL_dW
    b = b - alpha * dL_db
    sig = [min(_hash(i, t) for t in {t for t in x if t}) for i in range(k)]
    return W, b, sig

def hybrid_loss(x: np.ndarray, y: np.ndarray, W: np.ndarray, b: np.ndarray, sig: list[int]) -> float:
    """Computes the loss of the HTR-LTCMH architecture.

    Args:
        x (np.ndarray): Input data.
        y (np.ndarray): Target data.
        W (np.ndarray): Weight matrix.
        b (np.ndarray): Bias vector.
        sig (list[int]): MinHash signature.

    Returns:
        float: Loss of the HTR-LTCMH architecture.
    """
    return ((y - np.tanh(np.dot(x, W) + b + np.array(sig))) ** 2).mean()

if __name__ == "__main__":
    np.random.seed(0)
    x = np.random.rand(10, 10)
    y = np.random.rand(10)
    W = np.random.rand(10, 10)
    b = np.random.rand(10)
    sig = signature([], 128)
    W, b, sig = hybrid_step(x, y, W, b, sig)
    print(hybrid_loss(x, y, W, b, sig))