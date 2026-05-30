# DARWIN HAMMER — match 174, survivor 1
# gen: 4
# parent_a: liquid_time_constant.py (gen0)
# parent_b: hybrid_hybrid_hdc_serpentin_hybrid_sparse_wta_hy_m117_s2.py (gen3)
# born: 2026-05-29T23:27:16Z

"""
Hybrid Algorithm: Fusing Liquid Time-Constant Networks (LTCs) with Hyperdimensional Computing (HDC) and Sparse WTA

This module integrates the continuous-time recurrent neural network structure of Liquid Time-Constant Networks (LTCs)
with the hyperdimensional computing and sparse winner-take-all (WTA) mechanisms. The mathematical bridge between the two
parent algorithms lies in the use of the LTC's learned gating function to modulate the hyperdimensional binding and
bundle operations, effectively creating a dynamic, input-dependent representation.

The core idea is to use the LTC's gating function to compute a time-varying, input-dependent weight matrix that is then
used to modulate the HDC binding and bundle operations. This allows the system to adaptively reconfigure its internal
representation in response to changing inputs.

Parent Algorithms:
- liquid_time_constant.py: Liquid Time-Constant Networks (LTCs)
- hybrid_hybrid_hdc_serpentin_hybrid_sparse_wta_hy_m117_s2.py: Hybrid HDC and Sparse WTA

"""

import numpy as np
import random
import hashlib
from typing import Any, Dict, Iterable, List, Sequence, Tuple
from collections import OrderedDict
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Hyperdimensional primitives
# ---------------------------------------------------------------------------
Vector = np.ndarray  # bipolar hypervector of dtype int8

def random_vector(dim: int = 10000, seed: str | int | None = None) -> Vector:
    rng = random.Random(seed)
    data = np.fromiter(
        (1 if rng.getrandbits(1) else -1 for _ in range(dim)), dtype=np.int8, count=dim
    )
    return data

def symbol_vector(symbol: str, dim: int = 10000) -> Vector:
    seed = int.from_bytes(
        hashlib.sha256(symbol.encode("utf-8")).digest()[:8], byteorder="big"
    )
    return random_vector(dim, seed)

def bind(a: Vector, b: Vector) -> Vector:
    if a.shape != b.shape:
        raise ValueError("vectors must have equal shape")
    return a * b

def bundle(vectors: Iterable[Vector]) -> Vector:
    vecs = list(vectors)
    if not vecs:
        raise ValueError("bundle requires at least one vector")
    stacked = np.stack(vecs, axis=0).astype(np.int32)
    summed = stacked.sum(axis=0)
    return np.where(summed >= 0, 1, -1).astype(np.int8)

def cosine_similarity(a: Vector, b: Vector) -> float:
    if a.shape != b.shape:
        raise ValueError("vectors must have equal shape")
    dot = float(np.dot(a, b))
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)

# ---------------------------------------------------------------------------
# Liquid Time-Constant Networks (LTCs) primitives
# ---------------------------------------------------------------------------
def sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )

def ltc_f(
    x: np.ndarray,
    I: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    return sigmoid(np.dot(W, np.concatenate((x, I))) + b)

def ltc_step(
    x: np.ndarray,
    I: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    A: np.ndarray,
    tau: float,
) -> np.ndarray:
    f_x = ltc_f(x, I, W, b)
    dxdt = - (1.0 / tau + f_x) * x + f_x * A
    return x + dxdt * 0.01  # assuming a fixed time step of 0.01

# ---------------------------------------------------------------------------
# Hybrid HDC-LTC-WTA
# ---------------------------------------------------------------------------
def hybrid_hdc_ltc_wta(
    x: np.ndarray,
    I: np.ndarray,
    W_hdc: np.ndarray,
    b_hdc: np.ndarray,
    W_ltc: np.ndarray,
    b_ltc: np.ndarray,
    A: np.ndarray,
    tau: float,
    dim: int,
) -> Tuple[np.ndarray, np.ndarray]:
    f_x = ltc_f(x, I, W_ltc, b_ltc)
    modulated_W_hdc = W_hdc * sigmoid(f_x)
    bound_vector = bind(random_vector(dim), I)
    bundled_vector = bundle([bound_vector, modulated_W_hdc])
    next_x = ltc_step(x, I, W_ltc, b_ltc, A, tau)
    return next_x, bundled_vector

def expand(values: Sequence[float], m: int, salt: str = "") -> np.ndarray:
    # assuming a simple expansion for demonstration purposes
    return np.array(values)

def sparse_wta(values: np.ndarray, k: int) -> np.ndarray:
    # assuming a simple WTA for demonstration purposes
    return np.argsort(-values)[:k]

def main():
    np.random.seed(0)
    random.seed(0)

    # Set up dimensions and parameters
    hidden_dim = 10
    input_dim = 5
    hdc_dim = 10000
    tau = 1.0

    # Initialize weights and biases
    W_ltc = np.random.rand(hidden_dim, hidden_dim + input_dim)
    b_ltc = np.random.rand(hidden_dim)
    A = np.random.rand(hidden_dim)
    W_hdc = np.random.randint(-1, 2, size=(hdc_dim, hdc_dim))

    # Initialize state and input
    x = np.random.rand(hidden_dim)
    I = np.random.rand(input_dim)

    # Run hybrid HDC-LTC-WTA
    next_x, bundled_vector = hybrid_hdc_ltc_wta(
        x, I, W_hdc, np.zeros(hdc_dim), W_ltc, b_ltc, A, tau, hdc_dim
    )

    print("Next state:", next_x)
    print("Bundled vector:", bundled_vector)

if __name__ == "__main__":
    main()