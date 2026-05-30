# DARWIN HAMMER — match 174, survivor 0
# gen: 4
# parent_a: liquid_time_constant.py (gen0)
# parent_b: hybrid_hybrid_hdc_serpentin_hybrid_sparse_wta_hy_m117_s2.py (gen3)
# born: 2026-05-29T23:27:16Z

"""
This module implements a novel hybrid algorithm that mathematically fuses the core topologies of 
liquid_time_constant.py and hybrid_hybrid_hdc_serpentin_hybrid_sparse_wta_hy_m117_s2.py. 
The mathematical bridge between the two structures is the integration of the liquid time constant 
networks' input-dependent time constant with the hyperdimensional primitives' binding and bundling 
operations. Specifically, the hybrid algorithm uses the liquid time constant networks' ODE 
formulation to update the hidden state of the network, while employing the hyperdimensional 
primitives' binding and bundling operations to compute the input-dependent time constant.

The key innovation of this hybrid algorithm is the introduction of a new, hybrid operation that 
combines the strengths of both parent algorithms. This operation, called "hybrid_bind", takes the 
current hidden state and input as arguments and returns a bipolar hypervector that represents the 
input-dependent time constant. The hybrid_bind operation is then used to update the hidden state 
of the network using the ODE formulation of the liquid time constant networks.

The hybrid algorithm also includes a "hybrid_bundle" operation that takes a set of bipolar 
hypervectors as arguments and returns a single, bundled hypervector that represents the 
superposition of the input-dependent time constants. This operation is used to compute the 
asymptotic target state of the network.

Finally, the hybrid algorithm includes a "hybrid_step" operation that takes the current hidden 
state, input, and parameters as arguments and returns the updated hidden state of the network. This 
operation is used to simulate the dynamics of the hybrid network.
"""

import numpy as np
import random
import math
import sys
import pathlib

# Primitives
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

def random_vector(dim: int = 10000, seed: str | int | None = None) -> np.ndarray:
    rng = random.Random(seed)
    data = np.fromiter(
        (1 if rng.getrandbits(1) else -1 for _ in range(dim)), dtype=np.int8, count=dim
    )
    return data

def symbol_vector(symbol: str, dim: int = 10000) -> np.ndarray:
    seed = int.from_bytes(
        hashlib.sha256(symbol.encode("utf-8")).digest()[:8], byteorder="big"
    )
    return random_vector(dim, seed)

def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.shape != b.shape:
        raise ValueError("vectors must have equal shape")
    return a * b

def bundle(vectors: list) -> np.ndarray:
    vecs = list(vectors)
    if not vecs:
        raise ValueError("bundle requires at least one vector")
    stacked = np.stack(vecs, axis=0).astype(np.int32)
    summed = stacked.sum(axis=0)
    return np.where(summed >= 0, 1, -1).astype(np.int8)

# Hybrid operations
def hybrid_bind(x: np.ndarray, I: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    f = ltc_f(x, I, W, b)
    return symbol_vector(str(f), dim=10000)

def hybrid_bundle(vectors: list) -> np.ndarray:
    return bundle(vectors)

def hybrid_step(x: np.ndarray, I: np.ndarray, W: np.ndarray, b: np.ndarray, tau: float) -> np.ndarray:
    f = ltc_f(x, I, W, b)
    binding = hybrid_bind(x, I, W, b)
    dxdt = - (1 / tau + f) * x + f * binding
    return x + dxdt * 0.01  # Euler method with dt=0.01

if __name__ == "__main__":
    # Smoke test
    x = np.random.rand(10)
    I = np.random.rand(10)
    W = np.random.rand(20, 20)
    b = np.random.rand(20)
    tau = 1.0

    x_new = hybrid_step(x, I, W, b, tau)
    print(x_new)