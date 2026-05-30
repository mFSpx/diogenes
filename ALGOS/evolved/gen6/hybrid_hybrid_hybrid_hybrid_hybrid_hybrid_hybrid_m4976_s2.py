# DARWIN HAMMER — match 4976, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hdc_hy_hybrid_hybrid_fisher_m1920_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_rbf_su_hybrid_hybrid_ternar_m225_s2.py (gen3)
# born: 2026-05-29T23:59:04Z

"""
Hybrid Algorithm Fusion of:
- hybrid_hybrid_hybrid_hdc_hybrid_hy_hybrid_hybrid_fisher_m1920_s0.py (Parent A)
- hybrid_hybrid_hybrid_rbf_su_hybrid_hybrid_ternar_m225_s2.py (Parent B)

Mathematical Bridge:
Both parents employ Gaussian‑based kernels.  Parent A uses a Gaussian beam to
define a Fisher information term, while Parent B uses a Gaussian radial‑basis
function (RBF) of Euclidean distance.  The fusion multiplies these two Gaussian
components to obtain a *joint kernel*:

    K(x, c) = exp(‑½·((θ‑μ)/σ)²)   ×   exp(‑(ε·‖x‑c‖)²)

where the first factor is the Fisher‑Gaussian (θ, μ, σ) from Parent A and the
second factor is the RBF Gaussian (ε, distance) from Parent B.  This joint kernel
serves both as the similarity measure for bound vectors and as the basis for a
linear surrogate model solved via the exact Gaussian elimination routine from
Parent B.

The resulting system provides:
1. A vector‑binding similarity (Parent A).
2. A Fisher‑scaled RBF surrogate prediction (Parents A + B).
3. Linear weight solving (Parent B) to train the surrogate on arbitrary data.
"""

import math
import random
import sys
import pathlib
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence, List, Tuple

Vector = Sequence[float]

# ----------------------------------------------------------------------
# Utilities from Parent A
# ----------------------------------------------------------------------
def random_vector(dim: int = 10000, seed: str | int | None = None) -> List[int]:
    rng = random.Random(seed)
    return [1 if rng.getrandbits(1) else -1 for _ in range(dim)]

def symbol_vector(symbol: str, dim: int = 10000) -> List[int]:
    import hashlib
    seed = int.from_bytes(hashlib.sha256(symbol.encode('utf-8')).digest()[:8], 'big')
    return random_vector(dim, seed)

def bind(a: List[float], b: List[float]) -> List[float]:
    if len(a) != len(b):
        raise ValueError('vectors must have equal length')
    return [x * y for x, y in zip(a, b)]

def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

# ----------------------------------------------------------------------
# Utilities from Parent B
# ----------------------------------------------------------------------
def gaussian_rbf(r: float, epsilon: float = 1.0) -> float:
    """RBF Gaussian kernel (no distance squared term inside epsilon)."""
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: Vector, b: Vector) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def solve_linear(a: List[List[float]], b: List[float]) -> List[float]:
    """Exact Gaussian elimination with partial pivoting."""
    n = len(b)
    m = [row[:] + [rhs] for row, rhs in zip(a, b)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            raise ValueError("singular surrogate system")
        m[col], m[pivot] = m[pivot], m[col]
        div = m[col][col]
        m[col] = [v / div for v in m[col]]
        for row in range(n):
            if row == col:
                continue
            factor = m[row][col]
            m[row] = [v - factor * p for v, p in zip(m[row], m[col])]
    return [row[-1] for row in m]

# ----------------------------------------------------------------------
# Hybrid Core: Joint Gaussian Kernel
# ----------------------------------------------------------------------
def joint_kernel(x: Vector, c: Vector,
                 epsilon: float,
                 theta: float, mu: float, sigma: float) -> float:
    """
    Combined kernel K(x,c) = RBF_Gaussian(euclidean(x,c), epsilon) *
                               Fisher_Gaussian(theta, mu, sigma)
    """
    r = euclidean(x, c)
    rbf_part = gaussian_rbf(r, epsilon)
    fisher_part = gaussian_beam(theta, mu, sigma)  # intensity without derivative
    return rbf_part * fisher_part

# ----------------------------------------------------------------------
# Surrogate Model with Joint Kernel
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class JointRBFSurrogate:
    centers: List[Tuple[float, ...]]
    weights: List[float]
    epsilon: float
    theta: float
    mu: float
    sigma: float

    def predict(self, x: Vector) -> float:
        return sum(w * joint_kernel(x, c, self.epsilon,
                                    self.theta, self.mu, self.sigma)
                   for w, c in zip(self.weights, self.centers))

# ----------------------------------------------------------------------
# Training routine (uses solve_linear from Parent B)
# ----------------------------------------------------------------------
def train_joint_surrogate(samples: List[Vector],
                          targets: List[float],
                          epsilon: float = 1.0,
                          theta: float = 0.0,
                          mu: float = 0.0,
                          sigma: float = 1.0) -> JointRBFSurrogate:
    """
    Build a surrogate where the kernel matrix K_ij = joint_kernel(sample_i,
    sample_j, epsilon, theta, mu, sigma).  Solve K * w = targets for weights.
    """
    if len(samples) != len(targets):
        raise ValueError("samples and targets must have equal length")
    n = len(samples)
    K: List[List[float]] = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            K[i][j] = joint_kernel(samples[i], samples[j],
                                   epsilon, theta, mu, sigma)
    weights = solve_linear(K, targets)
    # Freeze centers as tuples for hashability
    centers = [tuple(s) for s in samples]
    return JointRBFSurrogate(centers, weights, epsilon, theta, mu, sigma)

# ----------------------------------------------------------------------
# Hybrid Operations Demonstrating Integration
# ----------------------------------------------------------------------
def hybrid_operation(vector1: List[float],
                    vector2: List[float],
                    theta: float,
                    mu: float,
                    sigma: float) -> float:
    """
    Original hybrid operation (Parent A) extended with a Fisher‑scaled RBF
    weighting.  Returns similarity * Fisher information.
    """
    bound = bind(vector1, vector2)
    similarity = sum(bound) / len(bound)
    fisher = fisher_score(theta, mu, sigma)
    return similarity * fisher

def hybrid_predict(samples: List[Vector],
                   targets: List[float],
                   query: Vector,
                   epsilon: float = 1.0,
                   theta: float = 0.0,
                   mu: float = 0.0,
                   sigma: float = 1.0) -> float:
    """
    Trains a JointRBFSurrogate on (samples, targets) and predicts the value at
    `query`.  Demonstrates the full pipeline from training to evaluation.
    """
    surrogate = train_joint_surrogate(samples, targets,
                                      epsilon, theta, mu, sigma)
    return surrogate.predict(query)

def hybrid_vector_fisher_score(vector: List[float],
                               theta: float,
                               mu: float,
                               sigma: float) -> float:
    """
    Computes a Fisher‑information based score for a single vector by first
    binding it with a fixed reference (all‑ones) and then applying the joint
    kernel against a dummy centre (zero vector).  This showcases the
    interplay of binding, Fisher Gaussian, and RBF Gaussian in a single
    scalar output.
    """
    reference = [1.0] * len(vector)
    bound = bind(vector, reference)
    similarity = sum(bound) / len(bound)

    dummy_center = tuple(0.0 for _ in bound)
    fisher_part = gaussian_beam(theta, mu, sigma)
    rbf_part = gaussian_rbf(euclidean(bound, dummy_center), epsilon=1.0)

    return similarity * fisher_part * rbf_part

# ----------------------------------------------------------------------
# Simple smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Small dimensionality for quick execution
    dim = 64
    vec_a = random_vector(dim, seed=42)
    vec_b = random_vector(dim, seed=99)

    # Parameters for Gaussian components
    theta, mu, sigma = 0.3, 0.0, 1.0

    # Demonstrate hybrid_operation
    op_val = hybrid_operation(vec_a, vec_b, theta, mu, sigma)
    print(f"Hybrid operation value: {op_val:.6f}")

    # Prepare training data for surrogate
    train_samples = [random_vector(dim, seed=i) for i in range(5)]
    # Simple target: sum of elements (just for testing)
    train_targets = [sum(v) / dim for v in train_samples]

    query_vec = random_vector(dim, seed=123)

    pred = hybrid_predict(train_samples, train_targets, query_vec,
                          epsilon=0.5, theta=theta, mu=mu, sigma=sigma)
    print(f"Hybrid surrogate prediction: {pred:.6f}")

    # Demonstrate hybrid_vector_fisher_score
    score = hybrid_vector_fisher_score(vec_a, theta, mu, sigma)
    print(f"Hybrid vector Fisher score: {score:.6f}")

    sys.exit(0)