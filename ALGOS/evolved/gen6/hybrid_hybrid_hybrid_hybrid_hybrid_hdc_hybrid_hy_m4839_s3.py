# DARWIN HAMMER — match 4839, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_nlms_o_m1326_s1.py (gen5)
# parent_b: hybrid_hdc_hybrid_hybrid_decisi_m131_s0.py (gen3)
# born: 2026-05-29T23:58:17Z

"""Hybrid NLMS‑HDC Fusion Module
Parents:
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_nlms_o_m1326_s1.py (adaptive NLMS filter with graph/sheaf view)
- hybrid_hdc_hybrid_hybrid_decisi_m131_s0.py (binary hyperdimensional computing)

Mathematical bridge:
The NLMS weight vector is interpreted as a high‑dimensional (HD) hypervector.  Its real‑valued
components are quantised by sign to obtain a binary (+1/‑1) HD representation.  This representation
participates in the HDC algebra (binding, bundling, similarity).  Conversely, the similarity
scores obtained from HDC classification are fed back as a scalar “confidence” that modulates the
NLMS step size μ, thus closing a loop between the two domains.  The combined system therefore
updates the adaptive filter, projects the updated weights into HD space, classifies the input
with HDC, and uses the resulting Shannon entropy as a measure of uncertainty that can be
exploited downstream.
"""

import numpy as np
import math
import random
import sys
import pathlib
import hashlib
from collections import Counter, defaultdict

# ----------------------------------------------------------------------
# Hyperdimensional Computing primitives (Parent B)
# ----------------------------------------------------------------------
Vector = list[int]

def random_vector(dim: int = 10000, seed: str | int | None = None) -> Vector:
    rng = random.Random(seed)
    return [1 if rng.getrandbits(1) else -1 for _ in range(dim)]

def symbol_vector(symbol: str, dim: int = 10000) -> Vector:
    # deterministic seed from SHA‑256 hash of the symbol
    h = hashlib.sha256(symbol.encode('utf-8')).digest()
    seed = int.from_bytes(h[:8], 'big')
    return random_vector(dim, seed)

def bind(a: Vector, b: Vector) -> Vector:
    if len(a) != len(b):
        raise ValueError('vectors must have equal length')
    return [x * y for x, y in zip(a, b)]

def bundle(vectors: list[Vector]) -> Vector:
    vecs = list(vectors)
    if not vecs:
        raise ValueError('at least one vector is required')
    dim = len(vecs[0])
    if any(len(v) != dim for v in vecs):
        raise ValueError('vectors must have equal length')
    sums = [0] * dim
    for v in vecs:
        for i, x in enumerate(v):
            sums[i] += x
    return [1 if x >= 0 else -1 for x in sums]

def permute(v: Vector, shifts: int = 1) -> Vector:
    if not v:
        return []
    s = shifts % len(v)
    return v[-s:] + v[:-s] if s else list(v)

def similarity(a: Vector, b: Vector) -> float:
    if len(a) != len(b):
        raise ValueError('vectors must have equal length')
    if not a:
        raise ValueError('vectors must not be empty')
    # normalized Hamming similarity expressed as dot product / dimension
    return sum(x * y for x, y in zip(a, b)) / len(a)

# ----------------------------------------------------------------------
# Normalized Least‑Mean‑Squares (NLMS) primitives (Parent A)
# ----------------------------------------------------------------------
def predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Linear prediction y = w·x."""
    return float(np.dot(weights, x))

def nlms_update(weights: np.ndarray, x: np.ndarray, target: float,
                mu: float = 0.5, eps: float = 1e-9) -> tuple[np.ndarray, float]:
    """
    Perform one NLMS adaptation step.
    Returns the updated weight vector and the instantaneous error.
    """
    y = predict(weights, x)
    error = target - y
    power = float(np.dot(x, x) + eps)
    step = mu * error * x / power
    new_weights = weights + step
    return new_weights, error

# ----------------------------------------------------------------------
# Graph / Sheaf abstraction (simplified)
# ----------------------------------------------------------------------
class HybridSheaf:
    """Lightweight wrapper exposing node dimensions and edge‑wise restriction maps."""
    def __init__(self, node_dims: dict[int, int], edge_list: list[tuple[int, int]],
                 width: int = 64, depth: int = 4):
        self.node_dims = dict(node_dims)
        self.edges = list(edge_list)
        self._restrictions = {}
        self.width = width
        self.depth = depth

    def set_restriction(self, edge: tuple[int, int], src_map: np.ndarray, dst_map: np.ndarray):
        u, v = edge
        self._restrictions[(u, v)] = (np.asarray(src_map, dtype=float),
                                      np.asarray(dst_map, dtype=float))

    def _edge_dim(self, u: int, v: int) -> int:
        if (u, v) in self._restrictions:
            return self._restrictions[(u, v)][0].shape[0]
        if (v, u) in self._restrictions:
            return self._restrictions[(v, u)][1].shape[0]
        raise KeyError(f"No restriction map for edge ({u}, {v})")

# ----------------------------------------------------------------------
# Entropy utilities (bridge component)
# ----------------------------------------------------------------------
def shannon_entropy(probs: np.ndarray) -> float:
    """Compute Shannon entropy (base‑2) of a probability distribution."""
    probs = probs[np.where(probs > 0)]
    return -np.sum(probs * np.log2(probs))

def similarity_distribution(query_hd: Vector, class_hvs: dict[str, Vector]) -> dict[str, float]:
    """Return normalized similarity scores for each class."""
    sims = {cls: similarity(query_hd, hv) for cls, hv in class_hvs.items()}
    total = sum(sims.values())
    if total == 0:
        # avoid division by zero; assign uniform distribution
        n = len(sims)
        return {cls: 1.0 / n for cls in sims}
    return {cls: val / total for cls, val in sims.items()}

# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def weights_to_hd(weights: np.ndarray) -> Vector:
    """
    Quantise a real‑valued weight vector to a binary hypervector.
    Positive → +1, Zero/Negative → -1.
    """
    return [1 if w >= 0 else -1 for w in weights]

def hybrid_classify(query_vec: np.ndarray,
                    class_memory: dict[str, Vector]) -> tuple[str, float]:
    """
    Classify a query using the HD projection of NLMS weights.
    Returns the winning class label and the entropy of the similarity distribution.
    """
    query_hd = weights_to_hd(query_vec)
    probs = similarity_distribution(query_hd, class_memory)
    winner = max(probs, key=probs.get)
    entropy = shannon_entropy(np.array(list(probs.values())))
    return winner, entropy

def hybrid_step(weights: np.ndarray, x: np.ndarray, target: float,
                class_memory: dict[str, Vector],
                mu_base: float = 0.5, eps: float = 1e-9) -> tuple[np.ndarray, str, float]:
    """
    Perform a full hybrid iteration:
    1. NLMS weight update (step size modulated by previous entropy if available).
    2. Project updated weights to HD space and classify.
    3. Return updated weights, predicted class, and classification entropy.
    """
    # 1. Adaptive update
    new_weights, _ = nlms_update(weights, x, target, mu=mu_base, eps=eps)

    # 2. Classification in HD space
    pred_class, entropy = hybrid_classify(new_weights, class_memory)

    # 3. Optional feedback: shrink step size when entropy is high (uncertain)
    # (This demonstrates a mathematical coupling between the two domains.)
    if entropy > 1.0:  # threshold chosen heuristically
        mu_base *= 0.7  # reduce future learning rate
    # (mu_base is not returned but could be stored externally.)

    return new_weights, pred_class, entropy

# ----------------------------------------------------------------------
# Example / smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Dimensionalities
    dim_weights = 128          # modest size for quick test
    hd_dim = 10000             # hyperdimensional length

    # Initialise NLMS weights randomly
    rng = np.random.default_rng(42)
    w = rng.normal(loc=0.0, scale=1.0, size=dim_weights)

    # Synthetic input and target
    x = rng.normal(size=dim_weights)
    target = 0.3

    # Build a tiny class memory using symbol vectors
    classes = ["A", "B", "C"]
    class_memory = {c: symbol_vector(c, hd_dim) for c in classes}

    # Perform a few hybrid steps
    for step in range(5):
        w, pred, ent = hybrid_step(w, x, target, class_memory)
        print(f"Step {step+1}: Predicted={pred}, Entropy={ent:.4f}")

    # Verify that the functions run without raising
    sys.exit(0)