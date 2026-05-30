# DARWIN HAMMER — match 5380, survivor 4
# gen: 5
# parent_a: hybrid_hybrid_privacy_sketc_hybrid_hybrid_hybrid_m1610_s2.py (gen4)
# parent_b: hybrid_hybrid_rbf_surrogate_hybrid_hard_truth_ma_m93_s2.py (gen3)
# born: 2026-05-30T00:01:30Z

"""Hybrid algorithm merging differential‑privacy Count‑Min sketching (Parent A) with
RBF‑surrogate modelling and Hoeffding‑bound driven incremental updates (Parent B).

Mathematical bridge
-------------------
* From Parent A we obtain a noisy histogram **f ∈ ℝ^{w}** of hashed
  quasi‑identifiers by building a Count‑Min sketch **C ∈ ℤ^{d×w}** and adding
  Laplace noise with scale Δ/ε_priv.

* Parent B supplies an RBF surrogate  

  **s = g(f) = [ Σ_k w_{k}^{(m)}·exp(‑(‖f‑c_{k}‖·ε_rbf)²) ]_{m=1..M}**  

  i.e. a *vector* of **M** predicted stylometric features (each component uses its
  own weight vector but shares the same centres).

* The fused model multiplies the two representations with a matrix product  

  **Z = s ⊗ f = s·fᵀ ∈ ℝ^{M×w}**  

  (outer product).  Z can be interpreted as a joint “risk‑frequency” matrix
  that simultaneously captures privacy‑risk (via the noisy sketch) and
  stylometric diversity (via the surrogate).

* An incremental Hoeffding bound  

  **Δₙ = √( (R²·ln(1/δ)) / (2·n) )**  

  monitors the change of the risk scalar **r̂ = Σ_m s_m**.  When the absolute
  change between successive estimates falls below Δₙ the sketch is considered
  stable and updates stop.

The module below implements this fusion and provides three public functions
demonstrating the hybrid workflow.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Differential‑privacy Count‑Min sketch utilities
# ----------------------------------------------------------------------
def _hash(item: str, seed: int, width: int) -> int:
    rnd = random.Random((item, seed))
    return rnd.randrange(width)

def laplace_noise(scale: float) -> float:
    """Generate a single Laplace(0, scale) sample."""
    u = random.random() - 0.5
    return -scale * math.copysign(1.0, u) * math.log(1 - 2 * abs(u))

class CountMinSketch:
    """Count‑Min sketch with Laplace DP noise."""
    def __init__(self, depth: int, width: int, eps_priv: float, delta_priv: float = 1e-5):
        self.d = depth
        self.w = width
        self.eps = eps_priv
        self.delta = delta_priv
        self.table = np.zeros((depth, width), dtype=np.int64)
        # hash seeds are fixed per row for reproducibility
        self.seeds = [random.randint(0, 2**31 - 1) for _ in range(depth)]

    def add(self, item: str, count: int = 1) -> None:
        for i, seed in enumerate(self.seeds):
            col = _hash(item, seed, self.w)
            self.table[i, col] += count

    def noisy_histogram(self) -> np.ndarray:
        """Return the DP‑noisy frequency estimate vector f (length w)."""
        scale = 1.0 / self.eps  # sensitivity of a single increment is 1
        noisy = self.table.astype(np.float64) + np.vectorize(laplace_noise)(scale)
        f = np.min(noisy, axis=0)   # min‑over‑rows estimator
        return np.clip(f, a_min=0.0, a_max=None)   # frequencies cannot be negative

# ----------------------------------------------------------------------
# Parent B – RBF surrogate utilities
# ----------------------------------------------------------------------
def _euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def _gaussian(r: float, epsilon: float) -> float:
    return math.exp(-((epsilon * r) ** 2))

@dataclass(frozen=True)
class RBFSurrogate:
    """Vector‑valued RBF surrogate.
    
    *centers* – list of centre vectors (shared across all output dimensions)
    *weights* – list of weight vectors, one per output dimension (length = M)
    *epsilon* – RBF shape parameter
    """
    centers: List[Tuple[float, ...]]
    weights: List[List[float]]   # shape (M, K)
    epsilon: float = 1.0

    def predict(self, x: Sequence[float]) -> List[float]:
        """Return a list of M predictions for input vector x."""
        out = []
        for w_vec in self.weights:
            val = sum(
                w * _gaussian(_euclidean(x, c), self.epsilon)
                for w, c in zip(w_vec, self.centers)
            )
            out.append(val)
        return out

# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def build_noisy_sketch(data: Iterable[str],
                       depth: int = 5,
                       width: int = 1024,
                       eps_priv: float = 1.0) -> Tuple[CountMinSketch, np.ndarray]:
    """
    Build a Count‑Min sketch from an iterable of identifiers, add DP noise,
    and return both the sketch object and its noisy histogram vector f.
    """
    sketch = CountMinSketch(depth, width, eps_priv)
    for item in data:
        sketch.add(item)
    f = sketch.noisy_histogram()
    return sketch, f

def rbf_predict_features(f: np.ndarray, surrogate: RBFSurrogate) -> List[float]:
    """
    Feed the noisy sketch vector f into the RBF surrogate and obtain a
    stylometric feature vector s.
    """
    # The surrogate expects a 1‑D sequence; we provide f as list.
    return surrogate.predict(f.tolist())

def combine_features_and_frequencies(s: Sequence[float],
                                     f: np.ndarray) -> np.ndarray:
    """
    Outer‑product fusion Z = s ⊗ f (matrix of shape M×w).
    """
    s_arr = np.array(s, dtype=np.float64).reshape(-1, 1)   # (M,1)
    return s_arr @ f.reshape(1, -1)                       # (M,w)

def incremental_sketch_with_hoeffding(data_iter: Iterable[str],
                                     surrogate: RBFSurrogate,
                                     depth: int = 5,
                                     width: int = 1024,
                                     eps_priv: float = 1.0,
                                     delta: float = 0.05,
                                     confidence: float = 0.99,
                                     max_passes: int = 10) -> Tuple[CountMinSketch, List[float]]:
    """
    Incrementally ingest records, recompute risk after each pass, and stop when
    the Hoeffding bound guarantees that the change in risk is negligible.
    
    Returns the final sketch and the last stylometric feature vector.
    """
    sketch = CountMinSketch(depth, width, eps_priv)
    prev_risk = float('inf')
    risk_history: List[float] = []

    n = 0
    for pass_no in range(max_passes):
        for item in data_iter:
            sketch.add(item)
            n += 1

        f = sketch.noisy_histogram()
        s = surrogate.predict(f.tolist())
        risk = sum(s)                       # scalar risk proxy
        risk_history.append(risk)

        # Hoeffding bound for the risk estimator (range R = max(s)-min(s) ≤ 2)
        R = 2.0
        hoeffding_delta = math.sqrt((R * R * math.log(1.0 / (1 - confidence))) / (2 * n))
        if abs(risk - prev_risk) <= hoeffding_delta:
            break
        prev_risk = risk

    return sketch, s

def gini_gain_on_sketch(sketch: CountMinSketch, threshold: float = 0.1) -> bool:
    """
    Compute a Gini‑gain like metric on the noisy histogram.
    Returns True if a split (e.g., further partitioning) is advisable.
    """
    f = sketch.noisy_histogram()
    total = f.sum()
    if total == 0:
        return False
    probs = f / total
    gini = 1.0 - np.sum(probs ** 2)
    return gini > threshold

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # synthetic identifiers (e.g., hashed user IDs)
    synthetic_data = [f"user_{random.randint(0, 5000)}" for _ in range(5000)]

    # ----- Build sketch and obtain noisy histogram -----
    sketch_obj, f_vec = build_noisy_sketch(synthetic_data, depth=4, width=256, eps_priv=0.8)

    # ----- Create a dummy RBF surrogate (M=3 output dims, K=5 centres) -----
    K = 5
    M = 3
    centres = [tuple(random.random() for _ in range(len(f_vec))) for _ in range(K)]
    weights = [[random.uniform(-1, 1) for _ in range(K)] for _ in range(M)]
    surrogate = RBFSurrogate(centers=centres, weights=weights, epsilon=0.5)

    # ----- Predict stylometric features and fuse -----
    styl_feat = rbf_predict_features(f_vec, surrogate)
    fused_matrix = combine_features_and_frequencies(styl_feat, f_vec)
    print("Fused matrix shape:", fused_matrix.shape)

    # ----- Incremental update with Hoeffding bound -----
    # reuse the same synthetic generator (restart iterator)
    data_iter = (f"user_{random.randint(0, 5000)}" for _ in range(3000))
    final_sketch, final_features = incremental_sketch_with_hoeffding(
        data_iter, surrogate, depth=4, width=256, eps_priv=0.8,
        delta=0.05, confidence=0.99, max_passes=5
    )
    print("Final stylometric feature vector (length {}):".format(len(final_features)),
          final_features[:5], "...")

    # ----- Gini‑gain decision example -----
    split_needed = gini_gain_on_sketch(final_sketch, threshold=0.12)
    print("Gini‑gain suggests split:", split_needed)