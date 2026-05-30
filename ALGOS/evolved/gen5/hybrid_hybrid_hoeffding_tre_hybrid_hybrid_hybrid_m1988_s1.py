# DARWIN HAMMER — match 1988, survivor 1
# gen: 5
# parent_a: hybrid_hoeffding_tree_tropical_maxplus_m18_s1.py (gen1)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_ssim_hybrid_d_m905_s1.py (gen4)
# born: 2026-05-29T23:40:23Z

import numpy as np
import math
import random
from dataclasses import dataclass
from typing import Tuple, List, Callable, Any

# ----------------------------------------------------------------------
# Global configuration (could be loaded from a config file)
# ----------------------------------------------------------------------
GROUPS: Tuple[str, ...] = ("codex", "groq", "cohere", "local_models")
BASE_TAU: float = 1.0          # baseline liquid time constant
ALPHA: float = 5.0             # gating steepness for liquid‑time constant
LAMBDA: float = 0.7            # VFE weighting factor
MINHASH_K: int = 64            # number of hash functions for MinHash
MAX64: int = (1 << 64) - 1     # mask for 64‑bit hashing
SEED: int = 12345              # reproducibility

random.seed(SEED)
np.random.seed(SEED)

# ----------------------------------------------------------------------
# Hoeffding bound utilities
# ----------------------------------------------------------------------
def hoeffding_bound(r: float, delta: float, n: int) -> float:
    """Classic Hoeffding bound for a range‑bounded random variable."""
    if r <= 0 or not (0.0 < delta < 1.0) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0 required")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))

@dataclass(frozen=True)
class SplitDecision:
    should_split: bool
    epsilon: float
    gain_gap: float
    reason: str

def should_split(best_gain: float,
                 second_best_gain: float,
                 r: float,
                 delta: float,
                 n: int,
                 tie_threshold: float = 0.05) -> SplitDecision:
    """Decide whether to split a node using a Hoeffding bound that is
    adaptively scaled by the current liquid‑time constant."""
    eps = hoeffding_bound(r, delta, n)
    gap = best_gain - second_best_gain
    split = gap > eps or eps < tie_threshold
    reason = ("gap_exceeds_bound" if gap > eps
              else ("tie_threshold" if eps < tie_threshold
                    else "wait"))
    return SplitDecision(split, eps, gap, reason)

# ----------------------------------------------------------------------
# Tropical max‑plus algebra
# ----------------------------------------------------------------------
def t_add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.maximum(x, y)

def t_mul(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.add(x, y)

def t_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Tropical matrix multiplication: (A ⊗ B)_{ij} = max_k (A_{ik}+B_{kj})"""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    # broadcasting to compute all pairwise sums
    # shape (A_rows, B_cols, A_cols)
    sums = A[:, :, np.newaxis] + B[np.newaxis, :, :]
    return np.max(sums, axis=1)

def t_polyval(coeffs: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Evaluate a tropical polynomial p(x) = max_i (c_i + i*x)
    where coeffs[i] = c_i.
    """
    coeffs = np.asarray(coeffs, dtype=float)
    x = np.asarray(x, dtype=float)
    exponents = np.arange(coeffs.shape[0], dtype=float).reshape((-1,) + (1,) * x.ndim)
    terms = coeffs.reshape((-1,) + (1,) * x.ndim) + exponents * x
    return np.max(terms, axis=0)

def relu_layer_to_tropical(W: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Exact tropicalisation of a ReLU affine layer:
        y = ReLU(Wx + b)  →  y_i = max_j (W_{ij}+x_j) + b_i
    In max‑plus algebra the ReLU is implicit because max already
    enforces non‑negativity of the offset.
    """
    return np.asarray(W, dtype=float), np.asarray(b, dtype=float)

def tropical_network_eval(x: np.ndarray, layers: List[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """
    Forward pass through a network whose linear parts are expressed in
    tropical max‑plus algebra. Each layer is a tuple (W, b) as returned by
    ``relu_layer_to_tropical``.
    """
    h = np.asarray(x, dtype=float).ravel()
    for W, b in layers:
        h = t_add(t_matmul(W, h), b)   # max(W·x + b)
    return h

# ----------------------------------------------------------------------
# MinHash utilities (simple 64‑bit universal hashing)
# ----------------------------------------------------------------------
def _hash_vector(v: np.ndarray, seed: int) -> np.ndarray:
    """Hash a vector to a 64‑bit integer using a linear congruential scheme."""
    a = np.uint64(seed | 1)                     # odd multiplier
    b = np.uint64((seed >> 32) ^ 0x5bf03635)    # scramble
    hashed = (a * v.astype(np.uint64) + b) & MAX64
    return hashed

def compute_minhash_signature(x: np.ndarray, k: int = MINHASH_K) -> np.ndarray:
    """Return a MinHash signature of length k for the input vector."""
    signatures = np.full(k, MAX64, dtype=np.uint64)
    for i in range(k):
        hashed = _hash_vector(x, SEED + i)
        signatures[i] = np.min(hashed)  # min over all dimensions
    return signatures

def minhash_similarity(sig1: np.ndarray, sig2: np.ndarray) -> float:
    """Jaccard‑like similarity based on MinHash signatures."""
    return np.mean(sig1 == sig2)

# ----------------------------------------------------------------------
# Day‑of‑week dependent weighting (deeper integration)
# ----------------------------------------------------------------------
def weekday_weight_vector(groups: Tuple[str, ...], dow: int) -> np.ndarray:
    """
    Produce a weight vector that varies smoothly over the week.
    Uses a sinusoidal schedule to avoid hard‑coded steps.
    """
    base = np.linspace(0.1, 0.7, len(groups))
    phase = (dow / 7.0) * 2 * np.pi
    modulation = 0.5 * (1 + np.sin(phase))  # in [0,1]
    return base * (0.5 + modulation)        # keep weights >0

def compute_similarity_vector(x: np.ndarray, groups: Tuple[str, ...], dow: int) -> np.ndarray:
    """
    Blend the raw feature vector with a weekday‑aware weighting.
    """
    w = weekday_weight_vector(groups, dow)
    return x * w  # element‑wise scaling

# ----------------------------------------------------------------------
# Liquid‑time constant and variational free‑energy integration
# ----------------------------------------------------------------------
def compute_liquid_time_constant(similarity: np.ndarray) -> float:
    """Dynamic time constant that grows with the magnitude of similarity."""
    raw = BASE_TAU + LAMBDA * np.sum(similarity)
    # Pass through a smooth gating function to keep it bounded
    return BASE_TAU + (raw - BASE_TAU) / (1 + np.exp(-ALPHA * (raw - BASE_TAU)))

def hybrid_decision_hygiene(x: np.ndarray,
                            groups: Tuple[str, ...],
                            dow: int) -> float:
    """
    Combine similarity, MinHash, and liquid‑time dynamics into a single
    scalar that will later modulate the Hoeffding bound.
    """
    sim_vec = compute_similarity_vector(x, groups, dow)
    minhash_sig = compute_minhash_signature(x)
    minhash_sim = minhash_similarity(minhash_sig, compute_minhash_signature(sim_vec))
    tau = compute_liquid_time_constant(sim_vec)
    # Convex combination controlled by τ
    return (1.0 - np.exp(-tau)) * minhash_sim + np.exp(-tau) * np.mean(sim_vec)

# ----------------------------------------------------------------------
# Deeper fusion: Hoeffding tree node that uses tropical network output
# ----------------------------------------------------------------------
class TropicalHoeffdingNode:
    """
    A leaf node that stores sufficient statistics and decides when to split.
    The split decision uses:
      * classic Hoeffding bound,
      * a tropical network that predicts the *expected* gain,
      * a dynamic scaling factor derived from hybrid_decision_hygiene.
    """
    def __init__(self,
                 feature_dim: int,
                 n_classes: int,
                 tropical_layers: List[Tuple[np.ndarray, np.ndarray]],
                 delta: float = 1e-7,
                 r: float = 1.0):
        self.n = 0
        self.feature_dim = feature_dim
        self.n_classes = n_classes
        self.delta = delta
        self.r = r
        self.tropical_layers = tropical_layers
        self.class_counts = np.zeros(n_classes, dtype=int)
        self.sum_features = np.zeros(feature_dim, dtype=float)

    def update(self, x: np.ndarray, y: int, dow: int) -> None:
        """Ingest a new sample."""
        self.n += 1
        self.class_counts[y] += 1
        self.sum_features += x
        # optional: store raw samples for more sophisticated split tests

    def _expected_gain(self, x: np.ndarray, dow: int) -> float:
        """
        Use the tropical network to estimate the information gain that would
        result from splitting on this instance. The network is trained
        offline; here we only evaluate it.
        """
        # Combine raw features with weekday‑aware similarity before feeding
        sim_vec = compute_similarity_vector(x, GROUPS, dow)
        net_input = np.concatenate([x, sim_vec])
        out = tropical_network_eval(net_input, self.tropical_layers)
        # Map network output (max‑plus value) to a pseudo‑gain in [0,1]
        gain = 1.0 / (1.0 + np.exp(-out))  # sigmoid squashing
        return float(gain)

    def try_split(self, candidate_gains: List[float], dow: int) -> SplitDecision:
        """
        ``candidate_gains`` should contain the estimated gains for each
        possible split attribute (e.g. numeric thresholds). The best two
        are used for the Hoeffding test.
        """
        if not candidate_gains:
            raise ValueError("candidate_gains must contain at least one element")
        best = max(candidate_gains)
        second = max([g for g in candidate_gains if g != best] or [0.0])

        # Adjust the Hoeffding range r using the hybrid hygiene score
        hygiene = hybrid_decision_hygiene(self.sum_features / max(self.n, 1),
                                          GROUPS,
                                          dow)
        adjusted_r = self.r * (1.0 + hygiene)  # larger r → looser bound when uncertainty high

        decision = should_split(best_gain=best,
                                second_best_gain=second,
                                r=adjusted_r,
                                delta=self.delta,
                                n=self.n)
        return decision

# ----------------------------------------------------------------------
# Example construction of a tropical network (randomised for demo)
# ----------------------------------------------------------------------
def _random_tropical_layer(in_dim: int, out_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create a random tropical affine layer."""
    W = np.random.randn(out_dim, in_dim)   # real‑valued weights become tropical coefficients
    b = np.random.randn(out_dim)
    return relu_layer_to_tropical(W, b)

def build_demo_tropical_network(input_dim: int, hidden_dims: List[int]) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Utility to build a small tropical network for testing."""
    dims = [input_dim] + hidden_dims
    layers = []
    for i in range(len(dims) - 1):
        layers.append(_random_tropical_layer(dims[i], dims[i + 1]))
    return layers

# ----------------------------------------------------------------------
# Smoke test (executed only when run as a script)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # synthetic data
    FEATURE_DIM = 8
    N_CLASSES = 3
    DOW = 2  # Tuesday

    # build a tiny tropical network: input = raw + similarity (2*FEATURE_DIM)
    net = build_demo_tropical_network(FEATURE_DIM * 2, [16, 8, 1])

    node = TropicalHoeffdingNode(feature_dim=FEATURE_DIM,
                                 n_classes=N_CLASSES,
                                 tropical_layers=net,
                                 delta=1e-5,
                                 r=1.0)

    # simulate a stream of 200 samples
    for _ in range(200):
        x = np.random.rand(FEATURE_DIM)
        y = np.random.randint(N_CLASSES)
        node.update(x, y, DOW)

    # dummy candidate gains (e.g. from evaluating split thresholds)
    dummy_gains = [node._expected_gain(np.random.rand(FEATURE_DIM), DOW) for _ in range(5)]
    decision = node.try_split(dummy_gains, DOW)
    print("Split decision:", decision)