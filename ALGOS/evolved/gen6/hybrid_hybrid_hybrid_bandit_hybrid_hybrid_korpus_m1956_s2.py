# DARWIN HAMMER — match 1956, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_bandit_router_hybrid_hybrid_hybrid_m896_s2.py (gen4)
# parent_b: hybrid_hybrid_korpus_text_h_hybrid_hybrid_rbf_su_m849_s0.py (gen5)
# born: 2026-05-29T23:40:01Z

"""Hybrid algorithm combining:
- Parent A: temperature‑dependent developmental rate (Schoolfield model) used to
  modulate pruning probabilities in a Bayesian bandit update.
- Parent B: MinHash signatures for text data feeding a Radial Basis Function (RBF)
  surrogate model.

Mathematical bridge:
The developmental rate ρ(T) is employed as a scaling factor for the RBF kernel
width ε.  The MinHash Jaccard estimate J(text_i, text_j) provides a distance
d = 1‑J which is supplied to the Gaussian RBF kernel
k(d; ε) = exp(‑(ε·d)²).  Thus the temperature‑dependent physiological model
directly influences the perceptual similarity surface of the surrogate model,
creating a unified system that adapts its smoothness to the current temperature
while leveraging compact text fingerprints.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Sequence, Hashable, Set, Mapping, Tuple, Callable

import numpy as np

# ---------- Parent A components (Schoolfield developmental rate) ----------

R_CAL = 1.987  # cal mol^-1 K^-1
K25 = 298.15   # reference temperature (Kelvin)

@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0          # rate at 25 °C (K25)
    delta_h_activation: float = 12_000.0   # J mol⁻¹
    t_low: float = 283.15        # lower temperature bound (K)
    t_high: float = 307.15       # upper temperature bound (K)
    delta_h_low: float = -45_000.0
    delta_h_high: float = 65_000.0
    r_cal: float = R_CAL

def c_to_k(celsius: float) -> float:
    """Convert Celsius to Kelvin."""
    return celsius + 273.15

def developmental_rate(temp_k: float,
                       params: SchoolfieldParams = SchoolfieldParams()) -> float:
    """Schoolfield temperature‑dependent developmental rate ρ(T).

    Args:
        temp_k: Temperature in Kelvin (must be > 0).
        params: Parameter set for the model.

    Returns:
        Scaled developmental rate.
    """
    if temp_k <= 0:
        raise ValueError("temperature must be Kelvin‑positive")
    # Helper exponent term
    def exp_term(delta_h: float, t: float) -> float:
        return math.exp(-delta_h / (params.r_cal * t))
    # Full formulation (simplified version of the classic Schoolfield equation)
    num = params.rho_25 * (temp_k / K25) * exp_term(params.delta_h_activation, temp_k)
    den_low = 1.0 + exp_term(params.delta_h_low, params.t_low)
    den_high = 1.0 + exp_term(params.delta_h_high, params.t_high)
    return num / (den_low * den_high)

# ---------- Parent B components (MinHash + RBF surrogate) ----------

Vector = Sequence[float]
Node = Hashable
Graph = Mapping[Node, Set[Node]]
FeatureVec = Sequence[float]

def _hash_with_seed(value: int, seed: int) -> int:
    """Simple deterministic hash mixing value with a seed."""
    return (hash((value, seed)) & 0xffffffff)

def minhash_signature(text: str, num_perm: int = 128) -> np.ndarray:
    """Compute a MinHash signature for a piece of text.

    The text is tokenised on whitespace; each token is hashed with a set of
    random seeds (generated once per call).  The signature is the minimum hash
    per permutation.

    Args:
        text: Input document.
        num_perm: Number of permutation functions (signature length).

    Returns:
        A NumPy array of shape (num_perm,) containing 32‑bit unsigned ints.
    """
    tokens = text.split()
    if not tokens:
        # Empty document – return maximal hash values
        return np.full(num_perm, np.iinfo(np.uint32).max, dtype=np.uint32)

    # Pre‑compute random seeds for reproducibility
    rng = np.random.default_rng(seed=42)
    seeds = rng.integers(low=0, high=2**31 - 1, size=num_perm, dtype=np.int64)

    signature = np.full(num_perm, np.iinfo(np.uint32).max, dtype=np.uint32)

    for token in tokens:
        token_hash = hash(token)
        for i, seed in enumerate(seeds):
            combined = _hash_with_seed(token_hash, int(seed))
            if combined < signature[i]:
                signature[i] = combined
    return signature

def jaccard_estimate(sig1: np.ndarray, sig2: np.ndarray) -> float:
    """Estimate Jaccard similarity from two MinHash signatures."""
    if sig1.shape != sig2.shape:
        raise ValueError("signatures must have the same length")
    return float(np.mean(sig1 == sig2))

def gaussian_rbf(distance: float, epsilon: float) -> float:
    """Gaussian RBF kernel."""
    return math.exp(-((epsilon * distance) ** 2))

# ---------- Hybrid functions ----------

def hybrid_rbf_epsilon(base_epsilon: float,
                       temp_k: float,
                       params: SchoolfieldParams = SchoolfieldParams()) -> float:
    """Scale the RBF kernel width ε by the developmental rate ρ(T).

    The effective epsilon is ε_eff = base_epsilon * ρ(T).  Higher temperature
    (larger ρ) yields a broader kernel, i.e., smoother surrogate surface.
    """
    rho = developmental_rate(temp_k, params)
    return base_epsilon * rho

def surrogate_predict(query_vec: Vector,
                      support_vectors: List[Vector],
                      support_values: List[float],
                      temp_k: float,
                      base_epsilon: float = 1.0,
                      params: SchoolfieldParams = SchoolfieldParams()) -> float:
    """Predict a scalar output for `query_vec` using an RBF surrogate model.

    The RBF kernel width is temperature‑dependent via `hybrid_rbf_epsilon`.

    Args:
        query_vec: Feature vector for which to predict.
        support_vectors: List of known feature vectors (training set).
        support_values: Corresponding scalar outputs.
        temp_k: Current temperature (Kelvin) influencing kernel width.
        base_epsilon: Baseline kernel width before temperature scaling.
        params: Schoolfield parameters.

    Returns:
        Weighted RBF prediction (kernel regression with unit weights).
    """
    if len(support_vectors) != len(support_values):
        raise ValueError("support vectors and values length mismatch")
    eps = hybrid_rbf_epsilon(base_epsilon, temp_k, params)

    # Compute kernel values between query and each support vector
    kernels = []
    for sv in support_vectors:
        dist = euclidean(query_vec, sv)
        kernels.append(gaussian_rbf(dist, eps))

    kernels = np.array(kernels, dtype=float)
    if kernels.sum() == 0:
        # Avoid division by zero – fallback to simple average
        return float(np.mean(support_values))
    weights = kernels / kernels.sum()
    prediction = float(np.dot(weights, support_values))
    return prediction

def euclidean(a: Vector, b: Vector) -> float:
    """Euclidean distance between two vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str

def pruning_probability(action: BanditAction,
                        temp_k: float,
                        params: SchoolfieldParams = SchoolfieldParams(),
                        alpha: float = 0.5) -> float:
    """Compute a temperature‑modulated pruning probability for a bandit action.

    The baseline probability is derived from the confidence bound; it is then
    scaled by the developmental rate ρ(T) and a user‑provided factor α.

    Returns a value in [0, 1].
    """
    base = max(0.0, min(1.0, 1.0 - action.confidence_bound))
    rho = developmental_rate(temp_k, params)
    prob = base * (alpha + (1 - alpha) * rho)  # blend base with temperature influence
    return max(0.0, min(1.0, prob))

def hybrid_text_surrogate(text_query: str,
                          support_texts: List[str],
                          support_vectors: List[Vector],
                          support_values: List[float],
                          temp_celsius: float,
                          base_epsilon: float = 1.0) -> Tuple[float, float]:
    """End‑to‑end hybrid operation:

    1. Convert the query text to a MinHash signature.
    2. Estimate Jaccard similarity to each support text.
    3. Transform similarity into a distance and feed it to the RBF surrogate
       (temperature‑scaled) to obtain a prediction.

    Returns:
        (prediction, average_jaccard)
    """
    temp_k = c_to_k(temp_celsius)
    query_sig = minhash_signature(text_query)

    # Compute Jaccard similarities
    jaccards = []
    for txt in support_texts:
        sig = minhash_signature(txt)
        jaccards.append(jaccard_estimate(query_sig, sig))
    avg_jaccard = float(np.mean(jaccards))

    # Convert similarity to a pseudo‑feature distance (1‑J)
    # Here we embed the distance as an additional dimension appended to each
    # support vector, and similarly extend the query vector.
    distances = [1.0 - j for j in jaccards]
    extended_support = [list(vec) + [d] for vec, d in zip(support_vectors, distances)]
    extended_query = list(np.zeros_like(support_vectors[0])) + [0.0]  # placeholder for query distance

    # The query distance to itself is zero (max similarity)
    # Predict using the extended vectors
    pred = surrogate_predict(
        query_vec=extended_query,
        support_vectors=extended_support,
        support_values=support_values,
        temp_k=temp_k,
        base_epsilon=base_epsilon,
        params=SchoolfieldParams()
    )
    return pred, avg_jaccard

# ---------- Smoke test ----------

if __name__ == "__main__":
    # Simple deterministic seed for reproducibility
    random.seed(0)
    np.random.seed(0)

    # Sample texts
    query = "the quick brown fox jumps over the lazy dog"
    support = [
        "the fast brown fox leaps over a sleepy dog",
        "a completely unrelated sentence about quantum mechanics",
        "the quick brown fox jumps over the lazy dog"
    ]

    # Random feature vectors (e.g., node embeddings) for each support text
    support_vecs = [np.random.rand(5).tolist() for _ in support]
    support_vals = [0.2, 0.9, 0.1]  # arbitrary target values

    # Temperature in Celsius
    temperature_c = 25.0

    # Run the hybrid pipeline
    prediction, avg_sim = hybrid_text_surrogate(
        text_query=query,
        support_texts=support,
        support_vectors=support_vecs,
        support_values=support_vals,
        temp_celsius=temperature_c,
        base_epsilon=1.0
    )
    print(f"Hybrid surrogate prediction: {prediction:.4f}")
    print(f"Average MinHash Jaccard similarity: {avg_sim:.4f}")

    # Demonstrate bandit pruning probability
    action = BanditAction(
        action_id="a1",
        propensity=0.6,
        expected_reward=0.3,
        confidence_bound=0.8,
        algorithm="hybrid"
    )
    prune_prob = pruning_probability(action, c_to_k(temperature_c))
    print(f"Pruning probability for action {action.action_id}: {prune_prob:.4f}")