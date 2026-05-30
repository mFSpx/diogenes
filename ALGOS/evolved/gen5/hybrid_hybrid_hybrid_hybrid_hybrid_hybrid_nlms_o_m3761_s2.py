# DARWIN HAMMER — match 3761, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_endpoi_hybrid_hybrid_rbf_su_m726_s1.py (gen4)
# parent_b: hybrid_hybrid_nlms_omni_cha_hybrid_gliner_zero_s_m26_s3.py (gen2)
# born: 2026-05-29T23:51:36Z

"""Hybrid Algorithm integrating:
- Parent A: hybrid_hybrid_hybrid_endpoi_hybrid_hybrid_rbf_su_m726_s1 (morphology, recovery priority, Shannon entropy, RSA surrogate)
- Parent B: hybrid_hybrid_nlms_omni_cha_hybrid_gliner_zero_s_m26_s3 (NLMS adaptive update, span graph)

Mathematical Bridge:
The NLMS weight vector is used to predict RSA encryption parameters (e, n) from
a morphology feature vector. The encrypted recovery‑priority value is then
evaluated with Shannon entropy; this entropy dynamically scales the NLMS
step‑size μ, closing the loop between the two parent formulations."""
import sys
import random
import math
from pathlib import Path
from collections import Counter
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

# ---------- Parent A components ------------------------------------------------

@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)

def righting_time_index(m: Morphology, b: float = 1.0 / 3.0,
                        k: float = 0.35, neck_lever: float = 1.0) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever

def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """Normalized priority in [0,1]."""
    return max(0.0, min(1.0, righting_time_index(m) / max_index))

def shannon_entropy(observations: List[float], is_distribution: bool = False) -> float:
    xs = list(observations)
    if not xs:
        return 0.0
    if is_distribution:
        probs = [float(x) for x in xs]
        if any(p < 0 for p in probs) or abs(sum(probs) - 1.0) > 1e-6:
            raise ValueError("distribution must sum to 1")
    else:
        c = Counter(xs)
        total = sum(c.values())
        probs = [v / total for v in c.values()]
    return -sum(p * math.log2(p) for p in probs if p > 0)

# ---------- Parent B components ------------------------------------------------

def predict(weights: np.ndarray, x: np.ndarray) -> float:
    return float(np.dot(weights, x))

def nlms_update(weights: np.ndarray, x: np.ndarray, target: float,
                mu: float = 0.5, eps: float = 1e-9) -> Tuple[np.ndarray, float]:
    """Normalized Least Mean Squares weight update."""
    y = predict(weights, x)
    error = target - y
    power = float(np.dot(x, x) + eps)
    next_weights = weights + mu * error * x / power
    return next_weights, error

class Span:
    def __init__(self, start: int, end: int, text: str,
                 label: str, score: float, backend: str):
        self.start = start
        self.end = end
        self.text = text
        self.label = label
        self.score = score
        self.backend = backend

def extract_spans(text: str) -> List[Span]:
    """Very naive span extractor: each word becomes a span."""
    spans = []
    pos = 0
    for token in text.split():
        start = text.find(token, pos)
        end = start + len(token)
        spans.append(Span(start, end, token, label="WORD",
                          score=random.random(), backend="naive"))
        pos = end
    return spans

def build_adjacency(spans: List[Span]) -> np.ndarray:
    """Create a symmetric adjacency matrix where weight = average score of two spans."""
    n = len(spans)
    if n == 0:
        return np.zeros((0, 0))
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            w = (spans[i].score + spans[j].score) / 2.0
            A[i, j] = A[j, i] = w
    return A

# ---------- Hybrid core ---------------------------------------------------------

def morphology_feature_vector(m: Morphology) -> np.ndarray:
    """Feature vector used by NLMS to predict RSA parameters."""
    return np.array([m.length, m.width, m.height, m.mass], dtype=float)

def predict_rsa_params(weights: np.ndarray, features: np.ndarray) -> Tuple[int, int]:
    """
    Predict RSA exponent (e) and modulus (n) from a linear model.
    The predictions are forced into valid RSA ranges.
    """
    # Linear predictions
    raw_e = predict(weights, features)          # may be negative or fractional
    raw_n = predict(weights, features[::-1])    # use reversed features for diversity

    # Map to integer domain
    e = max(3, int(abs(raw_e)) | 1)              # make odd and >=3
    n_candidate = max(15, int(abs(raw_n)))      # n must be > e

    # Ensure n is composite of two distinct small primes (for demo)
    # Simple deterministic prime list
    primes = [101, 103, 107, 109, 113, 127, 131, 137, 139, 149]
    p = primes[e % len(primes)]
    q = primes[(e * 3) % len(primes)]
    if p == q:
        q = primes[(e * 7) % len(primes)]
    n = p * q
    return e, n

def rsa_encrypt(value: float, e: int, n: int) -> int:
    """Encrypt a float (scaled to integer) with RSA."""
    m_int = max(0, int(value * 1e6)) % n  # scale & fit into modulus
    return pow(m_int, e, n)

def hybrid_step(morph: Morphology,
                weights: np.ndarray,
                mu_base: float = 0.5) -> Tuple[np.ndarray, float]:
    """
    Perform one hybrid iteration:
    1. Compute recovery priority (target).
    2. Predict RSA (e, n) from morphology via NLMS weights.
    3. Encrypt the priority and compute its Shannon entropy.
    4. Scale NLMS step size μ by (1 + entropy) and update weights.
    Returns updated weights and the entropy value.
    """
    # 1. Target priority
    target = recovery_priority(morph)

    # 2. Feature vector & RSA prediction
    x = morphology_feature_vector(morph)
    e, n = predict_rsa_params(weights, x)

    # 3. Encryption & entropy
    cipher = rsa_encrypt(target, e, n)
    entropy = shannon_entropy([int(d) for d in str(cipher)])

    # 4. Adaptive NLMS update
    mu = mu_base * (1.0 + entropy)  # larger entropy -> larger adaptation
    new_weights, _ = nlms_update(weights, x, target, mu=mu)

    return new_weights, entropy

def process_text_with_hybrid(text: str,
                             init_weights: np.ndarray) -> Tuple[np.ndarray, List[float]]:
    """
    Extract spans from text, build a graph, and run the hybrid NLMS/RSA step
    for each span using a shared morphology derived from span length statistics.
    Returns final weight vector and list of entropies per span.
    """
    spans = extract_spans(text)
    if not spans:
        return init_weights, []

    # Simple morphology derived from span statistics
    lengths = [len(s.text) for s in spans]
    avg_len = sum(lengths) / len(lengths)
    morph = Morphology(length=avg_len,
                       width=avg_len * 0.6,
                       height=avg_len * 0.4,
                       mass=avg_len * 0.2)

    entropies = []
    weights = init_weights.copy()
    for _ in spans:
        weights, ent = hybrid_step(morph, weights)
        entropies.append(ent)
    return weights, entropies

# ---------- Smoke test -----------------------------------------------------------

if __name__ == "__main__":
    # Initialize a random weight vector for 4‑dimensional feature space
    rng = np.random.default_rng(seed=42)
    init_w = rng.normal(size=4)

    sample_text = "The quick brown fox jumps over the lazy dog."
    final_w, ent_list = process_text_with_hybrid(sample_text, init_w)

    print("Initial weights :", init_w)
    print("Final weights   :", final_w)
    print("Entropies per span:", ent_list)
    sys.exit(0)