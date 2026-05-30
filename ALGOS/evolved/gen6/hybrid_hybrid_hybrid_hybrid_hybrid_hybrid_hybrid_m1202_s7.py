# DARWIN HAMMER — match 1202, survivor 7
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_decisi_m886_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_rbf_su_hybrid_capybara_opti_m52_s0.py (gen3)
# born: 2026-05-29T23:34:35Z

"""Hybrid Algorithm Fusion of DARWIN HAMMER Parents A & B

Parent A (hybrid_hybrid_hybrid_hybrid_hybrid_privacy_sketc_m28_s0.py) provides:
- A ternary‑linear transformation matrix **W** (TTT) that maps an input feature
  vector *x* → *W·x*.
- Extraction of textual cues into a ``ResourceVector`` (load, privacy).

Parent B (hybrid_hybrid_hybrid_rbf_su_hybrid_capybara_opti_m52_s0.py) provides:
- A radial‑basis‑function surrogate model ``RBFSurrogate`` that maps a vector
  *z* to a scalar prediction using Gaussian kernels.
- A lightweight signal‑score routine that derives entropy and keyword‑hit
  statistics from raw byte streams.

**Mathematical Bridge**
The bridge is the shared vector space of feature descriptors.  We first build a
feature vector

    x = [load, privacy, entropy, keyword_hits]

from the textual and byte‑level inputs.  The TTT matrix **W** transforms this
vector into a latent representation *y = W·x*.  The surrogate model then
evaluates *y*:

    s = surrogate.predict(y)

Finally, the reconstruction loss of the TTT step

    r = ttt_loss(W, x)

acts as a *risk‑attenuation* factor, scaling the surrogate output:

    output = s / (1 + r)

The resulting pipeline fuses linear‑algebraic transformation, privacy‑aware
textual analysis, and RBF‑based surrogate prediction into a single unified
operation.

The module implements this fusion with clear helper functions and a smoke
test.
"""

import math
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Sequence

import numpy as np

# ----------------------------------------------------------------------
# Parent A – TTT linear transformation and textual feature extraction
# ----------------------------------------------------------------------


def init_ttt(d_in: int, d_out: int | None = None, scale: float = 0.01, seed: int = 0) -> np.ndarray:
    """Create a random TTT weight matrix."""
    rng = np.random.default_rng(seed)
    if d_out is None:
        d_out = d_in
    return rng.standard_normal((d_out, d_in)) * scale


def ttt_transform(W: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Linear transformation y = W·x."""
    return W @ x


def ttt_loss(W: np.ndarray, x: np.ndarray, target: np.ndarray | None = None) -> float:
    """Squared reconstruction loss ‖W·x − target‖² (target defaults to x)."""
    if target is None:
        target = x
    diff = W @ x - target
    return float(np.sum(diff ** 2))


def ttt_step(W: np.ndarray, x: np.ndarray, eta: float, target: np.ndarray | None = None) -> np.ndarray:
    """One gradient‑descent step on the TTT loss."""
    if target is None:
        target = x
    grad = 2.0 * np.outer((W @ x - target), x)
    return W - eta * grad


@dataclass
class ResourceVector:
    load: float
    privacy: float


def extract_text_features(text: str) -> ResourceVector:
    """Regex‑based cue extraction producing load & privacy scores."""
    evidence_pat = r"\b(evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b"
    planning_pat = r"\b(plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b"

    evidence = bool(re.search(evidence_pat, text, re.I))
    planning = bool(re.search(planning_pat, text, re.I))

    load = 1.0 if evidence else 0.0
    privacy = 0.5 if planning else 0.0
    return ResourceVector(load=load, privacy=privacy)


# ----------------------------------------------------------------------
# Parent B – RBF surrogate and signal scoring
# ----------------------------------------------------------------------


def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian kernel φ(r) = exp(-(ε·r)²)."""
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    """Euclidean distance between two equal‑length vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


@dataclass(frozen=True)
class RBFSurrogate:
    centers: List[Tuple[float, ...]]
    weights: List[float]
    epsilon: float = 1.0

    def predict(self, x: Sequence[float]) -> float:
        """RBF surrogate prediction Σ w_i·φ(‖x−c_i‖)."""
        return sum(
            w * gaussian(euclidean(x, c), self.epsilon)
            for w, c in zip(self.weights, self.centers)
        )


def _byte_entropy(data: bytes) -> float:
    """Shannon entropy of a byte sequence."""
    if not data:
        return 0.0
    freq = [0] * 256
    for b in data:
        freq[b] += 1
    probs = [f / len(data) for f in freq if f > 0]
    return -sum(p * math.log2(p) for p in probs)


def signal_scores(
    data: bytes,
    keyword_hits: int = 0,
    structural_links: int = 0,
) -> Tuple[float, float]:
    """
    Produce a pair (entropy, keyword_score) from raw data.
    *entropy* – Shannon entropy of the byte payload.
    *keyword_score* – weighted count of domain‑specific keywords.
    """
    entropy = _byte_entropy(data)
    keyword_score = keyword_hits * 0.1 + structural_links * 0.05
    return entropy, keyword_score


# ----------------------------------------------------------------------
# Fusion layer – three core functions demonstrating the hybrid operation
# ----------------------------------------------------------------------


def transform_load(resource: ResourceVector, W: np.ndarray) -> float:
    """
    Apply the TTT matrix to the *load* component alone (treated as a 1‑D vector)
    and return the transformed scalar (first entry of the result).
    """
    x = np.array([resource.load, resource.privacy])
    y = ttt_transform(W[:1, :], x)  # take first row of W
    return float(y[0])


def update_privacy(resource: ResourceVector, risk: float) -> ResourceVector:
    """
    Attenuate the privacy score based on reconstruction risk.
    Higher risk reduces the effective privacy dimension.
    """
    new_privacy = resource.privacy / (1.0 + risk)
    return ResourceVector(load=resource.load, privacy=new_privacy)


def hybrid_score(
    text: str,
    data: bytes,
    W: np.ndarray,
    surrogate: RBFSurrogate,
) -> float:
    """
    Full hybrid pipeline:
    1. Extract textual load/privacy.
    2. Compute signal scores from *data*.
    3. Assemble feature vector x = [load, privacy, entropy, keyword_score].
    4. Transform x with the TTT matrix → y.
    5. Predict with the RBF surrogate → s.
    6. Scale by reconstruction risk → final output.
    """
    # 1. Textual features
    res_vec = extract_text_features(text)

    # 2. Signal scores
    entropy, keyword_score = signal_scores(data)

    # 3. Feature vector (as a column vector for matrix multiplication)
    x = np.array([res_vec.load, res_vec.privacy, entropy, keyword_score], dtype=float)

    # 4. TTT transformation (ensure compatible dimensions)
    if W.shape[1] != x.shape[0]:
        raise ValueError(f"W expects {W.shape[1]} features, got {x.shape[0]}")
    y = ttt_transform(W, x)

    # 5. Surrogate prediction (convert y to a plain sequence)
    s = surrogate.predict(y.tolist())

    # 6. Risk‑attenuated output
    r = ttt_loss(W, x)
    output = s / (1.0 + r)
    return output


# ----------------------------------------------------------------------
# Helper to construct a random hybrid model (TTT + RBF)
# ----------------------------------------------------------------------


def build_random_hybrid(
    feature_dim: int = 4,
    latent_dim: int = 6,
    n_centers: int = 8,
    seed: int = 42,
) -> Tuple[np.ndarray, RBFSurrogate]:
    """
    Generate a random TTT matrix and an RBF surrogate whose centers match the
    latent dimension.  The random state is seeded for reproducibility.
    """
    rng = np.random.default_rng(seed)
    W = init_ttt(feature_dim, latent_dim, scale=0.05, seed=seed)

    centers = [tuple(rng.normal(0, 1, size=latent_dim)) for _ in range(n_centers)]
    weights = list(rng.uniform(-1, 1, size=n_centers))
    surrogate = RBFSurrogate(centers=centers, weights=weights, epsilon=1.0)
    return W, surrogate


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Sample inputs
    sample_text = (
        "The evidence was verified and the plan was documented in the final report."
    )
    sample_data = b"The quick brown fox jumps over the lazy dog. " * 10

    # Build a deterministic hybrid model
    W_mat, rbf_model = build_random_hybrid()

    # Execute core functions
    load_transformed = transform_load(extract_text_features(sample_text), W_mat)
    risk = ttt_loss(W_mat, np.array([1.0, 0.5, _byte_entropy(sample_data), 0.2]))
    updated_resource = update_privacy(extract_text_features(sample_text), risk)

    # Full hybrid score
    score = hybrid_score(sample_text, sample_data, W_mat, rbf_model)

    print(f"Transformed load (scalar): {load_transformed:.6f}")
    print(f"Updated resource vector: {updated_resource}")
    print(f"Hybrid final score: {score:.6f}")