# DARWIN HAMMER — match 172, survivor 3
# gen: 4
# parent_a: hybrid_hybrid_fisher_locali_jepa_energy_m52_s2.py (gen2)
# parent_b: hybrid_hybrid_hdc_serpentin_hybrid_sparse_wta_hy_m117_s1.py (gen3)
# born: 2026-05-29T23:27:25Z

"""Hybrid Fisher‑JEPA Hyperdimensional Algorithm.

Parents:
- hybrid_fisher_locali_jepa_energy_m52_s2.py (Algorithm A) – provides a Gaussian
  beam, Fisher information score and the JEPA‑style energy formulation
  E = ‖ encoder(t) – predictor( encoder(t_prev), F(θ) ) ‖².
- hybrid_hdc_serpentin_hybrid_sparse_wta_hy_m117_s1.py (Algorithm B) – supplies
  hyperdimensional primitives (random/binding/bundling/similarity).

Mathematical bridge:
The scalar Fisher score *F(θ)* is turned into a bipolar hypervector *ẑ* by a
deterministic hash‑seeded random generator.  The timestamp *t* itself is also
encoded as a hypervector *x = encoder(t)* via a symbol‑based hash.  The JEPA
predictor is realised in hyperdimensional space as a bundling of the bound
pair *(x_prev ⊗ ẑ)* with the previous representation *x_prev*.  Energy is the
squared Euclidean distance between the true representation *x* and the predicted
representation *x̂* (equivalently a scaled Hamming distance).

The module implements the full hybrid pipeline and provides three high‑level
functions that showcase encoding, prediction and energy computation.
"""

import math
import random
import sys
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Iterable

import numpy as np

# ---------------------------------------------------------------------------
# Hyperdimensional primitives (from Algorithm B)
# ---------------------------------------------------------------------------

Vector = List[int]  # bipolar hypervector (elements are +1 or -1)


def random_vector(dim: int = 10000, seed: int | str | None = None) -> Vector:
    """Generate a random bipolar hypervector of length *dim*."""
    rng = random.Random(seed)
    return [1 if rng.getrandbits(1) else -1 for _ in range(dim)]


def symbol_vector(symbol: str, dim: int = 10000) -> Vector:
    """Deterministic hypervector for an arbitrary symbol using SHA‑256 as seed."""
    seed_bytes = hashlib.sha256(symbol.encode("utf-8")).digest()[:8]
    seed = int.from_bytes(seed_bytes, "big")
    return random_vector(dim, seed)


def bind(a: Vector, b: Vector) -> Vector:
    """Component‑wise binding (multiplication) of two hypervectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have equal length")
    return [x * y for x, y in zip(a, b)]


def bundle(vectors: Iterable[Vector]) -> Vector:
    """Superposition (majority vote) of a collection of hypervectors."""
    vecs = list(vectors)
    if not vecs:
        raise ValueError("bundle requires at least one vector")
    dim = len(vecs[0])
    for v in vecs:
        if len(v) != dim:
            raise ValueError("all vectors must have same dimension")
    summed = [sum(comp) for comp in zip(*vecs)]
    return [1 if s >= 0 else -1 for s in summed]


def similarity(a: Vector, b: Vector) -> float:
    """Cosine similarity for bipolar vectors (identical to normalized dot product)."""
    if len(a) != len(b):
        raise ValueError("vectors must have equal length")
    dot = sum(x * y for x, y in zip(a, b))
    return dot / len(a)  # |a| = |b| = sqrt(dim) for bipolar vectors


# ---------------------------------------------------------------------------
# Fisher‑information utilities (from Algorithm A)
# ---------------------------------------------------------------------------


def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float = 0.0, width: float = 1.0, eps: float = 1e-12) -> float:
    """Fisher information for a Gaussian beam evaluated at *theta*."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


# ---------------------------------------------------------------------------
# Hybrid building blocks
# ---------------------------------------------------------------------------


def encode_timestamp(ts: float, dim: int = 10000) -> Vector:
    """
    Encode a timestamp (seconds since epoch) as a deterministic hypervector.
    The timestamp is first turned into an ISO‑8601 string to ensure stable hashing.
    """
    iso = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    return symbol_vector(iso, dim)


def fisher_to_hypervector(score: float, dim: int = 10000) -> Vector:
    """
    Map a scalar Fisher score to a bipolar hypervector.
    The absolute value determines the seed; the sign determines a final flip.
    """
    # Normalise score to a printable representation
    score_str = f"{score:.12g}"
    seed = int.from_bytes(hashlib.sha256(score_str.encode()).digest()[:8], "big")
    hv = random_vector(dim, seed)
    # If the score is negative (theoretically impossible for Fisher info) flip sign
    if score < 0:
        hv = [-x for x in hv]
    return hv


def predictor(prev_vec: Vector, fisher_vec: Vector) -> Vector:
    """
    Hyperdimensional JEPA predictor.
    It binds the previous representation with the Fisher hypervector and bundles
    the result with the unchanged previous representation.
    """
    bound = bind(prev_vec, fisher_vec)
    return bundle([bound, prev_vec])


def energy(true_vec: Vector, pred_vec: Vector) -> float:
    """
    Squared Euclidean distance between two bipolar hypervectors.
    For bipolar vectors this equals 2 * (dim - dot)/dim.
    """
    if len(true_vec) != len(pred_vec):
        raise ValueError("vectors must have equal length")
    diff = np.array(true_vec, dtype=np.float32) - np.array(pred_vec, dtype=np.float32)
    return float(np.dot(diff, diff))


# ---------------------------------------------------------------------------
# High‑level hybrid operations
# ---------------------------------------------------------------------------


def hybrid_energy(candidate_ts: float, reference_ts: float,
                  center: float = 0.0, width: float = 1.0,
                  dim: int = 10000) -> float:
    """
    Compute the JEPA‑style energy for a candidate timestamp.

    Steps:
    1. Fisher score F = fisher_score(candidate_ts, center, width)
    2. Encode candidate and reference timestamps as hypervectors.
    3. Convert F to a hypervector ẑ.
    4. Predict candidate representation from the reference using predictor.
    5. Return squared Euclidean distance between true and predicted vectors.
    """
    f_score = fisher_score(candidate_ts, center, width)
    cand_vec = encode_timestamp(candidate_ts, dim)
    ref_vec = encode_timestamp(reference_ts, dim)
    fisher_vec = fisher_to_hypervector(f_score, dim)
    pred_vec = predictor(ref_vec, fisher_vec)
    return energy(cand_vec, pred_vec)


def hybrid_similarity(candidate_ts: float, reference_ts: float,
                      dim: int = 10000) -> float:
    """
    Compute cosine similarity between the encoded candidate timestamp and the
    predictor's output (without the Fisher weighting).  This illustrates the
    pure hyperdimensional relationship.
    """
    cand_vec = encode_timestamp(candidate_ts, dim)
    ref_vec = encode_timestamp(reference_ts, dim)
    # Use a unit Fisher vector (all +1) to isolate the binding effect
    unit_fisher = [1] * dim
    pred_vec = predictor(ref_vec, unit_fisher)
    return similarity(cand_vec, pred_vec)


def hybrid_predict_next(prev_ts: float, next_ts_estimate: float,
                       center: float = 0.0, width: float = 1.0,
                       dim: int = 10000) -> Vector:
    """
    Given a previous timestamp *prev_ts*, generate a predicted hypervector for a
    future timestamp using the Fisher score of an *estimate* (e.g. a coarse
    prediction).  The function returns the predicted hypervector, which can be
    compared against the true encoding of *next_ts_estimate*.
    """
    f_score = fisher_score(next_ts_estimate, center, width)
    prev_vec = encode_timestamp(prev_ts, dim)
    fisher_vec = fisher_to_hypervector(f_score, dim)
    return predictor(prev_vec, fisher_vec)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    # Simple deterministic timestamps (seconds since epoch)
    now = datetime.now(tz=timezone.utc).timestamp()
    later = now + 3600.0          # +1 hour
    even_later = now + 7200.0    # +2 hours

    # Compute hybrid energies
    e1 = hybrid_energy(later, now, center=now, width=1800.0)
    e2 = hybrid_energy(even_later, later, center=now, width=1800.0)

    # Compute hybrid similarities
    sim1 = hybrid_similarity(later, now)
    sim2 = hybrid_similarity(even_later, later)

    # Predict next representation and compare with true encoding
    pred_vec = hybrid_predict_next(now, later, center=now, width=1800.0)
    true_vec = encode_timestamp(later)
    pred_energy = energy(true_vec, pred_vec)

    print(f"Hybrid energy (now→+1h): {e1:.4f}")
    print(f"Hybrid energy (+1h→+2h): {e2:.4f}")
    print(f"Hybrid similarity (now vs +1h): {sim1:.4f}")
    print(f"Hybrid similarity (+1h vs +2h): {sim2:.4f}")
    print(f"Prediction energy for +1h given now: {pred_energy:.4f}")

    # Simple sanity check: energies should be non‑negative and similarities in [-1,1]
    assert e1 >= 0 and e2 >= 0
    assert -1.0 <= sim1 <= 1.0 and -1.0 <= sim2 <= 1.0
    assert pred_energy >= 0
    print("Smoke test passed.")