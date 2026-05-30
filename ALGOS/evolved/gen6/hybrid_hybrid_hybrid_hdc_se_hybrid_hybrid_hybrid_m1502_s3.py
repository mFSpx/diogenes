# DARWIN HAMMER — match 1502, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hdc_serpentin_hybrid_sparse_wta_hy_m117_s2.py (gen3)
# parent_b: hybrid_hybrid_hybrid_nlms_o_hybrid_hybrid_hybrid_m424_s0.py (gen5)
# born: 2026-05-29T23:36:50Z

"""Hybrid Hyperdimensional‑NLMS Fusion
===================================

This module fuses the two parent algorithms:

* **Parent A** – hyperdimensional computing primitives (random/bipolar vectors,
  binding, bundling, cosine similarity).
* **Parent B** – Normalised Least‑Mean‑Squares (NLMS) adaptive filter together
  with a simple language‑statistical‑model (LSM) used for epistemic‑certainty
  weighting.

**Mathematical bridge**

1. The NLMS prediction error `e` is used as a *scalar modulation* of the
   hyperdimensional superposition (`bundle`).  After an NLMS weight update the
   error‑scaled bound vector `e·bind(sym,ctx)` is bundled with the previous
   weight hypervector, thereby injecting the adaptive error signal into the
   high‑dimensional representation.

2. The LSM word‑frequency distribution of a text is mapped to a hypervector
   by deterministic symbol‑vector lookup and a weighted sum.  The resulting
   hypervector is employed as a *certainty mask* that scales the contribution
   of an edge (pair of coordinates) in a graph‑based computation.  The mask
   itself is weighted by the epistemic certainty derived from the NLMS error,
   closing the loop between the two subsystems.

The three public functions below showcase this hybrid operation:
`hybrid_predict`, `hybrid_update` and `hybrid_edge_contribution`.  They can be
used independently or combined in larger pipelines."""


from __future__ import annotations

import hashlib
import math
import random
from collections import Counter
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Hyperdimensional primitives (Parent A)
# ----------------------------------------------------------------------
Vector = np.ndarray  # bipolar hypervector of dtype int8


def random_vector(dim: int = 10000, seed: str | int | None = None) -> Vector:
    """Generate a bipolar random hypervector."""
    rng = random.Random(seed)
    data = np.fromiter(
        (1 if rng.getrandbits(1) else -1 for _ in range(dim)),
        dtype=np.int8,
        count=dim,
    )
    return data


def symbol_vector(symbol: str, dim: int = 10000) -> Vector:
    """Deterministically map a symbol to a bipolar hypervector."""
    seed = int.from_bytes(
        hashlib.sha256(symbol.encode("utf-8")).digest()[:8], byteorder="big"
    )
    return random_vector(dim, seed)


def bind(a: Vector, b: Vector) -> Vector:
    """Element‑wise binding (multiplication) of two hypervectors."""
    if a.shape != b.shape:
        raise ValueError("vectors must have equal shape")
    return a * b


def bundle(vectors: Iterable[Vector]) -> Vector:
    """Superposition of hypervectors followed by binarization (sign)."""
    vecs = list(vectors)
    if not vecs:
        raise ValueError("bundle requires at least one vector")
    stacked = np.stack(vecs, axis=0).astype(np.int32)
    summed = stacked.sum(axis=0)
    return np.where(summed >= 0, 1, -1).astype(np.int8)


def cosine_similarity(a: Vector, b: Vector) -> float:
    """Cosine similarity for bipolar vectors."""
    if a.shape != b.shape:
        raise ValueError("vectors must have equal shape")
    dot = float(np.dot(a, b))
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


# ----------------------------------------------------------------------
# NLMS and LSM primitives (Parent B)
# ----------------------------------------------------------------------
def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Dot‑product prediction w·x."""
    return float(weights @ x)


def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, float]:
    """
    One Normalised LMS update step.
    Returns the updated weight vector and the prediction error.
    """
    pred = nlms_predict(weights, x)
    error = target - pred
    update = mu * error * x / (float(x @ x) + eps)
    new_weights = weights + update
    return new_weights, error


def words(text: str) -> List[str]:
    """Very simple whitespace tokenizer."""
    return [w for w in text.lower().split() if w]


def lsm_vector(text: str, dim: int = 10000) -> Vector:
    """
    Convert a text into a hypervector by weighted superposition of
    deterministic symbol vectors (one per distinct word).
    """
    token_list = words(text)
    total = max(1, len(token_list))
    freq: Counter[str] = Counter(token_list)

    weighted_vectors: List[Vector] = []
    for token, count in freq.items():
        weight = count / total
        sv = symbol_vector(token, dim)
        # Scale the bipolar vector by the frequency (still bipolar after sign)
        weighted_vectors.append(np.where(sv >= 0, 1, -1).astype(np.int8) * (1 if weight >= 0.5 else -1))

    if not weighted_vectors:
        # fallback to a random vector if text is empty
        return random_vector(dim, seed="empty")
    return bundle(weighted_vectors)


def calculate_epistemic_certainty(factor: float, error: float) -> float:
    """Map NLMS error to a certainty factor in [0,1]."""
    # error may be negative; we use absolute value and clamp.
    err = min(abs(error), 1.0)
    return max(0.0, min(1.0, factor * (1.0 - err)))


def calculate_edge_weight(distance: float, epistemic_certainty: float) -> float:
    """Effective edge weight – distance attenuated by certainty."""
    return distance * (1.0 - epistemic_certainty)


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def hybrid_predict(
    symbol: str,
    context: str,
    weight_vec: np.ndarray,
    dim: int = 10000,
) -> Tuple[float, Vector]:
    """
    Predict using a hyperdimensional representation.

    1. Create hypervectors for ``symbol`` and ``context`` and bind them.
    2. Convert the NLMS weight vector to a bipolar hypervector (sign).
    3. Return the cosine similarity between the bound vector and the weight
       hypervector together with the bound vector itself.
    """
    sv = symbol_vector(symbol, dim)
    cv = symbol_vector(context, dim)  # deterministic context vector
    bound = bind(sv, cv)

    weight_hv = np.where(weight_vec >= 0, 1, -1).astype(np.int8)
    similarity = cosine_similarity(bound, weight_hv)
    return similarity, bound


def hybrid_update(
    weight_vec: np.ndarray,
    x: np.ndarray,
    target: float,
    bound_vec: Vector,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, Vector]:
    """
    Perform an NLMS update and inject the resulting error into the
    hyperdimensional space.

    The error ``e`` scales the bound vector before it is bundled with the
    previous weight hypervector (sign of the weight vector).  The bundled
    result can be used as a new hyper‑representation of the model state.
    """
    new_weights, error = nlms_update(weight_vec, x, target, mu, eps)

    # Scale the bound vector by the signed error (preserve bipolar nature)
    scaled_bound = np.where(bound_vec * (1 if error >= 0 else -1) >= 0, 1, -1).astype(np.int8)

    prev_weight_hv = np.where(weight_vec >= 0, 1, -1).astype(np.int8)
    updated_hv = bundle([prev_weight_hv, scaled_bound])
    return new_weights, updated_hv


def hybrid_edge_contribution(
    coord_a: Tuple[float, float],
    coord_b: Tuple[float, float],
    text: str,
    dim: int = 10000,
    nlms_error: float = 0.1,
    factor: float = 0.5,
) -> float:
    """
    Compute a weighted contribution for an edge (coord_a, coord_b) using
    hyperdimensional and LSM information.

    * The Euclidean distance of the edge is computed.
    * An epistemic certainty is derived from a supplied NLMS error.
    * The edge is encoded as a bound hypervector of its endpoint symbols.
    * The LSM hypervector of ``text`` is used as a mask; its cosine similarity
      with the edge hypervector modulates the final contribution.
    """
    # 1. geometric part
    dist = math.hypot(coord_a[0] - coord_b[0], coord_a[1] - coord_b[1])

    # 2. epistemic certainty from NLMS error
    epistemic = calculate_epistemic_certainty(factor, nlms_error)
    base_weight = calculate_edge_weight(dist, epistemic)

    # 3. hyperdimensional encoding of the edge
    sv_a = symbol_vector(f"({coord_a[0]:.2f},{coord_a[1]:.2f})", dim)
    sv_b = symbol_vector(f"({coord_b[0]:.2f},{coord_b[1]:.2f})", dim)
    edge_hv = bind(sv_a, sv_b)

    # 4. LSM‑derived mask
    lsm_hv = lsm_vector(text, dim)

    # Cosine similarity acts as a scaling factor in [‑1,1]; shift to [0,1]
    sim = cosine_similarity(edge_hv, lsm_hv)
    sim_scaled = (sim + 1.0) / 2.0

    return base_weight * sim_scaled


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    DIM = 2048

    # Initialise a random NLMS weight vector
    rng = np.random.default_rng(42)
    w = rng.standard_normal(DIM).astype(np.float64)

    # Example input for NLMS
    x_input = rng.standard_normal(DIM).astype(np.float64)
    target_val = 0.3

    # Hybrid predict
    sim, bound = hybrid_predict(
        symbol="alpha",
        context="beta",
        weight_vec=w,
        dim=DIM,
    )
    print(f"Hybrid cosine similarity (symbol‑context vs. weights): {sim:.4f}")

    # Hybrid update (injecting NLMS error into hyperdimensional space)
    w_new, hv_new = hybrid_update(
        weight_vec=w,
        x=x_input,
        target=target_val,
        bound_vec=bound,
        mu=0.4,
    )
    print(f"NLMS error‑aware hypervector norm: {np.linalg.norm(hv_new)}")

    # Edge contribution example
    edge_val = hybrid_edge_contribution(
        coord_a=(1.0, 2.0),
        coord_b=(4.0, 6.0),
        text="the quick brown fox jumps over the lazy dog",
        dim=DIM,
        nlms_error=0.05,
    )
    print(f"Hybrid edge contribution: {edge_val:.6f}")

    # Verify that the updated weight vector still has the correct shape
    assert w_new.shape == (DIM,)
    print("Smoke test completed successfully.")