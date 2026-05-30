# DARWIN HAMMER — match 1266, survivor 4
# gen: 6
# parent_a: hybrid_hybrid_hybrid_sketch_hybrid_hybrid_hybrid_m1144_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_ternar_hybrid_sparse_wta_hy_m656_s1.py (gen5)
# born: 2026-05-29T23:34:58Z

"""Hybrid Sketch‑Sheaf‑TTT‑WTA Algorithm.

Parents:
- hybrid_hybrid_sketches_rlct_sheaf_cohomology_m11_s2.py (sketch + sheaf cohomology)
- hybrid_hybrid_hybrid_ternar_hybrid_sparse_wta_hy_m656_s1.py (ternary‑router linear model + sparse winner‑take‑all)

Mathematical bridge:
The min‑hash signatures generated for a document act as deterministic hash
functions for a Count‑Min sketch.  Each sketch bucket stores a *hyper‑vector*
(via `random_hv`).  The collection of bucket‑hypervectors forms a cellular
sheaf; the coboundary operator computes residual hypervectors by fractional
binding of neighboring buckets.  Those residuals are flattened and fed to the
TTT linear model `W`.  The TTT loss (reconstruction error) supplies a gradient
that updates `W`, while the updated projection `W @ r` (where `r` is a residual)
produces a score vector.  A sparse winner‑take‑all (WTA) selects the top‑k
entries of this score vector, which can be interpreted as the most “active”
latent directions for the current document.  Thus the algorithm intertwines
sketch‑based dimensionality reduction, sheaf‑theoretic inconsistency measurement,
linear self‑supervision, and sparse competition in a single unified pipeline.
"""

import hashlib
import math
import random
import sys
import pathlib
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np

# ---------------------------------------------------------------------------
# Hypervector utilities (from Parent A)
# ---------------------------------------------------------------------------

def random_hv(d: int = 1024, kind: str = "complex", seed: int | None = None) -> np.ndarray:
    """Generate a random hypervector of dimension *d*."""
    rng = np.random.default_rng(seed)
    if kind == "complex":
        theta = rng.uniform(0.0, 2.0 * np.pi, size=d)
        return np.exp(1j * theta)
    elif kind == "bipolar":
        return rng.choice([-1, 1], size=d).astype(np.float64)
    elif kind == "real":
        vec = rng.normal(size=d)
        return vec / np.linalg.norm(vec)
    else:
        raise ValueError(f"unknown kind {kind!r}")

def _shingles(text: str, k: int = 5) -> List[str]:
    cleaned = "".join(ch.lower() for ch in text if not ch.isspace())
    if len(cleaned) < k:
        return [cleaned]
    return [cleaned[i:i + k] for i in range(len(cleaned) - k + 1)]

def minhash_for_text(text: str, num_hashes: int = 64, seed_offset: int = 0) -> List[int]:
    """Simple min‑hash of *k*-shingles using SHA‑256."""
    sh = _shingles(text)
    hashes = []
    for i in range(num_hashes):
        m = hashlib.sha256()
        # incorporate seed offset to obtain independent hash functions
        m.update((str(i + seed_offset) + "|" + "|".join(sh)).encode())
        # take the integer value of the digest
        h = int.from_bytes(m.digest()[:8], "big", signed=False)
        hashes.append(h)
    return hashes

# ---------------------------------------------------------------------------
# Count‑Min Sketch + Sheaf utilities
# ---------------------------------------------------------------------------

def init_sketch(width: int, depth: int) -> np.ndarray:
    """Initialize a Count‑Min sketch of shape (depth, width) with zero hypervectors."""
    return np.zeros((depth, width), dtype=object)

def update_sketch(
    sketch: np.ndarray,
    text: str,
    num_hashes: int = 64,
    hv_dim: int = 1024,
    hv_kind: str = "complex",
) -> None:
    """Update the sketch with hypervectors derived from min‑hash signatures."""
    depth, width = sketch.shape
    hashes = minhash_for_text(text, num_hashes=num_hashes)
    for i, h in enumerate(hashes):
        row = i % depth
        col = h % width
        hv = random_hv(d=hv_dim, kind=hv_kind, seed=h)
        if sketch[row, col] is None or not isinstance(sketch[row, col], np.ndarray):
            sketch[row, col] = hv
        else:
            # simple additive aggregation (preserve complex phase by vector addition)
            sketch[row, col] = sketch[row, col] + hv

def _fractional_bind(v1: np.ndarray, v2: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Fractional power binding between two hypervectors."""
    # Ensure same dtype (complex or real)
    if np.iscomplexobj(v1) or np.iscomplexobj(v2):
        # work in magnitude‑phase domain
        mag1, phase1 = np.abs(v1), np.angle(v1)
        mag2, phase2 = np.abs(v2), np.angle(v2)
        mag = (mag1 ** alpha) * (mag2 ** (1 - alpha))
        phase = (phase1 * alpha) + (phase2 * (1 - alpha))
        return mag * np.exp(1j * phase)
    else:
        return (v1 ** alpha) * (v2 ** (1 - alpha))

def sheaf_residuals(sketch: np.ndarray, alpha: float = 0.5) -> List[np.ndarray]:
    """
    Compute coboundary‑like residuals for each adjacent pair of rows
    (treated as edges of a 1‑dimensional cellular complex).
    Returns a list of residual hypervectors.
    """
    depth, width = sketch.shape
    residuals = []
    for col in range(width):
        for row in range(depth - 1):
            hv_u = sketch[row, col]
            hv_v = sketch[row + 1, col]
            if hv_u is None or hv_v is None:
                continue
            bound = _fractional_bind(hv_u, hv_v, alpha=alpha)
            # residual = (hv_u + hv_v) - bound  (a simple coboundary analogue)
            residual = (hv_u + hv_v) - bound
            residuals.append(residual)
    return residuals

# ---------------------------------------------------------------------------
# Ternary‑Router Linear Model (TTT) utilities (from Parent B)
# ---------------------------------------------------------------------------

def init_ttt(d_in: int, d_out: int | None = None, scale: float = 0.01, seed: int = 0) -> np.ndarray:
    """Initialize weight matrix W of shape (d_out, d_in)."""
    rng = np.random.default_rng(seed)
    if d_out is None:
        d_out = d_in
    return rng.standard_normal((d_out, d_in)) * scale

def ttt_loss(W: np.ndarray, x: np.ndarray, target: np.ndarray | None = None) -> float:
    """Reconstruction loss ||W x - target||², target defaults to x."""
    pred = W @ x
    t = x if target is None else target
    residual = pred - t
    return float(residual @ residual)

def ttt_grad(W: np.ndarray, x: np.ndarray, target: np.ndarray | None = None) -> np.ndarray:
    """Gradient of the reconstruction loss w.r.t. W."""
    pred = W @ x
    t = x if target is None else target
    residual = pred - t
    return 2.0 * np.outer(residual, x)

def ttt_update(W: np.ndarray, x: np.ndarray, lr: float = 0.001) -> np.ndarray:
    """One SGD step on the reconstruction loss."""
    grad = ttt_grad(W, x)
    return W - lr * grad

# ---------------------------------------------------------------------------
# Sparse Winner‑Take‑All (WTA) utilities (from Parent B)
# ---------------------------------------------------------------------------

def sparse_wta(scores: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return the indices and values of the top‑k entries of *scores*.
    If k exceeds the length, all entries are returned sorted descending.
    """
    if k >= scores.size:
        idx = np.argsort(-scores)
        return idx, scores[idx]
    idx = np.argpartition(-scores, k)[:k]
    # sort the selected indices
    idx = idx[np.argsort(-scores[idx])]
    return idx, scores[idx]

# ---------------------------------------------------------------------------
# Hybrid pipeline
# ---------------------------------------------------------------------------

def hybrid_process(
    text: str,
    sketch_width: int = 128,
    sketch_depth: int = 8,
    hv_dim: int = 1024,
    num_hashes: int = 64,
    ttt_dim: int = 512,
    wta_k: int = 10,
    alpha: float = 0.5,
    lr: float = 0.001,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    End‑to‑end hybrid operation:

    1. Build / update a Count‑Min sketch with hypervectors derived from min‑hashes.
    2. Compute sheaf residual hypervectors across neighboring rows.
    3. Flatten the residuals into a single vector and feed it to the TTT model.
    4. Apply a sparse WTA on the TTT output to obtain the most active latent indices.

    Returns:
        W            – updated TTT weight matrix,
        scores       – raw TTT output scores (shape (ttt_dim,)),
        top_indices  – indices selected by WTA.
    """
    # 1. Sketch
    sketch = init_sketch(sketch_width, sketch_depth)
    update_sketch(sketch, text, num_hashes=num_hashes, hv_dim=hv_dim)

    # 2. Sheaf residuals
    residuals = sheaf_residuals(sketch, alpha=alpha)
    if not residuals:
        # fallback: use a zero vector if no residuals were generated
        agg_residual = np.zeros(hv_dim, dtype=complex)
    else:
        # aggregate by simple mean (preserve complex nature)
        agg_residual = np.mean(np.stack(residuals), axis=0)

    # 3. TTT projection
    W = init_ttt(d_in=hv_dim, d_out=ttt_dim, scale=0.01, seed=42)
    # One update step using the aggregated residual as input
    W = ttt_update(W, agg_residual, lr=lr)
    scores = W @ agg_residual  # shape (ttt_dim,)

    # 4. Sparse WTA
    top_idx, _ = sparse_wta(scores.real, k=wta_k)  # use real part for ranking

    return W, scores, top_idx.tolist()

# ---------------------------------------------------------------------------
# Simple smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample_text = (
        "In the age of quantum computing, algorithms that blend algebraic topology "
        "with high‑dimensional embeddings become increasingly relevant."
    )
    W, scores, top = hybrid_process(
        sample_text,
        sketch_width=64,
        sketch_depth=4,
        hv_dim=256,
        num_hashes=32,
        ttt_dim=128,
        wta_k=5,
        alpha=0.6,
        lr=0.005,
    )
    print("TTT weight matrix shape:", W.shape)
    print("Scores (first 10):", scores[:10])
    print("Top‑k active indices:", top)