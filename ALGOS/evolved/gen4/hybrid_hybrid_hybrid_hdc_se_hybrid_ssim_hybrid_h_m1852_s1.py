# DARWIN HAMMER — match 1852, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_hdc_serpentin_hybrid_sparse_wta_hy_m117_s0.py (gen3)
# parent_b: hybrid_ssim_hybrid_hybrid_fracti_m934_s4.py (gen3)
# born: 2026-05-29T23:39:11Z

"""Hybrid Hyperdimensional‑Sparse‑SSIM Algorithm

Parents:
- PARENT A: hyperdimensional encoding (bipolar vectors) + sparse winner‑take‑all (WTA) pool.
- PARENT B: Structural Similarity Index (SSIM) + fractional‑Hoeffding bound with hypervector
  binding via circular convolution.

Mathematical bridge:
Both parents manipulate high‑dimensional vectors.  We treat a *query* hypervector `Q`
(encoded from morphology scalars, parent A) and a *model* hypervector `M`
(encoded from a list of real scores, parent B).  The dot‑product similarity
`⟨Q, M⟩` is the classic HDC similarity, while SSIM(`Q`, `M`) measures
structural agreement.  SSIM is injected as the uncertainty term `ε` of a
Hoeffding bound, yielding a fused priority

    priority = ⟨Q, M⟩ · (1 – ε) · SSIM(Q, M)

where `ε = sqrt( (1/(2·N))·ln(2/δ) )` and `N` is the hypervector dimensionality.
The WTA mask (`top_k_mask`) sparsifies `M` before the dot product, preserving the
salient dimensions of parent A.  The three functions below demonstrate this
integration."""

from __future__ import annotations

import hashlib
import math
import random
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Hyperdimensional primitives (shared)
# ----------------------------------------------------------------------


def random_vector(dim: int = 10000, seed: str | int | None = None) -> List[int]:
    """Bipolar hypervector (parent A primitive)."""
    rng = random.Random(seed)
    return [1 if rng.getrandbits(1) else -1 for _ in range(dim)]


def symbol_vector(symbol: str, dim: int = 10000) -> List[int]:
    """Deterministic bipolar vector for a symbol (parent A primitive)."""
    seed = int.from_bytes(
        hashlib.sha256(symbol.encode("utf-8")).digest()[:8], "big"
    )
    return random_vector(dim, seed)


def bind_bipolar(a: List[int], b: List[int]) -> List[int]:
    """Element‑wise multiplication (XOR‑like bind for bipolar vectors)."""
    return [x * y for x, y in zip(a, b)]


def random_hv(dim: int = 10000, kind: str = "complex", seed: int | None = None) -> np.ndarray:
    """Random hypervector (parent B primitive)."""
    rng = np.random.default_rng(seed)
    if kind == "complex":
        theta = rng.uniform(0.0, 2.0 * np.pi, size=dim)
        return np.exp(1j * theta)
    if kind == "bipolar":
        return rng.choice(np.array([-1.0, 1.0]), size=dim)
    if kind == "real":
        v = rng.standard_normal(dim)
        return v / (np.linalg.norm(v) + 1e-30)
    raise ValueError(f"Unsupported kind {kind!r}")


def bind_fft(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Circular convolution via FFT (parent B bind)."""
    return np.fft.ifft(np.fft.fft(x) * np.fft.fft(y))


def fractional_power(X: np.ndarray, alpha: float) -> np.ndarray:
    """Fractional power transformation (parent B component)."""
    # Preserve sign for real vectors, magnitude for complex.
    if np.iscomplexobj(X):
        mag = np.abs(X) ** alpha
        phase = np.exp(1j * np.angle(X))
        return mag * phase
    else:
        return np.sign(X) * (np.abs(X) ** alpha)


def calculate_ssim(x: Iterable[float], y: Iterable[float],
                   dynamic_range: float = 255.0,
                   k1: float = 0.01, k2: float = 0.03) -> float:
    """Structural Similarity Index (parent B primitive)."""
    x = list(x)
    y = list(y)
    if len(x) != len(y):
        raise ValueError("samples must have equal length")
    if not x:
        raise ValueError("samples must not be empty")
    n = len(x)
    mx = sum(x) / n
    my = sum(y) / n
    vx = sum((v - mx) ** 2 for v in x) / n
    vy = sum((v - my) ** 2 for v in y) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y)) / n
    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2
    return ((2 * mx * my + c1) * (2 * cov + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))


def top_k_mask(vec: np.ndarray, k: int) -> np.ndarray:
    """Sparse WTA mask: 1 on the k largest |values|, 0 elsewhere."""
    if k <= 0:
        return np.zeros_like(vec, dtype=float)
    flat = np.abs(vec).ravel()
    if k >= flat.size:
        return np.ones_like(vec, dtype=float)
    threshold = np.partition(flat, -k)[-k]
    mask = (np.abs(vec) >= threshold).astype(float)
    return mask


# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------


def encode_morphology(scalars: List[float],
                      dim: int = 10000) -> np.ndarray:
    """
    Encode a list of scalar morphological features into a bipolar hypervector.
    Each scalar is first turned into a deterministic symbol vector (via its
    string representation) and then bound together using element‑wise
    multiplication.  The result is returned as a float ndarray to be compatible
    with the FFT‑based operations of parent B.
    """
    hv = np.ones(dim, dtype=float)
    for i, val in enumerate(scalars):
        sym = f"feat_{i}_{val:.4f}"
        vec = np.array(symbol_vector(sym, dim), dtype=float)
        hv = bind_bipolar(hv.tolist(), vec.tolist())
    return hv


def expand_scores(scores: List[float],
                  dim: int = 10000,
                  alpha: float = 0.5,
                  seed: int | None = None) -> np.ndarray:
    """
    Map a list of real scores to a single hypervector.
    Each score generates a real‑valued random hypervector, weighted by the
    fractional power `alpha` (parent B), and all are combined by circular
    convolution (FFT bind).  The final vector is normalized.
    """
    if not scores:
        raise ValueError("scores list must not be empty")
    rng = np.random.default_rng(seed)
    # Initialise with identity element for convolution (delta at index 0)
    agg = np.zeros(dim, dtype=complex)
    agg[0] = 1.0 + 0j
    for s in scores:
        base = random_hv(dim, kind="real", seed=int(rng.integers(1e9)))
        weighted = fractional_power(base, alpha) * s
        agg = bind_fft(agg, weighted.astype(complex))
    # Normalise to unit length
    norm = np.linalg.norm(agg)
    if norm == 0:
        return agg
    return agg / norm


def hybrid_priority(query: np.ndarray,
                    model: np.ndarray,
                    k: int = 100,
                    delta: float = 0.05) -> float:
    """
    Compute a fused priority score.
    1. Apply a WTA mask to the model hypervector (parent A).
    2. Compute dot‑product similarity with the query.
    3. Compute SSIM between the real‑valued representations of the two vectors.
    4. Derive a Hoeffding bound epsilon using SSIM as the confidence factor.
    5. Return the combined priority.
    """
    if query.shape != model.shape:
        raise ValueError("query and model must share the same shape")
    # 1. Sparse WTA
    mask = top_k_mask(model, k)
    sparse_model = model * mask
    # 2. Dot‑product similarity (bipolar / real part)
    dot_sim = np.real(np.vdot(query, sparse_model))
    # 3. SSIM (operate on magnitudes to stay in real domain)
    ssim_val = calculate_ssim(np.abs(query), np.abs(model))
    # 4. Hoeffding epsilon (uncertainty shrinks as SSIM → 1)
    n = query.size
    epsilon = math.sqrt((1.0 / (2 * n)) * math.log(2.0 / delta)) * (1.0 - ssim_val)
    # 5. Fuse
    priority = dot_sim * (1.0 - epsilon) * ssim_val
    return float(priority)


# ----------------------------------------------------------------------
# Demonstration / smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    DIM = 4096
    # Morphology encoding (parent A side)
    morphology = encode_morphology([0.12, 3.4, 7.8], dim=DIM)

    # Model scores (parent B side)
    scores = [0.85, 0.33, 0.67, 0.91]
    model_hv = expand_scores(scores, dim=DIM, alpha=0.7, seed=42)

    # Hybrid priority computation
    prio = hybrid_priority(morphology, model_hv, k=200, delta=0.01)

    print(f"Hybrid priority: {prio:.6f}")
    sys.exit(0)