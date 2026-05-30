# DARWIN HAMMER — match 27, survivor 3
# gen: 2
# parent_a: hybrid_hard_truth_math_model_pool_m8_s4.py (gen1)
# parent_b: kan.py (gen0)
# born: 2026-05-29T23:25:23Z

"""Hybrid Stylometry–KAN Model
================================

This module fuses two distinct parent algorithms:

* **Parent A** – ``hard_truth_math.py`` provides stylometric feature extraction
  from raw text, yielding a fixed‑size numeric vector.
* **Parent B** – ``kan.py`` implements Kolmogorov‑Arnold Networks (KAN) where
  every edge carries a learnable univariate B‑spline.

**Mathematical bridge** – The stylometric vector `s ∈ ℝ^d` is a continuous
representation of a piece of text.  The KAN theorem guarantees that any
continuous mapping `f : ℝ^d → ℝ^m` can be expressed as a composition of
univariate spline functions.  By feeding the stylometric vector into a KAN
layer we obtain a unified system that maps raw text → stylometric features →
KAN‑parameterised function.  The hybrid therefore combines the discrete
linguistic counting of Parent A with the universal approximation power of
Parent B.

The code below implements:
* stylometric extraction (`stylometry_features`);
* B‑spline basis evaluation (`bspline_basis`);
* a single KAN layer (`kan_layer`);
* utilities to initialise hybrid parameters (`init_hybrid_layer`);
* high‑level hybrid pipelines (`hybrid_feature_vector`,
  `hybrid_predict`).

All operations are pure NumPy and rely only on the Python standard library."""

import hashlib
import math
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – stylometry utilities (adapted)
# ----------------------------------------------------------------------
FUNCTION_CATS: dict[str, set[str]] = {
    "pronoun": set(
        "i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()
    ),
    "article": set("a an the".split()),
    "preposition": set(
        "about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()
    ),
    "auxiliary": set(
        "am are be been being can could did do does had has have is may might must shall should was were will would".split()
    ),
    "conjunction": set(
        "and but or nor so yet because although while if when where whereas unless until".split()
    ),
    "negation": set(
        "no not never none neither cannot can't won't don't didn't isn't aren't was wasn't weren't".split()
    ),
    "quantifier": set(
        "all any both each few many more most much none several some such".split()
    ),
    "adverb_common": set(
        "very really just still already also even only then there here now often always sometimes".split()
    ),
}
PUNCT = "!?;:,.—-()[]{}\"'`/\\|@#$%^&*+=~"


def words(text: str) -> List[str]:
    """Split ``text`` into lowercase alphabetic tokens."""
    return [word for word in (text or "").lower().split() if word.isalpha()]


def lsm_vector(text: str) -> dict[str, float]:
    """Lexical‑style‑measure (LSM) vector – proportion of each function class."""
    ws = words(text)
    total = max(1, len(ws))
    cnt = Counter(ws)
    return {
        cat: sum(cnt[w] for w in vocab) / total
        for cat, vocab in FUNCTION_CATS.items()
    }


def stable_hash(text: str) -> int:
    """Deterministic 48‑bit integer hash of ``text``."""
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)


def stylometry_features(text: str, dim: int = 96) -> np.ndarray:
    """Return a fixed‑length (``dim``) float vector summarising ``text``.

    The first 8 components are handcrafted ratios; the remaining slots are
    filled with a simple linear projection of the LSM dictionary to meet the
    required dimensionality.
    """
    ws = words(text)
    total_words = max(1, len(ws))
    total_chars = max(1, len(text or ""))

    # Hand‑crafted scalar features (8 of them)
    handcrafted = [
        total_words / 500.0,
        sum(len(w) for w in ws) / total_words / 12.0,
        (text.count("\n") + 1) / 200.0,
        sum(text.count(p) for p in "!?") / total_chars,
        sum(text.count(p) for p in ";:") / total_chars,
        sum(text.count(p) for p in "-—") / total_chars,
        sum(1 for ch in text if ch.isupper()) / total_chars,
        sum(1 for ch in text if ch in PUNCT) / total_chars,
    ]

    # LSM part – flatten dict values, pad/truncate to ``dim - 8``.
    lsm = list(lsm_vector(text).values())
    needed = dim - len(handcrafted)
    if needed > 0:
        # repeat the pattern if not enough entries
        repeats = (needed + len(lsm) - 1) // len(lsm)
        lsm = (lsm * repeats)[:needed]
    else:
        lsm = lsm[: dim - len(handcrafted)]

    vec = np.asarray(handcrafted + lsm, dtype=np.float64)
    # Normalise to unit L2 norm to keep magnitudes comparable for KAN
    norm = np.linalg.norm(vec) + 1e-12
    return vec / norm


# ----------------------------------------------------------------------
# Parent B – Kolmogorov‑Arnold Network utilities (adapted)
# ----------------------------------------------------------------------


def bspline_basis(x: np.ndarray, grid: np.ndarray, k: int = 3) -> np.ndarray:
    """Evaluate B‑spline basis of order ``k`` at points ``x``.

    Parameters
    ----------
    x : (N,) array of evaluation points.
    grid : (G,) array of interior knots (uniform spacing assumed).
    k : spline order (default cubic).

    Returns
    -------
    B : (N, G - 1) array where each column is a basis function.
    """
    x = np.asarray(x, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)

    # Build clamped knot vector with (k) repeated boundary knots
    t = np.concatenate(
        (
            np.full(k, grid[0]),
            grid,
            np.full(k, grid[-1]),
        )
    )
    n_basis = len(grid) - 1  # number of non‑zero basis functions

    # Initialise degree‑0 basis (piecewise constant)
    B = np.where(
        (x[:, None] >= t[:-1]) & (x[:, None] < t[1:]), 1.0, 0.0
    )  # shape (N, len(t)-1)

    # Cox‑de Boor recursion
    for d in range(1, k):
        denom1 = t[d + np.arange(len(t) - k - 1)] - t[np.arange(len(t) - k - 1)]
        denom2 = t[d + 1 + np.arange(len(t) - k - 1)] - t[1 + np.arange(len(t) - k - 1)]

        term1 = np.where(
            denom1 > 0,
            ((x[:, None] - t[np.arange(len(t) - k - 1)]) / denom1) * B[:, :-1],
            0.0,
        )
        term2 = np.where(
            denom2 > 0,
            ((t[d + 1 + np.arange(len(t) - k - 1)] - x[:, None]) / denom2) * B[:, 1:],
            0.0,
        )
        B = term1 + term2

    # Trim to the first ``n_basis`` columns (the rest are zero)
    return B[:, :n_basis]


@dataclass
class KanLayerParams:
    """Learnable spline coefficients for a single KAN layer."""
    coeff: np.ndarray  # shape (out_dim, in_dim, n_basis)
    grid: np.ndarray   # 1‑D knot grid shared across dimensions
    order: int         # spline order (k)


def init_hybrid_layer(in_dim: int, out_dim: int, grid_size: int = 10, order: int = 3) -> KanLayerParams:
    """Randomly initialise a KAN layer suitable for stylometric vectors.

    The knot grid is uniformly spaced over the interval [-1, 1] – the typical
    range after normalising stylometry features.
    """
    # Uniform grid for interior knots
    grid = np.linspace(-1.0, 1.0, grid_size)
    n_basis = grid_size - 1
    rng = np.random.default_rng()
    coeff = rng.normal(loc=0.0, scale=0.1, size=(out_dim, in_dim, n_basis))
    return KanLayerParams(coeff=coeff, grid=grid, order=order)


def kan_layer(x: np.ndarray, params: KanLayerParams) -> np.ndarray:
    """Apply a single KAN layer to input ``x`` (shape (N, in_dim)).

    Returns an array of shape (N, out_dim).
    """
    N, in_dim = x.shape
    out_dim, _, n_basis = params.coeff.shape
    k = params.order

    # Compute basis once per input dimension
    # Result: (N, in_dim, n_basis)
    basis = np.empty((N, in_dim, n_basis), dtype=np.float64)
    for p in range(in_dim):
        basis[:, p, :] = bspline_basis(x[:, p], params.grid, k)

    # Contract: out_dim = Σ_p Σ_i coeff[q, p, i] * B_{p,i}(x)
    # Use einsum for clarity
    out = np.einsum("qpi, nip -> nq", params.coeff, basis)
    return out


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------


def hybrid_feature_vector(
    text: str,
    kan_params: KanLayerParams,
    dim: int = 96,
) -> np.ndarray:
    """Convert ``text`` into a KAN‑processed feature vector.

    1. Extract stylometric vector ``s`` of size ``dim``.
    2. Normalise ``s`` to the interval [-1, 1] (required for the spline grid).
    3. Feed ``s`` through a single KAN layer defined by ``kan_params``.
    4. Return the resulting activation (shape (out_dim,)).
    """
    s = stylometry_features(text, dim=dim)  # (dim,)
    # Scale from unit‑norm to roughly [-1, 1]
    s_scaled = np.clip(s * 2.0, -1.0, 1.0).reshape(1, -1)  # (1, dim)
    out = kan_layer(s_scaled, kan_params)  # (1, out_dim)
    return out.squeeze()


def hybrid_predict(
    texts: List[str],
    kan_params: KanLayerParams,
    dim: int = 96,
) -> np.ndarray:
    """Batch version of :func:`hybrid_feature_vector`.

    Returns a matrix of shape (len(texts), out_dim) where each row is the KAN
    representation of the corresponding text.
    """
    batch = np.vstack([stylometry_features(t, dim=dim) for t in texts])
    batch = np.clip(batch * 2.0, -1.0, 1.0)  # scale to [-1, 1]
    return kan_layer(batch, kan_params)


def hybrid_hash_seed(text: str) -> int:
    """Derive a deterministic integer seed from ``text`` for reproducible init."""
    return stable_hash(text) % (2 ** 31 - 1)


def init_hybrid_from_text(
    example_text: str,
    out_dim: int,
    grid_size: int = 10,
    order: int = 3,
) -> KanLayerParams:
    """Initialise KAN parameters using a hash‑derived random seed.

    This ties the model's initial state to a piece of text, demonstrating a
    concrete mathematical link between the stylometric domain and the spline
    coefficients.
    """
    seed = hybrid_hash_seed(example_text)
    rng = random.Random(seed)
    # Seed NumPy's Generator with a 64‑bit integer derived from the Python RNG
    np_seed = rng.getrandbits(64) % (2 ** 32 - 1)
    np.random.seed(np_seed)
    return init_hybrid_layer(in_dim=96, out_dim=out_dim, grid_size=grid_size, order=order)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    example = (
        "In the midst of the quiet forest, the wind whispered "
        "secrets to the ancient trees, and the sun painted "
        "golden patterns on the mossy floor."
    )
    # Initialise a hybrid KAN layer using the example text as a seed
    params = init_hybrid_from_text(example_text=example, out_dim=4)

    # Single‑text inference
    vec = hybrid_feature_vector(example, params)
    print("Hybrid feature vector (single):", vec)

    # Batch inference
    batch_texts = [
        example,
        "Mathematics is the language with which God has written the universe.",
        "Artificial intelligence blends statistics, logic, and engineering.",
    ]
    batch_out = hybrid_predict(batch_texts, params)
    print("Batch output shape:", batch_out.shape)
    print("Batch output:\n", batch_out)