# DARWIN HAMMER — match 2745, survivor 7
# gen: 7
# parent_a: hybrid_hybrid_indy_learning_hybrid_hybrid_fisher_m72_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_tropic_m2576_s0.py (gen6)
# born: 2026-05-29T23:45:38Z

"""
Hybrid Algorithm: Fusion of hybrid_indy_learning_vector (Parent A) and 
hybrid_hybrid_cockpit_fisher (Parent B)

Mathematical Bridge
-------------------
Both parent algorithms rely on *Fisher information* derived from a Gaussian 
profile:

* Parent A uses Fisher information as a weighting factor for tokenization and 
  chunking of text.
* Parent B defines ``fisher_score`` (the Fisher information of a Gaussian beam) 
  and combines it with tropical algebra (max‑plus matrix operations) and a 
  trust‑weighted velocity term.

The hybrid therefore **maps each textual token to a Fisher‑information value**
and then **propagates those values over a trust graph using tropical matrix 
multiplication**.  The result is a globally consistent token importance that 
respects both local statistical uncertainty (Fisher) and relational trust 
structure (tropical geometry).

The core functions below implement:
1. Fisher‑information computation (Gaussian beam).
2. Tokenisation with Fisher‑based weights.
3. Trust‑weighted tropical propagation of those weights.
"""

import json
import hashlib
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Utility functions (borrowed from Parent A)
# ----------------------------------------------------------------------


def sha256_json(value: Any) -> str:
    """Deterministic SHA‑256 of a JSON‑serialisable value."""
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
    ).hexdigest()


def load_go_terms(root: Path = Path(__file__).resolve().parents[1]) -> List[str]:
    """Load ontology terms; fall back to a small default list."""
    p = root / "OFFICIAL_ONTOLOGY.json"
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        terms = data.get("active_terms") or []
        return [str(t).upper() for t in terms if str(t).strip()]
    except Exception:
        return [
            "ENTITY",
            "ATTRIBUTE",
            "RELATIONSHIP",
            "ACTION",
            "EVENT",
            "TIME",
            "EVIDENCE",
            "CLAIM",
            "HYPOTHESIS",
            "SIGNAL",
            "PATTERN",
            "TOOL",
            "ALGORITHM",
            "BOOK",
            "SOURCE",
            "LEAD",
            "LOCATION",
            "LAW",
            "RULE",
        ]


def tokenize(text: str) -> List[Dict[str, Any]]:
    """Return a list of token dictionaries with character offsets."""
    WORD_RE = re.compile(r"\S+")
    return [
        {"token": m.group(0), "start": m.start(), "end": m.end()}
        for m in WORD_RE.finditer(text)
    ]


# ----------------------------------------------------------------------
# Fisher‑information and Gaussian beam (Parent B)
# ----------------------------------------------------------------------


def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity profile."""
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float = 0.0, width: float = 1.0, eps: float = 1e-12) -> float:
    """Fisher information of a Gaussian beam."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


# ----------------------------------------------------------------------
# Tropical algebra utilities (Parent B)
# ----------------------------------------------------------------------


def t_add(x, y):
    """Tropical addition: max(x, y). Works element‑wise on NumPy arrays."""
    return np.maximum(x, y)


def t_mul(x, y):
    """Tropical multiplication: x + y. Works element‑wise on NumPy arrays."""
    return np.add(x, y)


def t_matmul(A, B):
    """
    Tropical matrix multiplication (max‑plus).

    C[i, j] = max_k ( A[i, k] + B[k, j] )
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("Both A and B must be 2‑D arrays")
    m, p = A.shape
    p2, n = B.shape
    if p != p2:
        raise ValueError(f"Incompatible shapes {A.shape} and {B.shape}")

    # initialise result with -inf (identity for max)
    C = np.full((m, n), -np.inf, dtype=float)

    for k in range(p):
        # broadcasting: column k of A (m,1) + row k of B (1,n)
        C = np.maximum(C, A[:, k][:, np.newaxis] + B[k, :][np.newaxis, :])
    return C


# ----------------------------------------------------------------------
# Trust‑weighted velocity (Parent B)
# ----------------------------------------------------------------------


def trust_weighted_velocity(x0: float, x1: float, trust: float) -> float:
    """Linear displacement scaled by a trust factor."""
    return trust * (x1 - x0)


# ----------------------------------------------------------------------
# Hybrid core functions (new)
# ----------------------------------------------------------------------


def token_fisher_weights(text: str) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """
    Tokenise *text* and assign each token a Fisher‑information weight.

    The token length (in characters) is used as the Gaussian variable ``theta``.
    The Gaussian is centred at the mean token length of the document and its
    width is the standard deviation (with a small epsilon to avoid division by
    zero).  The resulting Fisher scores are returned as a 1‑D NumPy array.
    """
    tokens = tokenize(text)

    lengths = np.array([len(t["token"]) for t in tokens], dtype=float)
    if len(lengths) == 0:
        return tokens, np.array([], dtype=float)

    mean_len = float(np.mean(lengths))
    std_len = float(np.std(lengths)) + 1e-6  # avoid zero width

    fisher_vals = np.array(
        [fisher_score(theta=ln, center=mean_len, width=std_len) for ln in lengths],
        dtype=float,
    )
    # Normalise to [0,1] for stability
    if fisher_vals.max() > 0:
        fisher_vals /= fisher_vals.max()
    return tokens, fisher_vals


def build_trust_matrix(num_tokens: int, sparsity: float = 0.3, seed: int = None) -> np.ndarray:
    """
    Construct a random trust matrix ``T`` of shape (N, N).

    * ``sparsity`` controls the fraction of entries that are set to a positive
      trust value (the rest stay at ``-inf`` which is the tropical additive
      identity).
    * ``seed`` makes the generation deterministic for testing.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    T = np.full((num_tokens, num_tokens), -np.inf, dtype=float)
    for i in range(num_tokens):
        for j in range(num_tokens):
            if i == j:
                # self‑trust is maximal (0 in max‑plus, because 0 is the neutral element for addition)
                T[i, j] = 0.0
            elif random.random() < sparsity:
                # trust values are drawn from (0, 1]; higher means more trust
                T[i, j] = random.random()
    return T


def tropical_trust_propagation(initial_weights: np.ndarray, trust_matrix: np.ndarray, steps: int = 5) -> np.ndarray:
    """
    Propagate token weights over the trust graph using tropical matrix multiplication.

    The iteration is::

        w_{t+1} = t_mul( w_t , trust_matrix )
        w_{t+1} = t_add( w_{t+1} , w_t )   # accumulate previous knowledge

    ``steps`` controls how many hops of trust are considered.
    """
    if initial_weights.ndim != 1:
        raise ValueError("initial_weights must be a 1‑D array")
    if trust_matrix.shape[0] != trust_matrix.shape[1]:
        raise ValueError("trust_matrix must be square")
    if trust_matrix.shape[0] != initial_weights.shape[0]:
        raise ValueError("Dimension mismatch between weights and trust matrix")

    w = initial_weights.copy()
    for _ in range(steps):
        # Tropical multiplication of a row vector by the matrix
        w_propagated = t_mul(w[np.newaxis, :], trust_matrix)[0]
        # Accumulate (max) with previous knowledge
        w = t_add(w, w_propagated)
    # Normalise again to [0,1] for interpretability
    if w.max() > 0:
        w /= w.max()
    return w


def hybrid_token_importance(text: str, trust_sparsity: float = 0.3, steps: int = 5, seed: int = None) -> List[Tuple[str, float]]:
    """
    End‑to‑end hybrid routine:

    1. Tokenise *text* and compute Fisher‑information weights.
    2. Build a random trust matrix (simulating relational trust between tokens).
    3. Propagate the Fisher weights through the trust graph using tropical algebra.
    4. Return a list of ``(token, final_score)`` sorted by descending importance.
    """
    tokens, fisher_weights = token_fisher_weights(text)
    if len(tokens) == 0:
        return []

    T = build_trust_matrix(len(tokens), sparsity=trust_sparsity, seed=seed)
    final_weights = tropical_trust_propagation(fisher_weights, T, steps=steps)

    results = [(tok["token"], float(score)) for tok, score in zip(tokens, final_weights)]
    results.sort(key=lambda x: x[1], reverse=True)
    return results


# ----------------------------------------------------------------------
# Optional auxiliary metrics from Parent B (not mandatory but kept for completeness)
# ----------------------------------------------------------------------


def anti_slop_ratio(claims_with_evidence: int, total_claims_emitted: int) -> float:
    """Ratio of evidential claims to total claims, clamped to [0,1]."""
    return 1.0 if total_claims_emitted <= 0 else max(0.0, min(1.0, claims_with_evidence / total_claims_emitted))


def cockpit_honesty(displayed_ok: int, unknown_displayed_as_ok: int) -> float:
    """Proportion of displayed OK items among all displayed items."""
    total = displayed_ok + unknown_displayed_as_ok
    return 1.0 if total <= 0 else max(0.0, min(1.0, displayed_ok / total))


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    sample_text = (
        "The quick brown fox jumps over the lazy dog. "
        "Information theory connects Fisher metrics with token uncertainty."
    )
    importance = hybrid_token_importance(sample_text, trust_sparsity=0.4, steps=7, seed=42)

    print("Hybrid token importance (top 10):")
    for token, score in importance[:10]:
        print(f"{token!r:>12} : {score:.4f}")

    # Simple sanity checks
    assert isinstance(importance, list)
    assert all(isinstance(t, str) and isinstance(s, float) for t, s in importance)
    print("\nSmoke test completed successfully.")