# DARWIN HAMMER — match 2745, survivor 6
# gen: 7
# parent_a: hybrid_hybrid_indy_learning_hybrid_hybrid_fisher_m72_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_tropic_m2576_s0.py (gen6)
# born: 2026-05-29T23:45:38Z

"""
Hybrid Algorithm: Fusion of hybrid_indy_learning_vector (token‑Fisher weighting) 
and hybrid_hybrid_tropical_fisher (tropical algebra with Fisher score).

Mathematical Bridge
------------------
Both parent algorithms compute a *Fisher information* derived from a Gaussian
beam model:

* Parent A uses Fisher information to weight text tokens.
* Parent B provides a `fisher_score` function and tropical algebra
  (max‑plus semiring) for matrix propagation.

The hybrid algorithm therefore:
1. Tokenises input text and assigns each token a Fisher‑based weight
   (`token_fisher_weight`).
2. Constructs a trust‑weighted co‑occurrence matrix whose entries are
   `trust_weighted_velocity` scaled by the Fisher weights.
3. Propagates information through the matrix using tropical matrix
   multiplication (`tropical_matmul`), yielding a max‑plus “trust‑flow”
   that respects both the statistical (Fisher) and geometric (tropical)
   structures of the parents.
"""

import json
import hashlib
import math
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


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
    """Load ontology terms; fall back to a built‑in list."""
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


# ----------------------------------------------------------------------
# Parent A – tokenisation & Fisher weighting
# ----------------------------------------------------------------------
WORD_RE = re.compile(r"\S+")


def tokenize(text: str) -> List[Dict[str, Any]]:
    """Return a list of token dicts with start/end character offsets."""
    return [
        {"token": m.group(0), "start": m.start(), "end": m.end()}
        for m in WORD_RE.finditer(text)
    ]


def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity used for Fisher information."""
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float = 0.0, width: float = 1.0, eps: float = 1e-12) -> float:
    """Fisher information of a Gaussian beam."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def token_fisher_weight(token: str, avg_len: float, std_len: float) -> float:
    """
    Compute a Fisher‑based weight for a token.
    The token length is used as the “θ” argument of the Gaussian.
    """
    length = len(token)
    # Avoid zero std deviation
    width = max(std_len, 1e-6)
    return fisher_score(theta=length, center=avg_len, width=width)


# ----------------------------------------------------------------------
# Parent B – trust, velocity and tropical algebra
# ----------------------------------------------------------------------
def anti_slop_ratio(claims_with_evidence: int, total_claims_emitted: int) -> float:
    """Simple trust proxy based on evidence ratio."""
    return (
        1.0
        if total_claims_emitted <= 0
        else max(0.0, min(1.0, claims_with_evidence / total_claims_emitted))
    )


def trust_weighted_velocity(x0: float, x1: float, trust: float) -> float:
    """Velocity scaled by a trust factor."""
    return trust * (x1 - x0)


def t_add(x, y):
    """Tropical addition (max). Works element‑wise on numpy arrays."""
    return np.maximum(x, y)


def t_mul(x, y):
    """Tropical multiplication (plus). Works element‑wise on numpy arrays."""
    return np.add(x, y)


def tropical_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Tropical (max‑plus) matrix multiplication.
    C[i, j] = max_k ( A[i, k] + B[k, j] )
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    m, p = A.shape
    p2, n = B.shape
    if p != p2:
        raise ValueError("Inner dimensions must match for tropical matmul")
    # Initialise with -inf (the tropical additive identity)
    C = np.full((m, n), -np.inf, dtype=float)
    for k in range(p):
        C = t_add(C, t_mul(A[:, k][:, np.newaxis], B[k, :][np.newaxis, :]))
    return C


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def compute_token_weights(text: str) -> Tuple[List[str], np.ndarray]:
    """
    Tokenise *text*, compute Fisher‑based weights for each distinct token,
    and return the ordered list of unique tokens together with a weight vector.
    """
    tokens_info = tokenize(text)
    tokens = [info["token"] for info in tokens_info]

    if not tokens:
        return [], np.array([])

    lengths = np.array([len(t) for t in tokens], dtype=float)
    avg_len = float(lengths.mean())
    std_len = float(lengths.std(ddof=0))

    # Aggregate by token string
    uniq_tokens, inverse = np.unique(tokens, return_inverse=True)
    weight_vec = np.zeros(len(uniq_tokens), dtype=float)

    for idx, token in enumerate(uniq_tokens):
        # Fisher weight based on token length statistics
        weight_vec[idx] = token_fisher_weight(token, avg_len, std_len)

    # Normalise to [0,1] for stability
    if weight_vec.max() > 0:
        weight_vec /= weight_vec.max()
    return list(uniq_tokens), weight_vec


def build_trust_cooccurrence_matrix(
    tokens: List[str],
    weight_vec: np.ndarray,
    trust: float,
) -> np.ndarray:
    """
    Build a symmetric co‑occurrence matrix *M* where
    M[i, j] = trust_weighted_velocity(pos_i, pos_j, trust) * (w_i + w_j)

    *pos_i* is the first occurrence index of token *i* in the original text.
    The matrix is suitable for tropical propagation.
    """
    n = len(tokens)
    if n == 0:
        return np.array([[]], dtype=float)

    # Map token to first position
    token_positions = {tok: i for i, tok in enumerate(tokens)}

    M = np.full((n, n), -np.inf, dtype=float)  # -inf = tropical additive identity
    for i in range(n):
        for j in range(n):
            pos_i = token_positions[tokens[i]]
            pos_j = token_positions[tokens[j]]
            vel = trust_weighted_velocity(float(pos_i), float(pos_j), trust)
            w_sum = weight_vec[i] + weight_vec[j]
            M[i, j] = vel * w_sum
    return M


def tropical_propagate(matrix: np.ndarray, steps: int = 2) -> np.ndarray:
    """
    Apply tropical matrix multiplication repeatedly to spread trust‑weighted
    information across the token graph.

    Returns the matrix after ``steps`` multiplications (including the original).
    """
    result = matrix.copy()
    for _ in range(steps):
        result = tropical_matmul(result, matrix)
    return result


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_text = (
        "The quick brown fox jumps over the lazy dog. "
        "The quick brown fox is swift and clever."
    )
    # 1. Compute token weights
    uniq_tokens, weights = compute_token_weights(sample_text)
    print("Unique tokens:", uniq_tokens)
    print("Normalized Fisher weights:", weights)

    # 2. Derive a trust value (e.g., based on dummy evidence counts)
    trust = anti_slop_ratio(claims_with_evidence=8, total_claims_emitted=10)
    print("Trust factor:", trust)

    # 3. Build the trust‑weighted co‑occurrence matrix
    co_mat = build_trust_cooccurrence_matrix(uniq_tokens, weights, trust)
    print("Co‑occurrence matrix (raw tropical values):\n", co_mat)

    # 4. Propagate via tropical algebra
    propagated = tropical_propagate(co_mat, steps=3)
    print("Propagated matrix after tropical multiplication:\n", propagated)

    # Simple sanity check: no NaNs and shape preserved
    assert propagated.shape == co_mat.shape
    assert not np.isnan(propagated).any()
    print("Hybrid algorithm executed successfully.")