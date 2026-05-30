# DARWIN HAMMER — match 2086, survivor 0
# gen: 5
# parent_a: hybrid_semantic_neighbors_hybrid_endpoint_circ_m17_s5.py (gen2)
# parent_b: hybrid_hybrid_ternary_lens__hybrid_minhash_hybri_m395_s2.py (gen4)
# born: 2026-05-29T23:40:44Z

"""Hybrid Semantic‑Morphology‑MinHash System
Parents:
- hybrid_semantic_neighbors_hybrid_endpoint_circ_m17_s5.py (semantic similarity scaled by morphology‑derived recovery priority)
- hybrid_hybrid_ternary_lens__hybrid_minhash_hybri_m395_s2.py (MinHash signatures, RLCT‑adjusted learning rate, risk‑aware modulation)

Mathematical Bridge
------------------
1. **Morphology → Priority**  
   A document’s morphology `m` yields a recovery priority `p(m)∈[0,1]` via the
   right‑ing‑time index (Parent A).

2. **MinHash → Feature Vector**  
   Tokens of a document are transformed into a fixed‑length MinHash signature
   `s ∈ ℕ^k`.  Treating `s` as a real‑valued vector enables a cosine similarity
   `c = cos(s_q , s_c)` between a query (`q`) and a candidate (`c`).

3. **Hybrid Affinity**  
   The semantic affinity is modulated by the candidate’s morphology:
   `h = c · p(m_c)` (identical to Parent A’s definition, but `c` now stems from
   MinHash rather than dense embeddings).

4. **RLCT‑adjusted Learning Rate**  
   Signature complexity is measured by the normalized distinct‑hash ratio  
   `H(s) = |unique(s)| / k`.  The Real‑Log‑Canonical‑Threshold (RLCT) is
   `λ = 1 / (1 + H(s))`.

5. **Risk‑aware Effective Rate**  
   Given a risk vector `r ∈ [0,1]^d` from audit findings, the effective learning
   rate for an NLMS predictor becomes  
   `μ_eff = μ_base · λ · (1 – mean(r))`.

The module implements the above equations and provides three core functions
demonstrating the hybrid operation, plus an NLMS weight‑update routine that
uses the computed `μ_eff`.  A smoke test at the end validates the pipeline."""

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Morphology & Recovery Priority (Parent A)
# ----------------------------------------------------------------------
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


def righting_time_index(m: Morphology, b: float = 1.0 / 3.0, k: float = 0.35, neck_lever: float = 1.0) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever


def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """Map righting‑time index to a normalized priority in [0,1]."""
    return max(0.0, min(1.0, righting_time_index(m) / max_index))


# ----------------------------------------------------------------------
# MinHash utilities (Parent B)
# ----------------------------------------------------------------------
MAX64 = (1 << 64) - 1


def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big") & MAX64


def minhash_signature(tokens: Iterable[str], k: int = 128) -> np.ndarray:
    """
    Fixed‑length MinHash signature of length *k*.
    For each hash function i∈[0,k) the minimum hash over all tokens is taken.
    """
    token_set = set(tokens)
    if not token_set:
        # all‑zero signature for empty input
        return np.zeros(k, dtype=np.uint64)
    sig = np.empty(k, dtype=np.uint64)
    for i in range(k):
        min_hash = min(_hash(i, t) for t in token_set)
        sig[i] = min_hash
    return sig


def signature_complexity(sig: np.ndarray) -> float:
    """
    Normalized distinct‑hash ratio H(s) = |unique(sig)| / k.
    """
    if sig.size == 0:
        return 0.0
    return len(np.unique(sig)) / sig.size


def rlct(sig: np.ndarray) -> float:
    """Real‑Log‑Canonical‑Threshold λ = 1 / (1 + H(s))."""
    h = signature_complexity(sig)
    return 1.0 / (1.0 + h)


# ----------------------------------------------------------------------
# Risk handling (Parent B)
# ----------------------------------------------------------------------
def candidate_risk_vector(audit_findings: List[int]) -> np.ndarray:
    """Convert integer audit findings into a risk vector (values assumed 0‑1)."""
    if not audit_findings:
        return np.zeros(1)
    return np.array(audit_findings, dtype=np.float64)


def effective_learning_rate(mu_base: float, sig: np.ndarray, risk_vec: np.ndarray) -> float:
    """
    μ_eff = μ_base · λ · (1 – mean(risk_vec))
    where λ = RLCT(sig).
    """
    if mu_base < 0:
        raise ValueError("base learning rate must be non‑negative")
    lam = rlct(sig)
    risk_factor = 1.0 - float(np.mean(risk_vec))
    return mu_base * lam * risk_factor


# ----------------------------------------------------------------------
# Core hybrid operations
# ----------------------------------------------------------------------
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Standard cosine similarity, safe for zero vectors."""
    if a.shape != b.shape:
        raise ValueError("vectors must have the same shape")
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def hybrid_affinity(
    query_tokens: Iterable[str],
    query_morph: Morphology,
    cand_tokens: Iterable[str],
    cand_morph: Morphology,
    k: int = 128,
) -> float:
    """
    Compute hybrid affinity between a query document and a candidate:

        1. Build MinHash signatures for both token sets.
        2. Compute cosine similarity c between the signatures.
        3. Scale by the candidate’s recovery priority p(cand_morph).

    Returns h = c * p(cand_morph) ∈ [0,1].
    """
    sig_q = minhash_signature(query_tokens, k).astype(np.float64)
    sig_c = minhash_signature(cand_tokens, k).astype(np.float64)
    c = cosine_similarity(sig_q, sig_c)
    p = recovery_priority(cand_morph)
    return c * p


def nlms_update(
    weights: np.ndarray,
    input_vec: np.ndarray,
    desired: float,
    mu_eff: float,
) -> np.ndarray:
    """
    Normalized Least‑Mean‑Squares weight update with effective learning rate μ_eff.

    w_{new} = w + (μ_eff * e * x) / (‖x‖² + ε)
    where e = desired – w·x.
    """
    if weights.shape != input_vec.shape:
        raise ValueError("weights and input vector must share shape")
    eps = 1e-12
    y = float(np.dot(weights, input_vec))
    error = desired - y
    norm_sq = float(np.dot(input_vec, input_vec)) + eps
    adaptation = (mu_eff * error / norm_sq) * input_vec
    return weights + adaptation


def hybrid_training_step(
    weights: np.ndarray,
    query_tokens: Iterable[str],
    cand_tokens: Iterable[str],
    cand_morph: Morphology,
    audit_findings: List[int],
    mu_base: float = 0.5,
    k: int = 128,
    desired_similarity: float = 1.0,
) -> Tuple[np.ndarray, float]:
    """
    One training iteration that:

    1. Forms the candidate’s MinHash signature.
    2. Computes the effective learning rate using RLCT and risk.
    3. Updates NLMS weights with the signature as input.
    4. Returns the new weights and the hybrid affinity value.
    """
    # Signature for the candidate (used both as input vector and for μ_eff)
    sig_c = minhash_signature(cand_tokens, k).astype(np.float64)

    # Risk handling
    risk_vec = candidate_risk_vector(audit_findings)

    # Effective learning rate
    mu_eff = effective_learning_rate(mu_base, sig_c.astype(np.uint64), risk_vec)

    # NLMS weight update
    new_weights = nlms_update(weights, sig_c, desired_similarity, mu_eff)

    # Hybrid affinity (could be used as a loss term)
    affinity = hybrid_affinity(query_tokens, Morphology(0, 0, 0, 0), cand_tokens, cand_morph, k)

    return new_weights, affinity


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple synthetic documents
    doc_q = "the quick brown fox jumps over the lazy dog".split()
    doc_c = "quick brown dogs leap over lazy foxes in the park".split()

    # Morphologies (arbitrary but plausible)
    morph_q = Morphology(length=10.0, width=5.0, height=2.0, mass=3.0)
    morph_c = Morphology(length=8.0, width=4.0, height=3.0, mass=2.5)

    # Random audit findings (0 = low risk, 1 = high risk)
    audit = [random.choice([0, 1]) for _ in range(5)]

    # Initial NLMS weights (zero vector)
    k = 128
    w = np.zeros(k, dtype=np.float64)

    # Run a single hybrid training step
    w_new, aff = hybrid_training_step(
        weights=w,
        query_tokens=doc_q,
        cand_tokens=doc_c,
        cand_morph=morph_c,
        audit_findings=audit,
        mu_base=0.3,
        k=k,
        desired_similarity=1.0,
    )

    print(f"Hybrid affinity h = {aff:.4f}")
    print(f"Effective learning rate μ_eff = {effective_learning_rate(0.3, minhash_signature(doc_c, k).astype(np.uint64), candidate_risk_vector(audit)):.6f}")
    print(f"Weight norm before: {np.linalg.norm(w):.4f}, after: {np.linalg.norm(w_new):.4f}")
    sys.exit(0)