# DARWIN HAMMER — match 4906, survivor 4
# gen: 6
# parent_a: hybrid_hybrid_hybrid_ternar_hybrid_hybrid_ternar_m1557_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_decisi_hybrid_regret_engine_m2_s4.py (gen3)
# born: 2026-05-29T23:58:54Z

"""
Hybrid Algorithm: Fusion of Ternary Routing + MinHash‑Audit (Parent A) with
Decision‑Regret Engine (Parent B)

Mathematical Bridge
-------------------
* Parent A produces a *weighted MinHash signature* `s_w = r * s`,
  where `r` is the audit‑risk scalar and `s` is a MinHash vector.
  The ternary router then selects a configuration  
  `c = argmax_i (s_w · T_i)` using a matrix `T` of ternary patterns.

* Parent B builds a *utility vector* `u = p·c_vec – n·c_vec` from regex‑based
  feature counts, and transforms it into a regret‑weighted probability
  distribution `π` via a soft‑max shifted by `max(u)`.  The Gini coefficient
  `G(π)` quantifies inequality (regret).

The fusion treats the **utility vector `u` as the MinHash signature `s`**.
Thus the audit‑risk scalar `r` weights the decision utilities before routing,
and the resulting ternary configuration `c` is fed back into the regret
framework.  The final hybrid score mixes the circuit‑breaker term
`B = Σ(V·r)` (from Parent A) with the Gini‑based regret term
`G(π)` (from Parent B).

The core equations of the hybrid system are:

    r      = audit_risk(findings)
    s      = minhash_signature(tokens, k)
    s_w    = r * s                     # weighted signature (A)
    c_idx  = argmax_i (s_w · T_i)      # ternary router choice (A)
    c_vec  = one_hot(c_idx, len(T))   # selected ternary pattern
    V      = Σ (c_vec * x)             # Voronoi‑like aggregation (A)
    B      = Σ (V * r)                 # circuit‑breaker term (A)

    u      = p·feat_counts – n·feat_counts   # utility (B)
    π      = softmax_regret(u)                # regret‑weighted distribution (B)
    G      = gini_coefficient(π)              # regret measure (B)

    hybrid_score = α * B + β * G              # α,β are fusion weights
"""

import sys
import math
import random
import hashlib
from pathlib import Path
from typing import List, Iterable, Tuple, Dict

import numpy as np
import re
import datetime as dt

# ----------------------------------------------------------------------
# Parent A – Audit risk, MinHash, Ternary routing
# ----------------------------------------------------------------------


def audit_risk(findings: Iterable[str]) -> float:
    """Compute audit risk as the proportion of findings that contain a keyword."""
    keywords = {"critical", "high", "risk", "issue", "vulnerability"}
    total = sum(1 for _ in findings)
    if total == 0:
        return 0.0
    flagged = sum(1 for f in findings if any(k in f.lower() for k in keywords))
    return flagged / total


def minhash_signature(tokens: Iterable[str], k: int = 64) -> np.ndarray:
    """Simple MinHash: for each of k hash functions keep the minimal hash value."""
    # deterministic pseudo‑random seeds for reproducibility
    seeds = np.arange(k, dtype=np.uint32)
    min_hashes = np.full(k, np.iinfo(np.uint64).max, dtype=np.uint64)
    for token in tokens:
        token_bytes = token.encode("utf-8")
        for i, seed in enumerate(seeds):
            h = int(hashlib.blake2b(token_bytes, person=seed.to_bytes(4, "little")).hexdigest(), 16)
            if h < min_hashes[i]:
                min_hashes[i] = h
    # Normalise to [0,1] for later dot‑product stability
    return min_hashes.astype(np.float64) / np.iinfo(np.uint64).max


class TernaryRouter:
    """Generates a small set of ternary configuration vectors."""

    def __init__(self, num_outputs: int = 8, dim: int = 64):
        self.num_outputs = num_outputs
        self.dim = dim
        self.configs = self._generate_configs()

    def _generate_configs(self) -> np.ndarray:
        """Create a matrix T of shape (num_outputs, dim) with entries in {-1,0,1}."""
        rng = np.random.default_rng(42)
        return rng.integers(-1, 2, size=(self.num_outputs, self.dim), dtype=np.int8)

    def select_configuration(self, weighted_sig: np.ndarray) -> Tuple[int, np.ndarray]:
        """Select index with maximal dot product and return one‑hot vector."""
        scores = self.configs @ weighted_sig
        idx = int(np.argmax(scores))
        one_hot = np.zeros(self.num_outputs, dtype=np.int8)
        one_hot[idx] = 1
        return idx, one_hot


def voronoi_aggregation(one_hot: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Aggregate auxiliary vector x with the selected ternary pattern."""
    return one_hot @ x  # scalar product, returns a single number (wrapped in ndarray)


def circuit_breaker(V: np.ndarray, r: float) -> float:
    """Final circuit‑breaker term."""
    return float(np.sum(V * r))


# ----------------------------------------------------------------------
# Parent B – Feature extraction, utility, regret‑softmax, Gini
# ----------------------------------------------------------------------


EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)
PLANNING_RE = re.compile(
    r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
    re.I,
)
DELAY_RE = re.compile(
    r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|before i|first|after|review)\b",
    re.I,
)
SUPPORT_RE = re.compile(
    r"\b(?:ask|call|text|friend|friends|rowyn|kai|chance|doctor|therapist|lawyer|advocate|support|help|community|team|handoff|delegate)\b",
    re.I,
)


def extract_feature_counts(text: str) -> Dict[str, int]:
    """Count regex‑based features in the supplied text."""
    return {
        "evidence": len(EVIDENCE_RE.findall(text)),
        "planning": len(PLANNING_RE.findall(text)),
        "delay": len(DELAY_RE.findall(text)),
        "support": len(SUPPORT_RE.findall(text)),
    }


def compute_utility(counts: Dict[str, int],
                    pos_weights: Dict[str, float],
                    neg_weights: Dict[str, float]) -> np.ndarray:
    """Utility vector u = p·c – n·c."""
    u = np.array(
        [pos_weights[k] * counts[k] - neg_weights[k] * counts[k] for k in counts],
        dtype=np.float64,
    )
    return u


def regret_softmax(u: np.ndarray) -> np.ndarray:
    """Regret‑weighted softmax: shift by max(u) for numerical stability."""
    shifted = u - np.max(u)
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals)


def gini_coefficient(probs: np.ndarray) -> float:
    """Gini coefficient of a probability distribution."""
    if probs.size == 0:
        return 0.0
    sorted_probs = np.sort(probs)
    n = probs.size
    cumulative = np.cumsum(sorted_probs)
    gini = 1.0 - (2.0 / (n - 1)) * (np.sum(cumulative) / np.sum(sorted_probs) - (n + 1) / 2.0)
    return float(gini)


# ----------------------------------------------------------------------
# Hybrid Functions
# ----------------------------------------------------------------------


def hybrid_signature_from_text(text: str, audit_findings: List[str]) -> Tuple[np.ndarray, float]:
    """Create weighted MinHash signature using audit risk as scalar weight."""
    tokens = re.findall(r"\w+", text.lower())
    s = minhash_signature(tokens)                     # MinHash vector
    r = audit_risk(audit_findings)                   # audit risk scalar
    s_w = r * s                                       # weighted signature
    return s_w, r


def hybrid_routing_and_regret(s_w: np.ndarray, r: float, aux_vector: np.ndarray) -> Tuple[int, float, float]:
    """
    Run ternary routing on the weighted signature, compute circuit breaker,
    and also produce regret‑softmax distribution from the same signature.
    Returns (router_index, B, Gini).
    """
    router = TernaryRouter(num_outputs=aux_vector.shape[0], dim=s_w.shape[0])
    idx, one_hot = router.select_configuration(s_w)
    V = voronoi_aggregation(one_hot, aux_vector)   # scalar aggregation
    B = circuit_breaker(V, r)

    # Treat the raw (un‑weighted) MinHash as a proxy utility for regret step
    u = s_w  # already weighted, but serves as utility vector
    pi = regret_softmax(u)
    G = gini_coefficient(pi)
    return idx, B, G


def hybrid_decision_score(text: str,
                          audit_findings: List[str],
                          aux_vector: np.ndarray,
                          pos_w: Dict[str, float],
                          neg_w: Dict[str, float],
                          alpha: float = 0.7,
                          beta: float = 0.3) -> float:
    """
    End‑to‑end hybrid score:
      1. Extract text features → utility vector u.
      2. Compute audit‑risk weighted MinHash signature.
      3. Run ternary router, obtain circuit‑breaker term B.
      4. Compute Gini of regret‑softmax distribution G.
      5. Fuse: score = α·B + β·G.
    """
    # Step 1: feature‑based utility (Parent B)
    counts = extract_feature_counts(text)
    u_feature = compute_utility(counts, pos_w, neg_w)

    # Step 2: weighted MinHash (Parent A) – we reuse u_feature as token source for diversity
    # For demonstration we combine both token sources.
    tokens = re.findall(r"\w+", text.lower())
    s = minhash_signature(tokens)
    r = audit_risk(audit_findings)
    s_w = r * s

    # Step 3 & 4: routing + regret
    router = TernaryRouter(num_outputs=aux_vector.shape[0], dim=s_w.shape[0])
    _, one_hot = router.select_configuration(s_w)
    V = voronoi_aggregation(one_hot, aux_vector)
    B = circuit_breaker(V, r)

    pi = regret_softmax(u_feature)          # regret distribution based on textual utility
    G = gini_coefficient(pi)

    # Step 5: fusion
    return alpha * B + beta * G


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_text = (
        "The audit confirmed evidence of a critical vulnerability. "
        "We plan a roadmap and schedule the fix. "
        "Support will be delegated to the engineering team."
    )
    audit_findings = [
        "Critical issue found in module X",
        "High risk exposure in API",
        "Low priority note",
    ]

    # Auxiliary vector x for Voronoi aggregation – arbitrary positive values
    aux_vec = np.linspace(1, 8, 8, dtype=np.float64)

    pos_weights = {"evidence": 1.2, "planning": 0.8, "delay": -0.5, "support": 0.6}
    neg_weights = {"evidence": 0.4, "planning": 0.2, "delay": 1.0, "support": 0.1}

    score = hybrid_decision_score(
        text=sample_text,
        audit_findings=audit_findings,
        aux_vector=aux_vec,
        pos_w=pos_weights,
        neg_w=neg_weights,
        alpha=0.6,
        beta=0.4,
    )
    print(f"Hybrid decision score: {score:.4f}")