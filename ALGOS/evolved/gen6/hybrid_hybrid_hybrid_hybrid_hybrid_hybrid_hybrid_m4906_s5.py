# DARWIN HAMMER — match 4906, survivor 5
# gen: 6
# parent_a: hybrid_hybrid_hybrid_ternar_hybrid_hybrid_ternar_m1557_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_decisi_hybrid_regret_engine_m2_s4.py (gen3)
# born: 2026-05-29T23:58:54Z

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
    seeds = np.arange(k, dtype=np.uint32)
    min_hashes = np.full(k, np.iinfo(np.uint64).max, dtype=np.uint64)
    for token in tokens:
        token_bytes = token.encode("utf-8")
        for i, seed in enumerate(seeds):
            h = int(hashlib.blake2b(token_bytes, person=seed.to_bytes(4, "little")).hexdigest(), 16)
            if h < min_hashes[i]:
                min_hashes[i] = h
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
    return one_hot @ x


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

def hybrid_signature_from_text(text: str, audit_findings: List[str],
                                pos_weights: Dict[str, float], neg_weights: Dict[str, float]) -> Tuple[np.ndarray, float, np.ndarray]:
    """Create weighted MinHash signature using audit risk as scalar weight."""
    tokens = re.findall(r"\w+", text.lower())
    s = minhash_signature(tokens)
    r = audit_risk(audit_findings)
    s_w = r * s

    feature_counts = extract_feature_counts(text)
    u = compute_utility(feature_counts, pos_weights, neg_weights)
    return s_w, r, u


def hybrid_routing_and_regret(s_w: np.ndarray, r: float, u: np.ndarray, aux_vector: np.ndarray) -> Tuple[int, float, float]:
    """
    Run ternary routing on the weighted signature, compute circuit breaker,
    and also produce regret‑softmax distribution from the same signature.
    Returns (router_index, B, Gini).
    """
    router = TernaryRouter(num_outputs=aux_vector.shape[0], dim=s_w.shape[0])
    idx, one_hot = router.select_configuration(s_w)
    V = voronoi_aggregation(one_hot, aux_vector)
    B = circuit_breaker(V, r)

    π = regret_softmax(u)
    G = gini_coefficient(π)
    return idx, B, G


def hybrid_score(text: str, audit_findings: List[str], aux_vector: np.ndarray,
                 pos_weights: Dict[str, float], neg_weights: Dict[str, float],
                 alpha: float = 0.5, beta: float = 0.5) -> Tuple[int, float, float, float]:
    s_w, r, u = hybrid_signature_from_text(text, audit_findings, pos_weights, neg_weights)
    idx, B, G = hybrid_routing_and_regret(s_w, r, u, aux_vector)
    hybrid = alpha * B + beta * G
    return idx, B, G, hybrid


if __name__ == "__main__":
    pos_weights = {"evidence": 1.0, "planning": 0.8, "delay": 0.2, "support": 1.2}
    neg_weights = {"evidence": 0.2, "planning": 0.1, "delay": 0.8, "support": 0.1}
    text = "The evidence suggests a delay."
    audit_findings = ["critical", "high"]
    aux_vector = np.random.rand(8)
    idx, B, G, hybrid = hybrid_score(text, audit_findings, aux_vector, pos_weights, neg_weights)
    print(f"idx: {idx}, B: {B}, G: {G}, hybrid: {hybrid}")