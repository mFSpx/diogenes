# DARWIN HAMMER — match 3882, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_ternar_hybrid_hybrid_hybrid_m2737_s6.py (gen5)
# parent_b: hybrid_hybrid_decision_hygi_ternary_lens_audit_m19_s2.py (gen2)
# born: 2026-05-29T23:52:15Z

"""Hybrid Fusion Module

Parents:
* **hybrid_hybrid_hybrid_ternar_hybrid_hybrid_hybrid_m2737_s6.py** – provides
  - a stable sigmoid,
  - an SSIM‑like similarity on low‑dimensional feature vectors,
  - a regex‑based 2‑dimensional feature extractor,
  - a simple endpoint circuit‑breaker.
* **hybrid_hybrid_decision_hygi_ternary_lens_audit_m19_s2.py** – provides
  - a 9‑dimensional regex count feature extractor,
  - a hygiene score `s = w⁺·v − w⁻·v`,
  - a Shannon‑entropy term `H`,
  - a hybrid candidate metric `Sₕ = s·(1 + H/Hₘₐₓ)`.

Mathematical Bridge
-------------------
Both parents operate on feature vectors derived from the same textual input.
Parent A yields a *normalized* 2‑dimensional vector `f₂ ∈ ℝ²`,
while Parent B yields a *count* 9‑dimensional vector `v₉ ∈ ℕ⁹`.

We concatenate them into a single 11‑dimensional state  

  `x = [f₂ , v₉]ᵀ ∈ ℝ² × ℕ⁹`.

The hybrid algorithm then

1. **Computes the hygiene component** on `v₉` exactly as Parent B.
2. **Applies the sigmoid gate** on the normalized part `f₂` exactly as Parent A.
3. **Merges the two** by multiplying the gated sigmoid with the hygiene‑entropy
   metric, yielding a unified score  

  `S = σ( w_g·f₂ + b ) · ( s·(1 + H/Hₘₐₓ) )`.

4. **Routes decisions** using the SSIM‑like similarity on `f₂` together with a
   cosine similarity on `v₉`, providing a richer similarity measure for
   downstream routing.

The circuit‑breaker guards any external call that could raise an exception
(e.g. loading a manifest) and demonstrates integration of the failure‑count
logic from Parent A.

The module therefore fuses the governing equations of both parents into a
single, mathematically coherent system.
"""

import math
import random
import re
import sys
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Regex feature extraction (shared)
# ----------------------------------------------------------------------
EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)
PLANNING_RE = re.compile(
    r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
    re.I,
)

# Additional patterns to reach nine count‑features (Parent B)
DELAY_RE = re.compile(r"\b(?:pause|sleep|wait|later|hold|cool\s?down|de[- ]?escalate|not\s?now)\b", re.I)
RISK_RE = re.compile(r"\b(?:risk|threat|vulnerability|exposure|danger)\b", re.I)
COMPLIANCE_RE = re.compile(r"\b(?:compliance|regulation|policy|standard|governance)\b", re.I)
AUDIT_RE = re.compile(r"\b(?:audit|review|inspection|assessment)\b", re.I)
VERIFICATION_RE = re.compile(r"\b(?:verification|validate|validation|certify|certification)\b", re.I)
SCHEDULE_RE = re.compile(r"\b(?:schedule|timeline|deadline|milestone)\b", re.I)
PRIORITY_RE = re.compile(r"\b(?:high\s?priority|critical|urgent|important)\b", re.I)

COUNT_REGEXES: List[Tuple[str, re.Pattern]] = [
    ("evidence", EVIDENCE_RE),
    ("planning", PLANNING_RE),
    ("delay", DELAY_RE),
    ("risk", RISK_RE),
    ("compliance", COMPLIANCE_RE),
    ("audit", AUDIT_RE),
    ("verification", VERIFICATION_RE),
    ("schedule", SCHEDULE_RE),
    ("priority", PRIORITY_RE),
]

# ----------------------------------------------------------------------
# Core mathematical primitives (Parent A)
# ----------------------------------------------------------------------
def sigmoid(z: np.ndarray) -> np.ndarray:
    """Element‑wise sigmoid, numerically stable."""
    z = np.clip(z, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


def ssim_like(a: np.ndarray, b: np.ndarray) -> float:
    """Very small SSIM‑style similarity used for routing."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    mu_a = a.mean()
    mu_b = b.mean()
    sigma_a = a.var()
    sigma_b = b.var()
    sigma_ab = ((a - mu_a) * (b - mu_b)).mean()
    num = (2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2)
    den = (mu_a ** 2 + mu_b ** 2 + C1) * (sigma_a + sigma_b + C2)
    return float(num / den)


def extract_regex_features(text: str) -> np.ndarray:
    """2‑dimensional normalized feature vector (Parent A)."""
    length = max(len(text), 1)
    ev = len(EVIDENCE_RE.findall(text)) / length
    pl = len(PLANNING_RE.findall(text)) / length
    return np.array([ev, pl], dtype=np.float64)


def extract_count_features(text: str) -> np.ndarray:
    """9‑dimensional integer count vector (Parent B)."""
    counts = [len(regex.findall(text)) for _, regex in COUNT_REGEXES]
    return np.array(counts, dtype=np.int64)


# ----------------------------------------------------------------------
# Decision‑Hygiene (Parent B)
# ----------------------------------------------------------------------
# Fixed positive / negative weight vectors for reproducibility
W_POS = np.array([2.0, 1.5, 1.2, 1.0, 0.8, 1.3, 1.1, 0.9, 1.4], dtype=np.float64)
W_NEG = np.array([0.5, 0.7, 0.6, 0.4, 0.3, 0.5, 0.4, 0.2, 0.6], dtype=np.float64)
H_MAX = math.log2(len(W_POS))  # log2(9)


def hygiene_score(v: np.ndarray) -> Tuple[float, float]:
    """
    Compute the hygiene component.

    Returns
    -------
    s : float
        Weighted hygiene score (w⁺·v − w⁻·v).
    H : float
        Shannon entropy of the normalized count vector.
    """
    s = float(W_POS.dot(v) - W_NEG.dot(v))
    total = float(v.sum())
    if total == 0.0:
        H = 0.0
    else:
        p = v / total
        # avoid log(0) by masking zero entries
        mask = p > 0
        H = -float(np.sum(p[mask] * np.log2(p[mask])))
    return s, H


def hybrid_candidate_metric(v: np.ndarray) -> float:
    """
    Compute the Parent B hybrid candidate metric Sₕ = s·(1+H/Hₘₐₓ).
    """
    s, H = hygiene_score(v)
    return s * (1.0 + H / H_MAX)


# ----------------------------------------------------------------------
# Circuit Breaker (Parent A)
# ----------------------------------------------------------------------
class EndpointCircuitBreaker:
    """Simple failure‑count circuit breaker."""

    def __init__(self, failure_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False

    def record_success(self) -> None:
        self.failures = 0
        self.open = False

    def record_failure(self) -> None:
        self.failures += 1
        self.open = self.failures >= self.failure_threshold

    def allow(self) -> bool:
        """Return True if calls are allowed (circuit closed)."""
        return not self.open


# ----------------------------------------------------------------------
# Hybrid Operations
# ----------------------------------------------------------------------
def extract_combined_features(text: str) -> np.ndarray:
    """
    Return the concatenated feature vector x = [f₂ , v₉]ᵀ.

    The first two entries are the normalized regex frequencies (Parent A);
    the remaining nine are raw counts (Parent B).
    """
    f2 = extract_regex_features(text)          # shape (2,)
    v9 = extract_count_features(text).astype(np.float64)  # cast for later math
    return np.concatenate([f2, v9])  # shape (11,)


def unified_hybrid_score(text: str, gate_weights: np.ndarray, gate_bias: float) -> float:
    """
    Compute the fused hybrid score S for a given document.

    Parameters
    ----------
    text : str
        Input document.
    gate_weights : np.ndarray, shape (2,)
        Linear weights applied to the normalized 2‑dimensional part before sigmoid.
    gate_bias : float
        Bias term for the sigmoid gate.

    Returns
    -------
    S : float
        Final hybrid score = sigmoid(gate) * Sₕ.
    """
    # Extract features
    combined = extract_combined_features(text)
    f2 = combined[:2]          # normalized part
    v9 = combined[2:].astype(np.int64)  # count part

    # Gate via sigmoid (Parent A)
    gate_input = float(gate_weights.dot(f2) + gate_bias)
    gate = float(sigmoid(np.array([gate_input]))[0])

    # Hygiene‑entropy metric (Parent B)
    Sh = hybrid_candidate_metric(v9)

    # Unified score
    return gate * Sh


def similarity_metric(text_a: str, text_b: str) -> float:
    """
    Compute a composite similarity between two documents.

    - SSIM‑like similarity on the normalized 2‑dimensional features.
    - Cosine similarity on the 9‑dimensional count vectors.
    The two components are averaged.

    Returns
    -------
    sim : float
        Value in [0, 1] (higher means more similar).
    """
    # Feature extraction
    a = extract_combined_features(text_a)
    b = extract_combined_features(text_b)

    # SSIM‑like on the first two dimensions
    ssim = ssim_like(a[:2], b[:2])
    ssim = max(0.0, min(1.0, ssim))  # clamp for safety

    # Cosine similarity on the count part
    v_a = a[2:]
    v_b = b[2:]
    norm_a = np.linalg.norm(v_a)
    norm_b = np.linalg.norm(v_b)
    if norm_a == 0.0 or norm_b == 0.0:
        cosine = 0.0
    else:
        cosine = float(np.dot(v_a, v_b) / (norm_a * norm_b))
        cosine = max(0.0, min(1.0, cosine))

    # Simple average as the fused similarity
    return (ssim + cosine) / 2.0


def guarded_audit(texts: List[str]) -> List[Tuple[str, float]]:
    """
    Demonstrate the circuit‑breaker in a batch‑processing scenario.
    For each text we attempt to compute the unified hybrid score.
    If the breaker is open, processing stops and remaining items are marked
    as failed.

    Returns
    -------
    results : list of (text, score) tuples; score is ``float('nan')`` on failure.
    """
    breaker = EndpointCircuitBreaker(failure_threshold=2)
    results: List[Tuple[str, float]] = []

    for txt in texts:
        if not breaker.allow():
            results.append((txt, float('nan')))
            continue
        try:
            # Randomly inject an artificial failure to showcase the breaker
            if random.random() < 0.1:
                raise RuntimeError("Synthetic processing error")
            score = unified_hybrid_score(txt, gate_weights=np.array([1.2, -0.8]), gate_bias=0.1)
            breaker.record_success()
            results.append((txt, score))
        except Exception:
            breaker.record_failure()
            results.append((txt, float('nan')))

    return results


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_texts = [
        "The audit confirmed evidence of compliance and a high priority plan.",
        "We need to wait, schedule the next review, and verify the risk assessment.",
        "No relevant evidence or planning was found; the document is empty.",
    ]

    print("=== Unified Hybrid Scores ===")
    for txt in sample_texts:
        score = unified_hybrid_score(txt, gate_weights=np.array([1.0, -0.5]), gate_bias=0.0)
        print(f"Score: {score:.4f}  |  Text: {txt}")

    print("\n=== Pairwise Similarities ===")
    for i in range(len(sample_texts)):
        for j in range(i + 1, len(sample_texts)):
            sim = similarity_metric(sample_texts[i], sample_texts[j])
            print(f"Sim({i},{j}) = {sim:.4f}")

    print("\n=== Guarded Batch Audit ===")
    batch_results = guarded_audit(sample_texts * 5)  # repeat to increase chance of failure
    for idx, (txt, sc) in enumerate(batch_results):
        status = "FAIL" if math.isnan(sc) else f"{sc:.4f}"
        print(f"{idx:02d}: {status}")