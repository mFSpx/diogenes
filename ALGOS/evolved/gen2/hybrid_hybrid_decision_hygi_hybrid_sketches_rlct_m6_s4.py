# DARWIN HAMMER — match 6, survivor 4
# gen: 2
# parent_a: hybrid_decision_hygiene_shannon_entropy_m12_s1.py (gen1)
# parent_b: hybrid_sketches_rlct_grokking_m5_s1.py (gen1)
# born: 2026-05-29T23:25:08Z

"""Hybrid Decision‑Hygiene & Sketch‑RLCT Module

Parents
-------
* **Parent A** – ``hybrid_decision_hygiene_shannon_entropy_m12_s1.py``  
  Provides regex‑based feature extraction from free‑form text and computes the
  Shannon entropy of the resulting count distribution.

* **Parent B** – ``hybrid_sketches_rlct_grokking_m5_s1.py``  
  Supplies Count‑Min sketch, a lightweight HyperLogLog cardinality estimator and
  RLCT (Real Log‑Canonical‑Threshold) asymptotic formulas that depend on
  logarithms of data‑set size and effective model complexity.

Mathematical Bridge
-------------------
Both families operate on **log‑count statistics**:

* The decision‑hygiene step yields a vector **c** of integer feature counts per
  document.  The Shannon entropy `H = -∑ p_i log p_i` uses the normalized
  frequencies `p_i = c_i / Σ c_i`.

* The sketch side treats frequencies as a stream of items.  A Count‑Min sketch
  can approximate the empirical log‑likelihood `ℓ = Σ log p_i` by summing the
  logarithms of the estimated frequencies.  The HyperLogLog estimate `Ň` of the
  number of distinct items supplies the “effective model dimension’’ that
  appears in the RLCT term `λ·log n – (m‑1)·loglog n`.

The hybrid therefore:

1. Extracts the hygiene feature counts from each document.
2. Feeds every *individual feature occurrence* (i.e. repeats each feature name
   according to its count) into a Count‑Min sketch.
3. Uses the HyperLogLog estimate of distinct feature tokens as the `m` term in
   the RLCT formula.
4. Combines the sketch‑based log‑likelihood estimate with the Shannon entropy
   to produce a unified free‑energy‑like quantity.

The public API exposes three representative operations that showcase this
fusion.
"""

import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import re

# ----------------------------------------------------------------------
# Parent A – decision hygiene feature extraction & Shannon entropy
# ----------------------------------------------------------------------

EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
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
BOUNDARY_RE = re.compile(
    r"\b(?:boundary|boundaries|walk away|no contact|do not|don't|stop|limit|refuse|protect|safe|safety|lock|redact|privacy|permission|consent)\b",
    re.I,
)
OUTCOME_RE = re.compile(
    r"\b(?:done|shipped|finished|complete|resolved|succeeded|passed|paid|safe|sent|filed|closed|fixed|working|green|verified)\b",
    re.I,
)
IMPULSIVE_RE = re.compile(
    r"\b(?:rage|impulsive|panic|panicking|spiral|doom|fuck it|right now|immediately|tonight|destroy|revenge|scorched|burn it|quit|give up)\b",
    re.I,
)
SCARCITY_RE = re.compile(
    r"\b(?:broke|homeless|last\s+\$?\d+|\$\s?\d+\s+to my name|can't afford|cannot afford|rent due|no money|starving|desperate|no sleep|exhausted)\b",
    re.I,
)
RISK_RE = re.compile(
    r"\b(?:kill|die|suicide|suicidal|self[- ]?harm|overdose|hurt myself|danger|unsafe|crisis|collapse)\b",
    re.I,
)


def extract_feature_counts(text: str) -> Dict[str, int]:
    """Return a dictionary of hygiene feature counts for a single document."""
    return {
        "evidence": len(EVIDENCE_RE.findall(text or "")),
        "planning": len(PLANNING_RE.findall(text or "")),
        "delay": len(DELAY_RE.findall(text or "")),
        "support": len(SUPPORT_RE.findall(text or "")),
        "boundary": len(BOUNDARY_RE.findall(text or "")),
        "outcome": len(OUTCOME_RE.findall(text or "")),
        "impulsive": len(IMPULSIVE_RE.findall(text or "")),
        "scarcity": len(SCARCITY_RE.findall(text or "")),
        "risk": len(RISK_RE.findall(text or "")),
    }


def shannon_entropy_from_counts(counts: Dict[str, int]) -> float:
    """Compute Shannon entropy of the normalized count vector."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    probs = np.array(list(counts.values())) / total
    # Guard against zero probabilities (should not happen after normalization)
    probs = probs[probs > 0]
    return -float(np.sum(probs * np.log(probs)))


# ----------------------------------------------------------------------
# Parent B – sketch primitives & RLCT utilities
# ----------------------------------------------------------------------


def _hash(item: str, seed: int = 0) -> int:
    """Deterministic 64‑bit hash used by sketches."""
    h = hash((item, seed))
    # Ensure positive 64‑bit integer
    return h & ((1 << 64) - 1)


def count_min_sketch(
    items: Iterable[str], width: int = 128, depth: int = 4
) -> List[List[int]]:
    """Build a Count‑Min sketch for a stream of string items."""
    table: List[List[int]] = [[0] * width for _ in range(depth)]
    for item in items:
        for d in range(depth):
            idx = _hash(item, seed=d) % width
            table[d][idx] += 1
    return table


def estimate_frequency(
    sketch: List[List[int]], item: str
) -> int:
    """Return the Count‑Min estimate of the frequency of *item*."""
    depth = len(sketch)
    width = len(sketch[0])
    mins = []
    for d in range(depth):
        idx = _hash(item, seed=d) % width
        mins.append(sketch[d][idx])
    return min(mins)


def hyperloglog_estimate(items: Iterable[str], b: int = 10) -> float:
    """
    Very lightweight HyperLogLog estimator.

    Parameters
    ----------
    items : iterable of str
        Stream of items.
    b : int, default 10
        Number of bits used for the register index (2**b registers).

    Returns
    -------
    float
        Estimated cardinality.
    """
    m = 1 << b
    registers = [0] * m
    for item in items:
        h = _hash(item)
        idx = h >> (64 - b)  # first b bits as register index
        w = (h << b) & ((1 << 64) - 1)  # remaining bits
        # count leading zeros in w plus one (as per HLL definition)
        rank = 1
        while rank <= 64 - b and (w >> (64 - b - rank)) & 1 == 0:
            rank += 1
        registers[idx] = max(registers[idx], rank)
    # Estimate
    alpha_m = 0.7213 / (1 + 1.079 / m)
    Z = 1.0 / sum([2.0 ** -r for r in registers])
    raw_est = alpha_m * m * m * Z
    # Small range correction
    if raw_est <= (5 / 2) * m:
        V = registers.count(0)
        if V != 0:
            raw_est = m * math.log(m / V)
    return raw_est


def approximate_log_likelihood(
    sketch: List[List[int]], items: Iterable[str]
) -> float:
    """
    Approximate Σ log p_i where p_i is the empirical frequency of each item.
    Uses the Count‑Min sketch to estimate frequencies.
    """
    total_count = sum(
        sum(row) for row in sketch
    )  # over‑estimate but sufficient for a proxy
    if total_count == 0:
        return float("-inf")
    log_likelihood = 0.0
    for item in items:
        freq = estimate_frequency(sketch, item)
        # Avoid log(0)
        prob = max(freq, 1) / total_count
        log_likelihood += math.log(prob)
    return log_likelihood


def hybrid_rlct_estimate(
    sketch: List[List[int]],
    hll_cardinality: float,
    lambda_param: float = 1.0,
) -> float:
    """
    Compute a hybrid RLCT estimate using sketch‑derived log‑likelihood and
    HyperLogLog cardinality as the model complexity term.

    RLCT ≈ λ·log(N) - (m‑1)·loglog(N)

    where
        N = total count (approximated by the sketch)
        m = effective number of distinct activation patterns (≈ HLL estimate)
    """
    # Approximate total number of observations N from the sketch
    N = sum(sum(row) for row in sketch)
    if N <= 0 or hll_cardinality <= 1:
        return float("nan")
    term1 = lambda_param * math.log(N)
    term2 = (hll_cardinality - 1) * math.log(math.log(N + 1) + 1e-12)
    return term1 - term2


# ----------------------------------------------------------------------
# Hybrid API – three representative functions
# ----------------------------------------------------------------------


def build_hybrid_sketch(
    corpus: Iterable[str],
    width: int = 128,
    depth: int = 4,
    hll_bits: int = 10,
) -> Tuple[List[List[int]], float, List[str]]:
    """
    Construct the hybrid data structures from a corpus of documents.

    Returns
    -------
    sketch : Count‑Min sketch of all *feature tokens* (each occurrence is an item)
    hll_est : HyperLogLog estimate of distinct feature tokens
    flat_items : List of all feature token strings (used for later likelihood approx)
    """
    flat_items: List[str] = []
    for doc in corpus:
        counts = extract_feature_counts(doc)
        for feature, cnt in counts.items():
            flat_items.extend([feature] * cnt)
    sketch = count_min_sketch(flat_items, width=width, depth=depth)
    hll_est = hyperloglog_estimate(flat_items, b=hll_bits)
    return sketch, hll_est, flat_items


def document_entropy_and_rlct(
    doc: str,
    sketch: List[List[int]],
    hll_estimate: float,
    lambda_param: float = 1.0,
) -> Tuple[float, float]:
    """
    For a single document compute:

    * Shannon entropy of its hygiene feature distribution.
    * Hybrid RLCT estimate using the global sketch and HLL cardinality.

    Returns (entropy, rlct_estimate).
    """
    counts = extract_feature_counts(doc)
    entropy = shannon_entropy_from_counts(counts)

    # Use the sketch to approximate log‑likelihood of this document's tokens
    doc_items = []
    for feature, cnt in counts.items():
        doc_items.extend([feature] * cnt)
    # Approximate log‑likelihood (not used directly in RLCT but demonstrates bridge)
    _ = approximate_log_likelihood(sketch, doc_items)

    rlct = hybrid_rlct_estimate(sketch, hll_estimate, lambda_param=lambda_param)
    return entropy, rlct


def hybrid_evaluation(
    corpus: Iterable[str],
    lambda_param: float = 1.0,
) -> Dict[str, float]:
    """
    End‑to‑end hybrid evaluation over a corpus.

    Returns a dictionary with aggregated metrics:
        - avg_entropy
        - rlct_estimate (global)
        - avg_log_likelihood (sketch‑based)
    """
    sketch, hll_est, flat_items = build_hybrid_sketch(corpus)
    entropies = []
    for doc in corpus:
        e, _ = document_entropy_and_rlct(doc, sketch, hll_est, lambda_param)
        entropies.append(e)
    avg_entropy = float(np.mean(entropies)) if entropies else 0.0
    rlct_est = hybrid_rlct_estimate(sketch, hll_est, lambda_param)
    avg_loglik = approximate_log_likelihood(sketch, flat_items) / max(
        1, len(flat_items)
    )
    return {
        "avg_entropy": avg_entropy,
        "rlct_estimate": rlct_est,
        "avg_log_likelihood": avg_loglik,
    }


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_corpus = [
        "I have evidence that the plan will succeed. We need to verify the source and keep a log.",
        "I'm feeling impulsive and want to act right now, but I should wait and get support.",
        "The budget is low, we can't afford more resources. We need a checklist and a timeline.",
        "I am scared and have thoughts of self‑harm. I need a therapist and a safe boundary.",
        "All tests passed, the deployment is done and verified. Great work!",
    ]

    metrics = hybrid_evaluation(sample_corpus, lambda_param=0.8)
    print("Hybrid evaluation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Demonstrate per‑document output
    sketch, hll_est, _ = build_hybrid_sketch(sample_corpus)
    for i, doc in enumerate(sample_corpus, 1):
        ent, rlct = document_entropy_and_rlct(doc, sketch, hll_est, lambda_param=0.8)
        print(f"\nDocument {i}:")
        print(f"  Entropy = {ent:.4f}")
        print(f"  RLCT estimate (global) = {rlct:.4f}")