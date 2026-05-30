# DARWIN HAMMER — match 5128, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_decision_hygi_hybrid_rete_bandit_g_m544_s2.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_minimu_m1767_s0.py (gen6)
# born: 2026-05-29T23:59:59Z

"""Hybrid Decision‑Hygiene & Sheaf‑Based Certainty Allocator

Parents:
- hybrid_hybrid_decision_hygi_hybrid_rete_bandit_g_m544_s2.py (Feature extraction + deterministic work‑share)
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_minimu_m1767_s0.py (MinHash + sheaf coboundary + epistemic certainty)

Mathematical bridge:
1. Text → feature counts → weighted score **S** (Parent A).
2. Same text → MinHash vector **μ** → sheaf coboundary vector **δ** → normalized similarity  
   **σ = (μ·δ) / (‖μ‖‖δ‖)** (Parent B).
3. Combine **S**, **σ**, and a day‑of‑week factor **d∈[0,1]** into a deterministic‑lane target  

   **p = clip(p₀ + α·S + β·σ + γ·d, 0, 100)**  

   where *clip* bounds to [0,100] and α,β,γ are tunable scalars.
4. The deterministic share receives **p %** of the total work budget; the remainder is
   distributed to LLM‑based lanes proportionally to **σ** (higher similarity → larger share).

The module implements this fusion with three public functions:
`extract_features`, `compute_deterministic_pct`, and `hybrid_allocate`. """

from __future__ import annotations

import math
import random
import re
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – regex feature extraction and weighted scoring
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
    r"\b(?:ask|call|text|friend|frie|help|assist|support|consult|mentor)\b",
    re.I,
)

POS_WEIGHTS = {"evidence": 2.0, "planning": 1.5, "support": 1.0}
NEG_WEIGHTS = {"delay": -1.2, "support": -0.5}  # example negative influence


def extract_features(text: str) -> Dict[str, int]:
    """Count occurrences of each semantic class using the compiled regexes."""
    counts = {
        "evidence": len(EVIDENCE_RE.findall(text)),
        "planning": len(PLANNING_RE.findall(text)),
        "delay": len(DELAY_RE.findall(text)),
        "support": len(SUPPORT_RE.findall(text)),
    }
    return counts


def compute_weighted_score(counts: Dict[str, int]) -> float:
    """Scalar S = Σ w⁺·c⁺ − Σ w⁻·c⁻."""
    pos = sum(POS_WEIGHTS.get(k, 0.0) * v for k, v in counts.items())
    neg = sum(NEG_WEIGHTS.get(k, 0.0) * v for k, v in counts.items())
    return pos + neg  # neg weights already negative


# ----------------------------------------------------------------------
# Parent B – MinHash → Sheaf coboundary → similarity
# ----------------------------------------------------------------------
def minhash_for_text(text: str, k: int = 64) -> List[int]:
    """Very‑lightweight MinHash: keep the minimum hash per bucket."""
    cleaned = text.replace(" ", "").strip().lower()
    shingles = [cleaned[i : i + 5] for i in range(len(cleaned) - 4)]
    signature = [sys.maxsize] * k
    for s in shingles:
        h = hash(s) % k
        signature[h] = min(signature[h], hash(s) % 1_000_000)
    return signature


class Sheaf:
    """Placeholder sheaf with random coboundary operator."""

    def __init__(self, node_dims: Dict[str, int], edges: List[Tuple[str, str]]):
        self.node_dims = node_dims
        self.edges = edges

    def coboundary_operator(self) -> np.ndarray:
        # Random vector length equals the dimensionality of the signature
        dim = sum(self.node_dims.values())
        rng = np.random.default_rng(42)
        return rng.random(dim)


def sheaf_similarity(minhash: List[int], node_dims: Dict[str, int], edges: List[Tuple[str, str]]) -> float:
    """σ = (μ·δ) / (‖μ‖‖δ‖) where δ is the coboundary vector."""
    mu = np.array(minhash, dtype=float)
    delta = Sheaf(node_dims, edges).coboundary_operator()
    # Align lengths (pad the shorter vector with zeros)
    if delta.shape[0] < mu.shape[0]:
        delta = np.pad(delta, (0, mu.shape[0] - delta.shape[0]))
    elif mu.shape[0] < delta.shape[0]:
        mu = np.pad(mu, (0, delta.shape[0] - mu.shape[0]))
    dot = float(np.dot(mu, delta))
    norm = float(np.linalg.norm(mu) * np.linalg.norm(delta))
    return 0.0 if norm == 0 else dot / norm


def confidence_to_probability(confidence_bps: int) -> float:
    """Map basis‑points confidence to a probability in [0,1]."""
    return max(0.0, min(1.0, confidence_bps / 10_000.0))


# ----------------------------------------------------------------------
# Fusion core – deterministic share computation
# ----------------------------------------------------------------------
def day_of_week_factor(ref_date: date | None = None) -> float:
    """Map weekday (Mon=0 … Sun=6) linearly to [0,1]."""
    if ref_date is None:
        ref_date = datetime.utcnow().date()
    return ref_date.weekday() / 6.0


def compute_deterministic_pct(
    score: float,
    similarity: float,
    day_factor: float,
    base_pct: float = 30.0,
    alpha: float = 2.5,
    beta: float = 15.0,
    gamma: float = 10.0,
) -> float:
    """
    p = clip(base + α·S + β·σ + γ·d, 0, 100)

    - `score`      : weighted feature score S (Parent A)
    - `similarity` : sheaf‑based similarity σ (Parent B)
    - `day_factor` : d ∈ [0,1] from the weekday
    """
    raw = base_pct + alpha * score + beta * similarity + gamma * day_factor
    return max(0.0, min(100.0, raw))


def hybrid_allocate(text: str, total_units: int = 100) -> Dict[str, int]:
    """
    Full pipeline:
    1. Extract semantic counts → weighted score S.
    2. MinHash → sheaf similarity σ.
    3. Day factor d.
    4. Deterministic share p % → deterministic_units.
    5. Remaining units split between two LLM lanes proportionally to σ.
    Returns a dict mapping lane names to allocated integer units.
    """
    # 1. Feature extraction & scoring
    counts = extract_features(text)
    S = compute_weighted_score(counts)

    # 2. MinHash + sheaf similarity
    mh = minhash_for_text(text)
    # Dummy sheaf topology (could be derived from counts in a real system)
    node_dims = {"evidence": 1, "planning": 1, "delay": 1, "support": 1}
    edges = [("evidence", "planning"), ("planning", "support"), ("support", "delay")]
    sigma = sheaf_similarity(mh, node_dims, edges)

    # 3. Day factor
    d = day_of_week_factor()

    # 4. Deterministic percentage
    p = compute_deterministic_pct(S, sigma, d)

    deterministic_units = int(round(total_units * p / 100.0))
    remaining = total_units - deterministic_units

    # 5. LLM lane split (two lanes for illustration)
    # Use σ to bias the split; ensure non‑zero denominator.
    bias = sigma if sigma > 0 else 0.5
    llm_a = int(round(remaining * bias / (bias + (1 - bias))))
    llm_b = remaining - llm_a

    return {
        "deterministic": deterministic_units,
        "llm_a": llm_a,
        "llm_b": llm_b,
        "score_S": int(S),
        "similarity_sigma": round(sigma, 4),
        "day_factor": round(d, 4),
        "deterministic_pct": round(p, 2),
    }


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_text = (
        "The plan includes a checklist of steps, "
        "and we have verified the source and recorded the evidence. "
        "However, we might need to wait until tomorrow before proceeding."
    )
    allocation = hybrid_allocate(sample_text, total_units=120)
    for k, v in allocation.items():
        print(f"{k}: {v}")