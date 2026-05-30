# DARWIN HAMMER — match 5642, survivor 5
# gen: 6
# parent_a: hybrid_hybrid_ternary_lens__hybrid_hybrid_hybrid_m1518_s3.py (gen5)
# parent_b: hybrid_hybrid_hybrid_doomsd_hybrid_model_vram_sc_m2303_s0.py (gen3)
# born: 2026-05-30T00:03:46Z

"""Hybrid algorithm merging Candidate feature extraction (Parent A) with
hyperdimensional encoding and Gini‑based memory budgeting (Parent B).

Mathematical bridge:
- Parent A produces a low‑dimensional real‑valued feature vector **x**.
- Parent B provides a hyperdimensional space where each scalar feature
  can be represented by a random bipolar vector **sᵢ** (symbol_vector).
- We bind each **sᵢ** with the sign of the corresponding feature value
  (binary binding) to obtain **bᵢ = sᵢ ⊗ sign(xᵢ)**.
- The candidate hypervector **h** is the bundle (average) of all **bᵢ**.
- The Gini coefficient of the absolute values of **h** quantifies the
  inequality of memory usage across dimensions; multiplied by the
  dimensionality it yields a memory‑budget estimate.

The three core functions below demonstrate this fused pipeline:
1. `candidate_feature_vector` – builds **x** from audit and regex counts.
2. `encode_candidate_hypervector` – maps **x** → **h** using binding & bundling.
3. `evaluate_candidate` – computes risk score, memory estimate via Gini,
   and annotates the `Candidate` instance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Data structures (from Parent A)
# ----------------------------------------------------------------------


@dataclass
class Candidate:
    """Container for a single vendor/model candidate."""
    id: str
    audit_findings: List[int]                # binary list, 1 = risk, 0 = safe
    description: str                         # free‑text description
    unique_quasi_identifiers: int            # for reconstruction risk
    total_records: int                       # denominator for reconstruction risk
    # fields populated during processing
    feature_vector: np.ndarray = field(init=False, repr=False)
    risk_score: float = field(init=False, repr=False)
    memory_estimate: float = field(init=False, repr=False)


# ----------------------------------------------------------------------
# Hyperdimensional primitives (from Parent B)
# ----------------------------------------------------------------------


Vector = List[int]


def random_vector(dim: int = 10000, seed: str | int | None = None) -> Vector:
    rng = random.Random(seed)
    return [1 if rng.getrandbits(1) else -1 for _ in range(dim)]


def symbol_vector(symbol: str, dim: int = 10000) -> Vector:
    seed = int.from_bytes(
        hashlib.sha256(symbol.encode("utf-8")).digest()[:8], "big"
    )
    return random_vector(dim, seed)


def bind(a: Vector, b: Vector) -> Vector:
    """Binary binding (element‑wise multiplication)."""
    if len(a) != len(b):
        raise ValueError("vectors must have equal length")
    return [x * y for x, y in zip(a, b)]


def bundle(vectors: Iterable[Vector]) -> Vector:
    """Bundling (average) of a collection of vectors."""
    vecs = list(vectors)
    if not vecs:
        return []
    dim = len(vecs[0])
    # Sum across vectors and divide by count, then round to nearest int (+1 / -1)
    summed = [sum(vec[i] for vec in vecs) / len(vecs) for i in range(dim)]
    return [1 if v >= 0 else -1 for v in summed]


def gini_coefficient(values: Iterable[float]) -> float:
    """Gini coefficient measuring inequality of a non‑negative distribution."""
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0:
        return 1.0
    n = len(xs)
    cumulative = sum((i + 1) * x for i, x in enumerate(xs))
    g = 2 * cumulative / (n * sum(xs)) - (n + 1) / n
    return g


# ----------------------------------------------------------------------
# Core hybrid operations
# ----------------------------------------------------------------------


def candidate_feature_vector(candidate: Candidate) -> np.ndarray:
    """
    Build the combined feature vector X = [audit_binary, regex_counts].

    Returns a float NumPy array.
    """
    # Audit binary part
    audit_vec = np.asarray(candidate.audit_findings, dtype=float)

    # Regex‑based textual counts
    text = candidate.description.lower()

    evidence_cnt = len(re.findall(
        r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|"
        r"receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|"
        r"check|checked|audit)\b", text, re.I))

    planning_cnt = len(re.findall(
        r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|"
        r"prioritize|triage|schedule|milestone|goal|objective)\b", text, re.I))

    # Reconstruction risk ratio (unique identifiers / total records)
    recon_ratio = (
        candidate.unique_quasi_identifiers / candidate.total_records
        if candidate.total_records > 0 else 0.0
    )

    # Assemble full vector
    full_vec = np.concatenate(
        [audit_vec,
         np.array([evidence_cnt, planning_cnt, recon_ratio], dtype=float)]
    )
    return full_vec


def encode_candidate_hypervector(
    candidate: Candidate,
    dim: int = 10000
) -> Vector:
    """
    Encode the low‑dimensional feature vector into a high‑dimensional
    hypervector using binding (sign) and bundling.

    Each feature index i gets a deterministic symbol vector s_i.
    The sign of the feature value binds to s_i; the collection is bundled.
    """
    x = candidate.feature_vector
    bound_vectors: List[Vector] = []

    for i, val in enumerate(x):
        # Symbol for this dimension
        sym = symbol_vector(f"feat_{i}", dim)
        # Binary sign binding (1 for non‑negative, -1 for negative)
        sign_vec = [1 if val >= 0 else -1] * dim
        bound = bind(sym, sign_vec)
        bound_vectors.append(bound)

    # Bundle all bound vectors into a single hypervector
    hypervector = bundle(bound_vectors)
    return hypervector


def evaluate_candidate(candidate: Candidate, dim: int = 10000) -> Candidate:
    """
    Full evaluation pipeline:
    1. Compute low‑dimensional feature vector.
    2. Encode to hypervector.
    3. Derive a risk score (linear combination of feature norm and hypervector norm).
    4. Estimate memory footprint using Gini coefficient of the hypervector.
    The candidate instance is mutated in‑place and also returned.
    """
    # Step 1
    candidate.feature_vector = candidate_feature_vector(candidate)

    # Step 2
    hypervec = encode_candidate_hypervector(candidate, dim)

    # Step 3 – risk score
    #   - higher audit risk (sum) increases score
    #   - larger evidence/planning counts lower score (mitigation)
    #   - hypervector norm adds a regularisation term
    audit_risk = np.sum(candidate.feature_vector[: len(candidate.audit_findings)])
    evidence_planning = candidate.feature_vector[-3:-1].sum()
    recon_ratio = candidate.feature_vector[-1]

    hv_norm = math.sqrt(sum(v * v for v in hypervec))

    candidate.risk_score = (
        0.6 * audit_risk
        - 0.2 * evidence_planning
        + 0.1 * recon_ratio
        + 0.1 * hv_norm / math.sqrt(dim)
    )

    # Step 4 – memory estimate via Gini
    # Use absolute values of hypervector components as a proxy for memory load
    gini = gini_coefficient([abs(v) for v in hypervec])
    # Assume 4 bytes per int element
    candidate.memory_estimate = gini * dim * 4.0  # bytes

    return candidate


# ----------------------------------------------------------------------
# Demonstration / smoke test
# ----------------------------------------------------------------------


def _build_dummy_candidate() -> Candidate:
    """Create a simple candidate for quick testing."""
    return Candidate(
        id="demo-001",
        audit_findings=[0, 1, 0, 1, 0],
        description=(
            "The system includes audit logs, evidence receipts, and a "
            "detailed plan with milestones. Verification steps are documented."
        ),
        unique_quasi_identifiers=42,
        total_records=1000,
    )


if __name__ == "__main__":
    # Simple smoke test that runs without external input
    cand = _build_dummy_candidate()
    evaluate_candidate(cand, dim=4096)  # use a smaller dimension for speed
    print(f"Candidate ID: {cand.id}")
    print(f"Feature vector (len={len(cand.feature_vector)}): {cand.feature_vector}")
    print(f"Risk score: {cand.risk_score:.4f}")
    print(f"Memory estimate: {cand.memory_estimate / 1024:.2f} KiB")
    sys.exit(0)