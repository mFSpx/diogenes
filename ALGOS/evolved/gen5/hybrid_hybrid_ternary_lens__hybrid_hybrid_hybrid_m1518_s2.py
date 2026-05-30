# DARWIN HAMMER — match 1518, survivor 2
# gen: 5
# parent_a: hybrid_ternary_lens_audit_hybrid_hybrid_privac_m154_s2.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_perceptual_de_m264_s0.py (gen4)
# born: 2026-05-29T23:37:09Z

"""Hybrid Audit‑RBF Scheduler

Parents:
- `ternary_lens_audit.py` (Algorithm A) – produces binary audit risk vectors and a
  reconstruction risk score used for VRAM‑aware scheduling.
- `hybrid_perceptual_dedupe_hybrid_hybrid_rbf_su_m10_s5.py` (Algorithm B) – extracts
  regex‑based textual feature scores and feeds them to a radial‑basis‑function
  (RBF) surrogate model.

Mathematical Bridge
-------------------
For each candidate we build a *combined feature vector*  


X = [ a₁ … a_n ,  s₁ … s_m ] ∈ ℝ^{n+m}


where `a_i ∈ {0,1}` are the binary audit findings (Algorithm A) and `s_j ≥ 0`
are the regex‑derived feature counts (Algorithm B).  

A scalar *risk* is obtained by a weighted sum of the audit‑risk proportion
`r_a = (1/n) Σ a_i` and the reconstruction‑risk score
`r_r = unique_quasi_identifiers / total_records` :


R = α·r_a + β·r_r          (α,β ≥ 0, α+β = 1)


Memory consumption is estimated with an RBF surrogate.  Given a set of
prototype vectors `{P_k}` and associated memory targets `{M_k}`, the kernel
weight for prototype `k` is  


w_k = exp( - (ε·‖X‑P_k‖)² )


and the predicted memory usage is  


Ĥ = Σ_k w_k·M_k / Σ_k w_k .


The scheduler orders candidates by descending risk, then greedily loads them
while the cumulative predicted memory stays ≤ `memory_limit`.  Lower‑risk
candidates are evicted if needed, yielding a unified audit‑risk‑driven VRAM
scheduler.

The module provides three core hybrid functions:
- `candidate_feature_vector`
- `rbf_memory_estimate`
- `schedule_candidates`
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import math
import numpy as np

# ----------------------------------------------------------------------
# Data structures
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
# Core hybrid operations
# ----------------------------------------------------------------------


def candidate_feature_vector(candidate: Candidate) -> np.ndarray:
    """
    Build the combined feature vector X = [audit_binary, regex_counts].

    The audit part is taken as‑is (0/1).  The textual part uses the same
    regex patterns as the original Parent B to produce integer counts.
    The resulting vector is returned as a float NumPy array.
    """
    # --- audit part ----------------------------------------------------
    audit_vec = np.asarray(candidate.audit_findings, dtype=float)

    # --- textual regex feature extraction (Parent B) -------------------
    text = candidate.description

    # Feature: evidence‑related keywords
    evidence_cnt = len(re.findall(
        r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|"
        r"receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|"
        r"check|checked|audit)\b", text, re.I))

    # Feature: planning‑related keywords
    planning_cnt = len(re.findall(
        r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|"
        r"prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|"
        r"test|smoke)\b", text, re.I))

    # Feature: delay‑related keywords (captures “delay”, “slow”, etc.)
    delay_cnt = len(re.findall(r"\b(?:delay|slow|latency|wait|hold|postpone|defer)\b",
                               text, re.I))

    # Assemble textual part
    text_vec = np.array([evidence_cnt, planning_cnt, delay_cnt], dtype=float)

    # Concatenate
    combined = np.concatenate([audit_vec, text_vec])
    candidate.feature_vector = combined
    return combined


def compute_risk_score(candidate: Candidate,
                       alpha: float = 0.6,
                       beta: float = 0.4) -> float:
    """
    Compute the scalar risk R = α·r_a + β·r_r.

    * r_a = proportion of audit risk bits set to 1.
    * r_r = reconstruction risk = unique_quasi_identifiers / total_records.
    """
    if not candidate.audit_findings:
        r_a = 0.0
    else:
        r_a = sum(candidate.audit_findings) / len(candidate.audit_findings)

    if candidate.total_records <= 0:
        r_r = 0.0
    else:
        r_r = candidate.unique_quasi_identifiers / candidate.total_records

    risk = alpha * r_a + beta * r_r
    candidate.risk_score = risk
    return risk


def rbf_memory_estimate(feature_vec: np.ndarray,
                        prototypes: np.ndarray,
                        prototype_mem: np.ndarray,
                        epsilon: float = 1.0) -> float:
    """
    Estimate memory consumption using an RBF surrogate.

    Parameters
    ----------
    feature_vec : (d,) array
        Combined feature vector of the candidate.
    prototypes : (k, d) array
        Matrix of prototype vectors.
    prototype_mem : (k,) array
        Known memory usage associated with each prototype.
    epsilon : float
        RBF shape parameter.

    Returns
    -------
    float
        Predicted memory consumption.
    """
    # Euclidean distances to each prototype
    diffs = prototypes - feature_vec  # (k, d)
    dists = np.linalg.norm(diffs, axis=1)  # (k,)

    # Gaussian kernel weights
    weights = np.exp(- (epsilon * dists) ** 2)  # (k,)

    if weights.sum() == 0:
        # Avoid division by zero – fall back to simple average
        return float(prototype_mem.mean())

    estimate = np.dot(weights, prototype_mem) / weights.sum()
    return float(estimate)


def schedule_candidates(candidates: List[Candidate],
                        memory_limit: float,
                        prototypes: np.ndarray,
                        prototype_mem: np.ndarray,
                        epsilon: float = 1.0) -> List[Candidate]:
    """
    Greedy scheduler that respects the memory ceiling.

    1. Compute feature vectors, risk scores, and memory estimates for every
       candidate.
    2. Sort candidates by descending risk (high risk → higher priority for audit).
    3. Iterate, loading candidates while the cumulative predicted memory stays
       ≤ `memory_limit`.  If a candidate would exceed the limit, it is skipped.
    4. Return the list of loaded candidates (in load order).

    This mirrors the ModelPool logic of Algorithm A while using the RBF‑based
    memory estimate of Algorithm B.
    """
    # Pre‑process all candidates
    for cand in candidates:
        candidate_feature_vector(cand)
        compute_risk_score(cand)
        cand.memory_estimate = rbf_memory_estimate(
            cand.feature_vector, prototypes, prototype_mem, epsilon)

    # Sort by risk descending
    ordered = sorted(candidates, key=lambda c: c.risk_score, reverse=True)

    loaded: List[Candidate] = []
    used_memory = 0.0

    for cand in ordered:
        if used_memory + cand.memory_estimate <= memory_limit:
            loaded.append(cand)
            used_memory += cand.memory_estimate
        else:
            # Candidate cannot be loaded; continue to next (lower priority)
            continue

    return loaded


# ----------------------------------------------------------------------
# Helper to generate synthetic prototypes (for demonstration)
# ----------------------------------------------------------------------


def generate_prototypes(num: int, dim: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Produce `num` random prototype vectors and associated synthetic memory usage.
    Memory values are drawn from a uniform range [10, 50] (arbitrary units).
    """
    rng = np.random.default_rng(seed)
    prototypes = rng.uniform(0, 10, size=(num, dim))
    memory = rng.uniform(10, 50, size=num)
    return prototypes, memory


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Create a few dummy candidates
    demo_candidates = [
        Candidate(
            id="model_A",
            audit_findings=[0, 0, 1, 0, 0],
            description="Verified source with evidence and a clear roadmap. No delay observed.",
            unique_quasi_identifiers=5,
            total_records=100,
        ),
        Candidate(
            id="model_B",
            audit_findings=[1, 1, 1, 1, 1],
            description="Missing evidence, many delays, and no clear plan.",
            unique_quasi_identifiers=30,
            total_records=120,
        ),
        Candidate(
            id="model_C",
            audit_findings=[0, 0, 0, 0, 0],
            description="All checks passed, evidence present, schedule ready.",
            unique_quasi_identifiers=2,
            total_records=80,
        ),
        Candidate(
            id="model_D",
            audit_findings=[0, 1, 0, 1, 0],
            description="Partial verification, some planning steps, occasional latency.",
            unique_quasi_identifiers=10,
            total_records=200,
        ),
    ]

    # Generate synthetic prototypes matching the combined feature dimension
    # Dimension = len(audit_findings) + 3 regex features
    dim = len(demo_candidates[0].audit_findings) + 3
    protos, prot_mem = generate_prototypes(num=8, dim=dim)

    # Run scheduler with a memory ceiling
    loaded = schedule_candidates(
        candidates=demo_candidates,
        memory_limit=100.0,
        prototypes=protos,
        prototype_mem=prot_mem,
        epsilon=0.8,
    )

    print("Loaded candidates (in priority order):")
    for c in loaded:
        print(f"  • {c.id}: risk={c.risk_score:.3f}, mem_est={c.memory_estimate:.2f}")

    # Verify that total memory does not exceed the limit
    total_mem = sum(c.memory_estimate for c in loaded)
    print(f"Total predicted memory: {total_mem:.2f} / 100.0")