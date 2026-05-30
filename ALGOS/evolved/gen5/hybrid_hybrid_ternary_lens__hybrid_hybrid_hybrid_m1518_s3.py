# DARWIN HAMMER — match 1518, survivor 3
# gen: 5
# parent_a: hybrid_ternary_lens_audit_hybrid_hybrid_privac_m154_s2.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_perceptual_de_m264_s0.py (gen4)
# born: 2026-05-29T23:37:09Z

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
                        epsilon: float = 1.0,
                        fallback: bool = True) -> List[Candidate]:
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

    If fallback is True, use LRU eviction when memory limit is reached.
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
    lru: List[Candidate] = []

    for cand in ordered:
        if used_memory + cand.memory_estimate <= memory_limit:
            loaded.append(cand)
            used_memory += cand.memory_estimate
            lru.append(cand)
        else:
            if fallback:
                # Evict LRU candidate
                if lru:
                    evicted = lru.pop(0)
                    used_memory -= evicted.memory_estimate
                    if used_memory + cand.memory_estimate <= memory_limit:
                        loaded.append(cand)
                        used_memory += cand.memory_estimate
                        lru.append(cand)
            # Candidate cannot be loaded; continue to next (lower priority)
            continue

    return loaded


# ----------------------------------------------------------------------
# Helper to generate synthetic prototypes (for demonstration)
# ----------------------------------------------------------------------

def generate_prototypes(num: int, dim: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    np.random.seed(seed)
    prototypes = np.random.rand(num, dim)
    prototype_mem = np.random.rand(num) * 100  # Memory usage in MB
    return prototypes, prototype_mem

def main():
    parser = argparse.ArgumentParser(description='Hybrid Audit-RBF Scheduler')
    parser.add_argument('--memory_limit', type=float, default=1000.0, help='Memory limit in MB')
    parser.add_argument('--num_prototypes', type=int, default=10, help='Number of prototypes')
    parser.add_argument('--prototype_dim', type=int, default=6, help='Dimension of prototype vectors')
    args = parser.parse_args()

    prototypes, prototype_mem = generate_prototypes(args.num_prototypes, args.prototype_dim)

    candidates = [
        Candidate(id='candidate1', audit_findings=[1, 0, 1], description='example', unique_quasi_identifiers=10, total_records=100),
        Candidate(id='candidate2', audit_findings=[0, 1, 0], description='example2', unique_quasi_identifiers=20, total_records=200),
    ]

    loaded_candidates = schedule_candidates(candidates, args.memory_limit, prototypes, prototype_mem)
    for cand in loaded_candidates:
        print(f"Loaded candidate: {cand.id}, risk score: {cand.risk_score}, memory estimate: {cand.memory_estimate}")

if __name__ == '__main__':
    main()