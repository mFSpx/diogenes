# DARWIN HAMMER — match 2446, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_cockpit_metri_hard_truth_math_m27_s3.py (gen2)
# parent_b: hybrid_hybrid_regret_engine_hybrid_ternary_lens__m3_s5.py (gen3)
# born: 2026-05-29T23:42:18Z

"""
Hybrid Module: Fusing Hybrid Cockpit Metrics (Parent A) with Hybrid Regret-Weighted Ternary-Decision Analyzer (Parent B)

The mathematical bridge between the two parents lies in the treatment of their output probability distributions.
Parent A (Hybrid Cockpit Metrics) provides a trust value in the interval [0,1], which can be used to weight the similarity score.
Parent B (Hybrid Regret-Weighted Ternary-Decision Analyzer) provides a regret-weighted probability distribution over actions and a ternary vector.

The hybrid algorithm maps the regret-weighted probabilities onto the ternary alphabet by sign-quantisation, 
then uses the trust value from Parent A to modulate the MinHash similarity between two token sets.

This fusion yields a unified system that assesses both linguistic similarity and decision confidence, 
while incorporating evidence-coverage quality.
"""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Shared data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathAction:
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True)
class MathCounterfactual:
    action_id: str
    outcome_value: float
    probability: float = 1.0

# ---------------------------------------------------------------------------
# Parent A – cockpit metrics
# ---------------------------------------------------------------------------

def anti_slop_ratio(claims_with_evidence: int, total_claims_emitted: int) -> float:
    """Ratio of supported claims, clamped to [0, 1]."""
    return 1.0 if total_claims_emitted <= 0 else max(0.0, min(1.0,
                claims_with_evidence / total_claims_emitted))


def cockpit_honesty(displayed_ok: int, unknown_displayed_as_ok: int) -> float:
    """Fraction of displayed items that are known‑good, clamped to [0, 1]."""
    total = displayed_ok + unknown_displayed_as_ok
    return 1.0 if total <= 0 else max(0.0, min(1.0, displayed_ok / total))


def audit_debt(exports_missing_audit_step: int, total_exports: int) -> float:
    """Fraction of exports missing an audit step, clamped to [0, 1]."""
    return 1.0 if total_exports <= 0 else max(0.0, min(1.0,
                exports_missing_audit_step / total_exports))

# ---------------------------------------------------------------------------
# Parent B – regret-weighted ternary-decision analyzer
# ---------------------------------------------------------------------------

def minhash(token_set: Iterable[str], seed: int) -> str:
    """MinHash signature for a token set."""
    m = hashlib.md5()
    m.update(str(seed).encode())
    for token in sorted(token_set):
        m.update(token.encode())
    return m.hexdigest()

def regret_weighted_probabilities(actions: List[MathAction]) -> np.ndarray:
    """Regret-weighted probability distribution over actions."""
    exp_values = np.array([a.expected_value for a in actions])
    costs = np.array([a.cost for a in actions])
    risks = np.array([a.risk for a in actions])
    utilities = exp_values - costs - risks
    exp_utilities = np.exp(utilities)
    probabilities = exp_utilities / np.sum(exp_utilities)
    return probabilities

def ternary_decision_hygiene(payload_descriptor: Iterable[int]) -> np.ndarray:
    """Ternary vector from payload descriptor."""
    return np.array([1 if x > 0 else -1 if x < 0 else 0 for x in payload_descriptor])

# ---------------------------------------------------------------------------
# Hybrid functions
# ---------------------------------------------------------------------------

def hybrid_similarity(token_set1: Iterable[str], token_set2: Iterable[str], 
                       actions: List[MathAction], payload_descriptor: Iterable[int], 
                       claims_with_evidence: int, total_claims_emitted: int) -> float:
    """Hybrid similarity score."""
    # Compute regret-weighted probabilities
    probabilities = regret_weighted_probabilities(actions)
    
    # Sign-quantise probabilities to ternary symbols
    ternary_symbols = np.sign(probabilities - 0.5)
    
    # Compute ternary decision hygiene
    hygiene = ternary_decision_hygiene(payload_descriptor)
    
    # Concatenate ternary symbols and hygiene
    combined_ternary = np.concatenate((ternary_symbols, hygiene))
    
    # Compute Shannon entropy of combined ternary distribution
    entropy = -np.sum(np.unique(combined_ternary, return_counts=True)[1] / len(combined_ternary) * 
                       np.log2(np.unique(combined_ternary, return_counts=True)[1] / len(combined_ternary)))
    
    # Compute MinHash similarity
    minhash1 = minhash(token_set1, 42)
    minhash2 = minhash(token_set2, 42)
    minhash_similarity = int(minhash1 == minhash2)
    
    # Compute trust value
    trust = anti_slop_ratio(claims_with_evidence, total_claims_emitted)
    
    # Hybrid similarity score
    return trust * minhash_similarity * entropy

def hybrid_regret_weighted_lsm_score(actions: List[MathAction], 
                                     payload_descriptor: Iterable[int], 
                                     claims_with_evidence: int, total_claims_emitted: int) -> float:
    """Hybrid regret-weighted LSM score."""
    # Compute regret-weighted probabilities
    probabilities = regret_weighted_probabilities(actions)
    
    # Compute LSM score
    lsm_score = 1.0 - (np.abs(probabilities - 0.5) / (probabilities + 0.5 + 1e-6))
    
    # Compute trust value
    trust = anti_slop_ratio(claims_with_evidence, total_claims_emitted)
    
    # Hybrid regret-weighted LSM score
    return trust * np.mean(lsm_score)

def hybrid_cockpit_honesty(actions: List[MathAction], 
                           payload_descriptor: Iterable[int], 
                           displayed_ok: int, unknown_displayed_as_ok: int) -> float:
    """Hybrid cockpit honesty."""
    # Compute regret-weighted probabilities
    probabilities = regret_weighted_probabilities(actions)
    
    # Compute ternary decision hygiene
    hygiene = ternary_decision_hygiene(payload_descriptor)
    
    # Compute cockpit honesty
    honesty = cockpit_honesty(displayed_ok, unknown_displayed_as_ok)
    
    # Hybrid cockpit honesty
    return honesty * np.mean(probabilities) * np.mean(hygiene)

if __name__ == "__main__":
    # Smoke test
    actions = [MathAction("action1", 10.0), MathAction("action2", 20.0)]
    payload_descriptor = [1, -1, 0]
    token_set1 = ["token1", "token2"]
    token_set2 = ["token2", "token3"]
    claims_with_evidence = 10
    total_claims_emitted = 20
    displayed_ok = 10
    unknown_displayed_as_ok = 5
    
    print(hybrid_similarity(token_set1, token_set2, actions, payload_descriptor, 
                             claims_with_evidence, total_claims_emitted))
    print(hybrid_regret_weighted_lsm_score(actions, payload_descriptor, 
                                            claims_with_evidence, total_claims_emitted))
    print(hybrid_cockpit_honesty(actions, payload_descriptor, displayed_ok, unknown_displayed_as_ok))