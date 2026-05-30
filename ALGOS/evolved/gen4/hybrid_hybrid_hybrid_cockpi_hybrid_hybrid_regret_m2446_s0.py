# DARWIN HAMMER — match 2446, survivor 0
# gen: 4
# parent_a: hybrid_hybrid_cockpit_metri_hard_truth_math_m27_s3.py (gen2)
# parent_b: hybrid_hybrid_regret_engine_hybrid_ternary_lens__m3_s5.py (gen3)
# born: 2026-05-29T23:42:18Z

"""Hybrid Regret-Weighted Linguistic Similarity Analyzer (RW-LSA).

This module fuses:

* **Parent A** – hybrid_hybrid_cockpit_metri_hard_truth_math_m27_s3.py (RW-LSM):
  - Generates linguistic similarity measures (LSM) between text pairs.
  - Computes trust-weighted LSM scores using cockpit metrics.

* **Parent B** – hybrid_hybrid_regret_engine_hybrid_ternary_lens__m3_s5.py (RW-TD-H):
  - Produces regret-weighted probability distributions over actions.
  - Computes Shannon-entropy over a concatenated ternary vector.

**Mathematical bridge**

Both parents ultimately yield *discrete probability-mass samples*:

* RW-LSM provides a trust-weighted LSM score `h * lsm_score(a, b)`.
* RW-TD-H provides a regret-weighted probability distribution `p` over actions.

The hybrid algorithm maps the trust-weighted LSM scores onto the regret-weighted probabilities `p` by modulating the LSM scores with the regret weights, yielding a similarity score that is aware of both linguistic similarity and decision confidence.

The implementation below provides three core functions that realise this fusion and a small smoke-test when executed as a script.
"""

import math
import random
import sys
from pathlib import Path
from typing import Callable, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Parent A – RW-LSM (re-implemented for internal use)
# ---------------------------------------------------------------------------

def lsm_vector(text: str, vocab: list, cnt: dict) -> dict:
    """Compute linguistic similarity measure (LSM) vector."""
    total = sum(cnt[w] for w in vocab)
    return {cat: sum(cnt[w] for w in vocab) / total}

def lsm_score(a: str, b: str, vocab: list, cnt_a: dict, cnt_b: dict) -> float:
    """Compute LSM score between two text strings."""
    av = sum(cnt_a[w] for w in vocab)
    bv = sum(cnt_b[w] for w in vocab)
    return 1.0 - (abs(av - bv) / (av + bv + 1e-6))

def trust_weighted_lsm_score(a: str, b: str, h: float, vocab: list, cnt_a: dict, cnt_b: dict) -> float:
    """Compute trust-weighted LSM score."""
    return h * lsm_score(a, b, vocab, cnt_a, cnt_b)

def anti_slop_ratio(claims_with_evidence: int, total_claims_emitted: int) -> float:
    """Ratio of supported claims, clamped to [0, 1]."""
    return 1.0 if total_claims_emitted <= 0 else max(0.0, min(1.0, claims_with_evidence / total_claims_emitted))

# ---------------------------------------------------------------------------
# Parent B – RW-TD-H (re-implemented for internal use)
# ---------------------------------------------------------------------------

class MathAction:
    def __init__(self, id: str, expected_value: float, cost: float = 0.0, risk: float = 0.0):
        self.id = id
        self.expected_value = expected_value
        self.cost = cost
        self.risk = risk

class MathCounterfactual:
    def __init__(self, action_id: str, outcome_value: float, probability: float = 1.0):
        self.action_id = action_id
        self.outcome_value = outcome_value
        self.probability = probability

def regret_weighted_probability(actions: list, regrets: list) -> list:
    """Compute regret-weighted probability distribution over actions."""
    exp_values = [a.expected_value for a in actions]
    softmax = np.exp(np.array(regrets) - np.max(regrets)) / np.sum(np.exp(np.array(regrets) - np.max(regrets)))
    return softmax

# ---------------------------------------------------------------------------
# Hybrid RW-LSA
# ---------------------------------------------------------------------------

def hybrid_rw_lsa(text_a: str, text_b: str, actions: list, regrets: list, claims_with_evidence: int, total_claims_emitted: int, vocab: list, cnt_a: dict, cnt_b: dict) -> float:
    """Compute hybrid regret-weighted linguistic similarity score."""
    h = anti_slop_ratio(claims_with_evidence, total_claims_emitted)
    lsm = trust_weighted_lsm_score(text_a, text_b, h, vocab, cnt_a, cnt_b)
    p = regret_weighted_probability(actions, regrets)
    return np.sum(np.array(p) * np.array([lsm] * len(p)))

def hybrid_entropy_rw_lsa(text_a: str, text_b: str, actions: list, regrets: list, claims_with_evidence: int, total_claims_emitted: int, vocab: list, cnt_a: dict, cnt_b: dict) -> float:
    """Compute hybrid entropy of regret-weighted linguistic similarity score."""
    p = regret_weighted_probability(actions, regrets)
    lsm = trust_weighted_lsm_score(text_a, text_b, anti_slop_ratio(claims_with_evidence, total_claims_emitted), vocab, cnt_a, cnt_b)
    return -np.sum(np.array(p) * np.log2(np.array([lsm] * len(p))))

def smoke_test():
    vocab = ['apple', 'banana', 'orange']
    cnt_a = {'apple': 2, 'banana': 1}
    cnt_b = {'apple': 1, 'orange': 2}
    actions = [MathAction('action1', 0.5), MathAction('action2', 0.3)]
    regrets = [0.2, 0.1]
    text_a = 'I love apples and bananas.'
    text_b = 'I love apples and oranges.'
    claims_with_evidence = 10
    total_claims_emitted = 20

    print(hybrid_rw_lsa(text_a, text_b, actions, regrets, claims_with_evidence, total_claims_emitted, vocab, cnt_a, cnt_b))
    print(hybrid_entropy_rw_lsa(text_a, text_b, actions, regrets, claims_with_evidence, total_claims_emitted, vocab, cnt_a, cnt_b))

if __name__ == "__main__":
    smoke_test()