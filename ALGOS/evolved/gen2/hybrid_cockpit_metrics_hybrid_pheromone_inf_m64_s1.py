# DARWIN HAMMER — match 64, survivor 1
# gen: 2
# parent_a: cockpit_metrics.py (gen0)
# parent_b: hybrid_pheromone_infotaxis_m3_s1.py (gen1)
# born: 2026-05-29T23:25:30Z

"""Hybrid Cockpit‑Pheromone Metrics

This module fuses the *cockpit* honesty/evidence‑coverage metrics (Algorithm A) with the
surface‑pheromone / infotaxis entropy search (Algorithm B).  The mathematical bridge is
the observation that all cockpit metrics are normalized ratios in the interval [0, 1].
These ratios can be interpreted as prior probabilities that weight pheromone
signals and entropy calculations.  Consequently:

* A metric ratio *m* scales a pheromone signal *S* → *m·S*.
* A vector of metric ratios forms a probability distribution *p* that can be fed
  into the entropy functions of the infotaxis component.
* Expected entropy is then evaluated on a mixture of metric‑weighted hit/miss
  states, producing a single “trust‑entropy” score.

The three core functions below illustrate this integration.
"""

import math
import random
import sys
import pathlib
import numpy as np
from datetime import datetime, timezone
import uuid

# ---------- Parent A: cockpit metrics ----------
def anti_slop_ratio(claims_with_evidence: int, total_claims_emitted: int) -> float:
    """Fraction of claims that have supporting evidence."""
    return 1.0 if total_claims_emitted <= 0 else max(0.0, min(1.0,
        claims_with_evidence / total_claims_emitted))

def cockpit_honesty(displayed_ok: int, unknown_displayed_as_ok: int) -> float:
    """Fraction of displayed items that are known to be OK."""
    total = displayed_ok + unknown_displayed_as_ok
    return 1.0 if total <= 0 else max(0.0, min(1.0, displayed_ok / total))

def audit_debt(exports_missing_audit_step: int) -> float:
    """Raw count of missing audit steps (non‑negative)."""
    return float(max(0, exports_missing_audit_step))

# ---------- Parent B: pheromone + infotaxis ----------
def calculate_pheromone_signal(base_signal: float,
                               half_life_seconds: float,
                               elapsed_seconds: float) -> float:
    """Exponential decay of a pheromone signal."""
    if half_life_seconds <= 0:
        raise ValueError("half_life_seconds must be > 0")
    decay = math.pow(0.5, elapsed_seconds / half_life_seconds)
    return base_signal * decay

def calculate_entropy(probabilities: np.ndarray, eps: float = 1e-12) -> float:
    """Shannon entropy of a probability vector."""
    total = probabilities.sum()
    if total <= 0:
        raise ValueError('positive probability mass required')
    probs = probabilities / total
    return -np.sum(probs * np.log(np.maximum(probs, eps)))

def expected_entropy(p_hit: float,
                     hit_state: np.ndarray,
                     miss_state: np.ndarray) -> float:
    """Expected entropy after a binary outcome."""
    if not 0.0 <= p_hit <= 1.0:
        raise ValueError('p_hit must be in [0,1]')
    return p_hit * calculate_entropy(hit_state) + (1.0 - p_hit) * calculate_entropy(miss_state)

def best_action(actions: dict) -> str:
    """Select the action with minimal expected entropy."""
    if not actions:
        raise ValueError('actions required')
    return min(actions, key=lambda a: (expected_entropy(*actions[a]), a))

# ---------- Hybrid Functions ----------
def hybrid_pheromone_signal(metric_ratio: float,
                            base_signal: float,
                            half_life_seconds: float,
                            elapsed_seconds: float) -> float:
    """
    Combine a cockpit metric (as a prior weight) with pheromone decay.

    The metric_ratio ∈ [0,1] scales the initial signal before exponential decay.
    """
    if not 0.0 <= metric_ratio <= 1.0:
        raise ValueError('metric_ratio must be in [0,1]')
    weighted_signal = metric_ratio * base_signal
    return calculate_pheromone_signal(weighted_signal, half_life_seconds, elapsed_seconds)


def hybrid_entropy_from_metrics(metrics: dict,
                                hit_state: np.ndarray,
                                miss_state: np.ndarray) -> float:
    """
    Build a probability distribution from cockpit metrics and compute expected entropy.

    `metrics` maps metric names to raw integer counts.  Each count is transformed
    into a normalized ratio (using the appropriate cockpit function) and then
    assembled into a probability vector.
    """
    # Derive normalized ratios
    ratios = []
    # Expected keys: 'claims_with_evidence', 'total_claims_emitted',
    #                'displayed_ok', 'unknown_displayed_as_ok',
    #                'exports_missing_audit_step'
    # Missing keys are treated as zero.
    claims = metrics.get('claims_with_evidence', 0)
    total = metrics.get('total_claims_emitted', 0)
    ratios.append(anti_slop_ratio(claims, total))

    displayed_ok = metrics.get('displayed_ok', 0)
    unknown_ok = metrics.get('unknown_displayed_as_ok', 0)
    ratios.append(cockpit_honesty(displayed_ok, unknown_ok))

    # audit_debt is not a ratio; we convert it to a pseudo‑ratio by normalising
    # against a large sentinel (e.g., 1000) to keep it in [0,1].
    debt = metrics.get('exports_missing_audit_step', 0)
    ratios.append(1.0 - min(1.0, audit_debt(debt) / 1000.0))

    prob_vec = np.array(ratios, dtype=float)
    # Use the metric‑derived probabilities as the hit probability.
    p_hit = prob_vec.mean()
    return expected_entropy(p_hit, hit_state, miss_state)


def compute_hybrid_honesty_score(displayed_ok: int,
                                 unknown_displayed_as_ok: int,
                                 claims_with_evidence: int,
                                 total_claims_emitted: int,
                                 base_signal: float,
                                 half_life_seconds: float,
                                 elapsed_seconds: float) -> float:
    """
    Produce a single trust score that merges cockpit honesty, evidence coverage,
    and a decayed pheromone signal.

    The score is the product of:
        * cockpit_honesty
        * anti_slop_ratio
        * hybrid_pheromone_signal (using the product of the two ratios as weight)
    The result is clipped to [0,1].
    """
    honesty = cockpit_honesty(displayed_ok, unknown_displayed_as_ok)
    evidence = anti_slop_ratio(claims_with_evidence, total_claims_emitted)
    weight = honesty * evidence
    pheromone = hybrid_pheromone_signal(weight, base_signal,
                                        half_life_seconds, elapsed_seconds)
    return max(0.0, min(1.0, pheromone))


# ---------- Smoke test ----------
if __name__ == "__main__":
    # Simple deterministic parameters
    displayed_ok = 80
    unknown_ok = 20
    claims_with_evidence = 45
    total_claims = 60
    base_signal = 1.0
    half_life = 30.0          # seconds
    elapsed = 10.0            # seconds

    trust = compute_hybrid_honesty_score(
        displayed_ok,
        unknown_ok,
        claims_with_evidence,
        total_claims,
        base_signal,
        half_life,
        elapsed
    )
    print(f"Hybrid trust score: {trust:.4f}")

    # Prepare a tiny infotaxis scenario
    hit_state = np.array([0.7, 0.2, 0.1])
    miss_state = np.array([0.4, 0.4, 0.2])

    metrics = {
        'claims_with_evidence': claims_with_evidence,
        'total_claims_emitted': total_claims,
        'displayed_ok': displayed_ok,
        'unknown_displayed_as_ok': unknown_ok,
        'exports_missing_audit_step': 5
    }

    exp_ent = hybrid_entropy_from_metrics(metrics, hit_state, miss_state)
    print(f"Hybrid expected entropy: {exp_ent:.6f}")

    # Define actions with pre‑computed (p_hit, hit_state, miss_state)
    actions = {
        'move_north': (0.6, hit_state, miss_state),
        'move_south': (0.4, hit_state, miss_state),
        'stay': (0.5, hit_state, miss_state)
    }
    best = best_action(actions)
    print(f"Best action (minimum expected entropy): {best}")