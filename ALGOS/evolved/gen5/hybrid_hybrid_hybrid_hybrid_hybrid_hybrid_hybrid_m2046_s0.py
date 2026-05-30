# DARWIN HAMMER — match 2046, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_hybrid_decisi_hybrid_hard_truth_ma_m167_s1.py (gen3)
# parent_b: hybrid_hybrid_hybrid_regret_hybrid_hybrid_ternar_m241_s3.py (gen4)
# born: 2026-05-29T23:40:32Z

"""Hybrid Decision‑Regret Analyzer (HDR‑A)

Parents:
- hybrid_hybrid_hybrid_decisi_hybrid_hard_truth_ma_m167_s1.py (regex‑based feature extraction + expected edge lengths)
- hybrid_hybrid_hybrid_regret_hybrid_hybrid_ternar_m241_s3.py (regret‑weighted probabilities & audit‑signature pruning)

Mathematical bridge:
The parent‑A pipeline produces a *feature‑count vector* **f** ∈ ℝ⁶ (evidence, planning, delay, support, boundary, outcome).
Parent‑B supplies a *regret‑weighted probability vector* **p** ∈ ℝⁿ for n actions.
Both parents also expose an *edge‑expectation matrix* **E** ∈ ℝ⁶ˣⁿ derived from the expected edge‑lengths (or costs) of actions.
The hybrid cost is defined as the bilinear form

    C = fᵀ · E · p

which simultaneously weights stylometric evidence by the probabilistic importance of actions, while the audit‑signature pruning step
uses **p** to discard low‑impact actions before forming **E**. This unifies the deterministic stylometry side with the stochastic regret‑weighted side
into a single, differentiable decision metric.
"""

import re
import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Parent‑A: regex feature extraction
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
BOUNDARY_RE = re.compile(
    r"\b(?:boundary|boundaries|walk away|no contact|do not|don't|stop|limit|refuse|protect|safe|safety|lock|redact|privacy|permission|consent)\b",
    re.I,
)
OUTCOME_RE = re.compile(
    r"\b(?:done|shipped|finished|complete|resolved|succeeded|passed|paid|safe|sent|delivered|finished)\b",
    re.I,
)

def extract_feature_counts(text: str) -> Dict[str, int]:
    """Return a dict with counts for each parent‑A feature class."""
    return {
        "evidence": len(EVIDENCE_RE.findall(text)),
        "planning": len(PLANNING_RE.findall(text)),
        "delay":    len(DELAY_RE.findall(text)),
        "support":  len(SUPPORT_RE.findall(text)),
        "boundary": len(BOUNDARY_RE.findall(text)),
        "outcome":  len(OUTCOME_RE.findall(text)),
    }

# ----------------------------------------------------------------------
# Parent‑B: regret‑weighted probabilities and audit‑signature pruning
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathAction:
    """Atomic decision element."""
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0

def calculate_regret_weighted_probabilities(actions: List[MathAction]) -> np.ndarray:
    """
    Regret weighting: p_i ∝ exp( - (max_ev - ev_i) )
    where max_ev is the maximal expected value among actions.
    The resulting vector is normalized to sum to 1.
    """
    if not actions:
        return np.array([], dtype=float)

    ev = np.array([a.expected_value for a in actions], dtype=float)
    max_ev = ev.max()
    # Regret = max_ev - ev_i  (non‑negative)
    regret = max_ev - ev
    # Convert regret to weight via exponential decay
    weights = np.exp(-regret)
    probs = weights / weights.sum()
    return probs

def prune_actions_by_probability(actions: List[MathAction],
                                 probs: np.ndarray,
                                 threshold: float = 0.05) -> List[MathAction]:
    """
    Keep actions whose regret‑weighted probability exceeds *threshold*.
    This mimics the audit‑signature pruning step.
    """
    kept = [a for a, p in zip(actions, probs) if p >= threshold]
    return kept

# ----------------------------------------------------------------------
# Hybrid core: bridging structures
# ----------------------------------------------------------------------
def build_edge_expectation_matrix(features: Dict[str, int],
                                 actions: List[MathAction]) -> np.ndarray:
    """
    Construct E ∈ ℝ⁶ˣⁿ where each entry e_{k,i} = f_k * g(cost_i)
    with g(cost) = exp( -|cost - μ| ) and μ = mean(costs).
    This yields a matrix that couples feature magnitude with
    the expected edge length (cost) of each action.
    """
    if not actions:
        return np.empty((6, 0), dtype=float)

    f_vec = np.array(list(features.values()), dtype=float)  # shape (6,)
    costs = np.array([a.cost for a in actions], dtype=float)  # shape (n,)

    mu = costs.mean() if costs.size else 0.0
    g = np.exp(-np.abs(costs - mu))  # shape (n,)

    # Broadcast to shape (6, n)
    E = f_vec[:, None] * g[None, :]
    return E

def hybrid_cost(text: str,
                actions: List[MathAction],
                prune_threshold: float = 0.05) -> float:
    """
    Full hybrid evaluation:
    1. Extract feature counts → **f**
    2. Compute regret‑weighted probabilities → **p**
    3. Prune low‑probability actions (audit‑signature step)
    4. Build edge‑expectation matrix **E** from remaining actions
    5. Return bilinear cost C = fᵀ·E·p
    """
    # Step 1
    feats = extract_feature_counts(text)

    # Step 2
    probs = calculate_regret_weighted_probabilities(actions)

    # Step 3
    actions_pruned = prune_actions_by_probability(actions, probs, prune_threshold)
    if not actions_pruned:
        # No surviving actions → cost is simply the L2 norm of feature vector
        return float(np.linalg.norm(list(feats.values())))

    # Re‑compute probabilities on the pruned set (renormalize)
    probs_pruned = calculate_regret_weighted_probabilities(actions_pruned)

    # Step 4
    E = build_edge_expectation_matrix(feats, actions_pruned)

    # Step 5: bilinear form
    # fᵀ·E yields shape (n,), then dot with p gives scalar
    f_vec = np.array(list(feats.values()), dtype=float)
    intermediate = f_vec @ E                     # shape (n,)
    cost = float(intermediate @ probs_pruned)    # scalar
    return cost

def summarize_hybrid(text: str,
                     actions: List[MathAction],
                     prune_threshold: float = 0.05) -> Tuple[float, Dict[str, int], np.ndarray]:
    """
    Convenience wrapper returning:
    - hybrid cost,
    - feature count dict,
    - final regret‑weighted probability vector (post‑pruning).
    """
    feats = extract_feature_counts(text)
    probs = calculate_regret_weighted_probabilities(actions)
    actions_pruned = prune_actions_by_probability(actions, probs, prune_threshold)
    if not actions_pruned:
        return (float(np.linalg.norm(list(feats.values()))), feats, np.array([]))

    probs_pruned = calculate_regret_weighted_probabilities(actions_pruned)
    cost = hybrid_cost(text, actions, prune_threshold)
    return (cost, feats, probs_pruned)

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_text = (
        "We have verified the source and recorded the hash. "
        "The plan includes a checklist and timeline. "
        "Please wait until tomorrow before proceeding. "
        "Contact the doctor for support. "
        "Do not share the boundary details. "
        "The task is done and shipped."
    )

    # Create a diverse set of actions
    actions = [
        MathAction(id="A1", expected_value=0.9, cost=3.2, risk=0.1),
        MathAction(id="A2", expected_value=0.4, cost=7.5, risk=0.3),
        MathAction(id="A3", expected_value=0.6, cost=5.1, risk=0.2),
        MathAction(id="A4", expected_value=0.2, cost=9.8, risk=0.5),
        MathAction(id="A5", expected_value=0.75, cost=2.0, risk=0.15),
    ]

    cost = hybrid_cost(sample_text, actions, prune_threshold=0.1)
    print(f"Hybrid cost: {cost:.4f}")

    total, feats, probs = summarize_hybrid(sample_text, actions, prune_threshold=0.1)
    print("Feature counts:", feats)
    print("Final regret‑weighted probabilities:", probs)