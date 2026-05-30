# DARWIN HAMMER — match 2337, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_hybrid_decisi_hybrid_hybrid_path_s_m17_s3.py (gen4)
# parent_b: hybrid_hybrid_regret_engine_hybrid_decision_hygi_m993_s3.py (gen3)
# born: 2026-05-29T23:41:54Z

"""Hybrid Fusion Module
Parents:
- hybrid_hybrid_hybrid_decisi_hybrid_hybrid_path_s_m17_s3.py (Parent A)
- hybrid_hybrid_regret_engine_hybrid_decision_hygi_m993_s3.py (Parent B)

Mathematical Bridge
-------------------
Parent A maps each datum to a 2‑D resource vector **R_i = [L_i, P_i]** and a
high‑dimensional feature vector **v_i**.  Concatenating them yields an
augmented point **p_i ∈ ℝ^{2+d}**.  The ordered sequence **{p_i}** defines a
discrete path; a truncated path‑signature **S** (level‑1 and level‑2 iterated
integrals) aggregates all pairwise interactions of the augmented resources.

Parent B takes a set of expected values, applies a regret‑weighted exponential
soft‑max to obtain a probability distribution **π**, and then evaluates the
Shannon entropy **H(π)** (and optionally the Gini coefficient) to quantify
selection uncertainty.

The fusion treats each component of the signature **S** as an “action’’ with
expected value **E_k = S_k**.  The regret‑weighted soft‑max is applied to the
signature vector, producing **π_k**.  The entropy **H(π)** is fed back as a
budget‑aware scaling factor: a higher entropy relaxes the greedy budget
selector from Parent A, allowing more exploratory selections.  This creates a
single unified system where structural path information influences regret‑aware
probabilistic decision‑making and budget allocation.

The module implements:
1. Simple regex‑based resource extraction (Parent A).
2. Dummy high‑dimensional feature extraction.
3. Augmented path construction and truncated signature computation.
4. Regret‑weighted strategy derived from the signature.
5. Entropy/Gini evaluation.
6. Entropy‑scaled greedy budget selection.
"""

from __future__ import annotations

import math
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Iterable

import numpy as np

# ----------------------------------------------------------------------
# Parent A – resource extraction (regex based)
# ----------------------------------------------------------------------
EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|"
    r"hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)
PLANNING_RE = re.compile(
    r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|"
    r"triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
    re.I,
)
DELAY_RE = re.compile(r"\b(?:pause|sleep|wait|tomorrow|later|delay|postpone|defer)\b", re.I)


def extract_resources(text: str) -> Tuple[float, float]:
    """Return a simple (load, privacy) pair from regex counts."""
    load = len(EVIDENCE_RE.findall(text))
    privacy = len(PLANNING_RE.findall(text))
    # Scale to small positive floats to avoid zeros in later soft‑max
    return float(load + 1), float(privacy + 1)


# ----------------------------------------------------------------------
# Parent A – dummy high‑dimensional feature extraction
# ----------------------------------------------------------------------
def extract_features(text: str, dim: int = 5) -> np.ndarray:
    """Return a deterministic pseudo‑random vector based on text hash."""
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    return rng.random(dim)


# ----------------------------------------------------------------------
# Parent A – augmented path and truncated signature
# ----------------------------------------------------------------------
def build_augmented_path(texts: List[str]) -> np.ndarray:
    """Create an (n, 2+d) array where each row is [L, P, v...]."""
    points = []
    for t in texts:
        L, P = extract_resources(t)
        v = extract_features(t)
        points.append(np.concatenate(([L, P], v)))
    return np.stack(points)


def compute_path_signature(points: np.ndarray, level: int = 2) -> np.ndarray:
    """
    Truncated signature up to `level`.
    Level‑1: sum of increments (Δp).
    Level‑2: sum of outer products of successive increments.
    Returns a 1‑D vector concatenating level‑1 and flattened level‑2.
    """
    if points.shape[0] < 2:
        raise ValueError("At least two points required for signature.")
    increments = np.diff(points, axis=0)  # shape (n‑1, m)
    # Level‑1
    level1 = increments.sum(axis=0)  # shape (m,)
    # Level‑2
    if level >= 2:
        # Compute Σ_i Δp_i ⊗ Δp_i (outer product) and flatten
        outer_sum = np.einsum("ij,ik->jk", increments, increments)
        level2 = outer_sum.ravel()
        signature = np.concatenate((level1, level2))
    else:
        signature = level1
    return signature


# ----------------------------------------------------------------------
# Parent B – data structures
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


# ----------------------------------------------------------------------
# Parent B – regret‑weighted strategy
# ----------------------------------------------------------------------
def compute_regret_weighted_strategy(
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual] | None = None,
) -> Dict[str, float]:
    """Soft‑max over (expected − cost − risk + counterfactual) with regret scaling."""
    if not actions:
        return {}
    cf = {c.action_id: c.outcome_value * c.probability for c in (counterfactuals or [])}
    vals = {
        a.id: a.expected_value - a.cost - a.risk + cf.get(a.id, 0.0) for a in actions
    }
    best = max(vals.values())
    # Regret‑scaled exponentials
    weights = {k: math.exp(v - best) for k, v in vals.items()}
    total = sum(weights.values()) or 1.0
    return {k: w / total for k, w in weights.items()}


def shannon_entropy(probabilities: Iterable[float]) -> float:
    """Base‑2 Shannon entropy."""
    probs = [p for p in probabilities if p > 0]
    return -sum(p * math.log(p, 2) for p in probs)


def gini_coefficient(values: Iterable[float]) -> float:
    """Gini coefficient for non‑negative values."""
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0:
        return 0.0
    n = len(xs)
    cumulative = sum((2 * i - n - 1) * x for i, x in enumerate(xs, 1))
    return cumulative / (n * sum(xs))


# ----------------------------------------------------------------------
# Fusion – bridging the two parents
# ----------------------------------------------------------------------
def strategy_from_signature(
    signature: np.ndarray,
    resources: np.ndarray,
    counterfactuals: List[MathCounterfactual] | None = None,
) -> Tuple[Dict[str, float], float, float]:
    """
    Treat each component of `signature` as an action.
    Expected value = component value.
    Cost = corresponding load (first column of resources) normalized.
    Risk = corresponding privacy (second column) normalized.
    Returns (probability distribution, entropy, gini).
    """
    n_actions = signature.size
    # Align resources: repeat or truncate to match signature length
    loads = np.interp(
        np.linspace(0, resources.shape[0] - 1, n_actions),
        np.arange(resources.shape[0]),
        resources[:, 0],
    )
    privs = np.interp(
        np.linspace(0, resources.shape[0] - 1, n_actions),
        np.arange(resources.shape[0]),
        resources[:, 1],
    )
    actions = [
        MathAction(
            id=f"sig_{i}",
            expected_value=float(sig),
            cost=float(l),
            risk=float(p),
        )
        for i, (sig, l, p) in enumerate(zip(signature, loads, privs))
    ]
    probs = compute_regret_weighted_strategy(actions, counterfactuals)
    entropy = shannon_entropy(probs.values())
    gini = gini_coefficient(list(probs.values()))
    return probs, entropy, gini


def entropy_scaled_greedy_selection(
    resources: np.ndarray,
    budget: float,
    entropy: float,
    alpha: float = 0.5,
) -> List[int]:
    """
    Greedy selector that picks indices with smallest (load + privacy) ratio.
    The effective budget is inflated by `1 + alpha * entropy` to allow more
    selections when uncertainty (entropy) is high.
    """
    effective_budget = budget * (1.0 + alpha * entropy)
    ratios = (resources[:, 0] + resources[:, 1]) / (resources[:, 0] + 1e-9)
    order = np.argsort(ratios)
    selected = []
    spent = 0.0
    for idx in order:
        cost = resources[idx, 0] + resources[idx, 1]
        if spent + cost <= effective_budget:
            selected.append(int(idx))
            spent += cost
    return selected


def hybrid_fusion_pipeline(texts: List[str], budget: float) -> Dict[str, object]:
    """
    End‑to‑end pipeline:
    1. Build augmented path from texts.
    2. Compute truncated signature.
    3. Derive regret‑weighted strategy from signature.
    4. Compute entropy and Gini.
    5. Perform entropy‑scaled greedy budget selection.
    Returns a dictionary with all intermediate results.
    """
    points = build_augmented_path(texts)                 # (n, m)
    signature = compute_path_signature(points, level=2) # 1‑D vector
    probs, entropy, gini = strategy_from_signature(signature, points)
    selected = entropy_scaled_greedy_selection(points, budget, entropy)
    return {
        "points": points,
        "signature": signature,
        "strategy": probs,
        "entropy": entropy,
        "gini": gini,
        "selected_indices": selected,
    }


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_texts = [
        "The evidence was verified and the plan is to proceed after a short pause.",
        "We need to document the source and schedule the next test.",
        "Audit log shows the hash and confirms the procedure.",
        "Delay the rollout until tomorrow; prioritize privacy concerns.",
        "Checklist completed, budget approved, and the roadmap is clear.",
    ]
    result = hybrid_fusion_pipeline(sample_texts, budget=30.0)
    print("Signature shape:", result["signature"].shape)
    print("Strategy (first 5):", dict(list(result["strategy"].items())[:5]))
    print("Entropy:", result["entropy"])
    print("Gini:", result["gini"])
    print("Selected indices:", result["selected_indices"])