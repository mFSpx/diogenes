# DARWIN HAMMER — match 5375, survivor 0
# gen: 6
# parent_a: hybrid_decision_hygiene_shannon_entropy_m12_s1.py (gen1)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_decisi_m855_s0.py (gen5)
# born: 2026-05-30T00:01:35Z

"""Hybrid Decision Hygiene & Entropy–KL Fusion

Parents:
- hybrid_decision_hygiene_shannon_entropy_m12_s1.py (Decision Hygiene + Shannon Entropy)
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_decisi_m855_s0.py (KL Divergence + Pruning)

Mathematical Bridge:
The decision‑hygiene stage yields a categorical count vector **c** whose
Shannon entropy **H(c)** quantifies the uncertainty of the textual
decision context.  The KL‑divergence stage works with probability
distributions **p** (e.g. action‑signature vectors) and a target
distribution **q**.  We fuse the two by letting the entropy weight the
KL term:

    D̂_KL(p‖q; H) = (1 + H) · Σ_i p_i · log(p_i / q_i)

The weighted divergence then drives the pruning schedule:
signatures are multiplied element‑wise by a schedule that is scaled by
the weighted KL values.  This creates a single unified system where
textual complexity directly influences probabilistic pruning.

The module implements three core hybrid operations:
1. `decision_hygiene_entropy(text)` – extracts counts & computes H.
2. `weighted_kl_divergence(signatures, target, entropy)` – KL weighted by H.
3. `hybrid_prune(signatures, schedule, weighted_kl)` – applies entropy‑aware pruning.
"""

import re
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Decision Hygiene regexes and count extraction
# ----------------------------------------------------------------------
EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|"
    r"sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)

PLANNING_RE = re.compile(
    r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|"
    r"triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
    re.I,
)

DELAY_RE = re.compile(
    r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|"
    r"before i|first|after|review)\b",
    re.I,
)

SUPPORT_RE = re.compile(
    r"\b(?:ask|call|text|friend|friends|rowyn|kai|chance|doctor|therapist|lawyer|"
    r"advocate|support|help|community|team|handoff|delegate)\b",
    re.I,
)

BOUNDARY_RE = re.compile(
    r"\b(?:boundary|boundaries|walk away|no contact|do not|don't|stop|limit|refuse|"
    r"protect|safe|safety|lock|redact|privacy|permission|consent)\b",
    re.I,
)

OUTCOME_RE = re.compile(
    r"\b(?:done|shipped|finished|complete|resolved|succeeded|passed|paid|safe|sent|"
    r"filed|closed|fixed|working|green|verified)\b",
    re.I,
)

IMPULSIVE_RE = re.compile(
    r"\b(?:rage|impulsive|panic|panicking|spiral|doom|fuck it|right now|immediately|"
    r"tonight|destroy|revenge|scorched|burn it|quit|give up)\b",
    re.I,
)

SCARCITY_RE = re.compile(
    r"\b(?:broke|homeless|last\s+\$?\d+|\$\s?\d+\s+to my name|can't afford|cannot afford|"
    r"rent due|no money|starving|desperate|no sleep|exhausted)\b",
    re.I,
)

RISK_RE = re.compile(
    r"\b(?:kill|die|suicide|suicidal|self[- ]?harm|overdose|hurt myself|danger|unsafe|"
    r"crisis|collapse)\b",
    re.I,
)

def counts(text: str) -> Dict[str, int]:
    """Return a dictionary of category counts extracted from *text*."""
    return {
        "evidence_count": len(EVIDENCE_RE.findall(text or "")),
        "planning_count": len(PLANNING_RE.findall(text or "")),
        "delay_count": len(DELAY_RE.findall(text or "")),
        "support_count": len(SUPPORT_RE.findall(text or "")),
        "boundary_count": len(BOUNDARY_RE.findall(text or "")),
        "outcome_count": len(OUTCOME_RE.findall(text or "")),
        "impulsive_count": len(IMPULSIVE_RE.findall(text or "")),
        "scarcity_count": len(SCARCITY_RE.findall(text or "")),
        "risk_count": len(RISK_RE.findall(text or "")),
    }

def shannon_entropy_from_counts(cnt: Dict[str, int]) -> float:
    """Compute Shannon entropy (base‑2) of the non‑zero count distribution."""
    values = np.array(list(cnt.values()), dtype=float)
    total = values.sum()
    if total == 0:
        return 0.0
    probs = values[values > 0] / total
    return -np.sum(probs * np.log2(probs))

def decision_hygiene_entropy(text: str) -> float:
    """Convenience wrapper: extract counts from *text* and return H."""
    return shannon_entropy_from_counts(counts(text))

# ----------------------------------------------------------------------
# Parent B – KL divergence, lead‑lag transform, pruning utilities
# ----------------------------------------------------------------------
def lead_lag_transform(X: np.ndarray) -> np.ndarray:
    """Create a feature vector consisting of linear and quadratic sums per row."""
    linear = np.sum(X, axis=1, keepdims=True)
    quadratic = np.sum(X ** 2, axis=1, keepdims=True)
    return np.concatenate((linear, quadratic), axis=1)

def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute KL(p‖q) with natural logarithm.
    Both *p* and *q* must be 1‑D probability vectors (sum to 1, non‑negative).
    Zero entries in *p* are ignored (0·log0 = 0).  Zero entries in *q* where
    p>0 raise a ValueError because the divergence would be infinite.
    """
    if p.shape != q.shape:
        raise ValueError("p and q must have the same shape")
    mask = p > 0
    if np.any((q[mask] == 0)):
        raise ValueError("KL divergence undefined for q_i == 0 where p_i > 0")
    return np.sum(p[mask] * np.log(p[mask] / q[mask]))

def weighted_kl_divergence(
    signatures: np.ndarray,
    target: np.ndarray,
    entropy: float,
) -> np.ndarray:
    """
    Compute a weighted KL divergence for each row of *signatures* against
    *target*.  The weight is (1 + entropy) as described in the module docstring.
    Returns a 1‑D array of weighted divergences.
    """
    # Normalize each signature row to a probability distribution
    row_sums = signatures.sum(axis=1, keepdims=True)
    # Avoid division by zero – rows that sum to zero become uniform
    row_sums[row_sums == 0] = 1.0
    probs = signatures / row_sums

    # Ensure target is a proper distribution
    target = np.asarray(target, dtype=float)
    target = target / target.sum()

    weighted = np.empty(probs.shape[0], dtype=float)
    factor = 1.0 + entropy
    for i, p_vec in enumerate(probs):
        weighted[i] = factor * kl_divergence(p_vec, target)
    return weighted

def prune_candidates(signatures: np.ndarray, schedule: np.ndarray) -> np.ndarray:
    """
    Element‑wise multiplication of *signatures* by *schedule*.
    *schedule* can be a 1‑D vector (broadcast across columns) or a matrix
    matching *signatures* shape.
    """
    return signatures * schedule

def hybrid_prune(
    signatures: np.ndarray,
    schedule: np.ndarray,
    weighted_kl: np.ndarray,
) -> np.ndarray:
    """
    Apply entropy‑aware pruning:
    - Scale *schedule* by the normalized weighted KL values.
    - Broadcast the scaling factor across columns.
    """
    if weighted_kl.ndim != 1:
        raise ValueError("weighted_kl must be a 1‑D array")
    # Normalise weighted KL to [0,1] for safe scaling
    max_val = weighted_kl.max()
    scale = weighted_kl / max_val if max_val != 0 else weighted_kl
    scale = scale[:, np.newaxis]  # shape (n_rows, 1)
    adjusted_schedule = schedule * scale
    return prune_candidates(signatures, adjusted_schedule)

# ----------------------------------------------------------------------
# Hybrid Operations (exposed API)
# ----------------------------------------------------------------------
def hybrid_operation(
    text: str,
    signatures: np.ndarray,
    schedule: np.ndarray,
    target_distribution: np.ndarray,
) -> Dict[str, Any]:
    """
    Full hybrid pipeline:
    1. Compute decision‑hygiene entropy H from *text*.
    2. Compute weighted KL divergences between *signatures* and *target_distribution*.
    3. Prune *signatures* using the entropy‑aware schedule.
    4. Return a dictionary with intermediate and final results.
    """
    entropy = decision_hygiene_entropy(text)
    weighted_kl = weighted_kl_divergence(signatures, target_distribution, entropy)
    pruned = hybrid_prune(signatures, schedule, weighted_kl)

    return {
        "entropy": entropy,
        "weighted_kl": weighted_kl,
        "pruned_signatures": pruned,
    }

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Sample textual input
    sample_text = (
        "We have evidence that the plan was executed, but there is a delay. "
        "Please verify the screenshot and confirm the outcome. "
        "If risk arises, we should support the team and set a new boundary."
    )

    # Synthetic signatures matrix (5 candidates × 4 features)
    np.random.seed(42)
    signatures = np.random.rand(5, 4)

    # Random schedule matrix (same shape)
    schedule = np.random.rand(5, 4)

    # Target distribution (4‑dimensional)
    target = np.array([0.25, 0.25, 0.25, 0.25])

    result = hybrid_operation(sample_text, signatures, schedule, target)

    print("Decision‑hygiene entropy (bits):", result["entropy"])
    print("Weighted KL divergences:", result["weighted_kl"])
    print("Pruned signatures matrix:\n", result["pruned_signatures"])