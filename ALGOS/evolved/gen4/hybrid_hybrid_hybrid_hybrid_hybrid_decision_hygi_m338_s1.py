# DARWIN HAMMER — match 338, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_hybrid_worksh_hybrid_sheaf_cohomol_m179_s2.py (gen3)
# parent_b: hybrid_decision_hygiene_shannon_entropy_m12_s5.py (gen1)
# born: 2026-05-29T23:28:22Z

"""Hybrid allocation‑sheaf & decision‑hygiene module.

Parents:
- **Parent A** – weekday‑dependent weight vector, deterministic/residual resource
  allocation, and a trivial sheaf on a group graph with coboundary norm.
- **Parent B** – regex‑driven textual feature extraction, positive/negative cue
  weighting, and a Shannon‑entropy based quality metric.

Mathematical bridge:
The weight vector **w ∈ ℝⁿ** (n = number of groups) is a row‑stochastic linear
map that sends any scalar **r** (e.g. a residual resource or a feature‑derived
score) to the allocation vector **r·w**.  Treating the allocation as a sheaf
section over the group graph, the coboundary operator **δ** reduces to edge‑wise
differences **δ(a)_e = a_i – a_j**.  The L²‑norm **‖δ(a)‖** quantifies allocation
coherence, while the Shannon entropy of the normalized allocation measures
distributional fairness.  The hybrid algorithm therefore
1. extracts a scalar cue **s** from text (Parent B),
2. maps **s** linearly onto the group space via **w** (Parent A),
3. evaluates sheaf consistency and entropy on the resulting section.

The functions below implement this fused workflow.
"""

from __future__ import annotations

import datetime as dt
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Constants shared by both parents
# ----------------------------------------------------------------------
GROUPS: Tuple[str, ...] = ("codex", "groq", "cohere", "local_models")
# Simple undirected chain graph for the sheaf
EDGES: List[Tuple[str, str]] = [
    ("codex", "groq"),
    ("groq", "cohere"),
    ("cohere", "local_models"),
]

# ----------------------------------------------------------------------
# Parent A – weekday weight vector & allocation utilities
# ----------------------------------------------------------------------
def _pct(value: float) -> float:
    """Round a float to six decimal places."""
    return round(float(value), 6)


def weekday_weight_vector(groups: Sequence[str], date: dt.date) -> np.ndarray:
    """
    Produce a row‑stochastic weight vector based on the day of week.

    For illustration we assign a raw weight `w_i = 1 + dow` to every group,
    where `dow` is the weekday index (0 = Sunday … 6 = Saturday).  The vector
    is then normalised to sum to 1.
    """
    dow = (date.weekday() + 1) % 7  # 0 = Sunday … 6 = Saturday
    raw = np.full(len(groups), 1 + dow, dtype=np.float64)
    norm = raw / raw.sum()
    return np.vectorize(_pct)(norm)


def allocate_total_resource(total: int, weight_vec: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Split `total` into a deterministic chunk (60 % rounded down) and a residual.
    The residual is distributed across groups by the weight vector.
    Returns the allocation vector (deterministic + residual) and the residual amount.
    """
    deterministic = int(math.floor(0.6 * total))
    residual = total - deterministic
    # deterministic part is equally split (could also use weight_vec)
    det_vec = np.full_like(weight_vec, deterministic / len(weight_vec), dtype=np.float64)
    # residual part via linear map r·w
    res_vec = residual * weight_vec
    allocation = det_vec + res_vec
    return allocation, residual


def sheaf_coboundary_norm(allocation: np.ndarray, edges: List[Tuple[str, str]]) -> float:
    """
    Compute the L2 norm of the coboundary of the allocation section.
    Nodes are ordered as in GROUPS.
    """
    idx = {name: i for i, name in enumerate(GROUPS)}
    diffs = []
    for u, v in edges:
        diffs.append(allocation[idx[u]] - allocation[idx[v]])
    return float(np.linalg.norm(diffs))


def shannon_entropy(vector: np.ndarray) -> float:
    """
    Shannon entropy of a probability distribution derived from `vector`.
    Zero entries are ignored.
    """
    prob = vector / vector.sum()
    prob = prob[prob > 0]
    return -float(np.sum(prob * np.log2(prob)))


# ----------------------------------------------------------------------
# Parent B – regex feature extraction & cue scoring
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
    r"\b(?:done|shipped|finished|complete|resolved|succeeded|passed|paid|safe|sent|filed|closed|fixed|working|green|verified)\b",
    re.I,
)
IMPULSIVE_RE = re.compile(
    r"\b(?:rage|impulsive|panic|panicking|spiral|doom|fuck it|right now|immediately|tonight|destroy|revenge|scorched|burn it|quit|give up)\b",
    re.I,
)
SCARCITY_RE = re.compile(
    r"\b(?:broke|homeless|last\s+\$?\d+|\$\s?\d+\s+to my name|can't afford|cannot afford|rent due|no money|starving|desperate|no sleep|exhausted)\b",
    re.I,
)
RISK_RE = re.compile(
    r"\b(?:kill|die|suicide|suicidal|self[- ]?harm|overdose|hurt myself|danger|unsafe|crisis|collapse)\b",
    re.I,
)

_FEATURE_ORDER = [
    "evidence",
    "planning",
    "delay",
    "support",
    "boundary",
    "outcome",
    "impulsive",
    "scarcity",
    "risk",
]

_POSITIVE_WEIGHTS = np.array([1600, 1200, 1400, 1000, 1200, 800, 0, 0, 0], dtype=np.int64)
_NEGATIVE_WEIGHTS = np.array([0, 0, 0, 0, 0, 0, 2000, 1800, 2500], dtype=np.int64)


def _count_matches(pattern: re.Pattern, text: str) -> int:
    return len(pattern.findall(text))


def extract_feature_counts(text: str) -> Dict[str, int]:
    """
    Return a dictionary mapping each feature name to the number of regex matches
    found in `text`.
    """
    return {
        "evidence": _count_matches(EVIDENCE_RE, text),
        "planning": _count_matches(PLANNING_RE, text),
        "delay": _count_matches(DELAY_RE, text),
        "support": _count_matches(SUPPORT_RE, text),
        "boundary": _count_matches(BOUNDARY_RE, text),
        "outcome": _count_matches(OUTCOME_RE, text),
        "impulsive": _count_matches(IMPULSIVE_RE, text),
        "scarcity": _count_matches(SCARCITY_RE, text),
        "risk": _count_matches(RISK_RE, text),
    }


def compute_cue_score(counts: Dict[str, int]) -> float:
    """
    Linear combination of counts with positive and negative weights.
    Positive cues increase the score, negative cues decrease it.
    """
    vec = np.array([counts.get(name, 0) for name in _FEATURE_ORDER], dtype=np.float64)
    pos = _POSITIVE_WEIGHTS.astype(np.float64)
    neg = _NEGATIVE_WEIGHTS.astype(np.float64)
    return float(np.dot(pos, vec) - np.dot(neg, vec))


# ----------------------------------------------------------------------
# Hybrid operation
# ----------------------------------------------------------------------
def hybrid_process(
    text: str,
    total_resource: int,
    on_date: dt.date | None = None,
) -> Dict[str, object]:
    """
    End‑to‑end hybrid pipeline:

    1. Build a weekday‑dependent weight vector `w`.
    2. Extract textual feature counts and compute a cue score `s`.
    3. Allocate `total_resource` deterministically and distribute the residual `r`.
    4. Add the cue‑derived allocation `s·w` to the deterministic allocation.
    5. Evaluate sheaf coboundary norm and Shannon entropy of the final allocation.

    Returns a dictionary with intermediate and final results.
    """
    if on_date is None:
        on_date = dt.date.today()

    # 1. Weight vector
    w = weekday_weight_vector(GROUPS, on_date)

    # 2. Feature extraction & cue score
    counts = extract_feature_counts(text)
    cue_score = compute_cue_score(counts)

    # 3. Base allocation from total resource
    base_alloc, residual = allocate_total_resource(total_resource, w)

    # 4. Map cue score (treated as an additional residual) onto groups
    cue_allocation = cue_score * w
    final_alloc = base_alloc + cue_allocation

    # 5. Metrics
    cob_norm = sheaf_coboundary_norm(final_alloc, EDGES)
    entropy = shannon_entropy(final_alloc)

    return {
        "date": on_date.isoformat(),
        "weight_vector": w,
        "feature_counts": counts,
        "cue_score": cue_score,
        "base_allocation": base_alloc,
        "cue_allocation": cue_allocation,
        "final_allocation": final_alloc,
        "coboundary_norm": cob_norm,
        "entropy": entropy,
    }


# ----------------------------------------------------------------------
# Additional demonstration functions
# ----------------------------------------------------------------------
def demo_weight_vector():
    """Print the weight vector for today."""
    w = weekday_weight_vector(GROUPS, dt.date.today())
    print("Weekday weight vector:", w)


def demo_feature_extraction():
    """Run feature extraction on a sample sentence."""
    sample = (
        "We have evidence and a plan, but there is a delay. "
        "Support from friends is needed, and we must avoid risk."
    )
    counts = extract_feature_counts(sample)
    print("Feature counts:", counts)
    print("Cue score:", compute_cue_score(counts))


def demo_full_pipeline():
    """Execute the full hybrid process with synthetic inputs."""
    text = (
        "The team provided evidence and a detailed checklist. "
        "However, there is a delay and scarcity of resources, "
        "so we need support and must watch out for risk."
    )
    result = hybrid_process(text, total_resource=1000)
    for k, v in result.items():
        if isinstance(v, np.ndarray):
            v = v.tolist()
        print(f"{k}: {v}")


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Demo: Weight Vector ===")
    demo_weight_vector()
    print("\n=== Demo: Feature Extraction ===")
    demo_feature_extraction()
    print("\n=== Demo: Full Hybrid Pipeline ===")
    demo_full_pipeline()