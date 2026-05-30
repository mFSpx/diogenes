# DARWIN HAMMER — match 2604, survivor 0
# gen: 4
# parent_a: hybrid_hybrid_hybrid_worksh_hybrid_hybrid_sketch_m135_s0.py (gen3)
# parent_b: hybrid_ternary_lens_audit_decreasing_pruning_m15_s1.py (gen1)
# born: 2026-05-29T23:43:05Z

"""Hybrid Allocation‑Audit‑Sheaf Fusion

Parents:
- hybrid_hybrid_hybrid_worksh_hybrid_hybrid_sketch_m135_s0.py (allocation + sheaf cohomology)
- hybrid_ternary_lens_audit_decreasing_pruning_m15_s1.py (ternary lens audit + decreasing‑rate pruning)

Mathematical bridge:
The allocation routine produces a 0‑cochain **s** ∈ ℝⁿ over the set of groups G.
The audit routine yields a penalty vector **p** ∈ ℝⁿ where each entry aggregates
audit findings for the candidates belonging to the corresponding group.
We fuse them by forming a weighted section **w = s ⊙ p** (Hadamard product).
The coboundary operator δ : C⁰ → C¹ (implemented as a matrix B) maps **w**
to edge‑wise differences; its L²‑norm ‖δ w‖₂ quantifies topological inconsistency.
A decreasing‑rate pruning factor λ(step) attenuates the penalties before the
Hadamard product, giving a time‑varying hybrid score.  Candidates whose
group‑score exceeds a residual‑derived threshold are kept.

The module provides:
1. weekday_weight_vector – weekday‑dependent scalar weights.
2. allocate_hybrid – deterministic allocation per group using (1).
3. audit_penalty_vector – aggregates audit findings per group.
4. hybrid_prune – applies decreasing‑rate pruning, builds the weighted
   section, computes the sheaf residual, and selects survivors.

All operations rely only on numpy and the Python standard library.
"""

import datetime as dt
import hashlib
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Shared constants
# ----------------------------------------------------------------------
GROUPS: Tuple[str, ...] = ("codex", "groq", "cohere", "local_models")
MAX64 = (1 << 64) - 1
CLASSIFICATIONS = {
    "usable_now",
    "research_only",
    "needs_conversion",
    "unsafe_for_fastpath",
    "unsupported",
}
LOCAL_PATTERNS = [
    "*bitnet*",
    "*BitNet*",
    "*fairyfuse*",
    "*FairyFuse*",
    "*lora*",
    "*LoRA*",
    "*adapter*",
]

# ----------------------------------------------------------------------
# Parent A – allocation utilities
# ----------------------------------------------------------------------


def _pct(value: float) -> float:
    """Round a float to six decimal places (consistent with Parent A)."""
    return round(float(value), 6)


def doomsday(year: int, month: int, day: int) -> int:
    """Return a 0‑based weekday index (Monday=0 … Sunday=6) using
    Conway's Doomsday algorithm."""
    # Anchor days for centuries
    anchors = [2, 0, 5, 3]  # 1800,1900,2000,2100 → Tuesday, Sunday, Friday, Wednesday
    century = year // 100
    anchor = anchors[century % 4]

    # Year’s last two digits
    y = year % 100
    y_div12 = y // 12
    y_mod12 = y % 12
    y_div4 = y_mod12 // 4

    doom = (anchor + y_div12 + y_mod12 + y_div4) % 7

    # Month offsets (non‑leap year)
    month_offsets = [0, 3, 0, 3, 2, 5, 0, 3, 6, 1, 4, 6]
    offset = month_offsets[month - 1]

    # Leap year correction
    leap = int((year % 4 == 0) and (year % 100 != 0 or year % 400 == 0))
    if month < 3:
        offset -= leap

    return (doom + offset + day) % 7


def weekday_weight_vector(ref_date: dt.date | None = None) -> np.ndarray:
    """Build a 4‑element weight vector for the groups based on the weekday.

    The original algorithm uses a deterministic hash of the weekday index;
    we reproduce that behaviour.
    """
    if ref_date is None:
        ref_date = dt.date.today()
    wday = doomsday(ref_date.year, ref_date.month, ref_date.day)

    # Derive a 64‑bit integer from the weekday using a simple hash.
    h = hashlib.sha256(str(wday).encode()).digest()
    seed = int.from_bytes(h[:8], "little") & MAX64
    rng = random.Random(seed)

    # Generate four positive weights that sum to 1.
    raw = np.array([rng.random() for _ in GROUPS], dtype=float)
    weights = raw / raw.sum()
    return np.vectorize(_pct)(weights)


def allocate_hybrid() -> np.ndarray:
    """Deterministic allocation per group using the weekday weight vector.

    Returns a 1‑D numpy array of length ``len(GROUPS)`` whose entries sum to 1.
    """
    return weekday_weight_vector()


# ----------------------------------------------------------------------
# Parent B – audit and decreasing‑rate pruning utilities
# ----------------------------------------------------------------------


def utc_now() -> str:
    """Current UTC time in ISO‑8601 format (trailing Z)."""
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _matches_local_pattern(text: str) -> bool:
    """Return True if *text* matches any glob in ``LOCAL_PATTERNS``."""
    # Convert simple glob patterns to regex.
    regexes = [re.compile("^" + p.replace("*", ".*") + "$", re.IGNORECASE) for p in LOCAL_PATTERNS]
    return any(r.search(text) for r in regexes)


def audit_findings(candidate: Dict[str, Any]) -> List[str]:
    """Evaluate a single lens candidate and return a list of textual findings.

    The logic mirrors the original ternary lens audit: we flag violations
    based on key/family patterns and classification constraints.
    """
    findings: List[str] = []
    key = candidate.get("candidate_key", "")
    family = candidate.get("family", "")
    classification = candidate.get("classification", "")

    # Rule 1 – standard LoRA families must be marked unsafe_for_fastpath.
    if re.search(r"standard.*lora|peft|qlora", f"{key} {family}", re.I):
        if classification != "unsafe_for_fastpath" or not candidate.get("fast_path_compatible", False):
            findings.append("STANDARD_LORA_RULE_VIOLATION")

    # Rule 2 – local‑model patterns must be classified as usable_now.
    if _matches_local_pattern(key) or _matches_local_pattern(family):
        if classification != "usable_now":
            findings.append("LOCAL_MODEL_CLASSIFICATION_MISMATCH")

    # Rule 3 – any unsupported classification is a hard stop.
    if classification == "unsupported":
        findings.append("UNSUPPORTED_CLASSIFICATION")

    return findings


def decreasing_prune_factor(step: int, max_steps: int, base: float = 0.85) -> float:
    """Exponential decay factor λ(step) ∈ (0,1] decreasing with *step*."""
    if step < 0:
        raise ValueError("step must be non‑negative")
    # Normalise step to [0,1] then apply exponential decay.
    normalized = min(step / max_steps, 1.0)
    return base ** normalized


# ----------------------------------------------------------------------
# Hybrid core – sheaf construction, weighting, and pruning
# ----------------------------------------------------------------------


def build_coboundary_matrix(groups: Tuple[str, ...]) -> np.ndarray:
    """Return the coboundary matrix B for the complete graph on *groups*.

    For n groups there are n·(n‑1)/2 edges.  Each row corresponds to an edge
    (i,j) with +1 at column i and –1 at column j.
    """
    n = len(groups)
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    B = np.zeros((len(edges), n), dtype=float)
    for row, (i, j) in enumerate(edges):
        B[row, i] = 1.0
        B[row, j] = -1.0
    return B


def sheaf_residual(section: np.ndarray, coboundary: np.ndarray) -> float:
    """Compute the L2 norm of δ section."""
    diff = coboundary @ section
    return float(np.linalg.norm(diff, 2))


def audit_penalty_vector(candidates: List[Dict[str, Any]]) -> np.ndarray:
    """Aggregate audit findings per group into a penalty vector.

    Each finding contributes a unit penalty.  Candidates are mapped to groups
    via a simple heuristic based on their classification:
        usable_now            → local_models
        research_only        → cohere
        needs_conversion      → groq
        unsafe_for_fastpath   → codex
        unsupported           → codex (hard penalty)
    """
    mapping = {
        "usable_now": "local_models",
        "research_only": "cohere",
        "needs_conversion": "groq",
        "unsafe_for_fastpath": "codex",
        "unsupported": "codex",
    }
    penalties = np.zeros(len(GROUPS), dtype=float)
    group_index = {g: i for i, g in enumerate(GROUPS)}

    for cand in candidates:
        cls = cand.get("classification", "")
        grp = mapping.get(cls, "codex")
        idx = group_index[grp]
        findings = audit_findings(cand)
        penalties[idx] += len(findings)  # each finding adds 1.0 penalty

    return penalties


def hybrid_prune(
    candidates: List[Dict[str, Any]],
    step: int,
    max_steps: int,
    allocation: np.ndarray | None = None,
) -> Tuple[List[Dict[str, Any]], float]:
    """Hybrid pruning pipeline.

    1. Compute (or receive) a deterministic allocation vector **s**.
    2. Derive audit penalties **p** per group.
    3. Apply a decreasing‑rate factor λ(step) to the penalties.
    4. Form the weighted section **w = s ⊙ (λ · p)**.
    5. Compute the sheaf residual r = ‖δ w‖₂.
    6. Keep candidates whose group‑score (s_i · λ · p_i) exceeds r / √|G|.

    Returns the list of survivors and the residual value.
    """
    if allocation is None:
        allocation = allocate_hybrid()
    else:
        allocation = np.asarray(allocation, dtype=float)

    # Step 2‑3: audit penalties with decay
    raw_penalties = audit_penalty_vector(candidates)
    decay = decreasing_prune_factor(step, max_steps)
    weighted_penalties = decay * raw_penalties

    # Step 4: weighted section (Hadamard product)
    weighted_section = allocation * weighted_penalties

    # Step 5: residual
    B = build_coboundary_matrix(GROUPS)
    residual = sheaf_residual(weighted_section, B)

    # Step 6: thresholding
    threshold = residual / math.sqrt(len(GROUPS))
    survivors: List[Dict[str, Any]] = []
    group_index = {g: i for i, g in enumerate(GROUPS)}
    for cand in candidates:
        cls = cand.get("classification", "")
        grp = {
            "usable_now": "local_models",
            "research_only": "cohere",
            "needs_conversion": "groq",
            "unsafe_for_fastpath": "codex",
            "unsupported": "codex",
        }.get(cls, "codex")
        score = allocation[group_index[grp]] * weighted_penalties[group_index[grp]]
        if score >= threshold:
            survivors.append(cand)

    return survivors, residual


# ----------------------------------------------------------------------
# Demonstration / smoke test
# ----------------------------------------------------------------------


def _demo_candidates() -> List[Dict[str, Any]]:
    """Create a tiny manifest of synthetic candidates for the smoke test."""
    return [
        {
            "candidate_key": "standard_lora_v1",
            "family": "peft",
            "classification": "unsafe_for_fastpath",
            "fast_path_compatible": False,
        },
        {
            "candidate_key": "my_bitnet_adapter",
            "family": "adapter_family",
            "classification": "usable_now",
            "fast_path_compatible": True,
        },
        {
            "candidate_key": "experimental_cohere",
            "family": "cohere_v2",
            "classification": "research_only",
            "fast_path_compatible": True,
        },
        {
            "candidate_key": "legacy_model",
            "family": "old",
            "classification": "unsupported",
            "fast_path_compatible": False,
        },
    ]


if __name__ == "__main__":
    # 1. Allocation (weekday‑dependent)
    alloc = allocate_hybrid()
    print("Allocation vector:", dict(zip(GROUPS, map(_pct, alloc))))

    # 2. Audit penalties
    cands = _demo_candidates()
    penalties = audit_penalty_vector(cands)
    print("Raw audit penalties per group:", dict(zip(GROUPS, penalties)))

    # 3. Hybrid pruning at step 3 of a 10‑step schedule
    survivors, resid = hybrid_prune(candidates=cands, step=3, max_steps=10, allocation=alloc)
    print(f"Sheaf residual: {resid:.6f}")
    print("Surviving candidates:")
    for s in survivors:
        print(" -", s["candidate_key"])