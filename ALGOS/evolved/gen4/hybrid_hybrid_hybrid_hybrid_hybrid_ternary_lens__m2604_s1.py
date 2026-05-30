# DARWIN HAMMER — match 2604, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_hybrid_worksh_hybrid_hybrid_sketch_m135_s0.py (gen3)
# parent_b: hybrid_ternary_lens_audit_decreasing_pruning_m15_s1.py (gen1)
# born: 2026-05-29T23:43:05Z

"""Hybrid Allocation‑Sheaf‑Audit‑Pruning

This module fuses the *weekday‑weighted allocation* and *sheaf‑cohomology* logic
from ``hybrid_hybrid_hybrid_worksh_hybrid_hybrid_sketch_m135_s0.py`` (Parent A)
with the *ternary lens audit* and *decreasing‑rate pruning* machinery from
``hybrid_ternary_lens_audit_decreasing_pruning_m15_s1.py`` (Parent B).

Mathematical bridge
-------------------
* Parent A* produces a scalar allocation `s ∈ ℝⁿ` for a set of groups.
  Building the edge‑incidence (coboundary) matrix `Δ ∈ {‑1,0,1}^{m×n}` yields the
  1‑cochain `r = Δ s`, i.e. the vector of pairwise residuals.
* Parent B* evaluates a collection of lens candidates and assigns each an
  *audit score* `a_i ∈ ℕ` (the number of rule violations).  The pruning schedule
  is a decreasing probability `p(k) = p₀ / (1 + k)` after `k` iterations.

The hybrid algorithm treats the sheaf residual norm `‖r‖₂` as a *global risk
factor* that modulates the base pruning probability.  Concretely, the pruning
probability for candidate *i* at iteration *k* becomes


p_i(k) = (p₀ / (1 + k)) * (1 + β·‖r‖₂)   (clipped to [0,1])


Thus a highly inconsistent allocation (large residual) makes the system more
aggressive in discarding risky lens candidates, while a consistent allocation
keeps the pruning gentle.

The public API offers three representative functions that showcase the hybrid
behaviour:

1. ``weekday_weight_vector`` – builds the weekday‑dependent weight vector.
2. ``allocate_and_residual`` – performs allocation, builds the sheaf, and
   returns both the allocation and its residual norm.
3. ``audit_and_prune`` – loads a manifest, audits candidates, and prunes them
   using a schedule that is mathematically coupled to the residual norm.

A minimal smoke test at the bottom runs the full pipeline without external
files."""

import datetime as dt
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – allocation & sheaf utilities
# ----------------------------------------------------------------------
GROUPS: Tuple[str, ...] = ("codex", "groq", "cohere", "local_models")
MAX64 = (1 << 64) - 1


def _pct(value: float) -> float:
    """Round a float to six decimal places (consistent with Parent A)."""
    return round(float(value), 6)


def doomsday(year: int, month: int, day: int) -> int:
    """Return a 0‑based weekday index (Monday=0) using Conway's Doomsday algorithm."""
    # Zeller’s congruence variant for Gregorian calendar
    if month < 3:
        month += 12
        year -= 1
    K = year % 100
    J = year // 100
    f = day + (13 * (month + 1)) // 5 + K + K // 4 + J // 4 + 5 * J
    return (f % 7 + 6) % 7  # shift so Monday=0


def weekday_weight_vector(ref: dt.date | None = None) -> np.ndarray:
    """Return a 4‑element weight vector that depends on the weekday of *ref*.

    The original Parent A uses a deterministic pseudo‑random mix of three base
    weights.  Here we keep the same spirit while staying deterministic.
    """
    if ref is None:
        ref = dt.date.today()
    w = np.empty(len(GROUPS), dtype=float)
    base = [0.25, 0.35, 0.20, 0.20]  # sum to 1.0
    weekday = ref.weekday()  # Monday=0
    rng = random.Random(doomsday(ref.year, ref.month, ref.day) + weekday)
    for i in range(len(GROUPS)):
        jitter = rng.uniform(-0.03, 0.03)
        w[i] = _pct(base[i] + jitter)
    # Renormalise to exactly sum to 1
    w /= w.sum()
    return w


def allocate_hybrid(weights: np.ndarray) -> np.ndarray:
    """Deterministic allocation split between deterministic and LLM portions.

    Returns a per‑group allocation vector `s` (sum = 1.0).  The LLM portion is
    simulated by a second random draw that is mixed with the deterministic
    weights.
    """
    if len(weights) != len(GROUPS):
        raise ValueError("weights length mismatch")
    # deterministic part
    det = weights * 0.6
    # LLM‑simulated stochastic part
    rng = random.Random(int(weights.sum() * MAX64) & MAX64)
    llm = np.array([rng.random() for _ in GROUPS])
    llm = llm / llm.sum() * 0.4
    s = det + llm
    s = np.array([_pct(v) for v in s])
    s /= s.sum()  # final normalisation
    return s


def coboundary_matrix(groups: Tuple[str, ...]) -> np.ndarray:
    """Build the (oriented) edge‑incidence matrix Δ for the complete graph.

    Each row corresponds to an unordered pair (i, j) with i < j and contains
    +1 at column i and -1 at column j.
    """
    n = len(groups)
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    m = len(edges)
    Δ = np.zeros((m, n), dtype=float)
    for row, (i, j) in enumerate(edges):
        Δ[row, i] = 1.0
        Δ[row, j] = -1.0
    return Δ


def sheaf_residual(allocation: np.ndarray) -> Tuple[np.ndarray, float]:
    """Apply the coboundary operator to *allocation* and return (r, ‖r‖₂)."""
    Δ = coboundary_matrix(GROUPS)
    r = Δ @ allocation
    norm = float(np.linalg.norm(r, 2))
    return r, norm


def allocate_and_residual(ref: dt.date | None = None) -> Tuple[np.ndarray, float]:
    """Convenience wrapper: compute allocation vector and its sheaf residual norm."""
    w = weekday_weight_vector(ref)
    s = allocate_hybrid(w)
    _, norm = sheaf_residual(s)
    return s, norm


# ----------------------------------------------------------------------
# Parent B – audit & decreasing‑rate pruning utilities
# ----------------------------------------------------------------------
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


def load_manifest(path: Path) -> dict[str, Any]:
    """Load a JSON manifest and validate its classifications."""
    data = json.loads(path.read_text(encoding="utf-8"))
    for cand in data.get("vendors", []):
        cls = cand.get("classification")
        if cls not in CLASSIFICATIONS:
            raise ValueError(f"invalid classification {cls!r}")
    return data


def audit_candidate(candidate: dict[str, Any]) -> int:
    """Return an integer audit score (number of rule violations) for *candidate*.

    The score is the count of textual rule violations detected by simple
    regular‑expression checks.  This mirrors the spirit of the original
    ternary‑lens audit.
    """
    findings = 0
    key = candidate.get("candidate_key", "")
    family = candidate.get("family", "")
    notes = candidate.get("notes", "")

    # Example rule 1: presence of known local‑only patterns
    if any(p.strip("*").lower() in (key + " " + family + " " + notes).lower() for p in LOCAL_PATTERNS):
        findings += 1

    # Example rule 2: classification mismatch with fast‑path expectation
    if re.search(r"standard.*lora|peft|qlora", key + " " + family, re.I):
        if candidate.get("classification") != "unsafe_for_fastpath" or candidate.get(
            "fast_path_compatible"
        ):
            findings += 1

    # Example rule 3: explicit "unsupported" tag without notes
    if candidate.get("classification") == "unsupported" and not notes.strip():
        findings += 1

    return findings


def audit_manifest(manifest: dict[str, Any]) -> Dict[str, int]:
    """Audit every vendor entry and return a mapping ``candidate_key → score``."""
    scores: Dict[str, int] = {}
    for cand in manifest.get("vendors", []):
        key = cand.get("candidate_key", "<unknown>")
        scores[key] = audit_candidate(cand)
    return scores


def decreasing_prune(
    scores: Dict[str, int],
    residual_norm: float,
    max_iter: int = 5,
    base_prob: float = 0.5,
    beta: float = 0.3,
) -> List[str]:
    """Iteratively prune candidates using a schedule modulated by *residual_norm*.

    Parameters
    ----------
    scores: mapping candidate → audit score (higher = riskier)
    residual_norm: global sheaf residual L2 norm
    max_iter: number of pruning passes
    base_prob: initial pruning probability for a candidate with score = 0
    beta: scaling factor that amplifies the effect of the residual norm

    Returns
    -------
    List of candidate keys that survive all iterations.
    """
    survivors = list(scores.keys())
    for k in range(max_iter):
        prob_factor = base_prob / (1 + k)  # decreasing base
        # Residual‑driven amplification (clipped to keep probabilities ≤ 1)
        amp = min(1.0, 1.0 + beta * residual_norm)
        for cand in survivors[:]:  # iterate over a copy
            risk = scores[cand]
            # Higher audit score → higher chance of removal
            p_remove = prob_factor * (1 + risk / max(1, max(scores.values())))
            p_remove = min(1.0, p_remove * amp)
            if random.random() < p_remove:
                survivors.remove(cand)
    return survivors


def audit_and_prune(manifest_path: Path, residual_norm: float) -> List[str]:
    """Full hybrid audit‑prune pipeline driven by a sheaf residual."""
    manifest = load_manifest(manifest_path)
    scores = audit_manifest(manifest)
    kept = decreasing_prune(scores, residual_norm)
    return kept


# ----------------------------------------------------------------------
# Hybrid orchestrator
# ----------------------------------------------------------------------
def hybrid_pipeline(ref_date: dt.date | None = None) -> Tuple[np.ndarray, float, List[str]]:
    """Run the complete hybrid workflow.

    1. Compute weekday‑aware allocation and its sheaf residual norm.
    2. Create a temporary manifest (synthetic for the smoke test).
    3. Audit the manifest and prune candidates using the residual norm.

    Returns
    -------
    (allocation_vector, residual_norm, list_of_surviving_candidate_keys)
    """
    # Step 1 – allocation + sheaf
    allocation, resid_norm = allocate_and_residual(ref_date)

    # Step 2 – synthetic manifest generation
    tmp_path = Path("tmp_manifest.json")
    synthetic = {
        "vendors": [
            {
                "candidate_key": "lora_alpha",
                "family": "LoRA",
                "classification": "unsafe_for_fastpath",
                "fast_path_compatible": False,
                "notes": "",
            },
            {
                "candidate_key": "bitnet_v2",
                "family": "BitNet",
                "classification": "needs_conversion",
                "fast_path_compatible": True,
                "notes": "requires conversion",
            },
            {
                "candidate_key": "fairyfuse_x",
                "family": "FairyFuse",
                "classification": "unsupported",
                "fast_path_compatible": False,
                "notes": "",
            },
            {
                "candidate_key": "generic_model",
                "family": "Generic",
                "classification": "usable_now",
                "fast_path_compatible": True,
                "notes": "ready for production",
            },
        ]
    }
    tmp_path.write_text(json.dumps(synthetic, indent=2), encoding="utf-8")

    # Step 3 – audit & prune
    survivors = audit_and_prune(tmp_path, resid_norm)

    # Clean up the temporary file
    try:
        tmp_path.unlink()
    except OSError:
        pass

    return allocation, resid_norm, survivors


if __name__ == "__main__":
    # Smoke test: run the hybrid pipeline and print a concise report
    alloc, norm, kept = hybrid_pipeline()
    print("Allocation vector (per group):")
    for grp, val in zip(GROUPS, alloc):
        print(f"  {grp:12s}: {val:.6f}")
    print(f"\nSheaf residual L2 norm: {norm:.6f}")
    print("\nCandidates surviving pruning:")
    for cand in kept:
        print(f"  - {cand}")
    sys.exit(0)