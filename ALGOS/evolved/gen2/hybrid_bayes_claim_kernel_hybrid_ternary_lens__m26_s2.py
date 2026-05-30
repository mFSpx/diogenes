# DARWIN HAMMER — match 26, survivor 2
# gen: 2
# parent_a: bayes_claim_kernel.py (gen0)
# parent_b: hybrid_ternary_lens_audit_decreasing_pruning_m15_s2.py (gen1)
# born: 2026-05-29T23:23:04Z

"""Hybrid Bayesian‑Pruning Module.

Parents:
- **bayes_claim_kernel.py** (Parent A) – Bayesian update of a hypothesis given
  evidence and a likelihood ratio.
- **hybrid_ternary_lens_audit_decreasing_pruning_m15_s2.py** (Parent B) – A
  time‑decaying pruning schedule p(t)=min(1, λ·exp(−α·t)) that is modulated
  per‑candidate by a classification weight vector **w** derived from an audit
  manifest.

Mathematical Bridge:
For each piece of evidence we treat its audit classification *c* as a
component of the weight vector **w** (normalised counts of the manifest).  The
pruning probability for that evidence at time *t* is

    p_i(t) = p(t) · w_c   where   p(t)=min(1, λ·exp(−α·t))

We reinterpret *p_i(t)* as a *damping factor* on the likelihood ratio
ℓ_i supplied to the Bayesian update:

    ℓ_i' = ℓ_i · (1 − p_i(t))

Thus evidence that is likely to be pruned (high p_i) contributes less
information to the posterior, while evidence from abundant classifications
(low w_c) retains more influence.  The hybrid functions below implement this
fusion in a single unified workflow.
"""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Hashable, List, Mapping, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Simple domain types (stand‑ins for the original .types module)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathEvidence:
    id: str
    claim: str
    classification: str  # must be one of CLASSIFICATIONS


@dataclass(frozen=True)
class MathHypothesis:
    id: str
    prior: float          # prior probability before this evidence
    posterior: float      # current posterior probability
    evidence_ids: Tuple[str, ...] = ()


# ----------------------------------------------------------------------
# Parent A – Bayesian update (unchanged except for type hints)
# ----------------------------------------------------------------------
def update_hypothesis(
    hypothesis: MathHypothesis,
    evidence: MathEvidence,
    likelihood_ratio: float,
) -> MathHypothesis:
    """Return a new hypothesis with posterior updated by the given likelihood ratio."""
    if likelihood_ratio < 0:
        raise ValueError("likelihood_ratio must be non‑negative")
    p = max(0.0, min(1.0, hypothesis.posterior))
    if p <= 0.0 or likelihood_ratio == 0.0:
        posterior = 0.0
    elif p >= 1.0:
        posterior = 1.0
    else:
        odds = p / (1.0 - p)
        new_odds = odds * likelihood_ratio
        posterior = new_odds / (1.0 + new_odds)
    posterior = max(0.0, min(1.0, posterior))
    ids = tuple(list(hypothesis.evidence_ids) + [evidence.id])
    return MathHypothesis(
        id=hypothesis.id,
        prior=hypothesis.posterior,
        posterior=posterior,
        evidence_ids=ids,
    )


# ----------------------------------------------------------------------
# Parent B – decreasing‑rate pruning schedule
# ----------------------------------------------------------------------
def prune_probability(t: float, lam: float = 1.0, alpha: float = 0.2) -> float:
    """Base pruning probability p(t)=min(1, λ·exp(−α·t))."""
    if t < 0 or lam < 0 or alpha < 0:
        raise ValueError("t, lam and alpha must be non‑negative")
    return min(1.0, lam * math.exp(-alpha * t))


# ----------------------------------------------------------------------
# Shared constants (mirroring Parent B)
# ----------------------------------------------------------------------
CLASSIFICATIONS = {
    "usable_now",
    "research_only",
    "needs_conversion",
    "unsafe_for_fastpath",
    "unsupported",
}


# ----------------------------------------------------------------------
# Hybrid Core – classification weights & per‑evidence prune factors
# ----------------------------------------------------------------------
def classification_weight_vector(manifest: Mapping[str, Any]) -> np.ndarray:
    """
    Produce a normalised weight vector **w** ∈ ℝ^k where k = |CLASSIFICATIONS|.
    The i‑th component is the relative frequency of the i‑th classification
    (sorted alphabetically) in the manifest.
    """
    counts = np.array(
        [
            sum(
                1
                for cand in manifest.get("vendors", [])
                if cand.get("classification") == cls
            )
            for cls in sorted(CLASSIFICATIONS)
        ],
        dtype=float,
    )
    total = counts.sum()
    return counts / total if total > 0 else counts


def per_evidence_prune_factor(
    evidence: MathEvidence,
    weight_vec: np.ndarray,
    t: float,
    lam: float = 1.0,
    alpha: float = 0.2,
) -> float:
    """
    Compute the damping factor (1‑p_i(t)) for a single piece of evidence.

    p_i(t) = p(t)·w_c  where  c = evidence.classification
    """
    base_p = prune_probability(t, lam, alpha)
    # Map classification to index in the sorted weight vector
    cls_index = sorted(CLASSIFICATIONS).index(evidence.classification)
    p_i = base_p * weight_vec[cls_index]
    return 1.0 - p_i  # factor to multiply the likelihood ratio


def hybrid_update_hypothesis(
    hypothesis: MathHypothesis,
    evidence: MathEvidence,
    weight_vec: np.ndarray,
    t: float,
    lam: float = 1.0,
    alpha: float = 0.2,
    base_likelihood_ratio: float = 1.0,
) -> MathHypothesis:
    """
    Perform a Bayesian update where the supplied likelihood ratio is damped
    by the time‑dependent pruning factor derived from the audit classification.
    """
    damp = per_evidence_prune_factor(evidence, weight_vec, t, lam, alpha)
    effective_lr = base_likelihood_ratio * damp
    return update_hypothesis(hypothesis, evidence, effective_lr)


# ----------------------------------------------------------------------
# Public hybrid workflow
# ----------------------------------------------------------------------
def batch_hybrid_update(
    hypotheses: List[MathHypothesis],
    evidences: List[MathEvidence],
    manifest: Mapping[str, Any],
    t: float,
    lam: float = 1.0,
    alpha: float = 0.2,
    base_lr: float = 1.0,
) -> List[MathHypothesis]:
    """
    Apply the hybrid Bayesian‑pruning update to a collection of hypotheses.
    Each evidence item is matched to a hypothesis with the same ``id``; if no
    matching hypothesis exists a new one is created with a uniform prior of
    0.5.
    """
    weight_vec = classification_weight_vector(manifest)
    # Index hypotheses by id for O(1) lookup
    hypo_map = {h.id: h for h in hypotheses}
    updated: List[MathHypothesis] = []

    for ev in evidences:
        hyp = hypo_map.get(ev.id)
        if hyp is None:
            hyp = MathHypothesis(id=ev.id, prior=0.5, posterior=0.5, evidence_ids=())
        new_hyp = hybrid_update_hypothesis(
            hyp,
            ev,
            weight_vec,
            t,
            lam,
            alpha,
            base_lr,
        )
        hypo_map[ev.id] = new_hyp
        updated.append(new_hyp)

    return list(hypo_map.values())


def prune_candidates_by_classification(
    candidates: List[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    t: float,
    lam: float = 1.0,
    alpha: float = 0.2,
    seed: int | str | None = None,
) -> List[Mapping[str, Any]]:
    """
    Stochastically drop candidates according to the hybrid probability matrix
    P_i(t)=p(t)·w_{class(i)}.  Returns the surviving subset.
    """
    rng = random.Random(seed)
    weight_vec = classification_weight_vector(manifest)
    cls_order = sorted(CLASSIFICATIONS)
    survivors: List[Mapping[str, Any]] = []

    for cand in candidates:
        classification = cand.get("classification")
        if classification not in CLASSIFICATIONS:
            # Invalid classification – keep it unchanged (defensive)
            survivors.append(cand)
            continue
        idx = cls_order.index(classification)
        p_i = prune_probability(t, lam, alpha) * weight_vec[idx]
        if rng.random() >= p_i:
            survivors.append(cand)

    return survivors


def audit_and_hybrid_update(
    manifest_path: Path,
    evidence_list: List[MathEvidence],
    t: float,
    lam: float = 1.0,
    alpha: float = 0.2,
    seed: int | str | None = None,
) -> Tuple[List[Mapping[str, Any]], List[MathHypothesis]]:
    """
    End‑to‑end hybrid operation:
    1. Load the audit manifest.
    2. Prune the vendor candidates using the classification‑aware schedule.
    3. Update (or create) hypotheses for the provided evidences using the same
       schedule as a damping factor on the likelihood ratios.
    Returns a tuple ``(pruned_candidates, updated_hypotheses)``.
    """
    manifest = load_manifest(manifest_path)
    pruned = prune_candidates_by_classification(
        manifest.get("vendors", []),
        manifest,
        t,
        lam,
        alpha,
        seed,
    )
    # For the demo we start with an empty hypothesis list
    updated_hypos = batch_hybrid_update(
        hypotheses=[],
        evidences=evidence_list,
        manifest=manifest,
        t=t,
        lam=lam,
        alpha=alpha,
        base_lr=1.0,
    )
    return pruned, updated_hypos


# ----------------------------------------------------------------------
# Minimal manifest loader (mirrors Parent B helper)
# ----------------------------------------------------------------------
def load_manifest(path: Path) -> dict[str, Any]:
    """Load a JSON manifest and perform a light validation of classifications."""
    data = json.loads(path.read_text(encoding="utf-8"))
    for cand in data.get("vendors", []):
        if cand.get("classification") not in CLASSIFICATIONS:
            raise ValueError(
                f"Invalid classification {cand.get('classification')!r} in manifest"
            )
    return data


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a tiny in‑memory manifest
    dummy_manifest = {
        "vendors": [
            {"candidate_key": "v1", "classification": "usable_now"},
            {"candidate_key": "v2", "classification": "research_only"},
            {"candidate_key": "v3", "classification": "needs_conversion"},
            {"candidate_key": "v4", "classification": "usable_now"},
        ]
    }

    # Write it to a temporary file
    tmp_path = Path("tmp_manifest.json")
    tmp_path.write_text(json.dumps(dummy_manifest), encoding="utf-8")

    # Construct some evidence items
    evidences = [
        MathEvidence(id="v1", claim="claim_a", classification="usable_now"),
        MathEvidence(id="v2", claim="claim_b", classification="research_only"),
        MathEvidence(id="v5", claim="claim_c", classification="unsupported"),
    ]

    # Run the hybrid pipeline
    pruned, hypos = audit_and_hybrid_update(
        manifest_path=tmp_path,
        evidence_list=evidences,
        t=2.5,
        lam=1.0,
        alpha=0.3,
        seed=42,
    )

    print("Pruned candidates (ids):", [c["candidate_key"] for c in pruned])
    print("Updated hypotheses:")
    for h in hypos:
        print(f"  {h.id}: posterior={h.posterior:.4f}, evidence={h.evidence_ids}")

    # Clean up
    tmp_path.unlink(missing_ok=True)