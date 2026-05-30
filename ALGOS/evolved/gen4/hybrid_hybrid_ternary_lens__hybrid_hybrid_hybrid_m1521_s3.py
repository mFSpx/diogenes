# DARWIN HAMMER — match 1521, survivor 3
# gen: 4
# parent_a: hybrid_ternary_lens_audit_decreasing_pruning_m15_s2.py (gen1)
# parent_b: hybrid_hybrid_hybrid_decisi_hybrid_bayes_update__m1014_s1.py (gen3)
# born: 2026-05-29T23:37:00Z

"""Hybrid Audit‑Prune‑Bayes Module

Parents:
- **Parent A** (ternary_lens_audit / decreasing_pruning): provides a
  classification count vector **s**, normalised to a weight vector **w**
  (per‑class prevalence) and a time‑decaying prune probability
  p(t)=min(1, λ·exp(-α·t)).
- **Parent B** (regex feature extraction / Shannon entropy / Bayesian update):
  extracts a discrete feature histogram **f** from free‑form text,
  computes its Shannon entropy **H**, and supplies a Bayesian marginal/
  posterior formula.

Mathematical Bridge:
For each candidate *c* we treat the class‑weight *w_{class(c)}* as a
*prior* probability π_c.  The feature histogram of the candidate yields a
likelihood ℓ_c = Σ_i f_i / Σ_i f_i (i.e. the normalised count of the most
prominent evidence token).  A Bayesian marginal
M_c = π_c·ℓ_c + (1−π_c)·β (β = false‑positive rate) is computed, and the
posterior is

    posterior_c = (π_c·ℓ_c) / M_c .

The entropy **H** of the feature distribution modulates the decay
parameter λ → λ·exp(−γ·H) (γ>0).  The final stochastic keep‑probability for a
candidate at time *t* is

    keep_prob_c(t) = (1 − p(t)) + p(t)·posterior_c

where p(t)=min(1, λ·exp(−α·t)) with the entropy‑scaled λ.  Candidates are
kept with probability *keep_prob_c(t)*, yielding a unified audit‑prune‑bayes
pipeline.

The module implements this fusion with three public functions:
`load_manifest`, `compute_class_weights`, `hybrid_filter_candidates`.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Configuration (mirroring Parent A defaults)
# ----------------------------------------------------------------------
DEFAULT_MANIFEST = Path(__file__).with_name("sample_manifest.json")
DEFAULT_TIME = 0.0
DEFAULT_LAMBDA = 0.9   # base prune intensity
DEFAULT_ALPHA = 0.1    # exponential decay rate
DEFAULT_GAMMA = 0.5    # entropy scaling factor
DEFAULT_FALSE_POSITIVE = 0.05  # β in Bayesian marginal

# ----------------------------------------------------------------------
# Parent A – Audit helpers
# ----------------------------------------------------------------------
CLASSIFICATIONS = {
    "usable_now",
    "research_only",
    "needs_conversion",
    "unsafe_for_fastpath",
    "unsupported",
}


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    """
    Load a JSON manifest.  Expected schema (simplified):

    [
        {
            "id": "candidate‑1",
            "classification": "usable_now",
            "description": "Free‑form text containing evidence tokens ..."
        },
        ...
    ]
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Manifest not found: {path}")
    if not isinstance(data, list):
        raise ValueError("Manifest JSON must be a list of candidate objects")
    return data


def compute_class_weights(candidates: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Build the count vector **s** over CLASSIFICATIONS and normalise to obtain
    the weight vector **w**.
    """
    counts = {cls: 0 for cls in CLASSIFICATIONS}
    for cand in candidates:
        cls = cand.get("classification")
        if cls in counts:
            counts[cls] += 1
        else:
            # Unknown classes are ignored but counted as zero weight
            continue
    total = sum(counts.values())
    if total == 0:
        # Avoid division by zero – assign uniform tiny weight
        return {cls: 1.0 / len(CLASSIFICATIONS) for cls in CLASSIFICATIONS}
    return {cls: cnt / total for cls, cnt in counts.items()}


# ----------------------------------------------------------------------
# Parent B – Feature / Entropy / Bayesian helpers
# ----------------------------------------------------------------------
_EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|"
    r"receipt|sha256|screenshot|record|log|document|proof|fact|facts|"
    r"check|checked)\b",
    flags=re.IGNORECASE,
)


def extract_features(text: str) -> Dict[str, int]:
    """Regex‑based extraction of evidence‑type tokens."""
    feats: Dict[str, int] = {}
    for m in _EVIDENCE_RE.finditer(text):
        token = m.group().lower()
        feats[token] = feats.get(token, 0) + 1
    return feats


def compute_shannon_entropy(features: Dict[str, int]) -> float:
    """Shannon entropy H = -Σ p_i log p_i of the feature count distribution."""
    total = sum(features.values())
    if total == 0:
        return 0.0
    probs = np.array(list(features.values())) / total
    return -float(np.sum(probs * np.log(probs + 1e-12)))  # epsilon for safety


def bayesian_posterior(
    prior: float, likelihood: float, false_positive: float = DEFAULT_FALSE_POSITIVE
) -> float:
    """
    Compute posterior = (π·ℓ) / (π·ℓ + (1-π)·β)
    where π is the prior, ℓ the likelihood, β the false‑positive rate.
    """
    marginal = prior * likelihood + (1.0 - prior) * false_positive
    if marginal == 0.0:
        return 0.0
    return (prior * likelihood) / marginal


# ----------------------------------------------------------------------
# Hybrid core
# ----------------------------------------------------------------------
def prune_probability(
    t: float,
    lam: float = DEFAULT_LAMBDA,
    alpha: float = DEFAULT_ALPHA,
    entropy: float = 0.0,
    gamma: float = DEFAULT_GAMMA,
) -> float:
    """
    Time‑decaying prune probability p(t) with entropy‑scaled λ:

        λ' = λ·exp(−γ·H)
        p(t) = min(1, λ'·exp(−α·t))
    """
    lam_scaled = lam * math.exp(-gamma * entropy)
    prob = lam_scaled * math.exp(-alpha * t)
    return min(1.0, prob)


def hybrid_filter_candidates(
    candidates: List[Dict[str, Any]],
    t: float = DEFAULT_TIME,
    lam: float = DEFAULT_LAMBDA,
    alpha: float = DEFAULT_ALPHA,
    gamma: float = DEFAULT_GAMMA,
    false_positive: float = DEFAULT_FALSE_POSITIVE,
    rng: random.Random | None = None,
) -> List[Dict[str, Any]]:
    """
    Apply the hybrid audit‑prune‑bayes filter.

    Returns the subset of candidates kept after stochastic filtering.
    """
    rng = rng or random.Random()
    # 1️⃣ Compute class‑based priors w
    class_weights = compute_class_weights(candidates)

    # 2️⃣ Process each candidate
    kept: List[Dict[str, Any]] = []
    for cand in candidates:
        cls = cand.get("classification")
        prior = class_weights.get(cls, 0.0)

        # Feature extraction & entropy
        text = cand.get("description", "")
        feats = extract_features(text)
        entropy = compute_shannon_entropy(feats)

        # Likelihood: proportion of the most frequent evidence token
        if feats:
            max_count = max(feats.values())
            likelihood = max_count / sum(feats.values())
        else:
            likelihood = 0.0

        # Bayesian posterior
        post = bayesian_posterior(prior, likelihood, false_positive)

        # Prune probability at this time step, modulated by entropy
        p_t = prune_probability(t, lam, alpha, entropy, gamma)

        # Final keep probability: keep if not pruned OR survive pruning via posterior
        keep_prob = (1.0 - p_t) + p_t * post

        if rng.random() < keep_prob:
            # Optionally annotate the candidate with debug info
            cand["_debug"] = {
                "prior": prior,
                "likelihood": likelihood,
                "entropy": entropy,
                "posterior": post,
                "prune_prob": p_t,
                "keep_prob": keep_prob,
            }
            kept.append(cand)

    return kept


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
def _create_sample_manifest(path: Path) -> None:
    """Write a tiny manifest used for the __main__ smoke test."""
    sample = [
        {
            "id": "c1",
            "classification": "usable_now",
            "description": "This model has verified evidence and a sha256 hash.",
        },
        {
            "id": "c2",
            "classification": "research_only",
            "description": "No concrete evidence, just speculation.",
        },
        {
            "id": "c3",
            "classification": "needs_conversion",
            "description": "Source code confirmed, screenshot attached.",
        },
        {
            "id": "c4",
            "classification": "unsupported",
            "description": "Unsupported format, no evidence.",
        },
    ]
    path.write_text(json.dumps(sample, indent=2), encoding="utf-8")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Hybrid audit‑prune‑bayes demo")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Path to JSON manifest (will be created if missing)",
    )
    parser.add_argument("--time", type=float, default=DEFAULT_TIME, help="Time step t")
    parser.add_argument("--lambda", dest="lam", type=float, default=DEFAULT_LAMBDA)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument(
        "--false-positive",
        dest="fp",
        type=float,
        default=DEFAULT_FALSE_POSITIVE,
        help="False‑positive rate β for Bayesian update",
    )
    args = parser.parse_args(argv)

    if not args.manifest.exists():
        _create_sample_manifest(args.manifest)

    candidates = load_manifest(args.manifest)
    filtered = hybrid_filter_candidates(
        candidates,
        t=args.time,
        lam=args.lam,
        alpha=args.alpha,
        gamma=args.gamma,
        false_positive=args.fp,
    )

    print(f"Original candidates: {len(candidates)}")
    print(f"Filtered candidates: {len(filtered)}")
    for c in filtered:
        print(f"- {c['id']} (class={c['classification']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())