# DARWIN HAMMER — match 2433, survivor 2
# gen: 3
# parent_a: hybrid_bayes_claim_kernel_hybrid_ternary_lens__m26_s2.py (gen2)
# parent_b: hybrid_hybrid_decision_hygi_decreasing_pruning_m17_s1.py (gen2)
# born: 2026-05-29T23:42:18Z

"""Hybrid Bayesian‑Entropy Pruning Module.
Parents:
- hybrid_bayes_claim_kernel_hybrid_ternary_lens__m26_s2.py (Bayesian update with
  classification‑weighted pruning).
- hybrid_hybrid_decision_hygi_decreasing_pruning_m17_s1.py (Shannon‑entropy
  feature extraction + decreasing‑rate pruning schedule).

Mathematical Bridge:
For each piece of evidence *e* we compute three scalars:
1. **w_c** – the normalised classification weight derived from the audit
   manifest (parent A).
2. **H(e)** – the Shannon entropy of the regex‑extracted feature vector of the
   evidence text (parent B).  It is normalised to the unit interval
   `h = H(e) / log(|F|)` where `|F|` is the number of distinct regex feature
   families.
3. **p(t)** – a decreasing pruning probability
   `p(t)=min(1, λ·exp(−α·t))` (parent B).

These are fused into a single damping factor

    d(e,t) = 1 − p(t)·w_c·h

which multiplies the raw likelihood ratio ℓ supplied for the evidence:

    ℓ' = ℓ · d(e,t)

The damped likelihood ℓ' is then fed to the standard Bayesian update
`posterior = prior·ℓ' / (prior·ℓ' + (1−prior))`.  This single formula
integrates the core topologies of both parents: classification weighting,
entropy‑based relevance, and a time‑decaying pruning schedule.

The module implements this hybrid pipeline with three public functions:
`extract_features`, `hybrid_likelihood`, and `process_evidence_sequence`.
"""

from __future__ import annotations

import math
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Domain data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathEvidence:
    """A single piece of evidence."""
    id: str
    claim: str                     # textual claim the evidence refers to
    classification: str           # audit classification label
    text: str                      # raw textual content of the evidence
    timestamp: float = 0.0        # abstract time index (seconds or steps)


@dataclass(frozen=True)
class MathHypothesis:
    """A hypothesis about a claim."""
    id: str
    prior: float                  # prior probability before processing evidence
    posterior: float = 0.0        # current posterior probability
    evidence_ids: Tuple[str, ...] = ()


# ----------------------------------------------------------------------
# Parent‑B regex feature families (used for entropy calculation)
# ----------------------------------------------------------------------
FEATURE_REGEXES: Dict[str, re.Pattern] = {
    "EVIDENCE": re.compile(
        r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|"
        r"receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|"
        r"check|checked|audit)\b",
        re.IGNORECASE,
    ),
    "PLANNING": re.compile(
        r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|"
        r"prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|"
        r"test|smoke)\b",
        re.IGNORECASE,
    ),
    "DELAY": re.compile(
        r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool\s?down|de[- ]?escalate|"
        r"not\s?now|before\s?i|first|after|review)\b",
        re.IGNORECASE,
    ),
    "SUPPORT": re.compile(
        r"\b(?:ask|call|text|friend|friends|rowyn|kai|chance|doctor|therapist|"
        r"lawyer|advocate|support|help|community|team|handoff|delegate)\b",
        re.IGNORECASE,
    ),
    "BOUNDARY": re.compile(
        r"\b(?:boundary|boundaries|walk\s?away|no\s?contact|do\s?not|don't|"
        r"stop|limit|refuse|protect|safe|safety|lock|redact|privacy|permission|"
        r"consent)\b",
        re.IGNORECASE,
    ),
    "OUTCOME": re.compile(
        r"\b(?:done|shipped|finished|complete|resolved|succeeded|passed|paid|"
        r"safe|sent|filed|closed|fixed|working|green|verified)\b",
        re.IGNORECASE,
    ),
    "IMPULSIVE": re.compile(
        r"\b(?:rage|impulsive|panic|panicking|spiral|doom|fuck\s?it|right\s?now|"
        r"immediate|now|asap)\b",
        re.IGNORECASE,
    ),
}


# ----------------------------------------------------------------------
# Helper 1 – Feature extraction & Shannon entropy (Parent B)
# ----------------------------------------------------------------------
def extract_features(text: str) -> Counter:
    """Return a Counter mapping feature family names to occurrence counts in *text*."""
    counts = Counter()
    for name, pattern in FEATURE_REGEXES.items():
        matches = pattern.findall(text)
        if matches:
            counts[name] = len(matches)
    return counts


def shannon_entropy(counter: Counter) -> float:
    """Shannon entropy of a discrete distribution given by *counter*."""
    total = sum(counter.values())
    if total == 0:
        return 0.0
    probs = np.array([c / total for c in counter.values()], dtype=float)
    return -np.sum(probs * np.log(probs + 1e-12))


def normalised_entropy(counter: Counter) -> float:
    """
    Normalised entropy in [0,1].
    If there are *k* possible feature families, the maximum entropy is log(k).
    """
    if not counter:
        return 0.0
    max_entropy = math.log(len(FEATURE_REGEXES))
    return shannon_entropy(counter) / max_entropy if max_entropy > 0 else 0.0


# ----------------------------------------------------------------------
# Helper 2 – Classification weight vector (Parent A)
# ----------------------------------------------------------------------
def classification_weights(evidences: Iterable[MathEvidence]) -> Dict[str, float]:
    """
    Compute normalised weights w_c for each classification label.
    w_c = count_c / total_evidence.
    """
    counts = Counter(e.classification for e in evidences)
    total = sum(counts.values())
    if total == 0:
        return {}
    return {cls: cnt / total for cls, cnt in counts.items()}


# ----------------------------------------------------------------------
# Helper 3 – Decreasing pruning schedule (Parent B)
# ----------------------------------------------------------------------
def pruning_probability(t: float, lam: float = 0.9, alpha: float = 0.05) -> float:
    """
    Time‑decaying pruning probability p(t) = min(1, λ·exp(−α·t)).
    """
    return min(1.0, lam * math.exp(-alpha * t))


# ----------------------------------------------------------------------
# Core Fusion – Hybrid likelihood damping (Mathematical Bridge)
# ----------------------------------------------------------------------
def hybrid_likelihood(
    raw_likelihood: float,
    t: float,
    classification_weight: float,
    entropy_factor: float,
    lam: float = 0.9,
    alpha: float = 0.05,
) -> float:
    """
    Apply the hybrid damping factor:

        d = 1 − p(t)·w_c·h

    where
        p(t)   – decreasing pruning probability,
        w_c    – classification weight,
        h      – normalised Shannon entropy of the evidence's feature vector.

    The returned value is the damped likelihood ℓ' = ℓ·d.
    """
    p = pruning_probability(t, lam, alpha)
    damping = 1.0 - p * classification_weight * entropy_factor
    # Guard against negative damping due to extreme parameters
    damping = max(0.0, damping)
    return raw_likelihood * damping


# ----------------------------------------------------------------------
# Bayesian update (Parent A) – now using hybrid likelihood
# ----------------------------------------------------------------------
def bayesian_update(
    hypothesis: MathHypothesis,
    evidence: MathEvidence,
    raw_likelihood: float,
    t: float,
    class_weights: Dict[str, float],
    lam: float = 0.9,
    alpha: float = 0.05,
) -> MathHypothesis:
    """
    Return a new hypothesis with posterior updated using the hybrid likelihood.
    """
    w_c = class_weights.get(evidence.classification, 0.0)

    # Entropy factor from the evidence text
    feats = extract_features(evidence.text)
    h = normalised_entropy(feats)

    # Damped likelihood
    ell_prime = hybrid_likelihood(
        raw_likelihood,
        t,
        classification_weight=w_c,
        entropy_factor=h,
        lam=lam,
        alpha=alpha,
    )

    # Standard Bayesian update using odds form:
    # posterior = prior * ℓ' / (prior * ℓ' + (1 - prior))
    prior = hypothesis.posterior if hypothesis.posterior > 0 else hypothesis.prior
    numerator = prior * ell_prime
    denominator = numerator + (1.0 - prior)
    new_posterior = numerator / denominator if denominator != 0 else 0.0

    new_evidence_ids = hypothesis.evidence_ids + (evidence.id,)
    return replace(
        hypothesis,
        posterior=new_posterior,
        evidence_ids=new_evidence_ids,
    )


# ----------------------------------------------------------------------
# Public API – processing a sequence of evidence items
# ----------------------------------------------------------------------
def process_evidence_sequence(
    evidences: List[MathEvidence],
    hypothesis_id: str,
    prior: float = 0.5,
    lam: float = 0.9,
    alpha: float = 0.05,
    seed: int | None = None,
) -> MathHypothesis:
    """
    Process *evidences* in chronological order (according to .timestamp) and
    return the final hypothesis after hybrid Bayesian updating.

    A synthetic raw likelihood ratio is generated for each evidence:
        ℓ = 1 + 0.5·rand_uniform(0,1)   (i.e. in [1,1.5])
    This placeholder mimics the presence of an external likelihood estimator.
    """
    if seed is not None:
        random.seed(seed)

    # Ensure deterministic ordering
    sorted_evs = sorted(evidences, key=lambda e: e.timestamp)

    # Pre‑compute classification weights once (they are static in this design)
    class_w = classification_weights(sorted_evs)

    # Initialise hypothesis
    hypothesis = MathHypothesis(id=hypothesis_id, prior=prior, posterior=prior)

    for idx, ev in enumerate(sorted_evs):
        # Simulated raw likelihood ratio (must be >0)
        raw_lr = 1.0 + 0.5 * random.random()
        hypothesis = bayesian_update(
            hypothesis,
            ev,
            raw_likelihood=raw_lr,
            t=ev.timestamp,
            class_weights=class_w,
            lam=lam,
            alpha=alpha,
        )
    return hypothesis


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a tiny audit manifest of 5 evidences with varying classifications and texts
    demo_evidences = [
        MathEvidence(
            id="e1",
            claim="Claim A",
            classification="high",
            text="The source provided a verified receipt and a hash of the document.",
            timestamp=0.0,
        ),
        MathEvidence(
            id="e2",
            claim="Claim A",
            classification="medium",
            text="We plan to schedule a review after the test phase is complete.",
            timestamp=2.0,
        ),
        MathEvidence(
            id="e3",
            claim="Claim A",
            classification="low",
            text="User reported panic and immediate need for support.",
            timestamp=5.0,
        ),
        MathEvidence(
            id="e4",
            claim="Claim A",
            classification="high",
            text="Outcome: shipped and verified. All logs are clean.",
            timestamp=7.0,
        ),
        MathEvidence(
            id="e5",
            claim="Claim A",
            classification="medium",
            text="Boundary conditions were respected; no further action required.",
            timestamp=10.0,
        ),
    ]

    final_hyp = process_evidence_sequence(
        demo_evidences,
        hypothesis_id="hyp1",
        prior=0.4,
        lam=0.85,
        alpha=0.07,
        seed=42,
    )
    print(f"Final posterior for hypothesis '{final_hyp.id}': {final_hyp.posterior:.4f}")
    print(f"Evidence chain: {final_hyp.evidence_ids}")