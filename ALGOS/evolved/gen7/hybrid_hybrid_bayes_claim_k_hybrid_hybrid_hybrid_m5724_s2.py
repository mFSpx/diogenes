# DARWIN HAMMER — match 5724, survivor 2
# gen: 7
# parent_a: hybrid_bayes_claim_kernel_hybrid_ternary_route_m84_s1.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_distri_m1510_s5.py (gen6)
# born: 2026-05-30T00:04:28Z

"""
Hybrid algorithm fusing the Bayesian update from hybrid_bayes_claim_kernel_hybrid_ternary_route_m84_s1.py
and the stylometry-based weighted graph construction from hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_distri_m1510_s5.py.

The mathematical bridge lies in using the posterior probabilities from the Bayesian update as weights
in the weighted graph construction, effectively incorporating epistemic confidence into the stylometry features.
"""

import math
import random
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple
from pathlib import Path

@dataclass(frozen=True)
class MathEvidence:
    """An observation that can be used to update an edge hypothesis."""
    id: str
    measurement: float  # e.g., observed length or signal strength
    noise_std: float    # standard deviation of measurement noise

@dataclass(frozen=True)
class MathHypothesis:
    """Bayesian hypothesis attached to a tree edge."""
    id: str
    prior: float                # prior probability that the edge is reliable
    posterior: float            # current posterior after evidence
    evidence_ids: Tuple[str, ...] = ()

@dataclass(frozen=True)
class CertaintyFlag:
    """Epistemic confidence attached to an edge."""
    label: str
    confidence_bps: int  # basis points, 0-10000

def update_hypothesis(
    hypothesis: MathHypothesis,
    evidence: MathEvidence,
    likelihood_ratio: float,
) -> MathHypothesis:
    """Update posterior of a hypothesis using a likelihood ratio.

    The odds form is used to keep the operation numerically stable.
    """
    if likelihood_ratio < 0:
        raise ValueError("likelihood_ratio must be non-negative")

    p = max(0.0, min(1.0, hypothesis.posterior))
    if p <= 0.0 or likelihood_ratio == 0.0:
        posterior = 0.0
    elif p >= 1.0:
        posterior = 1.0
    else:
        posterior = (p * likelihood_ratio) / (p * likelihood_ratio + (1 - p))
    return MathHypothesis(
        hypothesis.id,
        hypothesis.prior,
        posterior,
        hypothesis.evidence_ids + (evidence.id,),
    )

def stylometry_features(text: str) -> Dict[str, int]:
    """Extract a tiny stylometric fingerprint from *text*."""
    pronouns = {
        "i", "you", "he", "she", "we", "they",
        "me", "him", "her", "us", "them"
    }
    conjunctions = {
        "and", "or", "but", "nor", "so", "yet",
        "for", "although"
    }
    verbs = {
        "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "do", "does"
    }

    tokens = [t.lower().strip(".,!?;:\"'()[]") for t in text.split()]
    counts = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1

    return {
        "pronoun": sum(counts.get(w, 0) for w in pronouns),
        "conj":    sum(counts.get(w, 0) for w in conjunctions),
        "verb":    sum(counts.get(w, 0) for w in verbs),
        "len":     len(tokens),
    }

def build_weighted_graph(
    features_list: List[Dict[str, int]],
    posteriors: List[float]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a symmetric similarity matrix W (cosine similarity) and node
    strengths (L2 norm of the feature vectors), incorporating posterior
    probabilities as weights.

    Returns
    -------
    W : (n, n) ndarray
        Cosine similarity in [0, 1].
    strengths : (n,) ndarray
        L2 norm of each feature vector (≥ 1e-12 to avoid division by zero).
    """
    n = len(features_list)
    # Preserve a deterministic ordering of feature dimensions
    keys = sorted({k for d in features_list for k in d})
    mat = np.zeros((n, len(keys)), dtype=float)

    for i, (d, p) in enumerate(zip(features_list, posteriors)):
        for j, k in enumerate(keys):
            mat[i, j] = p * d.get(k, 0)

    strengths = np.linalg.norm(mat, axis=1) + 1e-12
    normed = mat / strengths[:, None]
    W = np.dot(normed, normed.T)
    return W, strengths

def hybrid_operation(texts: List[str], 
                     hypotheses: List[MathHypothesis], 
                     evidences: List[MathEvidence], 
                     likelihood_ratios: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    features_list = [stylometry_features(t) for t in texts]
    posteriors = []
    for h, e, lr in zip(hypotheses, evidences, likelihood_ratios):
        updated_h = update_hypothesis(h, e, lr)
        posteriors.append(updated_h.posterior)
    return build_weighted_graph(features_list, posteriors)

if __name__ == "__main__":
    texts = ["This is a test.", "Another test.", "One more test."]
    hypotheses = [MathHypothesis("h1", 0.5, 0.5), MathHypothesis("h2", 0.6, 0.6), MathHypothesis("h3", 0.7, 0.7)]
    evidences = [MathEvidence("e1", 1.0, 0.1), MathEvidence("e2", 2.0, 0.2), MathEvidence("e3", 3.0, 0.3)]
    likelihood_ratios = [2.0, 3.0, 4.0]
    W, strengths = hybrid_operation(texts, hypotheses, evidences, likelihood_ratios)
    print(W)
    print(strengths)