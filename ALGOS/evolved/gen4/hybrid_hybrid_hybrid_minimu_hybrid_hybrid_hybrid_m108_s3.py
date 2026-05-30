# DARWIN HAMMER — match 108, survivor 3
# gen: 4
# parent_a: hybrid_hybrid_minimum_cost__epistemic_certainty_m48_s5.py (gen2)
# parent_b: hybrid_hybrid_hybrid_bandit_label_foundry_m21_s0.py (gen3)
# born: 2026-05-29T23:27:00Z

import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Iterable, Set, Callable, Union, Any
import numpy as np
import hashlib
from pathlib import Path

# ----------------------------------------------------------------------
# Epistemic certainty helpers
# ----------------------------------------------------------------------
EPISTEMIC_FLAGS = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")


@dataclass(frozen=True)
class CertaintyFlag:
    label: str
    confidence_bps: int
    authority_class: str
    rationale: str
    evidence_refs: Tuple[str, ...] = ()
    generated_at: str = ""

    def __post_init__(self) -> None:
        if self.label not in EPISTEMIC_FLAGS:
            raise ValueError(f"unknown epistemic flag: {self.label!r}")
        if not 0 <= int(self.confidence_bps) <= 10_000:
            raise ValueError("confidence_bps must be in 0..10000")
        if not self.generated_at:
            object.__setattr__(
                self,
                "generated_at",
                datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            )

    def as_dict(self) -> Dict[str, Union[str, int, Tuple[str, ...]]]:
        return {
            "label": self.label,
            "confidence_bps": self.confidence_bps,
            "authority_class": self.authority_class,
            "rationale": self.rationale,
            "evidence_refs": self.evidence_refs,
            "generated_at": self.generated_at,
        }


def certainty(
    label: str,
    *,
    confidence_bps: int,
    authority_class: str,
    rationale: str,
    evidence_refs: Iterable[str] = (),
) -> CertaintyFlag:
    return CertaintyFlag(
        label=label,
        confidence_bps=int(confidence_bps),
        authority_class=authority_class,
        rationale=rationale,
        evidence_refs=tuple(str(x) for x in evidence_refs if x is not None),
    )


# ----------------------------------------------------------------------
# Labeling function results
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class LabelingFunctionResult:
    lf_name: str
    doc_id: str
    label: int  # binary {0,1}


@dataclass(frozen=True)
class ProbabilisticLabel:
    doc_id: str
    label: int  # the *observed* label (0/1)
    confidence: float  # belief that the observed label is correct (0..1)


@dataclass(frozen=True)
class LabelError:
    doc_id: str
    given_label: int
    suggested_label: int
    error_probability: float


def labeling_function(name: str | None = None):
    def deco(fn: Callable[[dict], int]):
        fn.lf_name = name or fn.__name__
        return fn

    return deco


# ----------------------------------------------------------------------
# Simple Count‑Min Sketch for distinct‑context estimation
# ----------------------------------------------------------------------
class CountMinSketch:
    """
    A lightweight Count‑Min Sketch using pairwise‑independent hash functions.
    Provides an upper‑bound estimate of the frequency of a key and,
    via the `distinct_estimate` method, an approximation of the number of
    distinct keys (based on the LogLog algorithm).
    """

    def __init__(self, width: int = 2_048, depth: int = 4, seed: int = 0):
        self.width = width
        self.depth = depth
        self.tables = np.zeros((depth, width), dtype=np.int64)
        rng = np.random.default_rng(seed)
        self.seeds = rng.integers(1, 2**31 - 1, size=depth, dtype=np.int64)

    def _hash(self, key: str, i: int) -> int:
        h = hashlib.blake2b(digest_size=8, person=bytes([i]))
        h.update(key.encode("utf-8"))
        h.update(self.seeds[i].to_bytes(4, "little"))
        return int.from_bytes(h.digest(), "little") % self.width

    def add(self, key: str, increment: int = 1) -> None:
        for i in range(self.depth):
            idx = self._hash(key, i)
            self.tables[i, idx] += increment

    def estimate(self, key: str) -> int:
        return min(self.tables[i, self._hash(key, i)] for i in range(self.depth))

    def distinct_estimate(self) -> int:
        """
        Simple LogLog estimator using the minimum count across rows.
        Not as accurate as HyperLogLog but sufficient for a lightweight demo.
        """
        mins = self.tables.min(axis=0)
        # The probability that a bucket stays zero after n distinct insertions is (1-1/width)^n
        # Inverting gives n ≈ -width * ln(zero_fraction)
        zero_fraction = np.mean(mins == 0)
        if zero_fraction == 0:
            # All buckets non‑zero → we are beyond the reliable range; fallback to width
            return self.width
        return int(-self.width * math.log(zero_fraction))


# ----------------------------------------------------------------------
# Core fusion logic – deeper mathematical integration
# ----------------------------------------------------------------------
def _aggregate_confidences(confidences: List[float]) -> float:
    """
    Combine multiple confidence scores into a single probability.
    Uses the normalized product of odds (log‑odds addition) which is
    equivalent to a Bayesian update with independent evidence.
    """
    if not confidences:
        return 0.5
    # Convert to odds, multiply, then back to probability
    odds = 1.0
    for c in confidences:
        # Clamp to avoid division by zero
        c = min(max(c, 1e-12), 1 - 1e-12)
        odds *= c / (1 - c)
    prob = odds / (1 + odds)
    return prob


def guide_labeling_function_results_with_epistemic_certainty(
    labeling_function_results: List[LabelingFunctionResult],
    certainty_flags: List[CertaintyFlag],
) -> List[ProbabilisticLabel]:
    """
    For each labeling function result we collect *all* matching certainty flags
    (by authority_class) and combine their confidences using a Bayesian‑style
    product of odds.  If no flag matches we fall back to a neutral 0.5 confidence.
    """
    # Index flags by authority_class for fast lookup
    flags_by_authority: Dict[str, List[CertaintyFlag]] = defaultdict(list)
    for flag in certainty_flags:
        flags_by_authority[flag.authority_class].append(flag)

    guided: List[ProbabilisticLabel] = []
    for res in labeling_function_results:
        matching = flags_by_authority.get(res.lf_name, [])
        if matching:
            combined_conf = _aggregate_confidences([f.confidence_bps / 10_000 for f in matching])
        else:
            combined_conf = 0.5
        guided.append(ProbabilisticLabel(res.doc_id, res.label, combined_conf))
    return guided


def aggregate_probabilistic_labels(
    probabilistic_labels: List[ProbabilisticLabel],
) -> List[ProbabilisticLabel]:
    """
    Merge multiple probabilistic labels that refer to the same document.
    The posterior confidence is computed as a weighted average where the
    weight equals the confidence of each observation (i.e. more certain
    votes dominate).  The resulting label is the MAP (maximum a posteriori)
    decision.
    """
    grouped: Dict[str, List[ProbabilisticLabel]] = defaultdict(list)
    for pl in probabilistic_labels:
        grouped[pl.doc_id].append(pl)

    merged: List[ProbabilisticLabel] = []
    for doc_id, items in grouped.items():
        # Weighted average of the *label* (0/1) using confidence as weight
        weights = np.array([p.confidence for p in items], dtype=float)
        labels = np.array([p.label for p in items], dtype=float)
        if weights.sum() == 0:
            posterior = 0.5
            label = 0
        else:
            posterior = np.dot(labels, weights) / weights.sum()
            label = int(posterior >= 0.5)
        merged.append(ProbabilisticLabel(doc_id, label, float(posterior)))
    return merged


def estimate_mean_reward_and_variance(
    probabilistic_labels: List[ProbabilisticLabel],
) -> Tuple[float, float]:
    """
    Compute the *expected* reward (label) and its variance under the
    posterior distribution supplied by `ProbabilisticLabel.confidence`.
    The reward for a document is the label (0/1) multiplied by the belief
    that the label is correct.
    """
    if not probabilistic_labels:
        return 0.0, 0.0

    # Expected reward per document
    rewards = np.array(
        [pl.label * pl.confidence for pl in probabilistic_labels], dtype=float
    )
    mean_reward = float(rewards.mean())

    # Weighted (unbiased) variance: Var(X) = E[w*(x-μ)^2] / (sum w - (sum w^2 / sum w))
    weights = np.array([pl.confidence for pl in probabilistic_labels], dtype=float)
    if weights.sum() <= 1:
        variance = 0.0
    else:
        diff = rewards - mean_reward
        weighted_sq = np.dot(weights, diff ** 2)
        correction = weights.sum() - (np.dot(weights, weights) / weights.sum())
        variance = float(weighted_sq / correction)
    return mean_reward, variance


def estimate_number_of_distinct_contexts(
    probabilistic_labels: List[ProbabilisticLabel],
    sketch_width: int = 2_048,
    sketch_depth: int = 4,
) -> int:
    """
    Approximate the number of distinct document identifiers using a
    Count‑Min Sketch combined with a LogLog estimator.
    """
    cms = CountMinSketch(width=sketch_width, depth=sketch_depth)
    for pl in probabilistic_labels:
        cms.add(pl.doc_id)
    return cms.distinct_estimate()


def find_label_errors(
    docs: List[Dict[str, Any]],
    given: List[int],
    probs: List[float],
    threshold: float = 0.65,
) -> List[LabelError]:
    """
    Identify documents where the model's confidence in the given label
    falls below `threshold`.  The suggested label is simply the opposite
    of the given label (binary case) and the error probability is
    1 - model confidence.
    """
    if not (len(docs) == len(given) == len(probs)):
        raise ValueError("Input lists must have the same length")
    errors: List[LabelError] = []
    for doc, given_label, prob in zip(docs, given, probs):
        if prob < threshold:
            errors.append(
                LabelError(
                    doc_id=doc["id"],
                    given_label=given_label,
                    suggested_label=1 - given_label,
                    error_probability=1 - prob,
                )
            )
    return errors


# ----------------------------------------------------------------------
# Example usage (smoke test)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic labeling function results
    labeling_function_results = [
        LabelingFunctionResult("lf1", "doc1", 1),
        LabelingFunctionResult("lf2", "doc2", 0),
        LabelingFunctionResult("lf1", "doc3", 1),
        LabelingFunctionResult("lf2", "doc1", 0),  # second opinion on doc1
    ]

    # Epistemic certainty flags – possibly multiple per authority
    certainty_flags = [
        certainty(
            "FACT",
            confidence_bps=9_000,
            authority_class="lf1",
            rationale="highly trusted source",
        ),
        certainty(
            "PROBABLE",
            confidence_bps=6_000,
            authority_class="lf2",
            rationale="moderate trust",
        ),
        certainty(
            "POSSIBLE",
            confidence_bps=4_000,
            authority_class="lf2",
            rationale="additional evidence",
        ),
    ]

    # Step 1 – guide raw LF results with epistemic priors
    guided = guide_labeling_function_results_with_epistemic_certainty(
        labeling_function_results, certainty_flags
    )

    # Step 2 – merge multiple observations per document
    merged = aggregate_probabilistic_labels(guided)

    # Step 3 – compute reward statistics
    mean_reward, variance = estimate_mean_reward_and_variance(merged)

    # Step 4 – estimate distinct contexts via sketch
    distinct_contexts = estimate_number_of_distinct_contexts(merged)

    print(
        f"Mean reward: {mean_reward:.4f}, Variance: {variance:.6f}, Distinct contexts: {distinct_contexts}"
    )

    # Demonstrate error detection
    docs = [{"id": pl.doc_id, "text": "placeholder"} for pl in merged]
    given_labels = [pl.label for pl in merged]
    probs = [pl.confidence for pl in merged]
    errors = find_label_errors(docs, given_labels, probs, threshold=0.7)
    if errors:
        print("Potential label errors detected:")
        for e in errors:
            print(
                f"  Doc {e.doc_id}: given {e.given_label}, suggest {e.suggested_label} (error prob {e.error_probability:.2f})"
            )
    else:
        print("No label errors above the confidence threshold.")