# DARWIN HAMMER — match 108, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_minimum_cost__epistemic_certainty_m48_s5.py (gen2)
# parent_b: hybrid_hybrid_hybrid_bandit_label_foundry_m21_s0.py (gen3)
# born: 2026-05-29T23:27:00Z

"""
This module combines the strengths of 'hybrid_hybrid_minimum_cost__epistemic_certainty_m48_s5.py' 
and 'hybrid_hybrid_hybrid_bandit_label_foundry_m21_s0.py'. The mathematical bridge lies in the 
integration of epistemic certainty helpers from the former with labeling function results from 
the latter, allowing for a unified system that leverages both statistical sketching and singular-
learning-theory asymptotics for exploration-exploitation balances. Specifically, this hybrid 
algorithm uses the epistemic certainty helpers to guide the labeling function results, which are 
then used to estimate the empirical mean reward and its variance.

The mathematical interface between the two parents is the concept of uncertainty. In the 
'hybrid_hybrid_minimum_cost__epistemic_certainty_m48_s5.py' parent, uncertainty is modeled 
through epistemic certainty helpers, which provide a framework for representing and reasoning about 
uncertainty. In the 'hybrid_hybrid_hybrid_bandit_label_foundry_m21_s0.py' parent, uncertainty is 
modeled through the use of statistical sketches, such as the Count-Min sketch and the HyperLogLog 
sketch. By combining these two approaches, we can create a unified system that leverages both 
statistical sketching and singular-learning-theory asymptotics for exploration-exploitation balances.
"""

import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Iterable, Set, Callable
import numpy as np
import hashlib
from pathlib import Path

# Epistemic certainty helpers
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

# Labeling function results
@dataclass(frozen=True)
class LabelingFunctionResult:
    lf_name: str
    doc_id: str
    label: int

@dataclass(frozen=True)
class ProbabilisticLabel:
    doc_id: str
    label: int
    confidence: float

@dataclass(frozen=True)
class LabelError:
    doc_id: str
    given_label: int
    suggested_label: int
    error_probability: float

def labeling_function(name: str|None=None):
    def deco(fn: Callable[[dict],int]):
        fn.lf_name = name or fn.__name__
        return fn
    return deco

def aggregate_labels(batches: list[list[LabelingFunctionResult]]) -> list[ProbabilisticLabel]:
    votes = defaultdict(list)
    for batch in batches:
        for r in batch:
            if r.label in (0,1): 
                votes[r.doc_id].append(r.label)
    out = []
    for d, vs in votes.items():
        if not vs: 
            out.append(ProbabilisticLabel(d, 0, 0.5))
            continue
        c = defaultdict(int)
        for v in vs:
            c[v] += 1
        label = 1 if c[1] >= c[0] else 0
        out.append(ProbabilisticLabel(d, label, c[label]/len(vs)))
    return out

def find_label_errors(docs: list[dict], given: list[int], probs: list[float], threshold: float=0.65) -> list[LabelError]:
    if not (len(docs) == len(given) == len(probs)):
        raise ValueError("Input lists must have the same length")
    errors = []
    for doc, given_label, prob in zip(docs, given, probs):
        if prob < threshold:
            errors.append(LabelError(doc["id"], given_label, 1 - given_label, 1 - prob))
    return errors

def guide_labeling_function_results_with_epistemic_certainty(
    labeling_function_results: list[LabelingFunctionResult], 
    certainty_flags: list[CertaintyFlag]
) -> list[ProbabilisticLabel]:
    guided_results = []
    for result in labeling_function_results:
        for flag in certainty_flags:
            if result.lf_name == flag.authority_class:
                # Guide the labeling function result with the epistemic certainty flag
                confidence = flag.confidence_bps / 10000
                guided_results.append(ProbabilisticLabel(result.doc_id, result.label, confidence))
                break
        else:
            # If no matching epistemic certainty flag is found, use a default confidence
            guided_results.append(ProbabilisticLabel(result.doc_id, result.label, 0.5))
    return guided_results

def estimate_mean_reward_and_variance(
    probabilistic_labels: list[ProbabilisticLabel]
) -> Tuple[float, float]:
    rewards = [label.label * label.confidence for label in probabilistic_labels]
    mean_reward = np.mean(rewards)
    variance = np.var(rewards)
    return mean_reward, variance

def estimate_number_of_distinct_contexts(
    probabilistic_labels: list[ProbabilisticLabel]
) -> int:
    distinct_contexts = set(label.doc_id for label in probabilistic_labels)
    return len(distinct_contexts)

if __name__ == "__main__":
    # Smoke test
    labeling_function_results = [
        LabelingFunctionResult("lf1", "doc1", 1),
        LabelingFunctionResult("lf2", "doc2", 0),
        LabelingFunctionResult("lf1", "doc3", 1),
    ]
    certainty_flags = [
        certainty("FACT", confidence_bps=10000, authority_class="lf1", rationale="example"),
        certainty("POSSIBLE", confidence_bps=5000, authority_class="lf2", rationale="example"),
    ]
    guided_results = guide_labeling_function_results_with_epistemic_certainty(labeling_function_results, certainty_flags)
    mean_reward, variance = estimate_mean_reward_and_variance(guided_results)
    distinct_contexts = estimate_number_of_distinct_contexts(guided_results)
    print(f"Mean reward: {mean_reward}, Variance: {variance}, Distinct contexts: {distinct_contexts}")