# DARWIN HAMMER — match 144, survivor 2
# gen: 4
# parent_a: hybrid_xgboost_objective_hybrid_ternary_lens__m33_s1.py (gen2)
# parent_b: hybrid_hybrid_label_foundry_hybrid_hybrid_decisi_m172_s0.py (gen3)
# born: 2026-05-29T23:27:11Z

import numpy as np
import math
import random
import sys
from pathlib import Path
from typing import Callable, List, Dict, Tuple, Iterable
from dataclasses import dataclass, field

# ----------------------------------------------------------------------
# Constants & Utility Patterns
# ----------------------------------------------------------------------
CLASSIFICATIONS = {"usable_now", "research_only", "needs_conversion", "unsafe_for_fastpath", "unsupported"}
LOCAL_PATTERNS = ["*bitnet*", "*BitNet*", "*fairyfuse*", "*FairyFuse*", "*lora*", "*LoRA*", "*adapter*"]

# ----------------------------------------------------------------------
# Core XGBoost‑style primitives
# ----------------------------------------------------------------------
def sigmoid(margin: np.ndarray | float) -> np.ndarray | float:
    """Numerically stable sigmoid."""
    m = np.asarray(margin, dtype=np.float64)
    # avoid overflow
    pos_mask = m >= 0
    neg_mask = ~pos_mask
    out = np.empty_like(m, dtype=np.float64)
    out[pos_mask] = 1.0 / (1.0 + np.exp(-m[pos_mask]))
    exp_m = np.exp(m[neg_mask])
    out[neg_mask] = exp_m / (1.0 + exp_m)
    return out if isinstance(margin, np.ndarray) else out.item()


def binary_logistic_grad_hess(
    y_true: np.ndarray,
    margin: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    First‑order gradient and second‑order hessian for binary logistic loss.
    Returns vectors of shape ``y_true.shape``.
    """
    p = sigmoid(margin)
    grad = p - y_true
    hess = p * (1.0 - p)
    return grad, hess


def optimal_leaf_weight(
    gradient_sum: float,
    hessian_sum: float,
    reg_lambda: float = 1.0,
) -> float:
    """Closed‑form optimal leaf weight used by XGBoost."""
    return -gradient_sum / (hessian_sum + reg_lambda)


def split_gain(
    left_gradient: float,
    left_hessian: float,
    right_gradient: float,
    right_hessian: float,
    *,
    reg_lambda: float = 1.0,
    gamma: float = 0.0,
) -> float:
    """
    Gain of a binary split.  ``gamma`` is the minimum loss reduction
    required to make a split; we subtract it at the end.
    """
    parent_gain = (left_gradient + right_gradient) ** 2 / (
        left_hessian + right_hessian + reg_lambda
    )
    left_gain = left_gradient ** 2 / (left_hessian + reg_lambda)
    right_gain = right_gradient ** 2 / (right_hessian + reg_lambda)
    return max(0.0, parent_gain - (left_gain + right_gain) - gamma)


# ----------------------------------------------------------------------
# Data structures for the labeling‑function side
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class LabelingFunctionResult:
    lf_name: str
    doc_id: str
    label: int  # binary label {0,1}
    score: float = 1.0  # optional confidence score supplied by the LF


@dataclass
class ProbabilisticLabel:
    doc_id: str
    label: int
    prob: float = field(default=0.0)


# ----------------------------------------------------------------------
# Decorator helper (kept for API compatibility)
# ----------------------------------------------------------------------
def labeling_function(name: str | None = None):
    """
    Simple decorator that tags a callable as a labeling function.
    The original function is returned unchanged; the name is stored
    as an attribute for later introspection.
    """

    def deco(fn: Callable[[Dict], int]) -> Callable[[Dict], int]:
        fn.lf_name = name or fn.__name__
        return fn

    return deco


# ----------------------------------------------------------------------
# Information‑richness utilities
# ----------------------------------------------------------------------
def shannon_entropy(logits: np.ndarray) -> float:
    """Shannon entropy of a probability distribution given un‑normalized logits."""
    # Use log‑sum‑exp for numerical stability
    max_logit = np.max(logits)
    shifted = logits - max_logit
    exp_shifted = np.exp(shifted)
    probs = exp_shifted / np.sum(exp_shifted)
    return -np.sum(probs * np.log(probs + 1e-12))


def information_weight(richness: float) -> float:
    """
    Convert a scalar ``richness`` (e.g. 0‑1) into a multiplicative weight.
    The sigmoid squashes extreme values while preserving monotonicity.
    """
    return 1.0 / (1.0 + np.exp(-richness))


# ----------------------------------------------------------------------
# Core hybrid operations
# ----------------------------------------------------------------------
def hybrid_labeling_function(
    loss_fn: Callable[[int], float],
    lf_result: LabelingFunctionResult,
    information_richness: float,
) -> int:
    """
    Decide whether to keep the original label or flip it.
    The decision balances the raw loss of the label with the
    information richness of the labeling function.
    """
    # raw loss for the current label
    raw_loss = loss_fn(lf_result.label)

    # richer LFs are trusted more → lower effective loss
    weight = information_weight(information_richness)
    adjusted_loss = raw_loss * weight

    # Threshold tuned for binary labels; 0.5 is the neutral point.
    return lf_result.label if adjusted_loss <= 0.5 else 1 - lf_result.label


def hybrid_prune_schedule(
    loss_fn: Callable[[int], float],
    lf_results: List[LabelingFunctionResult],
    information_richness: float,
    prune_gain_threshold: float = 0.0,
) -> List[LabelingFunctionResult]:
    """
    Prune labeling functions whose *potential* split gain,
    when weighted by information richness, falls below a threshold.
    This ties the XGBoost split‑gain calculation directly to the LF
    pruning decision.
    """
    # Compute gradient/hessian for each LF result using binary logistic loss
    # Treat the LF label as the "prediction" and the loss function as a proxy for true label.
    y_true = np.array([lf.label for lf in lf_results], dtype=np.float64)
    # Use the loss function to produce a pseudo‑margin; we map loss∈[0,1] → margin∈[-inf,inf]
    pseudo_margin = np.log1p(np.array([loss_fn(l) for l in y_true]))  # safe monotonic transform
    grad, hess = binary_logistic_grad_hess(y_true, pseudo_margin)

    # Aggregate gradients/hessians per labeling function name
    agg: Dict[str, Tuple[float, float, List[LabelingFunctionResult]]] = {}
    for lf, g, h in zip(lf_results, grad, hess):
        key = lf.lf_name
        if key not in agg:
            agg[key] = (0.0, 0.0, [])
        g_sum, h_sum, bucket = agg[key]
        agg[key] = (g_sum + g, h_sum + h, bucket + [lf])

    # Decide pruning based on split gain of a hypothetical split that isolates this LF
    kept: List[LabelingFunctionResult] = []
    for lf_name, (g_sum, h_sum, bucket) in agg.items():
        # Simulate a split where "left" = this LF, "right" = all others
        # Compute total gradient/hessian
        total_g = np.sum(grad)
        total_h = np.sum(hess)

        left_gain = split_gain(
            left_gradient=g_sum,
            left_hessian=h_sum,
            right_gradient=total_g - g_sum,
            right_hessian=total_h - h_sum,
            reg_lambda=1.0,
            gamma=0.0,
        )

        # Weight gain by information richness of the LF group (average)
        avg_richness = np.mean([lf.score for lf in bucket])
        weighted_gain = left_gain * information_weight(information_richness + avg_richness)

        if weighted_gain >= prune_gain_threshold:
            kept.extend(bucket)  # keep all results from this LF
        # else: drop them (pruned)

    return kept


def aggregate_labels(
    batches: Iterable[Iterable[LabelingFunctionResult]],
) -> List[ProbabilisticLabel]:
    """
    Aggregate votes across batches and produce a probability per (doc, label).
    The probability is the square‑root of the relative frequency, mirroring the
    original implementation but with a clear, type‑safe flow.
    """
    vote_counter: Dict[Tuple[str, str, int], int] = {}
    doc_counts: Dict[str, int] = {}

    for batch in batches:
        for result in batch:
            key = (result.lf_name, result.doc_id, result.label)
            vote_counter[key] = vote_counter.get(key, 0) + 1
            doc_counts[result.doc_id] = doc_counts.get(result.doc_id, 0) + 1

    aggregated: List[ProbabilisticLabel] = []
    for (lf_name, doc_id, label), cnt in vote_counter.items():
        prob = math.sqrt(cnt / doc_counts[doc_id])
        aggregated.append(ProbabilisticLabel(doc_id=doc_id, label=label, prob=prob))

    return aggregated


# ----------------------------------------------------------------------
# Example usage (kept for sanity‑check; not executed on import)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)

    # Dummy loss: identity (higher label → higher loss)
    def loss_fn(x: int) -> float:
        return float(x)

    # Create a tiny synthetic set of LF results
    lf_results = [
        LabelingFunctionResult(lf_name="lf1", doc_id="doc1", label=0, score=0.7),
        LabelingFunctionResult(lf_name="lf2", doc_id="doc1", label=1, score=0.4),
        LabelingFunctionResult(lf_name="lf1", doc_id="doc2", label=1, score=0.9),
        LabelingFunctionResult(lf_name="lf3", doc_id="doc2", label=0, score=0.2),
    ]

    # Prune using the hybrid schedule
    pruned = hybrid_prune_schedule(
        loss_fn,
        lf_results,
        information_richness=0.5,
        prune_gain_threshold=0.01,
    )
    print("Kept after pruning:", [r.lf_name for r in pruned])

    # Apply hybrid labeling to the first kept result
    if pruned:
        new_label = hybrid_labeling_function(loss_fn, pruned[0], information_richness=0.5)
        print("Adjusted label:", new_label)

    # Aggregate across a single batch for demonstration
    aggregated = aggregate_labels([pruned])
    for pl in aggregated:
        print(f"Doc {pl.doc_id} label {pl.label} prob {pl.prob:.3f}")