# DARWIN HAMMER — match 4752, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2588_s0.py (gen6)
# parent_b: hybrid_xgboost_objective_hybrid_ternary_lens__m33_s2.py (gen2)
# born: 2026-05-29T23:58:00Z

import math
import sys
import pathlib
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Any, Iterable, Tuple, List, Sequence

import numpy as np

# ----------------------------------------------------------------------
# Epistemic flag definitions
# ----------------------------------------------------------------------
EPISTEMIC_FLAGS: Tuple[str, ...] = (
    "FACT",
    "PROBABLE",
    "POSSIBLE",
    "BULLSHIT",
    "SURE_MAYBE",
)

# ----------------------------------------------------------------------
# Core data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class CertaintyFlag:
    """
    Immutable container for a single epistemic assessment.
    """
    label: str
    confidence_bps: int
    authority_class: str
    rationale: str
    evidence_refs: Tuple[str, ...] = ()
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc)
                                            .isoformat()
                                            .replace("+00:00", "Z"))

    def __post_init__(self) -> None:
        if self.label not in EPISTEMIC_FLAGS:
            raise ValueError(f"unknown epistemic flag: {self.label!r}")
        if not 0 <= int(self.confidence_bps) <= 10000:
            raise ValueError("confidence_bps must be in 0..10000")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def certainty(
    label: str,
    *,
    confidence_bps: int,
    authority_class: str,
    rationale: str,
    evidence_refs: Iterable[str] = (),
) -> CertaintyFlag:
    """
    Helper to create a CertaintyFlag with minimal boilerplate.
    """
    return CertaintyFlag(
        label=label,
        confidence_bps=int(confidence_bps),
        authority_class=authority_class,
        rationale=rationale,
        evidence_refs=tuple(str(x) for x in evidence_refs if x is not None),
    )


# ----------------------------------------------------------------------
# Geometric‑algebra inspired multivector wrapper
# ----------------------------------------------------------------------
@dataclass
class Multivector:
    """
    Light‑weight multivector that stores a sequence of CertaintyFlag objects
    and provides vectorised algebraic operations on their numeric
    representations (confidence, margin, etc.).
    """
    flags: List[CertaintyFlag]

    def __post_init__(self) -> None:
        if not self.flags:
            raise ValueError("Multivector must contain at least one CertaintyFlag")

    # ------------------------------------------------------------------
    # Numeric projections
    # ------------------------------------------------------------------
    @property
    def confidence(self) -> np.ndarray:
        """Confidence expressed as a probability in (0, 1)."""
        raw = np.array([f.confidence_bps for f in self.flags], dtype=float) / 10000.0
        # Clip to avoid exact 0 / 1 which break the logit transform
        return np.clip(raw, 1e-12, 1 - 1e-12)

    @property
    def margin(self) -> np.ndarray:
        """Logit (log‑odds) of the confidence vector."""
        return np.log(self.confidence / (1.0 - self.confidence))

    @property
    def label_weight(self) -> np.ndarray:
        """
        Map epistemic labels to a target probability y ∈ [0, 1].
        The mapping is heuristic but respects the intended ordering.
        """
        mapping = {
            "FACT": 1.0,
            "PROBABLE": 0.85,
            "POSSIBLE": 0.60,
            "SURE_MAYBE": 0.40,
            "BULLSHIT": 0.0,
        }
        return np.array([mapping[f.label] for f in self.flags], dtype=float)

    # ------------------------------------------------------------------
    # Algebraic operations
    # ------------------------------------------------------------------
    def logistic_grad_hess(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Vectorised first and second derivatives of binary logistic loss
        for the whole multivector.
        """
        p = sigmoid(self.margin)
        g = p - self.label_weight
        h = p * (1.0 - p)
        return g, h

    def aggregate(self) -> tuple[float, float]:
        """
        Sum of gradients and hessians – the scalar “blade” used by the
        split‑gain formula.
        """
        g, h = self.logistic_grad_hess()
        return float(g.sum()), float(h.sum())

    # ------------------------------------------------------------------
    # Partitioning utilities (used for a more realistic split gain)
    # ------------------------------------------------------------------
    def partition(self, threshold: float) -> Tuple["Multivector", "Multivector"]:
        """
        Split the multivector into two based on confidence threshold.
        Returns (left, right) where left.confidence >= threshold.
        """
        left_flags = [f for f in self.flags if f.confidence_bps / 10000.0 >= threshold]
        right_flags = [f for f in self.flags if f not in left_flags]
        return Multivector(left_flags), Multivector(right_flags)


# ----------------------------------------------------------------------
# Core mathematical primitives
# ----------------------------------------------------------------------
def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    """Numerically stable sigmoid."""
    x = np.asarray(x, dtype=float)
    # For large negative values, exp(-x) overflows; use alternative formulation
    pos_mask = x >= 0
    neg_mask = ~pos_mask
    out = np.empty_like(x, dtype=float)
    out[pos_mask] = 1.0 / (1.0 + np.exp(-x[pos_mask]))
    exp_x = np.exp(x[neg_mask])
    out[neg_mask] = exp_x / (1.0 + exp_x)
    return out if out.shape else out.item()


def optimal_leaf_weight(gradient_sum: float, hessian_sum: float, reg_lambda: float = 1.0) -> float:
    """Closed‑form optimal leaf weight for XGBoost‑style regularised loss."""
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
    XGBoost split‑gain formula. Returns the improvement in objective if a node
    is split into left/right children.
    """
    left_term = left_gradient ** 2 / (left_hessian + reg_lambda)
    right_term = right_gradient ** 2 / (right_hessian + reg_lambda)
    parent_term = (left_gradient + right_gradient) ** 2 / (left_hessian + right_hessian + reg_lambda)
    return left_term + right_term - parent_term - gamma


# ----------------------------------------------------------------------
# Hybrid pruning operation – the integration point
# ----------------------------------------------------------------------
def hybrid_pruning_operation(
    certainty_flags: List[CertaintyFlag],
    lambda_: float,
    alpha: float,
    reg_lambda: float = 1.0,
    gamma: float = 0.0,
    split_threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a per‑instance pruning margin that respects both the geometric
    (multivector) and XGBoost‑style split‑gain perspectives.

    Returns
    -------
    modulated_pruning_margin : np.ndarray
        The final pruning probability for each flag (values in (0, 1)).
    margins : np.ndarray
        The raw exponential‑decay schedule before modulation.
    """
    # 1️⃣ Build the multivector representation
    mv = Multivector(certainty_flags)

    # 2️⃣ Aggregate gradient/hessian for the whole node (parent)
    G_parent, H_parent = mv.aggregate()

    # 3️⃣ Partition the node to obtain a realistic left/right split
    left_mv, right_mv = mv.partition(split_threshold)

    # Guard against empty partitions – fall back to a no‑split scenario
    if not left_mv.flags or not right_mv.flags:
        gain = 0.0
    else:
        G_left, H_left = left_mv.aggregate()
        G_right, H_right = right_mv.aggregate()
        gain = split_gain(
            left_gradient=G_left,
            left_hessian=H_left,
            right_gradient=G_right,
            right_hessian=H_right,
            reg_lambda=reg_lambda,
            gamma=gamma,
        )

    # 4️⃣ Exponential‑decay schedule (one value per flag)
    t = np.arange(len(certainty_flags), dtype=float)
    pruning_margin = lambda_ * np.exp(-alpha * t)          # shape (n,)

    # 5️⃣ Modulation by split gain – sigmoid maps gain ∈ ℝ to (0,1)
    modulation = sigmoid(gain)                           # scalar
    modulated_pruning_margin = pruning_margin * modulation

    # Clip to a sensible probability range
    modulated_pruning_margin = np.clip(modulated_pruning_margin, 0.0, 1.0)

    return modulated_pruning_margin, pruning_margin


# ----------------------------------------------------------------------
# Demonstration entry‑point
# ----------------------------------------------------------------------
def main() -> None:
    certainty_flags = [
        certainty(
            "FACT",
            confidence_bps=8000,
            authority_class="high",
            rationale="very certain",
        ),
        certainty(
            "PROBABLE",
            confidence_bps=6000,
            authority_class="medium",
            rationale="somewhat certain",
        ),
        certainty(
            "POSSIBLE",
            confidence_bps=4000,
            authority_class="low",
            rationale="not very certain",
        ),
    ]

    lambda_ = 1.0
    alpha = 0.1

    modulated, raw = hybrid_pruning_operation(
        certainty_flags,
        lambda_=lambda_,
        alpha=alpha,
        reg_lambda=1.0,
        gamma=0.0,
        split_threshold=0.55,
    )

    # Simple human‑readable output
    for flag, prob in zip(certainty_flags, modulated):
        print(f"{flag.label:10s} → pruning probability: {prob:.4f}")

    # Show the underlying schedule for curiosity
    print("\nRaw exponential schedule:", raw)


if __name__ == "__main__":
    main()