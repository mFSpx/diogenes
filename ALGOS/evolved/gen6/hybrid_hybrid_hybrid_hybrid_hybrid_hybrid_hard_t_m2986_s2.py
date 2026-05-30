# DARWIN HAMMER — match 2986, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_geomet_m530_s2.py (gen5)
# parent_b: hybrid_hybrid_hard_truth_ma_hybrid_hybrid_endpoi_m127_s0.py (gen3)
# born: 2026-05-29T23:47:11Z

import hashlib
import math
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np


# ----------------------------------------------------------------------
# Data structures with basic validation
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathAction:
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("MathAction.id must be a non‑empty string")
        if not math.isfinite(self.expected_value):
            raise ValueError("MathAction.expected_value must be a finite number")
        if not math.isfinite(self.cost):
            raise ValueError("MathAction.cost must be a finite number")
        if not math.isfinite(self.risk):
            raise ValueError("MathAction.risk must be a finite number")


@dataclass(frozen=True)
class MathCounterfactual:
    action_id: str
    outcome_value: float
    probability: float = 1.0

    def __post_init__(self) -> None:
        if not self.action_id:
            raise ValueError("MathCounterfactual.action_id must be a non‑empty string")
        if not math.isfinite(self.outcome_value):
            raise ValueError("MathCounterfactual.outcome_value must be a finite number")
        if not (0.0 <= self.probability <= 1.0):
            raise ValueError("MathCounterfactual.probability must be in [0, 1]")


# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def _hash(seed: int, token: str) -> int:
    """Deterministic 64‑bit hash based on Blake2b."""
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def signature(tokens: Iterable[str], k: int = 128) -> List[int]:
    """Min‑hash style signature of length *k*."""
    toks = sorted({t for t in tokens if t})
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [(1 << 64) - 1] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]


def similarity(sig_a: List[int], sig_b: List[int]) -> float:
    """Jaccard‑like similarity for equal‑length signatures."""
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have equal length")
    if not sig_a:
        raise ValueError("signatures must not be empty")
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)


def ternary_vector_similarity(vector_a: List[int], vector_b: List[int]) -> float:
    """Exact‑match similarity for integer vectors."""
    if len(vector_a) != len(vector_b):
        raise ValueError("vectors must have equal length")
    return sum(1 for a, b in zip(vector_a, vector_b) if a == b) / len(vector_a)


def brainmap_curvature(stylometry_features: List[float], recovery_priority: float) -> float:
    """Projects stylometry onto a uniform direction scaled by recovery priority."""
    if not stylometry_features:
        return 0.0
    return float(np.dot(stylometry_features, np.full(len(stylometry_features), recovery_priority)))


def stylometry_analysis(text: str) -> List[float]:
    """Very light stylometry: normalized token frequencies."""
    words = re.findall(r"[a-z]+(?:'[a-z]+)?", (text or "").lower())
    if not words:
        return []
    unique = set(words)
    return [words.count(w) / len(words) for w in unique]


# ----------------------------------------------------------------------
# Core hybrid model with Least‑Squares integration
# ----------------------------------------------------------------------
class HybridRegretLensModel:
    """
    Encapsulates the Regret‑Weighted Ternary Lens (RW‑TL) fused with
    Brain‑Map Curvature (BMC) using a Least‑Squares fit for the
    mixing coefficient *α*.
    """

    def __init__(self, alpha: float = 0.5):
        """
        α ∈ [0,1] balances regret‑weight (RW) and curvature (BMC):
            output = α·RW + (1‑α)·BMC
        The default 0.5 gives an equal blend.
        """
        self.alpha = float(np.clip(alpha, 0.0, 1.0))

    @staticmethod
    def _regret_weight(action: MathAction, counterfactual: MathCounterfactual) -> float:
        """Scalar regret weight via sigmoid of the value gap."""
        gap = action.expected_value - counterfactual.outcome_value
        return float(sigmoid(np.array([gap]))[0])

    def predict(
        self,
        action: MathAction,
        counterfactual: MathCounterfactual,
        stylometry_features: List[float],
        recovery_priority: float,
    ) -> float:
        """Compute the hybrid output for a single observation."""
        rw = self._regret_weight(action, counterfactual)
        bmc = brainmap_curvature(stylometry_features, recovery_priority)
        return self.alpha * rw + (1.0 - self.alpha) * bmc

    def fit(
        self,
        dataset: List[Tuple[MathAction, MathCounterfactual, List[float], float, float]],
        regularization: float = 1e-6,
    ) -> None:
        """
        Fit α by minimizing the squared error between the model prediction
        and the provided target output.

        *dataset* entries are:
            (action, counterfactual, stylometry_features, recovery_priority, target_output)

        The solution is closed‑form because the model is linear in α.
        """
        if not dataset:
            raise ValueError("Training dataset cannot be empty")

        # Build design matrix A where column 0 = RW, column 1 = BMC
        A = []
        y = []
        for action, cf, feats, rp, target in dataset:
            rw = self._regret_weight(action, cf)
            bmc = brainmap_curvature(feats, rp)
            A.append([rw, bmc])
            y.append(target)

        A = np.asarray(A, dtype=float)
        y = np.asarray(y, dtype=float)

        # Least‑Squares with simple linear constraint 0 ≤ α ≤ 1
        # Solve min‖A·[α, 1‑α]ᵀ ‑ y‖²  →  α = ( (rw‑bmc)·(y‑bmc) ) / ‖rw‑bmc‖²
        diff = A[:, 0] - A[:, 1]
        numerator = np.dot(diff, y - A[:, 1])
        denominator = np.dot(diff, diff) + regularization
        alpha_opt = np.clip(numerator / denominator, 0.0, 1.0)

        self.alpha = float(alpha_opt)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(alpha={self.alpha:.4f})"


# ----------------------------------------------------------------------
# Demonstration / simple sanity test
# ----------------------------------------------------------------------
def _demo() -> None:
    # Synthetic tiny dataset
    actions = [
        MathAction("a1", expected_value=10.0),
        MathAction("a2", expected_value=5.0),
        MathAction("a3", expected_value=7.5),
    ]
    counterfactuals = [
        MathCounterfactual("a1", outcome_value=8.0),
        MathCounterfactual("a2", outcome_value=6.0),
        MathCounterfactual("a3", outcome_value=7.0),
    ]
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
        "Data science blends statistics, mathematics, and programming.",
    ]
    recovery_priorities = [0.3, 0.7, 0.5]

    # Build training tuples with a hand‑crafted target (for illustration)
    dataset = []
    for act, cf, txt, rp in zip(actions, counterfactuals, texts, recovery_priorities):
        feats = stylometry_analysis(txt)
        # Target is a convex combination with unknown α (we use 0.6 for generation)
        target = 0.6 * HybridRegretLensModel._regret_weight(act, cf) + 0.4 * brainmap_curvature(feats, rp)
        dataset.append((act, cf, feats, rp, target))

    model = HybridRegretLensModel(alpha=0.5)
    print("Before fit:", model)
    model.fit(dataset)
    print("After fit :", model)

    # Predict on a fresh example
    new_action = MathAction("new", expected_value=9.0)
    new_cf = MathCounterfactual("new", outcome_value=7.5)
    new_feats = stylometry_analysis("Artificial intelligence drives modern research.")
    new_rp = 0.6
    pred = model.predict(new_action, new_cf, new_feats, new_rp)
    print("Prediction on new sample:", pred)


if __name__ == "__main__":
    _demo()