# DARWIN HAMMER — match 1431, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_regret_m946_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hard_t_hybrid_sparse_wta_hy_m1093_s2.py (gen3)
# born: 2026-05-29T23:36:11Z

import numpy as np
import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

# ----------------------------------------------------------------------
# Module Docstring
# ----------------------------------------------------------------------
"""
This module integrates the mathematical structures of two parent algorithms:
- `hybrid_hybrid_hybrid_rbf_su_hybrid_capybara_opti_m52_s1.py` (Parent A)
- `hybrid_hybrid_hybrid_hard_t_hybrid_sparse_wta_hy_m1093_s2.py` (Parent B)

The fusion is based on the interface between the `RBFSurrogate` class in Parent A
and the `stylometry_features` function in Parent B. Specifically, we use the output
of `stylometry_features` as the input to the `predict` method of `RBFSurrogate`.

This hybrid algorithm combines the strengths of both parents: the Gaussian process
regression in Parent A and the stylometry features in Parent B.
"""

# ----------------------------------------------------------------------
# Gaussian process regression (Parent A)
# ----------------------------------------------------------------------
Vector = Sequence[float]

def gaussian(r: float, epsilon: float = 1.0) -> float:
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: Vector, b: Vector) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

@dataclass(frozen=True)
class RBFSurrogate:
    centers: list[tuple[float, ...]]
    weights: list[float]
    epsilon: float = 1.0

    def predict(self, x: Vector) -> float:
        return sum(w * gaussian(euclidean(x, c), self.epsilon) for w, c in zip(self.weights, self.centers))

# ----------------------------------------------------------------------
# Stylometry features (Parent B)
# ----------------------------------------------------------------------
FUNCTION_CATS = {
    "pronoun": set(
        "i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()
    ),
    "article": set("a an the".split()),
    "preposition": set(
        "about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()
    ),
    "auxiliary": set(
        "am are be been being can could did do does had has have is may might must shall should was were will would".split()
    ),
    "conjunction": set("and but or nor so yet because although while".split()),
}


def stylometry_features(text: str) -> np.ndarray:
    """Return a fixed‑size numeric vector describing ``text``."""
    tokens = text.lower().split()
    n_words = len(tokens)
    n_chars = len(text)
    avg_word_len = np.mean([len(t) for t in tokens]) if tokens else 0.0

    # Category frequencies
    cat_counts = {cat: 0 for cat in FUNCTION_CATS}
    for token in tokens:
        for cat, vocab in FUNCTION_CATS.items():
            if token in vocab:
                cat_counts[cat] += 1

    cat_freqs = [cat_counts[cat] / n_words if n_words else 0.0 for cat in sorted(FUNCTION_CATS)]

    return np.array([n_words, n_chars, avg_word_len, *cat_freqs], dtype=float)

# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------
def hybrid_predict(text: str, surrogate: RBFSurrogate) -> float:
    """Use stylometry features as input to the RBFSurrogate."""
    features = stylometry_features(text)
    return surrogate.predict(features)

def hybrid_compute_regret_weighted_strategy(actions: list, counterfactuals: list, surrogate: RBFSurrogate) -> dict[str, float]:
    """Compute the regret-weighted strategy using the hybrid predict function."""
    cf = {c.action_id: c.outcome_value * c.probability for c in counterfactuals}
    vals = {a.id: hybrid_predict(a.expected_value, surrogate) for a in actions}
    best = max(vals.values())
    w = {k: math.exp(v - best) for k, v in vals.items()}
    total = sum(w.values()) or 1.0
    return {k: v / total for k, v in w.items()}

def hybrid_social_interaction(x: np.ndarray, g_best: np.ndarray, k: int = 1, r1: float | None = None, seed: int | str | None = None) -> np.ndarray:
    """Social interaction function using the hybrid predict function."""
    if r1 is None:
        raise ValueError("r1 must be provided")
    return np.array([x[i] + r1 * (g_best[i] - x[i]) for i in range(len(x))])

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    surrogate = RBFSurrogate(centers=[(1.0, 2.0), (3.0, 4.0)], weights=[0.5, 0.5])
    actions = [MathAction(id="action1", expected_value=10.0, cost=5.0), MathAction(id="action2", expected_value=20.0, cost=10.0)]
    counterfactuals = [MathCounterfactual(action_id="action1", outcome_value=15.0, probability=0.5), MathCounterfactual(action_id="action2", outcome_value=25.0, probability=0.5)]
    print(hybrid_compute_regret_weighted_strategy(actions, counterfactuals, surrogate))
    text = "This is a sample text."
    print(hybrid_predict(text, surrogate))
    x = np.random.rand(10)
    g_best = np.random.rand(10)
    print(hybrid_social_interaction(x, g_best, r1=0.5))