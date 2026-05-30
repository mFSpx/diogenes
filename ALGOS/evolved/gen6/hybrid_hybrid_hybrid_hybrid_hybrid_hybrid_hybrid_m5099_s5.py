# DARWIN HAMMER — match 5099, survivor 5
# gen: 6
# parent_a: hybrid_hybrid_hybrid_nlms_o_hybrid_hybrid_hybrid_m1475_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hoeffding_tree_m1996_s5.py (gen5)
# born: 2026-05-29T23:59:56Z

"""Hybrid NLMS‑Hoeffding Tree
================================

Parent A: *hybrid_hybrid_hybrid_nlms_o_hybrid_hybrid_hybrid_m1475_s0.py* – provides a
Normalized Least‑Mean‑Squares (NLMS) adaptive linear model.

Parent B: *hybrid_hybrid_hybrid_hybrid_hoeffding_tree_m1996_s5.py* – provides a
streaming Hoeffding‑bound split decision based on variance‑reduction gain.

Mathematical bridge
-------------------
Both parents operate on a **feature vector  x ∈ ℝᵈ**.  
The NLMS part learns a weight vector **w** that predicts a scalar target **y** by
`ŷ = w·x`.  The prediction error **e = y – ŷ** can be interpreted as a
*residual* that measures how well the current linear model explains the data
in a leaf.

The Hoeffding‑tree part decides whether a leaf should be split by comparing
the variance‑reduction gain **G(k,θ)** of a candidate threshold **θ** on dimension
*k** with the Hoeffding bound **ε**.  By feeding the *residuals* **e** as the
streaming target for the gain computation, the split decision becomes
aware of the NLMS modelling error: dimensions that consistently produce large
errors obtain a higher variance and are therefore more likely to be split.

The fused algorithm therefore:

1. extracts a high‑dimensional textual feature vector,
2. updates an NLMS predictor inside each leaf,
3. accumulates sufficient statistics of the *residuals* per dimension,
4. periodically evaluates Hoeffding‑bound split candidates on those residuals,
5. creates child nodes that inherit fresh NLMS weights.

The code below implements this hybrid system in a compact, testable form.
"""

import sys
import math
import random
from pathlib import Path
from collections import Counter
from typing import List, Tuple, Dict, Any

import numpy as np

# ----------------------------------------------------------------------
# Parent A – textual feature extraction (simplified)
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
    "conjunction": set(
        "and but or nor so yet because although while if when where whereas unless until".split()
    ),
    "negation": set(
        "no not never none neither cannot cant wont dont didnt isnt arent wasnt werent".split()
    ),
}


def extract_features(text: str) -> np.ndarray:
    """
    Very lightweight bag‑of‑categories representation.
    Returns a normalized vector of length ``len(FUNCTION_CATS)``.
    """
    tokens = [t.strip(".,!?;:()[]\"'").lower() for t in text.split()]
    counts = np.zeros(len(FUNCTION_CATS), dtype=float)
    for idx, (cat, vocab) in enumerate(FUNCTION_CATS.items()):
        counts[idx] = sum(1 for t in tokens if t in vocab)
    norm = np.linalg.norm(counts) + 1e-12
    return counts / norm


# ----------------------------------------------------------------------
# Parent A – NLMS core (unchanged)
# ----------------------------------------------------------------------
def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Dot‑product prediction w·x."""
    return float(weights @ x)


def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, float]:
    """
    One NLMS update step.

    Returns
    -------
    new_weights : np.ndarray
        Updated weight vector.
    error : float
        Prediction error ``target – w·x``.
    """
    if not (0.0 < mu < 2.0):
        raise ValueError("mu must be in the interval (0, 2)")

    y_hat = nlms_predict(weights, x)
    error = target - y_hat
    norm_x = np.dot(x, x) + eps
    new_weights = weights + (mu * error / norm_x) * x
    return new_weights, error


# ----------------------------------------------------------------------
# Parent B – Hoeffding‑tree utilities (gain & bound)
# ----------------------------------------------------------------------
def variance(arr: np.ndarray) -> float:
    """Sample variance (unbiased)."""
    if arr.size == 0:
        return 0.0
    return float(arr.var(ddof=1))


def gain_variance_reduction(
    parent_vals: np.ndarray,
    left_vals: np.ndarray,
    right_vals: np.ndarray,
) -> float:
    """Reduction of variance achieved by a split."""
    n = parent_vals.size
    nL = left_vals.size
    nR = right_vals.size
    if n == 0:
        return 0.0
    parent_var = variance(parent_vals)
    left_var = variance(left_vals)
    right_var = variance(right_vals)
    return parent_var - (nL / n) * left_var - (nR / n) * right_var


def hoeffding_epsilon(n: int, R: float = 1.0, delta: float = 0.05) -> float:
    """
    Hoeffding bound ε = sqrt( (R^2 * ln(1/δ)) / (2n) )
    """
    if n == 0:
        return float("inf")
    return math.sqrt((R * R * math.log(1.0 / delta)) / (2.0 * n))


# ----------------------------------------------------------------------
# Hybrid node that couples NLMS with Hoeffding split logic
# ----------------------------------------------------------------------
class HybridNode:
    """
    A leaf node that maintains:
    * NLMS weights (linear predictor)
    * Streaming statistics of residuals per feature dimension
    * Counters for Hoeffding split evaluation
    """

    def __init__(self, dim: int, mu: float = 0.5):
        self.dim = dim
        self.mu = mu
        self.weights = np.zeros(dim, dtype=float)  # start with zero predictor
        self.n_samples = 0

        # Statistics of residuals per dimension for split evaluation
        self.residuals: List[float] = []  # residuals of the target (y)
        self.feature_history: List[np.ndarray] = []  # raw x vectors

        # Children (None means leaf)
        self.left: "HybridNode | None" = None
        self.right: "HybridNode | None" = None
        self.split_dim: int | None = None
        self.split_thr: float | None = None

    # ------------------------------------------------------------------
    # NLMS step (prediction + weight update)
    # ------------------------------------------------------------------
    def nlms_step(self, x: np.ndarray, y: float) -> float:
        """
        Predict with current weights, update them, and store residual.
        Returns the prediction error.
        """
        self.n_samples += 1
        self.weights, error = nlms_update(self.weights, x, y, mu=self.mu)
        self.residuals.append(error)
        self.feature_history.append(x.copy())
        return error

    # ------------------------------------------------------------------
    # Hoeffding split evaluation
    # ------------------------------------------------------------------
    def try_split(self, delta: float = 0.05, min_samples: int = 30) -> bool:
        """
        Attempt a split using Hoeffding bound on residual variance.
        Returns True if a split was performed.
        """
        if self.n_samples < min_samples or self.left is not None:
            return False

        # Convert stored data to arrays for vectorised ops
        X = np.vstack(self.feature_history)  # shape (n, dim)
        R = np.array(self.residuals)        # shape (n,)

        best_gain = -float("inf")
        best_dim = None
        best_thr = None

        # Examine each dimension independently; candidate thresholds are
        # the median of observed values (simple but effective).
        for d in range(self.dim):
            values = X[:, d]
            if values.size < 2:
                continue
            thr = np.median(values)
            left_mask = values <= thr
            right_mask = ~left_mask
            if left_mask.sum() == 0 or right_mask.sum() == 0:
                continue

            # Gain is computed on residuals, not on raw values
            gain = gain_variance_reduction(
                R,
                R[left_mask],
                R[right_mask],
            )
            if gain > best_gain:
                best_gain, best_dim, best_thr = gain, d, thr

        if best_dim is None:
            return False

        # Hoeffding bound on the gain (range R = 1 because residuals are bounded
        # by the target range after normalisation – we assume unit range here).
        epsilon = hoeffding_epsilon(self.n_samples, R=1.0, delta=delta)

        # We also need the second‑best gain to apply the classic Hoeffding test.
        # For simplicity we recompute the second best in a second pass.
        second_best_gain = -float("inf")
        for d in range(self.dim):
            if d == best_dim:
                continue
            values = X[:, d]
            if values.size < 2:
                continue
            thr = np.median(values)
            left_mask = values <= thr
            right_mask = ~left_mask
            if left_mask.sum() == 0 or right_mask.sum() == 0:
                continue
            gain = gain_variance_reduction(
                R,
                R[left_mask],
                R[right_mask],
            )
            if gain > second_best_gain:
                second_best_gain = gain

        # Split if the observed gap exceeds epsilon
        if (best_gain - second_best_gain) > epsilon:
            self.split_dim = best_dim
            self.split_thr = best_thr
            self.left = HybridNode(self.dim, mu=self.mu)
            self.right = HybridNode(self.dim, mu=self.mu)

            # Distribute stored samples to children (re‑play them)
            for x_vec, y_res in zip(self.feature_history, self.residuals):
                target = y_res  # residual acts as new target for children
                if x_vec[best_dim] <= best_thr:
                    self.left.nlms_step(x_vec, target)
                else:
                    self.right.nlms_step(x_vec, target)

            # Clear leaf‑specific buffers to free memory
            self.residuals.clear()
            self.feature_history.clear()
            return True

        return False

    # ------------------------------------------------------------------
    # Routing of a new sample through the tree
    # ------------------------------------------------------------------
    def route(self, x: np.ndarray, y: float) -> float:
        """
        Process a sample (x, y).  Returns the prediction error at the leaf
        where the sample is finally handled.
        """
        if self.left is not None and self.right is not None:
            # internal node – forward to the appropriate child
            if x[self.split_dim] <= self.split_thr:
                return self.left.route(x, y)
            else:
                return self.right.route(x, y)
        else:
            # leaf node – NLMS update
            error = self.nlms_step(x, y)
            # attempt a split after the update
            self.try_split()
            return error


class HybridNLMSHoeffdingTree:
    """
    Wrapper exposing a simple ``fit`` / ``predict_error`` API.
    """

    def __init__(self, dim: int, mu: float = 0.5):
        self.root = HybridNode(dim, mu=mu)

    def process(self, x: np.ndarray, y: float) -> float:
        """
        Feed a single (feature, target) pair into the streaming tree.
        Returns the prediction error after the update.
        """
        return self.root.route(x, y)

    def predict(self, x: np.ndarray) -> float:
        """
        Produce a prediction by traversing to the appropriate leaf and
        applying its NLMS weights.
        """
        node = self.root
        while node.left is not None and node.right is not None:
            if x[node.split_dim] <= node.split_thr:
                node = node.left
            else:
                node = node.right
        return nlms_predict(node.weights, x)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    # Generate synthetic textual data with a simple sentiment target
    texts = [
        "I love this product, it is amazing and wonderful!",
        "This is terrible, I hate it and will never buy again.",
        "The item is okay, not good but not bad either.",
        "Absolutely fantastic experience, would recommend to everyone.",
        "Worst purchase ever, completely disappointing.",
    ]

    # Assign synthetic sentiment scores in [-1, 1]
    targets = [0.9, -0.9, 0.0, 0.95, -0.95]

    dim = len(FUNCTION_CATS)
    tree = HybridNLMSHoeffdingTree(dim=dim, mu=0.3)

    # Stream the data many times to trigger splits
    for epoch in range(200):
        idx = random.randrange(len(texts))
        txt = texts[idx]
        y = targets[idx]
        x = extract_features(txt)
        err = tree.process(x, y)
        if epoch % 50 == 0:
            pred = tree.predict(x)
            print(f"Epoch {epoch:3d} | err {err: .4f} | pred {pred: .4f}")

    print("\nTree statistics:")
    def count_nodes(node: HybridNode) -> int:
        if node.left is None and node.right is None:
            return 1
        return 1 + count_nodes(node.left) + count_nodes(node.right)
    print("Total nodes:", count_nodes(tree.root))
    print("Root split dim:", tree.root.split_dim, "thr:", tree.root.split_thr)