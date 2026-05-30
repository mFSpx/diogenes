# DARWIN HAMMER — match 5099, survivor 6
# gen: 6
# parent_a: hybrid_hybrid_hybrid_nlms_o_hybrid_hybrid_hybrid_m1475_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hoeffding_tree_m1996_s5.py (gen5)
# born: 2026-05-29T23:59:56Z

import sys
import math
import random
from pathlib import Path
from collections import Counter
from typing import List, Tuple, Dict, Any

import numpy as np

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
    tokens = [t.strip(".,!?;:()[]\"'").lower() for t in text.split()]
    counts = np.zeros(len(FUNCTION_CATS), dtype=float)
    for idx, (cat, vocab) in enumerate(FUNCTION_CATS.items()):
        counts[idx] = sum(1 for t in tokens if t in vocab)
    norm = np.linalg.norm(counts) + 1e-12
    return counts / norm


def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    return float(weights @ x)


def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, float]:
    if not (0.0 < mu < 2.0):
        raise ValueError("mu must be in the interval (0, 2)")

    y_hat = nlms_predict(weights, x)
    error = target - y_hat
    norm_x = np.dot(x, x) + eps
    new_weights = weights + (mu * error / norm_x) * x
    return new_weights, error


def variance(arr: np.ndarray) -> float:
    if arr.size == 0:
        return 0.0
    return float(arr.var(ddof=1))


def gain_variance_reduction(
    parent_vals: np.ndarray,
    left_vals: np.ndarray,
    right_vals: np.ndarray,
) -> float:
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
    if n == 0:
        return float("inf")
    return math.sqrt((R * R * math.log(1.0 / delta)) / (2.0 * n))


class HybridNode:
    def __init__(self, dim: int, mu: float = 0.5):
        self.dim = dim
        self.mu = mu
        self.weights = np.zeros(dim, dtype=float)
        self.n_samples = 0
        self.residuals: List[float] = []
        self.feature_history: List[np.ndarray] = []
        self.left: "HybridNode | None" = None
        self.right: "HybridNode | None" = None
        self.split_dim: int | None = None
        self.split_thr: float | None = None

    def nlms_step(self, x: np.ndarray, y: float) -> float:
        self.n_samples += 1
        self.weights, error = nlms_update(self.weights, x, y, mu=self.mu)
        self.residuals.append(error)
        self.feature_history.append(x.copy())
        return error

    def try_split(self, delta: float = 0.05, min_samples: int = 30) -> bool:
        if self.n_samples < min_samples or self.left is not None:
            return False

        X = np.vstack(self.feature_history)
        R = np.array(self.residuals)

        best_gain = -float("inf")
        best_dim = None
        best_thr = None

        for d in range(self.dim):
            values = X[:, d]
            if values.size < 2:
                continue
            thr = np.median(values)
            left_mask = values <= thr
            right_mask = ~left_mask

            if np.any(left_mask) and np.any(right_mask):
                left_vals = R[left_mask]
                right_vals = R[right_mask]
                gain = gain_variance_reduction(R, left_vals, right_vals)

                epsilon = hoeffding_epsilon(self.n_samples, R=np.max(np.abs(R)))
                if gain > best_gain and gain > epsilon:
                    best_gain = gain
                    best_dim = d
                    best_thr = thr

        if best_dim is not None:
            self.split_dim = best_dim
            self.split_thr = best_thr
            self.left = HybridNode(self.dim, self.mu)
            self.right = HybridNode(self.dim, self.mu)

            left_mask = X[:, best_dim] <= best_thr
            right_mask = ~left_mask

            self.left.feature_history = [x for x, m in zip(self.feature_history, left_mask) if m]
            self.left.residuals = [r for r, m in zip(self.residuals, left_mask) if m]
            self.right.feature_history = [x for x, m in zip(self.feature_history, right_mask) if m]
            self.right.residuals = [r for r, m in zip(self.residuals, right_mask) if m]

            # Initialize child nodes' weights
            self.left.weights = self.weights
            self.right.weights = self.weights

            return True
        return False


class HybridTree:
    def __init__(self, dim: int, mu: float = 0.5):
        self.root = HybridNode(dim, mu)

    def update(self, x: np.ndarray, y: float) -> None:
        current_node = self.root
        while current_node.left is not None:
            if x[current_node.split_dim] <= current_node.split_thr:
                current_node = current_node.left
            else:
                current_node = current_node.right

        error = current_node.nlms_step(x, y)
        current_node.try_split()

    def predict(self, x: np.ndarray) -> float:
        current_node = self.root
        while current_node.left is not None:
            if x[current_node.split_dim] <= current_node.split_thr:
                current_node = current_node.left
            else:
                current_node = current_node.right

        return nlms_predict(current_node.weights, x)


# Example usage
if __name__ == "__main__":
    dim = len(FUNCTION_CATS)
    tree = HybridTree(dim)

    text = "This is an example sentence."
    x = extract_features(text)
    y = 1.0  # dummy target
    tree.update(x, y)
    print(tree.predict(x))