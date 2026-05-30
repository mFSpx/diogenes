# DARWIN HAMMER — match 5099, survivor 4
# gen: 6
# parent_a: hybrid_hybrid_hybrid_nlms_o_hybrid_hybrid_hybrid_m1475_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hoeffding_tree_m1996_s5.py (gen5)
# born: 2026-05-29T23:59:56Z

"""
Hybrid Algorithm: Fusing NLMS with Hybrid Decision-Hygiene & Minimum-Cost Epistemic Tree and Bandit Router & Path Signature with Hoeffding Bound

Parents
-------
* **hybrid_hybrid_nlms_omni_cha_hybrid_hybrid_hybrid_m1475_s0.py** – A NLMS (Normalized Least Mean Squares) algorithm 
  with chaotic sprint mechanism and hybrid decision-hygiene & minimum-cost epistemic tree.
* **hybrid_hybrid_hybrid_hoeffding_tree_m1996_s5.py** – A hybrid algorithm combining textual feature extraction 
  with Hoeffding-bound split decisions.

The mathematical bridge between these two structures lies in the representation of the extracted feature vector as a 
high-dimensional random feature vector **x ∈ ℝ^d**, where each component of the vector is treated as a streaming numeric 
attribute. We use the Hoeffding bound to decide whether to split each component, and integrate the NLMS algorithm to 
adapt to changing conditions by updating the weights based on the prediction error.

Mathematical Interface:
- Given a high-dimensional random feature vector **x ∈ ℝ^d**, we compute the reduction of variance (gain) for each 
  component **G(k,θ) = Var(x_k) – (n_L/n)·Var(x_k | x_k ≤ θ) – (n_R/n)·Var(x_k | x_k > θ)**.
- The Hoeffding bound gives a confidence interval on **G** using the range **r** of the gain (here bounded by the 
  variance of a unit-range feature, i.e. **r = 1**).
- If the observed gap between the best and second-best candidate exceeds **ε**, we perform the split and update the 
  weights using the NLMS algorithm.
"""

import numpy as np
import math
import random
import sys
import pathlib

from collections import Counter
from typing import Any, Dict, Iterable, List, Tuple

# ----------------------------------------------------------------------
# Core NLMS utilities
# ----------------------------------------------------------------------

def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Return the dot-product prediction w·x."""
    return float(weights @ x)


def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, float]:
    """
    Perform one NLMS weight update.

    Parameters
    ----------
    weights : np.ndarray
        Current weight vector (1-D).
    x : np.ndarray
        Input feature vector.
    target : float
        Desired scalar output.
    mu : float
        Learning rate in (0, 2).
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    new_weights : np.ndarray
        Updated weight vector.
    error : float
        Prediction error `target – w·x`.
    """
    if not (0.0 < mu < 2.0):
        raise ValueError("mu must be in the interval (0, 2)")

    # Compute prediction error
    error = target - nlms_predict(weights, x)

    # Update weights
    new_weights = weights + mu * x * error

    return new_weights, error


# ----------------------------------------------------------------------
# Parent A – feature extraction utilities
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
        "no not never none neither cannot can't won't don't didn't isn't aren't was wasn't weren't".split()
    ),
}

def extract_features(text: str) -> np.ndarray:
    """Extract features from text using Parent A's method."""
    # Tokenize text
    tokens = text.split()

    # Initialize feature vector
    features = np.zeros((len(tokens), len(FUNCTION_CATS)))

    # Iterate over tokens
    for i, token in enumerate(tokens):
        # Get function category of token
        category = get_function_category(token)

        # Update feature vector
        features[i, category] = 1

    return features


def get_function_category(token: str) -> int:
    """Return the function category of a token."""
    for category, words in FUNCTION_CATS.items():
        if token in words:
            return list(FUNCTION_CATS.keys()).index(category)

    return 0


# ----------------------------------------------------------------------
# Hoeffding Tree utilities
# ----------------------------------------------------------------------
def hoeffding_bound(gain: float, variance: float, sample_size: int, confidence: float = 0.95) -> float:
    """Return the Hoeffding bound for a given gain and variance."""
    r = 1  # Range of gain (unit-range feature)
    epsilon = 2 * r * np.sqrt(np.log(2 / confidence) / (2 * sample_size))
    return epsilon


def hoeffding_tree(features: np.ndarray, target: float, sample_size: int = 100, confidence: float = 0.95) -> Tuple[int, float]:
    """Perform Hoeffding tree split decision."""
    # Compute gain for each feature
    gains = np.zeros(len(features[0]))
    for i in range(len(features[0])):
        gain = np.mean(features[:, i]) - np.var(features[:, i])
        gains[i] = gain

    # Get best and second-best candidates
    best_candidate = np.argmax(gains)
    second_best_candidate = np.argsort(gains)[-2]

    # Compute Hoeffding bound
    epsilon = hoeffding_bound(np.max(gains), np.var(features[:, best_candidate]), sample_size, confidence)

    # Check if observed gap exceeds Hoeffding bound
    observed_gap = gains[best_candidate] - gains[second_best_candidate]
    if observed_gap > epsilon:
        return best_candidate, observed_gap

    return -1, 0.0


# ----------------------------------------------------------------------
# Hybrid utilities
# ----------------------------------------------------------------------
def hybrid_nlms_hoeffding(features: np.ndarray, target: float, sample_size: int = 100, confidence: float = 0.95) -> Tuple[np.ndarray, float]:
    """Perform hybrid NLMS-Hoeffding decision."""
    # Extract features
    extracted_features = extract_features(features)

    # Perform Hoeffding tree split decision
    candidate, gap = hoeffding_tree(extracted_features, target, sample_size, confidence)

    # Update weights using NLMS algorithm
    if candidate != -1:
        weights = np.random.rand(len(extracted_features[0]))
        x = extracted_features[:, candidate]
        new_weights, error = nlms_update(weights, x, target)
        return new_weights, error
    else:
        return np.random.rand(len(extracted_features[0])), 0.0


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Smoke test
    features = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    target = 1.0
    sample_size = 100
    confidence = 0.95

    new_weights, error = hybrid_nlms_hoeffding(features, target, sample_size, confidence)

    print("New Weights:", new_weights)
    print("Error:", error)