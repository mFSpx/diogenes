# DARWIN HAMMER — match 5582, survivor 0
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1235_s1.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_ternar_m690_s0.py (gen4)
# born: 2026-05-30T00:03:09Z

"""Hybrid Algorithm integrating:
- Parent A: Regret‑aware Hoeffding bound + NLMS adaptive filter.
- Parent B: Regex‑based feature extraction, epistemic certainty flags, ternary routing, and bandit‑style policy update.

Mathematical Bridge
------------------
The bridge is a two‑fold coupling:

1. **Regret‑aware confidence scaling** – The Hoeffding bound computed from the
   observed gain and accumulated regret (Parent A) yields a scalar
   ``confidence_factor`` ∈ (0, 1].  This factor scales the NLMS step size,
   making the adaptive filter more conservative when regret (i.e. past error)
   is large.

2. **Feature‑driven policy augmentation** – Textual context is transformed
   into a feature vector via the regexes of Parent B.  The normalized feature
   frequencies constitute epistemic certainty flags ``c_i``.  These flags are
   concatenated to the original signal vector, expanding the NLMS input space.
   The same feature vector is used in a bandit‑style gradient update of the
   weight matrix, where the reward is derived from the instantaneous NLMS
   error.  Thus the graph‑like decision logic of Parent B directly influences
   the weight adaptation of the NLMS filter.

The resulting system performs a single hybrid step:

features   = extract_features(text)
c          = certainty_flags(features)
x_aug      = concat(input_vector, c)
bound      = regret_aware_hoeffding(observed_gain, delta, n, regret)
step_adj   = step_size * (1.0 / (1.0 + bound))
w_new      = nlms_update(w, x_aug, desired, step_adj)
reward     = -abs(desired - np.dot(w_new, x_aug))
w_new      = bandit_policy_update(w_new, c, reward, lr_bandit)
output     = ternary_route(np.dot(w_new, x_aug))

All operations are pure NumPy / standard‑library math, satisfying the
environment constraints.
"""

import math
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent‑A building blocks
# ----------------------------------------------------------------------
def compute_hoeffding_bound(observed_gain: float, delta: float, n: int) -> float:
    """Hoeffding bound for bounded gain."""
    if n <= 0:
        raise ValueError("n must be positive")
    return math.sqrt((observed_gain * math.log(2.0 / delta)) / (2.0 * n))


def compute_regret_aware_hoeffding_bound(
    observed_gain: float, delta: float, n: int, regret: float
) -> float:
    """Regret‑aware Hoeffding bound (Parent A)."""
    base = compute_hoeffding_bound(observed_gain, delta, n)
    return base * (1.0 + regret)


def nlms_update(
    weight_matrix: np.ndarray,
    input_vector: np.ndarray,
    desired_output: float,
    step_size: float,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """Standard Normalized LMS weight update."""
    # prediction
    y = float(weight_matrix @ input_vector)
    # error
    e = desired_output - y
    # normalization term
    norm = float(input_vector @ input_vector) + epsilon
    # weight correction
    delta_w = (step_size / norm) * e * input_vector
    return weight_matrix + delta_w


# ----------------------------------------------------------------------
# Parent‑B building blocks
# ----------------------------------------------------------------------
# Regexes for feature extraction (excerpt, can be extended)
EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)
PLANNING_RE = re.compile(
    r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
    re.I,
)
DELAY_RE = re.compile(r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now)\b", re.I)
QUALITY_RE = re.compile(r"\b(?:quality|high|low|grade|rating)\b", re.I)
SECURITY_RE = re.compile(r"\b(?:security|secure|vulnerability|exploit)\b", re.I)

FEATURE_REGEXES: List[Tuple[str, re.Pattern]] = [
    ("evidence", EVIDENCE_RE),
    ("planning", PLANNING_RE),
    ("delay", DELAY_RE),
    ("quality", QUALITY_RE),
    ("security", SECURITY_RE),
]


def extract_features(text: str) -> Counter:
    """Count occurrences of each feature regex in *text*."""
    counts = Counter()
    for name, pattern in FEATURE_REGEXES:
        matches = pattern.findall(text)
        if matches:
            counts[name] = len(matches)
    return counts


def certainty_flags(feature_counts: Counter) -> np.ndarray:
    """Normalize feature counts to obtain epistemic certainty flags."""
    total = sum(feature_counts.values())
    if total == 0:
        # No evidence – return a uniform low‑certainty vector
        return np.full(len(FEATURE_REGEXES), 1.0 / len(FEATURE_REGEXES))
    return np.array([feature_counts.get(name, 0) / total for name, _ in FEATURE_REGEXES])


def ternary_route(value: float, low: float = -0.5, high: float = 0.5) -> int:
    """Map a continuous scalar to {-1, 0, +1}."""
    if value < low:
        return -1
    if value > high:
        return 1
    return 0


def bandit_policy_update(
    weight_matrix: np.ndarray,
    certainty_vec: np.ndarray,
    reward: float,
    lr: float = 0.01,
) -> np.ndarray:
    """
    Simple bandit‑style gradient ascent on the weight matrix.
    The certainty vector is treated as a context; the outer product
    with the current weight row produces a low‑rank update.
    """
    # reshape for broadcasting
    c = certainty_vec.reshape(-1, 1)  # (features, 1)
    # gradient estimate proportional to reward and context
    grad = reward * c @ c.T  # rank‑1 matrix of shape (features, features)
    # expand to match weight_matrix shape if necessary
    if grad.shape != weight_matrix.shape:
        # Pad or truncate to fit
        grad_resized = np.zeros_like(weight_matrix)
        rows = min(grad.shape[0], weight_matrix.shape[0])
        cols = min(grad.shape[1], weight_matrix.shape[1])
        grad_resized[:rows, :cols] = grad[:rows, :cols]
        grad = grad_resized
    return weight_matrix + lr * grad


# ----------------------------------------------------------------------
# Hybrid core functions (minimum three)
# ----------------------------------------------------------------------
def hybrid_regret_aware_step(
    weight_matrix: np.ndarray,
    input_vector: np.ndarray,
    desired_output: float,
    step_size: float,
    observed_gain: float,
    delta: float,
    n: int,
    regret: float,
) -> np.ndarray:
    """
    Perform an NLMS update where the step size is modulated by the
    regret‑aware Hoeffding bound (Parent A).
    """
    bound = compute_regret_aware_hoeffding_bound(observed_gain, delta, n, regret)
    adj_step = step_size / (1.0 + bound)  # smaller step when bound is large
    return nlms_update(weight_matrix, input_vector, desired_output, adj_step)


def hybrid_feature_augmented_input(
    raw_input: np.ndarray, text_context: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert *text_context* into certainty flags (Parent B) and concatenate
    them to the raw signal vector, producing an augmented NLMS input.
    Returns (augmented_vector, certainty_vector).
    """
    feats = extract_features(text_context)
    cert = certainty_flags(feats)  # shape (F,)
    # Ensure both are 1‑D arrays
    raw = np.asarray(raw_input, dtype=float).ravel()
    aug = np.concatenate([raw, cert])
    return aug, cert


def hybrid_step(
    weight_matrix: np.ndarray,
    raw_input: np.ndarray,
    desired_output: float,
    text_context: str,
    step_size: float = 0.1,
    observed_gain: float = 1.0,
    delta: float = 0.05,
    n: int = 100,
    regret: float = 0.0,
    lr_bandit: float = 0.01,
) -> Tuple[np.ndarray, int, float]:
    """
    One complete hybrid iteration.

    Returns
    -------
    new_weights : np.ndarray
        Updated weight matrix after NLMS and bandit updates.
    ternary_decision : int
        {-1, 0, +1} decision obtained from the filtered output.
    error : float
        Absolute prediction error used as a proxy for regret.
    """
    # 1️⃣ Feature‑augmented signal
    x_aug, cert = hybrid_feature_augmented_input(raw_input, text_context)

    # 2️⃣ Regret‑aware NLMS adaptation
    w_nlms = hybrid_regret_aware_step(
        weight_matrix,
        x_aug,
        desired_output,
        step_size,
        observed_gain,
        delta,
        n,
        regret,
    )

    # 3️⃣ Compute instantaneous error → reward
    prediction = float(w_nlms @ x_aug)
    error = abs(desired_output - prediction)
    reward = -error  # negative error as reward (higher is better)

    # 4️⃣ Bandit policy refinement using certainty flags
    w_updated = bandit_policy_update(w_nlms, cert, reward, lr=lr_bandit)

    # 5️⃣ Ternary routing of the final output
    final_output = float(w_updated @ x_aug)
    decision = ternary_route(final_output)

    return w_updated, decision, error


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple synthetic scenario
    rng = np.random.default_rng(42)

    # Initial weight matrix (signal length 3 + 5 certainty flags = 8)
    w0 = rng.normal(scale=0.1, size=8)

    # Random signal vector (e.g., sensor readings)
    signal = rng.normal(size=3)

    # Desired scalar output (e.g., target voltage)
    target = 0.7

    # Example textual context containing some features
    text = (
        "The plan was verified with evidence. "
        "We need to check quality and security before deployment."
    )

    # Run a few hybrid steps to demonstrate stability
    w = w0
    cumulative_regret = 0.0
    for t in range(5):
        w, decision, err = hybrid_step(
            weight_matrix=w,
            raw_input=signal,
            desired_output=target,
            text_context=text,
            step_size=0.2,
            observed_gain=1.0,
            delta=0.05,
            n=100 + t * 10,
            regret=cumulative_regret,
            lr_bandit=0.02,
        )
        cumulative_regret += err  # simple accumulation as regret proxy
        print(
            f"Iter {t+1}: decision={decision}, error={err:.4f}, "
            f"regret={cumulative_regret:.4f}"
        )
    sys.exit(0)