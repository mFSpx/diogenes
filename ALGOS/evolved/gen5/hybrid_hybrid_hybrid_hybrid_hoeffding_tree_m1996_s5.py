# DARWIN HAMMER — match 1996, survivor 5
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hard_t_hybrid_hybrid_bayes__m19_s3.py (gen4)
# parent_b: hoeffding_tree.py (gen0)
# born: 2026-05-29T23:40:18Z

"""Hybrid algorithm combining textual feature extraction (Parent A) with Hoeffding‑bound split decisions (Parent B).

Mathematical bridge:
- Parent A transforms a raw text into a high‑dimensional random feature vector  **x ∈ ℝ^d**  (stylometry, LSM, etc.).
- Parent B decides whether a numeric attribute of a streaming data set should be split by comparing the observed gain **G** with the Hoeffding bound **ε**.
- The hybrid treats each component of the extracted feature vector as a streaming numeric attribute.  For a candidate threshold **θ** on dimension *k* we compute the reduction of variance (gain) before/after the split:
  
  **G(k,θ) = Var(x_k) – (n_L/n)·Var(x_k | x_k ≤ θ) – (n_R/n)·Var(x_k | x_k > θ)**  

  The Hoeffding bound then gives a confidence interval on **G** using the range **r** of the gain (here bounded by the variance of a unit‑range feature, i.e. **r = 1**).  If the observed gap between the best and second‑best candidate exceeds **ε**, the split is performed.

The code below implements this fusion: feature extraction, gain computation for candidate splits, and Hoeffding‑bound split decision."""

import sys
import math
import random
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from dataclasses import dataclass

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
    "quantifier": set("all any both each few many more most much none several some such".split()),
    "adverb_common": set(
        "very really just still already also even only then there here now often always sometimes".split()
    ),
}


def _deterministic_hash(text: str) -> int:
    h = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big", signed=False)


def words(text: str) -> List[str]:
    return [w for w in text.lower().split() if w.isalpha()]


def stylometry_features(text: str, dim: int = 96) -> np.ndarray:
    """Produce a deterministic random vector of length *dim* for *text*."""
    seed = _deterministic_hash(text)
    rng = np.random.default_rng(seed)
    return rng.random(dim)


def lsm_vector(text: str) -> np.ndarray:
    """Lexical style matrix (LSM) – proportion of function‑category words."""
    wlist = words(text)
    lsm = np.zeros(len(FUNCTION_CATS))
    for i, (_, cat_words) in enumerate(FUNCTION_CATS.items()):
        lsm[i] = sum(1 for w in wlist if w in cat_words) / len(wlist) if wlist else 0.0
    return lsm


def extract_full_features(text: str, num_features: int = 15) -> Dict[str, float]:
    """Generate a small dictionary of pseudo‑features using a deterministic RNG."""
    seed = _deterministic_hash(text) % (2**32)
    rnd = random.Random(seed)
    return {f"feature_{i}": rnd.random() for i in range(num_features)}


def hybrid_feature_vector(text: str) -> np.ndarray:
    """Combine the three families of features into a single vector."""
    s = stylometry_features(text)          # 96‑dim random vector
    l = lsm_vector(text)                  # len(FUNCTION_CATS) dim
    f = np.array(list(extract_full_features(text).values()))  # num_features dim
    return np.concatenate([s, l, f])


# ----------------------------------------------------------------------
# Parent B – Hoeffding bound utilities
# ----------------------------------------------------------------------
def hoeffding_bound(r: float, delta: float, n: int) -> float:
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0 required")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))


@dataclass(frozen=True)
class SplitDecision:
    should_split: bool
    epsilon: float
    gain_gap: float
    reason: str


def should_split(best_gain: float, second_best_gain: float,
                 r: float, delta: float, n: int,
                 tie_threshold: float = 0.05) -> SplitDecision:
    eps = hoeffding_bound(r, delta, n)
    gap = best_gain - second_best_gain
    split = gap > eps or eps < tie_threshold
    reason = "gap_exceeds_bound" if gap > eps else ("tie_threshold" if eps < tie_threshold else "wait")
    return SplitDecision(split, eps, gap, reason)


# ----------------------------------------------------------------------
# Hybrid core: gain computation on extracted feature vectors + Hoeffding decision
# ----------------------------------------------------------------------
def variance_gain(feature_column: np.ndarray, threshold: float) -> float:
    """
    Compute reduction of variance when splitting *feature_column* at *threshold*.
    Gain = Var(parent) - weighted sum of child variances.
    """
    if feature_column.size == 0:
        return 0.0
    parent_var = np.var(feature_column, ddof=0)
    left = feature_column[feature_column <= threshold]
    right = feature_column[feature_column > threshold]
    n = feature_column.size
    n_l, n_r = left.size, right.size
    if n_l == 0 or n_r == 0:
        return 0.0
    left_var = np.var(left, ddof=0)
    right_var = np.var(right, ddof=0)
    gain = parent_var - (n_l / n) * left_var - (n_r / n) * right_var
    return gain


def evaluate_candidate_splits(feature_matrix: np.ndarray,
                              candidate_dims: List[int],
                              num_thresholds: int = 3) -> Tuple[float, float, int, float]:
    """
    Scan a small set of dimensions and thresholds, returning:
    (best_gain, second_best_gain, best_dim, best_threshold)
    """
    best_gain = -np.inf
    second_gain = -np.inf
    best_dim = -1
    best_thr = np.nan

    n_samples = feature_matrix.shape[0]
    if n_samples == 0:
        return 0.0, 0.0, best_dim, best_thr

    for dim in candidate_dims:
        column = feature_matrix[:, dim]
        # Use quantiles as candidate thresholds
        thresholds = np.quantile(column, np.linspace(0.2, 0.8, num_thresholds))
        for thr in thresholds:
            g = variance_gain(column, thr)
            if g > best_gain:
                second_gain = best_gain
                best_gain = g
                best_dim = dim
                best_thr = thr
            elif g > second_gain:
                second_gain = g

    return best_gain, second_gain, best_dim, best_thr


def hybrid_stream_update(text_stream: List[str],
                         delta: float = 0.05,
                         r: float = 1.0,
                         tie_threshold: float = 0.05) -> List[SplitDecision]:
    """
    Process a stream of texts, extract feature vectors, and at each step
    evaluate whether a split on any dimension should be performed.
    Returns the list of SplitDecision objects (one per processed instance).
    """
    decisions: List[SplitDecision] = []
    feature_buffer: List[np.ndarray] = []
    max_buffer = 200  # after which we evaluate a split

    for i, txt in enumerate(text_stream, 1):
        fv = hybrid_feature_vector(txt)
        feature_buffer.append(fv)

        if i % max_buffer == 0 or i == len(text_stream):
            mat = np.stack(feature_buffer, axis=0)
            # Randomly pick a few dimensions to test (to keep O(1) work)
            dims = random.sample(range(mat.shape[1]), k=min(5, mat.shape[1]))
            best_gain, second_gain, dim, thr = evaluate_candidate_splits(mat, dims)

            decision = should_split(best_gain, second_gain, r, delta, n=mat.shape[0], tie_threshold=tie_threshold)
            decisions.append(decision)

            # In a real tree we would create child nodes here; for the demo we just clear the buffer.
            feature_buffer.clear()

    return decisions


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "In a hole in the ground there lived a hobbit.",
        "Artificial intelligence is transforming many industries.",
        "To be, or not to be: that is the question.",
        "She sells seashells by the seashore.",
        "Data streams require incremental learning algorithms.",
        "Python's simplicity makes it a popular teaching language.",
        "Quantum mechanics challenges our intuition about reality.",
        "The rain in Spain stays mainly in the plain.",
        "Economics studies the allocation of scarce resources."
    ] * 30  # replicate to have enough instances

    decisions = hybrid_stream_update(sample_texts, delta=0.01, r=1.0)
    for idx, d in enumerate(decisions, 1):
        print(f"Decision {idx}: split={d.should_split}, eps={d.epsilon:.5f}, gap={d.gain_gap:.5f}, reason={d.reason}")

    print("Hybrid processing completed without errors.")