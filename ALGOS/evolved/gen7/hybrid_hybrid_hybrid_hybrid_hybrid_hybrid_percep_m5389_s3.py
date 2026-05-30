# DARWIN HAMMER — match 5389, survivor 3
# gen: 7
# parent_a: hybrid_hybrid_hybrid_rectif_hybrid_hybrid_hybrid_m1689_s1.py (gen6)
# parent_b: hybrid_hybrid_perceptual_de_hybrid_hybrid_ternar_m1907_s6.py (gen4)
# born: 2026-05-30T00:01:34Z

"""Hybrid Algorithm combining LSM lexical semantics, regret‑epistemic weighting, and RBF surrogate hashing.

Parents:
- hybrid_hybrid_hybrid_rectif_hybrid_hybrid_hybrid_m1689_s1.py (LSM vectors + regret‑epistemic pruning)
- hybrid_hybrid_perceptual_de_hybrid_hybrid_ternar_m1907_s6.py (hashing + Gaussian RBF surrogate)

Mathematical Bridge:
1. Text → LSM vector  (categorical frequency distribution, parent A).
2. LSM vector → combined hash (dhash + phash, parent B) – the hash is a deterministic,
   compact representation of the continuous LSM feature.
3. Hash‑derived feature matrix is fed to a Gaussian RBF kernel (parent B) to obtain a
   surrogate predictor 𝑓̂(·).
4. The raw prediction 𝑓̂ is modulated by a regret‑weight 𝑅 and an epistemic‑certainty
   flag 𝑈, both defined on the same LSM space (parent A).  The final posterior weight is

        w = 𝑓̂ × 𝑅 × 𝑈 .

Thus the three core topologies are mathematically fused into a single pipeline.
"""

import math
import random
import sys
import pathlib
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple, Dict, Sequence

import numpy as np

# ----------------------------------------------------------------------
# Parent A – lexical categories and LSM vector
# ----------------------------------------------------------------------
FUNCTION_CATS: dict[str, set[str]] = {
    "pronoun": set("i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()),
    "article": set("a an the".split()),
    "preposition": set("about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()),
    "auxiliary": set("am are be been being can could did do does had has have is may might must shall should was were will would".split()),
    "conjunction": set("and but or nor so yet because although while if when where whereas unless until".split()),
    "negation": set("no not never none neither cannot cant wont dont didnt isnt arent was wasnt werent".split()),
    "quantifier": set("all any both each few many more most much none several some such".split()),
    "adverb_common": set("very really just still already also even only then there here now often always sometimes".split()),
}
CATEGORY_ORDER = list(FUNCTION_CATS.keys())

def words(text: str) -> List[str]:
    """Simple tokenizer returning lowercase alphabetic tokens."""
    return [w for w in (text or "").lower().split() if w.isalpha()]

def lsm_vector(text: str) -> np.ndarray:
    """
    Compute the LSM (Lexical‑Semantic‑Morph) vector for *text*.
    The i‑th component is the proportion of words belonging to CATEGORY_ORDER[i].
    """
    ws = words(text)
    total = max(1, len(ws))
    cnt = Counter(ws)
    vec = np.zeros(len(CATEGORY_ORDER), dtype=float)
    for i, cat in enumerate(CATEGORY_ORDER):
        vocab = FUNCTION_CATS[cat]
        vec[i] = sum(cnt[w] for w in vocab) / total
    return vec

# ----------------------------------------------------------------------
# Parent B – hashing utilities and Gaussian RBF surrogate
# ----------------------------------------------------------------------
def compute_dhash(values: List[float]) -> int:
    """Difference hash: 1 bit per adjacent pair, 1 if decreasing."""
    bits = 0
    for i in range(len(values) - 1):
        bits = (bits << 1) | int(values[i] > values[i + 1])
    return bits

def compute_phash(values: List[float]) -> int:
    """Average hash limited to first 64 values."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits

def compute_combined_hash(values: List[float]) -> int:
    """Merge dhash (high‑order bits) and phash (low‑order bits) into one integer."""
    dh = compute_dhash(values)
    ph = compute_phash(values)
    return (dh << 64) | ph

def hamming_distance(a: int, b: int) -> int:
    """Return Hamming distance between two integers."""
    return bin(a ^ b).count("1")

class RBFSurrogate:
    """Gaussian RBF surrogate trained on (X, y)."""

    def __init__(self, X: np.ndarray, y: np.ndarray, gamma: float = 1.0, reg: float = 1e-6):
        """
        Parameters
        ----------
        X : (n_samples, n_features) array
        y : (n_samples,) array
        gamma : kernel width parameter
        reg : Tikhonov regularisation coefficient
        """
        self.gamma = float(gamma)
        self.reg = float(reg)
        self.X = X.astype(float)
        self.y = y.astype(float)
        self._fit()

    def _kernel(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Gaussian kernel matrix between rows of A and B."""
        sq_norms_A = np.sum(A ** 2, axis=1)[:, None]
        sq_norms_B = np.sum(B ** 2, axis=1)[None, :]
        dists = sq_norms_A + sq_norms_B - 2 * A @ B.T
        return np.exp(-self.gamma * dists)

    def _fit(self):
        K = self._kernel(self.X, self.X)
        n = K.shape[0]
        K_reg = K + self.reg * np.eye(n)
        self.alpha = np.linalg.solve(K_reg, self.y)

    def predict(self, X_new: np.ndarray) -> np.ndarray:
        """Predict for one or many new points."""
        K_new = self._kernel(X_new, self.X)
        return K_new @ self.alpha

# ----------------------------------------------------------------------
# Hybrid components: regret, epistemic certainty, and the unified model
# ----------------------------------------------------------------------
def regret_weight(pred: float, baseline: float) -> float:
    """
    Simple regret weight: 1 - normalized absolute error.
    The baseline is the mean of training targets.
    """
    if baseline == 0:
        return 1.0
    err = abs(pred - baseline)
    # Normalise by a heuristic range (max of baseline and err)
    norm = max(abs(baseline), err)
    return max(0.0, 1.0 - err / norm)

def epistemic_certainty(feature: np.ndarray, training_X: np.ndarray) -> float:
    """
    Epistemic certainty derived from distance to the nearest training point.
    Closer points → higher certainty. Returned in [0,1].
    """
    if training_X.size == 0:
        return 0.0
    dists = np.linalg.norm(training_X - feature, axis=1)
    min_dist = np.min(dists)
    # Convert distance to certainty with an exponential decay
    return math.exp(-min_dist)

@dataclass
class HybridRegretEpistemicRBF:
    """
    Unified model:
    1. Convert text → LSM vector.
    2. Hash the vector → integer (used only for diagnostics here).
    3. Train Gaussian RBF surrogate on raw LSM vectors.
    4. Predict and modulate by regret weight and epistemic certainty.
    """
    gamma: float = 1.0
    reg: float = 1e-6
    surrogate: RBFSurrogate | None = None
    baseline: float = 0.0
    training_X: np.ndarray = np.empty((0, len(CATEGORY_ORDER)))

    def fit(self, texts: List[str], targets: List[float]) -> None:
        """Train the surrogate on LSM vectors derived from *texts*."""
        X = np.vstack([lsm_vector(t) for t in texts])
        y = np.array(targets, dtype=float)
        self.surrogate = RBFSurrogate(X, y, gamma=self.gamma, reg=self.reg)
        self.baseline = float(np.mean(y))
        self.training_X = X

    def predict(self, text: str) -> Tuple[float, int, float, float]:
        """
        Return a tuple (final_weight, hash, raw_pred, certainty).

        - final_weight = raw_pred * regret * certainty
        - hash        = combined hash of the LSM vector (diagnostic)
        - raw_pred    = surrogate prediction before modulation
        - certainty   = epistemic certainty factor
        """
        if self.surrogate is None:
            raise RuntimeError("Model has not been fitted.")
        feat = lsm_vector(text)
        raw = float(self.surrogate.predict(feat.reshape(1, -1))[0])
        reg = regret_weight(raw, self.baseline)
        cert = epistemic_certainty(feat, self.training_X)
        final = raw * reg * cert
        h = compute_combined_hash(feat.tolist())
        return final, h, raw, cert

# ----------------------------------------------------------------------
# Demonstration functions
# ----------------------------------------------------------------------
def hybrid_feature_hash(text: str) -> int:
    """Convenience wrapper returning the combined hash of a text's LSM vector."""
    return compute_combined_hash(lsm_vector(text).tolist())

def hybrid_train_demo() -> HybridRegretEpistemicRBF:
    """Create a tiny training set, fit the model, and return it."""
    corpus = [
        "I am happy and excited about the new project",
        "He does not like the bad weather",
        "They will go to the market tomorrow",
        "She cannot attend the meeting because she is ill",
        "We have many options and can choose wisely",
    ]
    # Synthetic targets: arbitrary scores between 0 and 1
    targets = [0.9, 0.2, 0.6, 0.1, 0.8]
    model = HybridRegretEpistemicRBF(gamma=5.0, reg=1e-5)
    model.fit(corpus, targets)
    return model

def hybrid_predict_demo(model: HybridRegretEpistemicRBF, text: str) -> None:
    """Run prediction on *text* and print the intermediate values."""
    final, hsh, raw, cert = model.predict(text)
    print(f"Text: {text}")
    print(f"Combined hash: {hsh:#018x}")
    print(f"Raw RBF prediction: {raw:.4f}")
    print(f"Regret weight (implicit): {final / (raw * cert) if raw * cert != 0 else 0:.4f}")
    print(f"Epistemic certainty: {cert:.4f}")
    print(f"Final posterior weight: {final:.4f}")

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    demo_model = hybrid_train_demo()
    test_sentences = [
        "I cannot wait for the upcoming conference",
        "They are not interested in the proposal",
        "We have enough resources to succeed",
    ]
    for s in test_sentences:
        hybrid_predict_demo(demo_model, s)
        print("-" * 40)