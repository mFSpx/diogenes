# DARWIN HAMMER — match 5641, survivor 3
# gen: 4
# parent_a: hybrid_hybrid_hybrid_decisi_ternary_lens_audit_m733_s0.py (gen3)
# parent_b: hybrid_hard_truth_math_model_pool_m8_s1.py (gen1)
# born: 2026-05-30T00:03:43Z

"""Hybrid Fusion of Decision‑Hygiene Entropy & Stylometry‑Based Model Pool

Parents
-------
* hybrid_hybrid_hybrid_decisi_ternary_lens_audit_m733_s0.py (Decision‑hygiene counts,
  Shannon entropy, ternary‑lens classifications)
* hybrid_hard_truth_math_model_pool_m8_s1.py (Stylometry feature extraction,
  similarity‑driven model loading/eviction)

Mathematical Bridge
-------------------
The decision‑hygiene subsystem yields a probability distribution **p** over token
categories; its Shannon entropy **H(p)** quantifies linguistic uncertainty.
The stylometry subsystem produces a high‑dimensional feature vector **v** for a
text; similarity to a model’s feature vector **m** is measured by the cosine
similarity **S(v,m)**.

We fuse the two worlds by defining a *Hybrid Score*:

    Σ(text, m) = α · Ĥ(p) + β · S(v, m)

where **Ĥ(p)** is the entropy normalised to [0,1] (by dividing by log₂|C|, the
maximum possible entropy for the count categories) and **α,β∈ℝ⁺** are
tunable weights.  This scalar drives model selection and eviction, while the
ternary‑lens audit supplies a coarse classification that can modulate the
weights per class.

The implementation below embeds the core equations from both parents,
exposes three public functions demonstrating the hybrid operation, and
provides a lightweight `ModelPool` that uses the hybrid score for loading
and eviction decisions.
"""

import math
import random
import sys
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import re

# ----------------------------------------------------------------------
# Parent A – Decision‑Hygiene & Ternary Lens
# ----------------------------------------------------------------------
EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|"
    r"hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)
PLANNING_RE = re.compile(r"\b(?:plan|checklist|steps?|seq)\b", re.I)

CLASSIFICATIONS = {
    "usable_now",
    "research_only",
    "needs_conversion",
    "unsafe_for_fastpath",
    "unsupported",
}


def extract_hygiene_counts(text: str) -> Dict[str, int]:
    """Count decision‑hygiene tokens in *text*."""
    counts = defaultdict(int)
    counts["evidence"] = len(EVIDENCE_RE.findall(text))
    counts["planning"] = len(PLANNING_RE.findall(text))
    return dict(counts)


def calculate_shannon_entropy(counts: Dict[str, int]) -> float:
    """Shannon entropy H(p) for the categorical distribution defined by *counts*."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def approximate_log_likelihood(counts: Dict[str, int]) -> float:
    """Simple log‑likelihood proxy: sum_i log(count_i + 1)."""
    ll = 0.0
    for c in counts.values():
        ll += math.log(c + 1)
    return ll


def ternary_lens_classify(text: str) -> str:
    """Very coarse classification based on keyword presence."""
    lowered = text.lower()
    if any(tok in lowered for tok in ("unsafe", "danger", "risk")):
        return "unsafe_for_fastpath"
    if any(tok in lowered for tok in ("convert", "transform")):
        return "needs_conversion"
    if any(tok in lowered for tok in ("research", "explore", "study")):
        return "research_only"
    if any(tok in lowered for tok in ("ready", "immediate", "now")):
        return "usable_now"
    return "unsupported"


# ----------------------------------------------------------------------
# Parent B – Stylometry & Model Pool
# ----------------------------------------------------------------------
FUNCTION_CATS: dict[str, set[str]] = {
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
        "no not never none neither cannot can't won't don't didn't isn't aren't wasn't weren't".split()
    ),
    "quantifier": set(
        "all any both each few many more most much none several some such".split()
    ),
    "adverb_common": set(
        "very really just still already also even only then there here now often always sometimes".split()
    ),
}


def stylometry_features(text: str) -> np.ndarray:
    """Return a normalized feature vector counting occurrences of FUNCTION_CATS."""
    tokens = re.findall(r"\b\w+\b", text.lower())
    counter = Counter(tokens)
    feature_vec = []
    for cat, vocab in FUNCTION_CATS.items():
        count = sum(counter[word] for word in vocab)
        feature_vec.append(count)
    arr = np.array(feature_vec, dtype=float)
    norm = np.linalg.norm(arr)
    return arr / norm if norm > 0 else arr


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors (assumes non‑zero vectors)."""
    if a.size == 0 or b.size == 0:
        return 0.0
    dot = float(np.dot(a, b))
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return dot / denom if denom != 0 else 0.0


# ----------------------------------------------------------------------
# Hybrid Core – Entropy‑Weighted Similarity
# ----------------------------------------------------------------------
ALPHA = 0.6  # weight for normalized entropy
BETΑ = 0.4   # weight for cosine similarity


def hybrid_score(text: str, model_vec: np.ndarray) -> float:
    """
    Compute the hybrid score Σ(text, model) = α·Ĥ + β·S.

    * Ĥ – normalized Shannon entropy of decision‑hygiene counts.
    * S   – cosine similarity between stylometry vectors.
    """
    # Decision‑hygiene part
    counts = extract_hygiene_counts(text)
    raw_entropy = calculate_shannon_entropy(counts)
    # maximum entropy for two categories = log2(2) = 1
    normalized_entropy = raw_entropy / math.log2(len(counts)) if len(counts) > 1 else 0.0

    # Stylometry similarity part
    text_vec = stylometry_features(text)
    similarity = cosine_similarity(text_vec, model_vec)

    # Weighted combination
    return ALPHA * normalized_entropy + BETΑ * similarity


# ----------------------------------------------------------------------
# Hybrid Model Pool – Loading/Eviction driven by hybrid_score
# ----------------------------------------------------------------------
class ModelPool:
    """
    Simple in‑memory model pool.

    Each model is represented by a tuple (name, feature_vector, last_used_timestamp).
    The pool keeps at most *capacity* models; when the limit is exceeded the model
    with the lowest hybrid_score for the current query is evicted.
    """

    def __init__(self, capacity: int = 5):
        self.capacity = capacity
        self._models: Dict[str, Tuple[np.ndarray, float]] = {}  # name → (vec, last_used)

    def add_model(self, name: str, feature_vec: np.ndarray) -> None:
        """Insert a new model; if capacity is exceeded, evict the least‑similar model."""
        self._models[name] = (feature_vec, 0.0)

    def _update_timestamp(self, name: str) -> None:
        self._models[name] = (self._models[name][0], dt_now())

    def best_model(self, text: str) -> Tuple[str, float]:
        """Return the model name with the highest hybrid_score for *text*."""
        if not self._models:
            raise RuntimeError("ModelPool is empty.")
        scores = {
            name: hybrid_score(text, vec) for name, (vec, _) in self._models.items()
        }
        best_name = max(scores, key=scores.get)
        self._update_timestamp(best_name)
        return best_name, scores[best_name]

    def evict_if_necessary(self, text: str) -> None:
        """If the pool exceeds capacity, evict the model with the lowest hybrid_score."""
        while len(self._models) > self.capacity:
            scores = {
                name: hybrid_score(text, vec) for name, (vec, _) in self._models.items()
            }
            worst_name = min(scores, key=scores.get)
            del self._models[worst_name]

    def __len__(self) -> int:
        return len(self._models)


def dt_now() -> float:
    """Return a monotonic timestamp (seconds since epoch)."""
    return float(np.datetime64('now').astype('int64') // 1_000_000_000)


# ----------------------------------------------------------------------
# Public API – Demonstrating the Hybrid Operation
# ----------------------------------------------------------------------
def analyze_text(text: str) -> Dict[str, float]:
    """
    Perform a full hybrid analysis:
      * entropy of decision‑hygiene tokens
      * approximate log‑likelihood
      * ternary‑lens classification
    Returns a dictionary of the three metrics.
    """
    counts = extract_hygiene_counts(text)
    entropy = calculate_shannon_entropy(counts)
    log_likelihood = approximate_log_likelihood(counts)
    classification = ternary_lens_classify(text)
    return {
        "entropy": entropy,
        "log_likelihood": log_likelihood,
        "classification": classification,
    }


def select_model_for_text(pool: ModelPool, text: str) -> Tuple[str, float]:
    """
    Choose the best model from *pool* for *text* using the hybrid_score.
    The pool is also trimmed to respect its capacity.
    """
    best_name, score = pool.best_model(text)
    pool.evict_if_necessary(text)
    return best_name, score


def generate_dummy_models(num: int, dim: int = len(FUNCTION_CATS)) -> List[Tuple[str, np.ndarray]]:
    """Create *num* dummy models with random stylometry vectors."""
    models = []
    rng = np.random.default_rng(seed=42)
    for i in range(num):
        vec = rng.random(dim)
        vec /= np.linalg.norm(vec) if np.linalg.norm(vec) != 0 else 1.0
        models.append((f"model_{i}", vec))
    return models


# ----------------------------------------------------------------------
# Smoke Test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_text = (
        "The plan includes verifying sources, checking logs, and confirming the "
        "accuracy of each step. This research is ready for immediate use."
    )

    # 1. Hybrid analysis of the raw text
    analysis = analyze_text(sample_text)
    print("Hybrid Analysis:", analysis)

    # 2. Build a pool with dummy models
    pool = ModelPool(capacity=3)
    for name, vec in generate_dummy_models(5):
        pool.add_model(name, vec)

    # 3. Select the best model for the sample text
    best_name, best_score = select_model_for_text(pool, sample_text)
    print(f"Selected Model: {best_name} (Hybrid Score: {best_score:.4f})")

    # 4. Show remaining models in the pool after possible eviction
    print("Models remaining in pool:", list(pool._models.keys()))