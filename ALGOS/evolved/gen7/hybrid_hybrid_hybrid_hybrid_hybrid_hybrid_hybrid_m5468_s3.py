# DARWIN HAMMER — match 5468, survivor 3
# gen: 7
# parent_a: hybrid_hybrid_hybrid_krampu_hybrid_hybrid_hybrid_m1463_s1.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_rectif_m1967_s0.py (gen6)
# born: 2026-05-30T00:02:17Z

"""Hybrid Krampus‑Brainmap‑Indy‑Learning & Regex‑GA‑Fisher Fusion
================================================================

This module fuses the mathematics of the two parent algorithms:

* **Parent A** – provides a high‑dimensional term‑frequency vector *v*,
  an infotaxis‑style pheromone store, a perceptual binary hash and an
  RBF surrogate whose raw output is modulated by SSIM against a prototype
  vector.

* **Parent B** – supplies a regex‑driven categorical feature vector,
  a geometric‑algebra (GA) transformation of weight matrices and a Fisher
  score that re‑weights those matrices according to the categorical
  evidence.  An endpoint “circuit‑breaker” decides whether the update
  should be applied.

**Mathematical bridge**

1. The term‑frequency vector *v* (size *n*) is first **rotated** by a GA
   rotor *R* (an orthogonal matrix) derived from the categorical feature
   vector *c*.  This maps the lexical space into a geometry‑aware space
   that respects the regex‑derived semantics.

2. The rotated vector *vʹ = R · v* is used to update the pheromone store.
   The update amount is scaled by a **Fisher weight** *F* that measures
   the discriminative power of the regex categories for the current
   input.

3. The binary perceptual hash *h* (derived from *vʹ*) feeds an RBF
   surrogate *φ*.  The surrogate output is multiplied by the
   structural‑similarity index (SSIM) between *vʹ* and a fixed prototype
   vector *p*.

4. Finally the **fused score** combines the infotaxis information gain
   *IG* with the modulated RBF output *φ·SSIM*, both weighted by the
   Fisher score *F*.  If the circuit‑breaker deems *IG* too low, the
   pheromone store is left unchanged.

The result is a single coherent system that simultaneously exploits
lexical statistics, perceptual similarity, geometric‑algebraic
transformations and statistical discriminability.
"""

import sys
import math
import random
import re
from pathlib import Path
from collections import Counter
from typing import List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Constants & regex feature sets (Parent B)
# ----------------------------------------------------------------------
EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|"
    r"screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)
PLANNING_RE = re.compile(
    r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|"
    r"triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
    re.I,
)
DELAY_RE = re.compile(
    r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|"
    r"not now|before i|first|after|review)\b",
    re.I,
)
SUPPORT_RE = re.compile(
    r"\b(?:ask|call|text|friend|friends|rowyn|kai|chance|doctor|therapist|lawyer|"
    r"advocate|support|help|community|team|handoff|delegate)\b",
    re.I,
)
BOUNDARY_RE = re.compile(
    r"\b(?:boundary|boundaries|walk away|no contact|do not|don't|stop|limit|refuse|"
    r"protect|safe|safety|lock|redact|privacy|permission|consent)\b",
    re.I,
)
OUTCOME_RE = re.compile(
    r"\b(?:done|shipped|finished|complete|resolved|succeeded|passed|paid|safe|sent|"
    r"filed|closed|fixed|work)\b",
    re.I,
)

REGEX_CATEGORIES = {
    "evidence": EVIDENCE_RE,
    "planning": PLANNING_RE,
    "delay": DELAY_RE,
    "support": SUPPORT_RE,
    "boundary": BOUNDARY_RE,
    "outcome": OUTCOME_RE,
}

# ----------------------------------------------------------------------
# Constants (Parent A)
# ----------------------------------------------------------------------
DEFAULT_TERMS = ("ENTITY", "ATTRIBUTE", "RELATIONSHIP", "ACTION", "EVENT")
PROTOTYPE_VECTOR = np.array([0.2, 0.5, 0.3, 0.7, 0.1], dtype=np.float64)

# ----------------------------------------------------------------------
# Helper mathematics
# ----------------------------------------------------------------------
def compute_ssim(x: List[float], y: List[float],
                 dynamic_range: float = 1.0,
                 k1: float = 0.01,
                 k2: float = 0.03) -> float:
    """Structural Similarity Index (SSIM) for 1‑D signals."""
    if len(x) != len(y):
        raise ValueError("samples must have equal length")
    if not x:
        raise ValueError("samples must not be empty")

    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    mx = x_arr.mean()
    my = y_arr.mean()
    vx = ((x_arr - mx) ** 2).mean()
    vy = ((y_arr - my) ** 2).mean()
    cov = ((x_arr - mx) * (y_arr - my)).mean()

    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2

    numerator = (2 * mx * my + c1) * (2 * cov + c2)
    denominator = (mx ** 2 + my ** 2 + c1) * (vx + vy + c2)
    return float(numerator / denominator)


def rbf_surrogate(x: np.ndarray,
                  centers: np.ndarray,
                  gamma: float = 0.5) -> float:
    """Radial‑Basis‑Function surrogate (Gaussian)."""
    diff = centers - x
    sq_norm = np.sum(diff ** 2, axis=1)
    return float(np.exp(-gamma * sq_norm).mean())


def fisher_score(features: np.ndarray,
                 labels: np.ndarray) -> float:
    """
    Classic Fisher discriminant ratio.
    features: (m, d) matrix, m samples.
    labels:   (m,) binary class labels (0/1).
    Returns a scalar score (higher = better separation).
    """
    if features.shape[0] != labels.shape[0]:
        raise ValueError("features and labels must have same number of rows")
    class0 = features[labels == 0]
    class1 = features[labels == 1]
    if class0.size == 0 or class1.size == 0:
        return 0.0
    mean0 = class0.mean(axis=0)
    mean1 = class1.mean(axis=0)
    mean_diff = np.linalg.norm(mean1 - mean0) ** 2

    var0 = class0.var(axis=0).sum()
    var1 = class1.var(axis=0).sum()
    within = var0 + var1
    if within == 0:
        return 0.0
    return float(mean_diff / within)


def circuit_breaker(info_gain: float, threshold: float = 0.01) -> bool:
    """Return True if the update should be applied."""
    return info_gain > threshold


# ----------------------------------------------------------------------
# Core hybrid functions
# ----------------------------------------------------------------------
def term_frequency_vector(text: str,
                          vocab: Tuple[str, ...] = DEFAULT_TERMS) -> np.ndarray:
    """
    Build a simple term‑frequency vector over a fixed vocabulary.
    """
    tokens = re.findall(r"\w+", text.lower())
    counter = Counter(tokens)
    vec = np.array([counter.get(term.lower(), 0) for term in vocab],
                   dtype=np.float64)
    if vec.sum() > 0:
        vec = vec / vec.sum()          # normalise to probabilities
    return vec


def regex_feature_vector(text: str) -> np.ndarray:
    """
    Produce a binary feature vector for the six regex categories.
    Order follows REGEX_CATEGORIES keys.
    """
    feats = [1.0 if pattern.search(text) else 0.0
             for pattern in REGEX_CATEGORIES.values()]
    return np.array(feats, dtype=np.float64)


def ga_rotor_from_features(features: np.ndarray) -> np.ndarray:
    """
    Construct an orthogonal rotation matrix (GA rotor) whose
    orientation is driven by the categorical features.
    For reproducibility a deterministic pseudo‑random seed is derived
    from the feature vector.
    """
    dim = features.shape[0]
    # Derive a seed from the feature vector (hash‑like)
    seed = int(np.round(features.sum() * 1e6)) % (2 ** 32)
    rng = np.random.default_rng(seed)
    # Generate a random matrix and orthogonalise it via QR decomposition
    random_mat = rng.normal(size=(dim, dim))
    q, _ = np.linalg.qr(random_mat)
    return q.astype(np.float64)


def infotaxis_info_gain(v: np.ndarray,
                        pheromone: np.ndarray,
                        epsilon: float = 1e-12) -> float:
    """
    Compute information gain as the reduction in entropy after adding
    the current vector *v* to the pheromone store (treated as a probability
    distribution).
    """
    new_store = pheromone + v
    prob = new_store / (new_store.sum() + epsilon)
    entropy_before = -np.sum(pheromone / (pheromone.sum() + epsilon) *
                             np.log2(pheromone / (pheromone.sum() + epsilon) + epsilon))
    entropy_after = -np.sum(prob * np.log2(prob + epsilon))
    return float(entropy_before - entropy_after)


def fused_score(text: str,
                pheromone_store: np.ndarray,
                rbf_centers: np.ndarray,
                labels: np.ndarray) -> float:
    """
    End‑to‑end hybrid scoring:
      1. TF vector → GA rotation (driven by regex features)
      2. Info‑gain from infotaxis, scaled by Fisher score
      3. Perceptual hash → RBF surrogate, modulated by SSIM
      4. Combine (2) and (3) with circuit‑breaker logic.
    """
    # 1. lexical → geometric
    tf_vec = term_frequency_vector(text)                     # shape (n,)
    cat_vec = regex_feature_vector(text)                     # shape (c,)
    rotor = ga_rotor_from_features(cat_vec)                  # (c, c)
    # Pad tf_vec / rotor to compatible size if needed
    if rotor.shape[0] != tf_vec.shape[0]:
        # simple projection: repeat or truncate
        if rotor.shape[0] > tf_vec.shape[0]:
            tf_vec = np.pad(tf_vec,
                            (0, rotor.shape[0] - tf_vec.shape[0]),
                            constant_values=0.0)
        else:
            tf_vec = tf_vec[:rotor.shape[0]]
    v_rot = rotor @ tf_vec                                     # rotated vector

    # 2. infotaxis info gain
    ig = infotaxis_info_gain(v_rot, pheromone_store)

    # Fisher weight – we treat the rotated vector as a feature and the
    # provided *labels* (binary) as class information.
    fisher_w = fisher_score(v_rot.reshape(1, -1), labels.reshape(-1))

    weighted_ig = ig * fisher_w

    # 3. perceptual hash → binary vector (simple threshold)
    binary_hash = (v_rot > v_rot.mean()).astype(float)        # binary-like
    # RBF surrogate
    rbf_out = rbf_surrogate(binary_hash, rbf_centers)

    # SSIM modulation
    ssim_mod = compute_ssim(v_rot.tolist(), PROTOTYPE_VECTOR.tolist())
    modulated_rbf = rbf_out * ssim_mod

    # 4. Combine with circuit‑breaker
    if circuit_breaker(weighted_ig):
        pheromone_store += v_rot * fisher_w                     # update store
        final = weighted_ig + modulated_rbf
    else:
        final = modulated_rbf                                   # no IG contribution

    return float(final)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_text = (
        "The evidence was verified and the plan was documented. "
        "We will schedule the next checkpoint tomorrow."
    )
    # Initialise a pheromone store matching the rotated dimension (6 categories)
    pheromone = np.zeros(len(REGEX_CATEGORIES), dtype=np.float64)

    # Create dummy RBF centers (binary hash space dimension equals category count)
    rng = np.random.default_rng(42)
    centers = rng.integers(0, 2, size=(10, len(REGEX_CATEGORIES))).astype(float)

    # Dummy binary labels for Fisher score (alternating 0/1)
    dummy_labels = np.array([0, 1] * 5, dtype=int)

    score = fused_score(sample_text, pheromone, centers, dummy_labels)
    print(f"Fused score: {score:.4f}")
    print(f"Pheromone store after update: {pheromone}")