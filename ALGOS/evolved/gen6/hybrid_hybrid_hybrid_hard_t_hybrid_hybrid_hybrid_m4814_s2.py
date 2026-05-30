# DARWIN HAMMER — match 4814, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hard_truth_ma_hybrid_infotaxis_hyb_m2474_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_minimu_hybrid_hybrid_percep_m1424_s0.py (gen5)
# born: 2026-05-29T23:58:20Z

"""Hybrid Algorithm: LSM‑Bayesian‑RBF‑Morphology Fusion

Parents:
- **Parent A** – stylometry / LSM utilities with a recovery priority *p* derived from
  document morphology, used to scale LSM vectors before feeding them into an
  infotaxis‑style entropy engine.
- **Parent B** – Bayesian posterior similarity combined with a Gaussian RBF kernel,
  further modulated by a morphology‑derived recovery priority ρ to produce a hybrid
  edge cost.

Mathematical Bridge:
1. The LSM cosine similarity between two texts is interpreted as a *prior* probability.
2. A simple length‑difference based likelihood together with a configurable false‑positive
   rate yields the Bayesian posterior  

   \\hat p = \\frac{prior \\cdot likelihood}
                {likelihood \\cdot prior + false\\_positive \\cdot (1-prior)}.
3. The posterior amplitude feeds an RBF kernel  

   K = exp(-(ε·‖v_a‑v_b‖)²),

   where *v_a*, *v_b* are the LSM vectors.
4. The combined similarity  

   S = \\hat p \\cdot K

   is finally scaled by the morphology‑derived recovery priority  

   ρ = righting_time / max_index ∈ [0,1]

   to obtain a hybrid edge cost  

   cost = (1‑ρ)·(1‑S)·‖v_a‑v_b‖.

The module implements this pipeline and provides three public functions that
demonstrate the hybrid operation.
"""

import math
import random
import sys
import pathlib
import re
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Type aliases
# ----------------------------------------------------------------------
Vector = np.ndarray
Morphology = Dict[str, float]

# ----------------------------------------------------------------------
# Core utilities (Parent A fragments)
# ----------------------------------------------------------------------
def words(text: str) -> List[str]:
    """Extract lower‑cased alphabetic tokens."""
    return re.findall(r"[a-z]+(?:'[a-z]+)?", (text or "").lower())


def lsm_vector(text: str) -> Vector:
    """
    Very simple LSM‑like representation: frequency of each word token,
    normalised to unit L2 norm.  In a full implementation this would be
    replaced by a true Latent Semantic Model.
    """
    ws = words(text)
    total = max(1, len(ws))
    cnt = Counter(ws)
    # Build a deterministic ordering based on sorted vocabulary
    vocab = sorted(cnt.keys())
    vec = np.array([cnt[w] / total for w in vocab], dtype=float)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def cosine_similarity(a: Vector, b: Vector) -> float:
    """Cosine similarity, safe for zero‑vectors."""
    if a.size == 0 or b.size == 0:
        return 0.0
    # Pad to same length
    if a.size < b.size:
        a = np.pad(a, (0, b.size - a.size))
    elif b.size < a.size:
        b = np.pad(b, (0, a.size - b.size))
    dot = float(np.dot(a, b))
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    return dot / norm if norm > 0 else 0.0


def compute_morphology(text: str) -> Morphology:
    """
    Derive a lightweight morphology dictionary from raw text.
    Keys required downstream:
        - length, width, height (scaled by 100)
        - mass (character count)
        - righting_time (proxy for recovery effort)
        - max_index   (normaliser for ρ)
    """
    chars = len(text)
    words_cnt = len(words(text))
    sentences = max(1, text.count('.') + text.count('!') + text.count('?'))
    morphology: Morphology = {
        "length": chars / 100.0,
        "width":  words_cnt / 100.0,
        "height": sentences / 10.0,
        "mass": chars,
        # heuristic recovery priority components
        "righting_time": chars * 0.001,          # arbitrary scaling
        "max_index": max(chars, 1),              # avoid division by zero
    }
    return morphology


def recovery_priority(morph: Morphology) -> float:
    """Map morphology to a value in [0,1] (Parent A's p)."""
    # Simple linear normalisation of mass to [0,1] with a soft cap at 10 k chars
    cap = 10000.0
    p = min(1.0, morph["mass"] / cap)
    return p


# ----------------------------------------------------------------------
# Bayesian‑RBF utilities (Parent B fragments)
# ----------------------------------------------------------------------
def bayes_posterior(prior: float, likelihood: float, false_positive: float) -> float:
    """Posterior probability per Parent B."""
    denominator = likelihood * prior + false_positive * (1.0 - prior)
    return (likelihood * prior) / denominator if denominator != 0 else 0.0


def rbf_kernel(a: Vector, b: Vector, epsilon: float) -> float:
    """Gaussian RBF kernel."""
    diff = a - b
    return math.exp(-(epsilon * np.linalg.norm(diff)) ** 2)


def euclidean_distance(a: Vector, b: Vector) -> float:
    """Plain Euclidean distance (fallback when vectors differ in size)."""
    # Pad to same length as in cosine_similarity
    if a.size < b.size:
        a = np.pad(a, (0, b.size - a.size))
    elif b.size < a.size:
        b = np.pad(b, (0, a.size - b.size))
    return float(np.linalg.norm(a - b))


# ----------------------------------------------------------------------
# Hybrid operations (newly invented)
# ----------------------------------------------------------------------
def hybrid_similarity(
    text_a: str,
    text_b: str,
    epsilon: float = 0.1,
    false_positive: float = 0.05,
) -> float:
    """
    Compute the fused similarity S_ab.

    Steps:
        1. LSM vectors → cosine similarity → prior.
        2. Length‑difference → exponential likelihood.
        3. Bayesian posterior using the prior, likelihood and false_positive.
        4. RBF kernel on the raw LSM vectors.
        5. Multiply posterior by kernel to obtain S_ab.
    """
    # 1. Prior from LSM cosine similarity
    v_a = lsm_vector(text_a)
    v_b = lsm_vector(text_b)
    prior = cosine_similarity(v_a, v_b)  # in [0,1]

    # 2. Likelihood: exponential decay with absolute length difference
    len_a = len(text_a)
    len_b = len(text_b)
    delta = abs(len_a - len_b)
    # scale delta so that typical differences (~100 chars) give a reasonable likelihood
    sigma = 200.0
    likelihood = math.exp(-delta / sigma)  # also in (0,1]

    # 3. Bayesian posterior
    post = bayes_posterior(prior, likelihood, false_positive)

    # 4. RBF kernel on LSM vectors
    kernel = rbf_kernel(v_a, v_b, epsilon)

    # 5. Combined similarity
    return post * kernel


def hybrid_edge_cost(
    text_a: str,
    text_b: str,
    epsilon: float = 0.1,
    false_positive: float = 0.05,
) -> float:
    """
    Compute the hybrid edge cost as described in the bridge:

        ρ = righting_time / max_index  ∈ [0,1]
        d = Euclidean distance between LSM vectors
        S = hybrid_similarity(...)
        cost = (1‑ρ)·(1‑S)·d
    """
    # Morphology and recovery priority ρ (Parent A)
    morph_a = compute_morphology(text_a)
    morph_b = compute_morphology(text_b)
    # Average morphology for the edge
    avg_morph = {
        k: (morph_a[k] + morph_b[k]) / 2.0 for k in ("righting_time", "max_index")
    }
    rho = avg_morph["righting_time"] / avg_morph["max_index"]
    rho = max(0.0, min(1.0, rho))  # clip

    # Distance between LSM vectors
    v_a = lsm_vector(text_a)
    v_b = lsm_vector(text_b)
    d = euclidean_distance(v_a, v_b)

    # Hybrid similarity S
    S = hybrid_similarity(text_a, text_b, epsilon, false_positive)

    # Final cost
    cost = (1.0 - rho) * (1.0 - S) * d
    return cost


def hybrid_infotaxis_entropy(
    target_text: str,
    neighbor_texts: List[str],
    epsilon: float = 0.1,
    false_positive: float = 0.05,
) -> float:
    """
    Treat hybrid similarities to a set of neighbor candidates as a
    probability‑like mass distribution and compute the Shannon entropy.
    This mirrors Parent A's infotaxis step but uses the fused similarity S.
    """
    sims = np.array(
        [hybrid_similarity(target_text, nb, epsilon, false_positive) for nb in neighbor_texts],
        dtype=float,
    )
    total = sims.sum()
    if total == 0:
        # Uniform minimal entropy if no similarity information is present
        probs = np.full_like(sims, 1.0 / max(1, len(sims)))
    else:
        probs = sims / total

    # Shannon entropy (base‑e)
    entropy = -float(np.sum(probs * np.log(probs + 1e-12)))  # add epsilon to avoid log(0)
    return entropy


# ----------------------------------------------------------------------
# Minimal fitting routine (illustrative, not required by spec)
# ----------------------------------------------------------------------
def fit_epsilon(
    samples: List[Tuple[str, str, float]],
    false_positive: float = 0.05,
    init_eps: float = 0.1,
    lr: float = 0.01,
    epochs: int = 200,
) -> float:
    """
    Simple gradient‑free optimisation to find an ε that minimizes the squared
    error between observed costs and `hybrid_edge_cost` predictions.
    Returns the fitted ε.
    """
    eps = init_eps
    for _ in range(epochs):
        grads = []
        for a, b, observed in samples:
            pred = hybrid_edge_cost(a, b, epsilon=eps, false_positive=false_positive)
            # Numerical derivative (finite difference)
            delta = 1e-6
            pred_eps = hybrid_edge_cost(a, b, epsilon=eps + delta, false_positive=false_positive)
            grad = (pred_eps - pred) / delta
            grads.append(2 * (pred - observed) * grad)
        avg_grad = sum(grads) / max(1, len(grads))
        eps -= lr * avg_grad
        eps = max(1e-6, eps)  # keep positive
    return eps


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    txt1 = "The quick brown fox jumps over the lazy dog."
    txt2 = "A fast auburn fox leaped above a sleepy canine."
    txt3 = "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
    txt4 = "In a distant galaxy, the starship glided silently."

    # Demonstrate hybrid similarity
    sim12 = hybrid_similarity(txt1, txt2)
    sim13 = hybrid_similarity(txt1, txt3)
    print(f"Hybrid similarity (txt1, txt2): {sim12:.4f}")
    print(f"Hybrid similarity (txt1, txt3): {sim13:.4f}")

    # Edge cost between two texts
    cost12 = hybrid_edge_cost(txt1, txt2)
    cost13 = hybrid_edge_cost(txt1, txt3)
    print(f"Hybrid edge cost (txt1 ↔ txt2): {cost12:.4f}")
    print(f"Hybrid edge cost (txt1 ↔ txt3): {cost13:.4f}")

    # Entropy over a candidate set
    entropy = hybrid_infotaxis_entropy(txt1, [txt2, txt3, txt4])
    print(f"Infotaxis‑style entropy for txt1 against three neighbours: {entropy:.4f}")

    # Fit epsilon on synthetic data (no real ground truth, just demo)
    synthetic_samples = [
        (txt1, txt2, cost12),
        (txt1, txt3, cost13),
        (txt2, txt4, hybrid_edge_cost(txt2, txt4)),
    ]
    fitted_eps = fit_epsilon(synthetic_samples)
    print(f"Fitted epsilon after mock optimisation: {fitted_eps:.6f}")