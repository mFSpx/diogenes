# DARWIN HAMMER — match 5543, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_perceptual_de_hybrid_hybrid_rlct_g_m578_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_gini_c_m2063_s1.py (gen6)
# born: 2026-05-30T00:04:17Z

"""
Hybrid NLMS‑Tropical‑Graph Algorithm
===================================

This module fuses the core topologies of the two parent algorithms:

* **Parent A** – *hybrid_hybrid_perceptual_de_hybrid_hybrid_rlct_g_m578_s0.py*  
  Provides a Normalised Least Mean Squares (NLMS) adaptive filter together with
  graph‑based matrix updates (weight matrix *W*, adjacency, Laplacian).

* **Parent B** – *hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_gini_c_m2063_s1.py*  
  Introduces tropical (max‑plus) algebra and a Gini‑coefficient based similarity
  measure derived from high‑dimensional text‑category histograms.

**Mathematical bridge** – The NLMS step size ``μ`` is adapted by a tropical
combination of a base learning rate and the Gini coefficient of the current
text token distribution:


μₜ = μ_base ⊕ Gini   (tropical addition = max)


The regularisation term that injects graph smoothness into the weight update
is scaled by a tropical multiplication (which in the max‑plus semiring is
ordinary addition):


Δ_reg = (λ ⊗ L)·w   where λ ⊗ L = λ + L   (tropical multiplication)


Thus the NLMS weight update becomes a *hybrid* operation that simultaneously
leverages (i) adaptive learning via tropical algebra, (ii) inequality
information from text via the Gini coefficient, and (iii) distributed
graph‑based smoothing.

The module supplies three public functions that demonstrate this hybrid
behaviour and a small smoke test.
"""

import math
import random
import sys
from pathlib import Path
from typing import List, Sequence, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Tropical algebra (max‑plus semiring)
# ----------------------------------------------------------------------
def t_add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Tropical addition (⊕): element‑wise max."""
    return np.maximum(x, y)


def t_mul(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Tropical multiplication (⊗): element‑wise addition."""
    return np.add(x, y)


# ----------------------------------------------------------------------
# Utility functions from Parent A
# ----------------------------------------------------------------------
def euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian kernel used for graph edge weights."""
    return math.exp(-((epsilon * r) ** 2))


def build_adjacency(features: np.ndarray, epsilon: float = 1.0) -> np.ndarray:
    """
    Construct a fully‑connected adjacency matrix using a Gaussian kernel
    on Euclidean distances between feature vectors.
    """
    n = features.shape[0]
    adj = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            dist = euclidean(features[i], features[j])
            w = gaussian(dist, epsilon)
            adj[i, j] = adj[j, i] = w
    return adj


def laplacian(adj: np.ndarray) -> np.ndarray:
    """Unnormalised graph Laplacian L = D - A."""
    degree = np.diag(adj.sum(axis=1))
    return degree - adj


# ----------------------------------------------------------------------
# Text‑category utilities and Gini coefficient (Parent B)
# ----------------------------------------------------------------------
FUNCTION_CATS: Dict[str, set] = {
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
    "conjunction": set("and but or nor so yet because although while if when where whereas unless until".split()),
    "negation": set("no not never none neither cannot can't won't don't didn't isn't aren't was wasn't weren't".split()),
    "quantifier": set("all any both each few many more most much none several some such".split()),
    "adverb_common": set(
        "very really just still already also even only then there here now often always sometimes".split()
    ),
}


def token_to_category(token: str) -> str:
    """Return the first category a token belongs to, or 'other'."""
    token_l = token.lower()
    for cat, vocab in FUNCTION_CATS.items():
        if token_l in vocab:
            return cat
    return "other"


def gini_coefficient(counts: np.ndarray) -> float:
    """
    Compute the Gini coefficient for a 1‑D array of non‑negative counts.
    """
    if counts.ndim != 1:
        raise ValueError("counts must be 1‑D")
    if np.any(counts < 0):
        raise ValueError("counts must be non‑negative")
    total = counts.sum()
    if total == 0:
        return 0.0
    sorted_counts = np.sort(counts)
    n = len(counts)
    cum = np.cumsum(sorted_counts, dtype=float)
    gini = (n + 1 - 2 * np.sum(cum) / total) / n
    return gini


def gini_from_tokens(tokens: List[str]) -> float:
    """
    Build a histogram over the predefined categories and return its Gini
    coefficient.
    """
    cat_counts = {}
    for tok in tokens:
        cat = token_to_category(tok)
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    # Ensure a fixed order for reproducibility
    categories = sorted(cat_counts.keys())
    counts = np.array([cat_counts[c] for c in categories], dtype=float)
    return gini_coefficient(counts)


# ----------------------------------------------------------------------
# Hybrid NLMS update that mixes tropical algebra, Gini, and graph regularisation
# ----------------------------------------------------------------------
def tropical_nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    text_tokens: List[str],
    lap: np.ndarray,
    base_mu: float = 0.5,
    base_lambda: float = 0.1,
    eps: float = 1e-9,
) -> np.ndarray:
    """
    Perform a single NLMS weight update where:
      * The learning rate is tropical‑combined with the Gini coefficient of the
        current text token distribution.
      * A graph‑Laplacian regularisation term is scaled by tropical multiplication
        with ``base_lambda``.
    """
    # 1️⃣ Gini‑driven tropical learning rate
    gini = gini_from_tokens(text_tokens)          # in [0,1]
    mu_tropical = float(t_add(np.array([base_mu]), np.array([gini]))[0])

    # 2️⃣ Standard NLMS error term
    y = float(np.dot(weights, x))
    error = target - y

    # 3️⃣ Normalisation factor
    norm = float(np.dot(x, x) + eps)

    # 4️⃣ Primary NLMS delta
    delta = (mu_tropical / norm) * error * x

    # 5️⃣ Tropical‑scaled Laplacian regularisation
    #    λ ⊗ L = λ + L (element‑wise addition) → then multiply by w
    lambda_tropical = t_mul(np.full(lap.shape, base_lambda), lap)  # λ + L
    reg = lambda_tropical @ weights

    # 6️⃣ Updated weights
    new_weights = weights + delta - reg
    return new_weights


def hybrid_predict(
    weights: np.ndarray,
    x: np.ndarray,
    lap: np.ndarray,
    base_alpha: float = 0.3,
) -> float:
    """
    Predict using the adapted weights and a graph‑informed bias term.
    The bias is tropical‑added to the linear prediction:
        y = w·x ⊕ (α ⊗ (L·1))
    where ``1`` is a vector of ones and ``α`` is a scalar.
    """
    linear_part = float(np.dot(weights, x))
    graph_bias = float(t_add(np.array([base_alpha]), (lap @ np.ones(lap.shape[0]))) [0])
    return max(linear_part, graph_bias)  # tropical addition = max


def train_hybrid_nlms(
    features: np.ndarray,
    targets: np.ndarray,
    texts: List[List[str]],
    epochs: int = 5,
    base_mu: float = 0.5,
    base_lambda: float = 0.1,
) -> np.ndarray:
    """
    Train the hybrid filter over a dataset.

    Parameters
    ----------
    features : (N, d) array of input vectors.
    targets  : (N,)   array of desired outputs.
    texts    : list of N token‑lists (one per sample).
    epochs   : number of passes over the data.
    base_mu  : base NLMS step size.
    base_lambda : base graph regularisation coefficient.

    Returns
    -------
    weights : final weight vector (d,)
    """
    n_samples, dim = features.shape
    # Initialise weights to small random values
    rng = np.random.default_rng()
    w = rng.normal(scale=0.01, size=dim)

    # Build a static graph over the *feature* space (shared across epochs)
    adj = build_adjacency(features, epsilon=1.0)
    L = laplacian(adj)

    for epoch in range(epochs):
        # Simple online pass
        for i in range(n_samples):
            w = tropical_nlms_update(
                weights=w,
                x=features[i],
                target=float(targets[i]),
                text_tokens=texts[i],
                lap=L,
                base_mu=base_mu,
                base_lambda=base_lambda,
            )
    return w


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic numeric data
    np.random.seed(42)
    X = np.random.randn(20, 4)          # 20 samples, 4‑dimensional
    true_w = np.array([1.5, -2.0, 0.7, 0.3])
    y = X @ true_w + np.random.randn(20) * 0.1

    # Synthetic token data (randomly pick from a small pool)
    vocab = list(
        sum([list(v) for v in FUNCTION_CATS.values()], [])
    ) + ["randomword", "another"]
    token_data = [
        random.choices(vocab, k=random.randint(5, 12)) for _ in range(20)
    ]

    # Train the hybrid model
    learned_w = train_hybrid_nlms(
        features=X,
        targets=y,
        texts=token_data,
        epochs=3,
        base_mu=0.4,
        base_lambda=0.05,
    )

    # Predict on a new random sample
    x_new = np.random.randn(4)
    adj = build_adjacency(np.vstack([X, x_new]), epsilon=1.0)
    L_full = laplacian(adj)
    # Use the last row of L (corresponding to the new point) for bias
    L_new = L_full[-1, :-1]  # connections to training nodes
    pred = hybrid_predict(learned_w, x_new, L_new, base_alpha=0.2)

    print("Learned weights:", learned_w)
    print("Prediction for new sample:", pred)
    sys.exit(0)