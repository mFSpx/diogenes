# DARWIN HAMMER — match 5543, survivor 3
# gen: 7
# parent_a: hybrid_hybrid_perceptual_de_hybrid_hybrid_rlct_g_m578_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_gini_c_m2063_s1.py (gen6)
# born: 2026-05-30T00:04:17Z

import math
import random
from pathlib import Path
from typing import List, Sequence, Tuple, Dict, Iterable

import numpy as np

# ----------------------------------------------------------------------
# Tropical algebra (max‑plus semiring) utilities
# ----------------------------------------------------------------------
def t_add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Tropical addition (⊕): element‑wise maximum."""
    return np.maximum(x, y)


def t_mul(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Tropical multiplication (⊗): element‑wise addition."""
    return np.add(x, y)


# ----------------------------------------------------------------------
# Linear‑algebra utilities (Parent A)
# ----------------------------------------------------------------------
def euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    """Euclidean distance between two equal‑length vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian kernel used for graph edge weights."""
    return math.exp(-((epsilon * r) ** 2))


def build_adjacency(features: np.ndarray, epsilon: float = 1.0) -> np.ndarray:
    """
    Fully‑connected adjacency matrix using a Gaussian kernel on Euclidean
    distances between feature vectors.

    Parameters
    ----------
    features : np.ndarray, shape (n_samples, n_features)
        Feature matrix.
    epsilon : float, optional
        Bandwidth of the Gaussian kernel.

    Returns
    -------
    np.ndarray, shape (n_samples, n_samples)
        Symmetric adjacency matrix with values in (0, 1].
    """
    if features.ndim != 2:
        raise ValueError("features must be a 2‑D array")
    # Pairwise Euclidean distances via broadcasting – O(n²) but vectorised
    diff = features[:, None, :] - features[None, :, :]  # (n, n, d)
    dists = np.linalg.norm(diff, axis=2)               # (n, n)
    adj = np.vectorize(gaussian)(dists, epsilon)
    np.fill_diagonal(adj, 1.0)  # self‑loops have maximal similarity
    return adj


def laplacian(adj: np.ndarray) -> np.ndarray:
    """Unnormalised graph Laplacian L = D - A."""
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError("adjacency must be a square matrix")
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

    The implementation follows the definition based on the Lorenz curve.
    """
    if counts.ndim != 1:
        raise ValueError("counts must be 1‑D")
    if np.any(counts < 0):
        raise ValueError("counts must be non‑negative")
    total = counts.sum()
    if total == 0:
        return 0.0
    sorted_counts = np.sort(counts)                     # ascending
    n = len(counts)
    cum = np.cumsum(sorted_counts, dtype=float)        # cumulative sum
    gini = (n + 1 - 2 * np.sum(cum) / total) / n
    return float(gini)


def gini_from_tokens(tokens: Iterable[str]) -> float:
    """
    Build a histogram over the predefined categories and return its Gini
    coefficient.
    """
    cat_counts: Dict[str, int] = {}
    for tok in tokens:
        cat = token_to_category(tok)
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    # Fixed order for reproducibility
    categories = sorted(cat_counts.keys())
    counts = np.array([cat_counts[c] for c in categories], dtype=float)
    return gini_coefficient(counts)


# ----------------------------------------------------------------------
# Hybrid NLMS update – mathematically tightened
# ----------------------------------------------------------------------
def _tropical_learning_rate(base_mu: float, gini: float) -> float:
    """
    Compute a learning‑rate that respects the tropical semiring while still
    allowing the Gini coefficient to influence the step size even when
    ``base_mu`` > ``gini``.

    The rule is:
        μ = max(base_mu, μ_max * gini)

    where ``μ_max`` is a user‑defined ceiling (default 1.0).  This preserves the
    max‑plus semantics (tropical addition) but rescales the Gini contribution
    into the same numeric range as ``base_mu``.
    """
    mu_max = 1.0
    return max(base_mu, mu_max * gini)


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
    Perform a single NLMS weight update with a *tropically‑aware* learning rate
    and a *standard* graph‑Laplacian regularisation term.

    Parameters
    ----------
    weights : np.ndarray, shape (n_features,)
        Current filter coefficients.
    x : np.ndarray, shape (n_features,)
        Input vector.
    target : float
        Desired scalar output.
    text_tokens : List[str]
        Token list for the current time step – used to compute the Gini coefficient.
    lap : np.ndarray, shape (n_features, n_features)
        Graph Laplacian matrix.
    base_mu : float, optional
        Base learning‑rate (must be > 0).
    base_lambda : float, optional
        Regularisation strength (must be ≥ 0).
    eps : float, optional
        Small constant to avoid division by zero.

    Returns
    -------
    np.ndarray, shape (n_features,)
        Updated weight vector.
    """
    if weights.shape != x.shape:
        raise ValueError("weights and x must have the same shape")
    if lap.shape[0] != lap.shape[1] or lap.shape[0] != weights.shape[0]:
        raise ValueError("laplacian dimensions must match weight vector length")

    # 1️⃣ Gini‑driven tropical learning rate
    gini = gini_from_tokens(text_tokens)                     # ∈ [0, 1]
    mu_t = _tropical_learning_rate(base_mu, gini)           # scalar

    # 2️⃣ NLMS error term
    y = float(weights @ x)
    error = target - y

    # 3️⃣ Normalisation factor (standard NLMS)
    norm = float(x @ x + eps)

    # 4️⃣ Primary NLMS delta
    delta = (mu_t / norm) * error * x

    # 5️⃣ Graph‑Laplacian regularisation (standard algebra)
    #    λ * (L @ w) penalises deviation from smoothness on the graph.
    reg = base_lambda * (lap @ weights)

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

    The bias is computed as a tropical addition between a scalar ``base_alpha``
    and the graph‑driven term ``L·1`` (where ``1`` is the all‑ones vector).  The
    final output is the tropical maximum of the linear prediction and this bias.

    Parameters
    ----------
    weights : np.ndarray, shape (n_features,)
    x : np.ndarray, shape (n_features,)
    lap : np.ndarray, shape (n_features, n_features)
    base_alpha : float, optional

    Returns
    -------
    float
        Predicted scalar.
    """
    if weights.shape != x.shape:
        raise ValueError("weights and x must have the same shape")
    if lap.shape[0] != lap.shape[1] or lap.shape[0] != weights.shape[0]:
        raise ValueError("laplacian dimensions must match weight vector length")

    linear_part = float(weights @ x)

    # Graph bias: α ⊗ (L·1) = α + (L·1) in max‑plus; then tropical addition (max) with linear part
    graph_bias = base_alpha + float(lap @ np.ones(lap.shape[0]))
    return max(linear_part, graph_bias)


def train_hybrid_nlms(
    features: np.ndarray,
    targets: np.ndarray,
    texts: List[List[str]],
    epochs: int = 5,
    base_mu: float = 0.5,
    base_lambda: float = 0.1,
    eps: float = 1e-9,
) -> np.ndarray:
    """
    Train the hybrid NLMS filter on a dataset.

    Parameters
    ----------
    features : np.ndarray, shape (n_samples, n_features)
        Input matrix where each row is a feature vector.
    targets : np.ndarray, shape (n_samples,)
        Desired scalar outputs.
    texts : List[List[str]]
        Token lists aligned with ``features`` (one per sample).
    epochs : int, optional
        Number of passes over the data.
    base_mu : float, optional
        Base learning‑rate for the tropical combination.
    base_lambda : float, optional
        Regularisation strength.
    eps : float, optional
        Small constant for NLMS normalisation.

    Returns
    -------
    np.ndarray, shape (n_features,)
        Final weight vector after training.
    """
    if features.shape[0] != targets.shape[0] or features.shape[0] != len(texts):
        raise ValueError("features, targets, and texts must have the same number of samples")
    n_features = features.shape[1]

    # Initialise weights to small random values for symmetry breaking
    rng = np.random.default_rng()
    weights = rng.normal(loc=0.0, scale=0.01, size=n_features)

    # Pre‑compute the graph Laplacian once (fully‑connected graph)
    adjacency = build_adjacency(features)
    lap = laplacian(adjacency)

    for epoch in range(epochs):
        # Shuffle indices to avoid ordering bias
        indices = np.arange(features.shape[0])
        rng.shuffle(indices)

        for idx in indices:
            x = features[idx]
            y_target = float(targets[idx])
            tokens = texts[idx]

            weights = tropical_nlms_update(
                weights=weights,
                x=x,
                target=y_target,
                text_tokens=tokens,
                lap=lap,
                base_mu=base_mu,
                base_lambda=base_lambda,
                eps=eps,
            )
    return weights


# ----------------------------------------------------------------------
# Simple smoke test (executed when the module is run directly)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic dataset: 20 samples, 4‑dimensional features
    rng = np.random.default_rng(42)
    X = rng.normal(size=(20, 4))
    true_w = np.array([0.2, -0.5, 0.3, 0.1])
    y = X @ true_w + rng.normal(scale=0.05, size=20)

    # Generate dummy token lists (randomly pick from FUNCTION_CATS keys)
    vocab_keys = list(FUNCTION_CATS.keys())
    token_lists = [
        [random.choice(FUNCTION_CATS[random.choice(vocab_keys)]) for _ in range(10)]
        for _ in range(20)
    ]

    learned_w = train_hybrid_nlms(
        features=X,
        targets=y,
        texts=token_lists,
        epochs=10,
        base_mu=0.4,
        base_lambda=0.05,
    )

    # Evaluate on training data
    preds = np.array([hybrid_predict(learned_w, X[i], laplacian(build_adjacency(X))) for i in range(len(X))])
    mse = np.mean((preds - y) ** 2)
    print(f"Training MSE after hybrid NLMS: {mse:.6f}")