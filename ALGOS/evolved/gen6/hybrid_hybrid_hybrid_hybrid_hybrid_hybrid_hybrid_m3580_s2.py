# DARWIN HAMMER — match 3580, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2002_s5.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_privac_m1448_s0.py (gen5)
# born: 2026-05-29T23:50:57Z

"""Hybrid algorithm combining DARWIN HAMMER parent A (distance, kernel, entropy)
and parent B (lead‑lag transform, Kan basis, KL‑divergence risk).

Mathematical bridge:
- Parent A produces scalar similarity measures (Gaussian, SSIM) that can be
  interpreted as a probability mass over a small discrete support.
- Parent B operates on feature matrices with a lead‑lag transformation and
  projects them onto a Kan basis; it also evaluates reconstruction risk via
  Kullback‑Leibler (KL) divergence.
The fusion builds a feature vector from A, normalises it to a probability
distribution, feeds it through the lead‑lag → Kan pipeline of B, and finally
combines the KL‑divergence of that distribution against a reference with the
Shannon entropy of an auxiliary text payload. The resulting scalar is the
hybrid risk‑similarity score.
"""

import math
import random
import sys
import pathlib
import json
import hashlib
import numpy as np
from collections import Counter
import re
from dataclasses import dataclass
from typing import Sequence, Iterable

# ----------------------------------------------------------------------
# Types and data structures (from parent B)
# ----------------------------------------------------------------------
Vector = Sequence[float]

@dataclass(frozen=True)
class MathAction:
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0

@dataclass(frozen=True)
class MathCounterfactual:
    action_id: str
    outcome_value: float
    probability: float = 1.0

@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str

@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

# ----------------------------------------------------------------------
# Utilities from parent A
# ----------------------------------------------------------------------
def _to_numpy(v: Vector) -> np.ndarray:
    """Convert any sequence of numbers to a 1‑D float ndarray."""
    return np.asarray(v, dtype=float).ravel()


def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Isotropic Gaussian kernel."""
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: Vector, b: Vector) -> float:
    """Euclidean distance between two vectors."""
    a_arr, b_arr = _to_numpy(a), _to_numpy(b)
    if a_arr.shape != b_arr.shape:
        raise ValueError("vectors must have same dimension")
    return float(np.linalg.norm(a_arr - b_arr))


def ssim(v1: Vector, v2: Vector) -> float:
    """
    Lightweight structural similarity proxy for 1‑D vectors.
    Cosine similarity scaled to [0, 1].
    """
    a, b = _to_numpy(v1), _to_numpy(v2)
    if a.shape != b.shape:
        raise ValueError("vectors must have same dimension")
    dot = float(np.dot(a, b))
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    cosine = dot / (norm_a * norm_b)
    return (cosine + 1.0) / 2.0


def shannon_entropy(text: str) -> float:
    """Shannon entropy of token frequencies in *text*."""
    tokens = re.findall(r"\w+", text.lower())
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    total = float(sum(counts.values()))
    probs = np.fromiter((c / total for c in counts.values()), dtype=float)
    return float(-np.sum(probs * np.log(probs)))


# ----------------------------------------------------------------------
# Utilities from parent B
# ----------------------------------------------------------------------
def lead_lag_transform(X: np.ndarray) -> np.ndarray:
    """
    Lead‑lag transformation: concatenates linear (sum) and quadratic (sum of squares)
    statistics of each row.
    """
    linear = np.sum(X, axis=1, keepdims=True)
    quadratic = np.sum(X ** 2, axis=1, keepdims=True)
    return np.hstack((linear, quadratic))


def kan_basis(grid_size: int) -> np.ndarray:
    """
    Simple Kan‑basis generator.
    Returns a (grid_size, grid_size) matrix whose columns are exponentially
    decaying vectors and are orthonormalised via QR decomposition.
    """
    points = np.linspace(0, 1, grid_size)
    raw = np.exp(-np.outer(points, np.arange(grid_size)))
    # QR to obtain an orthonormal basis
    q, _ = np.linalg.qr(raw)
    return q


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Kullback‑Leibler divergence D_KL(p || q) for discrete distributions.
    Both inputs must be non‑negative and sum to 1.
    """
    if p.shape != q.shape:
        raise ValueError("Distributions must have the same shape")
    eps = np.finfo(float).eps
    p_safe = np.clip(p, eps, 1)
    q_safe = np.clip(q, eps, 1)
    return float(np.sum(p_safe * np.log(p_safe / q_safe)))


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def build_similarity_distribution(a: Vector, b: Vector) -> np.ndarray:
    """
    Constructs a 3‑component probability distribution from parent‑A similarities:
        [Gaussian(euclidean), SSIM, normalized Euclidean complement]
    The vector is normalised to sum to 1.
    """
    d = euclidean(a, b)
    g = gaussian(d, epsilon=1.0)
    s = ssim(a, b)
    # Complement of Euclidean distance (larger distance → smaller weight)
    max_dist = np.linalg.norm(np.ones_like(_to_numpy(a)))  # max possible in unit cube
    e_comp = max(0.0, 1.0 - d / max_dist)
    raw = np.array([g, s, e_comp], dtype=float)
    if raw.sum() == 0:
        return np.full_like(raw, 1.0 / raw.size)
    return raw / raw.sum()


def lead_lag_kan_projection(dist: np.ndarray, grid_size: int = 5) -> np.ndarray:
    """
    Applies the lead‑lag transform to a 2‑D matrix built from the distribution,
    then projects the result onto a Kan basis.
    """
    # reshape distribution to a column matrix (n, 1)
    X = dist.reshape(-1, 1).astype(float)
    # Lead‑lag yields a (n, 2) matrix
    ll = lead_lag_transform(X)
    # Pad/trim to match grid size
    if ll.shape[0] < grid_size:
        pad = np.zeros((grid_size - ll.shape[0], ll.shape[1]))
        ll = np.vstack((ll, pad))
    elif ll.shape[0] > grid_size:
        ll = ll[:grid_size, :]
    # Kan basis
    K = kan_basis(grid_size)
    # Project each column onto the basis
    projected = K.T @ ll
    return projected  # shape (grid_size, 2)


def hybrid_risk_score(a: Vector, b: Vector, text: str) -> float:
    """
    Final hybrid metric:
        1. Build similarity distribution from vectors (parent A).
        2. Compute KL divergence against a uniform reference.
        3. Transform via lead‑lag + Kan basis (parent B).
        4. Combine KL, entropy of *text*, and the Frobenius norm of the projected matrix.
    The result is a scalar risk‑similarity score; larger values indicate higher
    combined risk / dissimilarity.
    """
    # Step 1
    p = build_similarity_distribution(a, b)

    # Step 2 – uniform reference
    q = np.full_like(p, 1.0 / p.size)
    kl = kl_divergence(p, q)

    # Step 3
    proj = lead_lag_kan_projection(p, grid_size=5)

    # Step 4 – combine
    proj_norm = np.linalg.norm(proj, ord='fro')
    entropy = shannon_entropy(text)
    # Weighted sum (weights chosen heuristically)
    score = 0.6 * kl + 0.3 * proj_norm + 0.1 * entropy
    return score


# ----------------------------------------------------------------------
# Additional demonstration functions
# ----------------------------------------------------------------------
def sample_actions(num: int = 3) -> list[MathAction]:
    """Generate a list of random MathAction objects for testing."""
    actions = []
    for i in range(num):
        actions.append(
            MathAction(
                id=f"act_{i}",
                expected_value=random.uniform(-10, 10),
                cost=random.uniform(0, 5),
                risk=random.uniform(0, 1),
            )
        )
    return actions


def evaluate_counterfactuals(
    base_vec: Vector, actions: Iterable[MathAction]
) -> dict[str, float]:
    """
    For each action, create a counterfactual outcome by perturbing the base vector,
    then compute a hybrid risk score against the original vector.
    Returns a mapping from action id to score.
    """
    base_arr = _to_numpy(base_vec)
    results = {}
    for act in actions:
        perturb = np.random.normal(scale=act.risk, size=base_arr.shape)
        new_vec = base_arr + perturb
        # dummy text derived from action id
        txt = f"action {act.id} performed"
        score = hybrid_risk_score(base_arr, new_vec, txt)
        results[act.id] = score
    return results


def serialize_score_map(score_map: dict[str, float], path: pathlib.Path) -> None:
    """Write the score dictionary as JSON to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(score_map, fp, indent=2, sort_keys=True)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple deterministic vectors
    vec1 = [0.2, 0.5, 0.7]
    vec2 = [0.3, 0.4, 0.9]
    sample_text = "The quick brown fox jumps over the lazy dog."

    # Compute core hybrid score
    core_score = hybrid_risk_score(vec1, vec2, sample_text)
    print(f"Hybrid risk‑similarity score: {core_score:.4f}")

    # Generate actions and evaluate
    actions = sample_actions(5)
    scores = evaluate_counterfactuals(vec1, actions)
    print("Per‑action hybrid scores:")
    for aid, sc in scores.items():
        print(f"  {aid}: {sc:.4f}")

    # Serialize results
    out_path = pathlib.Path("hybrid_output.json")
    serialize_score_map(scores, out_path)
    print(f"Scores written to {out_path.resolve()}")