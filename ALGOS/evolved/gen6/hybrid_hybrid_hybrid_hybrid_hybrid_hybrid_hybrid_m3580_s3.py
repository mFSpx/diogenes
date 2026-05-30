# DARWIN HAMMER — match 3580, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2002_s5.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_privac_m1448_s0.py (gen5)
# born: 2026-05-29T23:50:57Z

import math
import random
import re
from collections import Counter
from dataclasses import dataclass
from typing import Sequence, Iterable, List

import numpy as np

# ----------------------------------------------------------------------
# Core data structures (kept from parent B for compatibility)
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
# Utilities from Parent A (distance / similarity)
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
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    cosine = float(np.dot(a, b) / (norm_a * norm_b))
    return (cosine + 1.0) / 2.0


def shannon_entropy(text: str) -> float:
    """Shannon entropy of token frequencies in *text*."""
    tokens = re.findall(r"\w+", text.lower())
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    total = float(sum(counts.values()))
    probs = np.fromiter((c / total for c in counts.values()), dtype=float)
    return float(-np.sum(probs * np.log(probs + np.finfo(float).eps)))


# ----------------------------------------------------------------------
# Utilities from Parent B (lead‑lag, Kan basis, KL)
# ----------------------------------------------------------------------
def lead_lag_transform(X: np.ndarray) -> np.ndarray:
    """
    Lead‑lag transformation: for each row of X compute the sum and sum‑of‑squares.
    Returns an (n, 2) matrix.
    """
    linear = np.sum(X, axis=1, keepdims=True)
    quadratic = np.sum(X ** 2, axis=1, keepdims=True)
    return np.hstack((linear, quadratic))


def kan_basis(grid_size: int) -> np.ndarray:
    """
    Generate an orthonormal Kan basis of shape (grid_size, grid_size).
    The basis vectors are exponentially decaying and orthonormalised via QR.
    """
    points = np.linspace(0, 1, grid_size)
    raw = np.exp(-np.outer(points, np.arange(grid_size)))
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
# Improved Hybrid Core
# ----------------------------------------------------------------------
def _max_possible_distance(dim: int) -> float:
    """
    Upper bound on Euclidean distance for vectors whose components lie in [0, 1].
    This replaces the ad‑hoc unit‑cube norm used previously and makes the
    complement term dimension‑aware.
    """
    return math.sqrt(dim)  # distance between (0,…,0) and (1,…,1)


def build_similarity_distribution(a: Vector, b: Vector) -> np.ndarray:
    """
    Construct a probability distribution that fuses three complementary
    similarity signals:

        1. Gaussian kernel on Euclidean distance.
        2. Cosine‑based SSIM proxy.
        3. Complement of the (scaled) Euclidean distance.

    The three raw scores are non‑negative; they are finally normalised to sum
    to one.  If all scores are zero, a uniform distribution is returned.
    """
    a_arr, b_arr = _to_numpy(a), _to_numpy(b)
    dim = a_arr.size

    # 1. Gaussian on Euclidean distance
    d = euclidean(a_arr, b_arr)
    g = gaussian(d, epsilon=1.0)

    # 2. SSIM (cosine‑based)
    s = ssim(a_arr, b_arr)

    # 3. Distance complement (larger distance → smaller weight)
    max_dist = _max_possible_distance(dim)
    e_comp = max(0.0, 1.0 - d / max_dist)

    raw = np.array([g, s, e_comp], dtype=float)
    total = raw.sum()
    if total == 0.0:
        return np.full_like(raw, 1.0 / raw.size)
    return raw / total


def lead_lag_kan_projection(
    dist: np.ndarray, grid_size: int = 5, pad_mode: str = "zero"
) -> np.ndarray:
    """
    Deeply integrate the similarity distribution with Parent‑B machinery:

    1. Reshape the distribution into a column vector.
    2. Apply the lead‑lag transform (producing a (n, 2) matrix).
    3. Pad/truncate the matrix to *grid_size* rows using the chosen *pad_mode*.
    4. Project each column onto the Kan basis of the same size.

    Parameters
    ----------
    dist: np.ndarray
        Probability vector (must sum to 1).
    grid_size: int
        Desired dimensionality of the Kan basis.
    pad_mode: str
        Either ``"zero"`` (pad with zeros) or ``"repeat"`` (repeat last row).

    Returns
    -------
    np.ndarray
        The projected matrix of shape (grid_size, 2).
    """
    if dist.ndim != 1:
        raise ValueError("dist must be a 1‑D probability vector")
    X = dist.reshape(-1, 1)  # (n, 1)

    ll = lead_lag_transform(X)  # (n, 2)

    # Pad / trim to grid_size rows
    if ll.shape[0] < grid_size:
        if pad_mode == "zero":
            pad = np.zeros((grid_size - ll.shape[0], ll.shape[1]), dtype=float)
        elif pad_mode == "repeat":
            repeat_row = ll[-1, :][np.newaxis, :]
            pad = np.repeat(repeat_row, grid_size - ll.shape[0], axis=0)
        else:
            raise ValueError(f"Unsupported pad_mode: {pad_mode}")
        ll = np.vstack((ll, pad))
    elif ll.shape[0] > grid_size:
        ll = ll[:grid_size, :]

    K = kan_basis(grid_size)               # (grid_size, grid_size)
    projected = K.T @ ll                    # (grid_size, 2)
    return projected


def hybrid_risk_score(
    a: Vector,
    b: Vector,
    text: str,
    *,
    grid_size: int = 5,
    kl_weight: float = 0.5,
    proj_weight: float = 0.35,
    ent_weight: float = 0.15,
    reference: Iterable[float] | None = None,
) -> float:
    """
    Compute a unified risk‑similarity scalar that tightly couples Parent‑A
    similarity with Parent‑B transformation and information‑theoretic measures.

    The pipeline is:

    1. Build a robust similarity distribution ``p`` from the two vectors.
    2. Compute KL divergence against a *reference* distribution.
       If ``reference`` is ``None`` a uniform distribution of matching size is used.
    3. Transform ``p`` via lead‑lag → Kan projection, obtaining ``P``.
    4. Compute the Frobenius norm of ``P`` (captures the energy after projection).
    5. Compute Shannon entropy of the auxiliary text payload.
    6. Return a weighted sum of the three quantities.

    The weights are configurable and sum to 1.0 by default, but the function
    does not enforce this – callers may experiment with different trade‑offs.
    """
    # 1. similarity distribution
    p = build_similarity_distribution(a, b)

    # 2. KL divergence
    if reference is None:
        q = np.full_like(p, 1.0 / p.size)
    else:
        q_arr = np.asarray(list(reference), dtype=float)
        if q_arr.shape != p.shape:
            raise ValueError("reference distribution must match shape of p")
        q = q_arr / q_arr.sum()  # ensure proper normalisation
    kl = kl_divergence(p, q)

    # 3. Lead‑lag + Kan projection
    proj = lead_lag_kan_projection(p, grid_size=grid_size, pad_mode="zero")
    proj_norm = np.linalg.norm(proj, ord="fro")

    # 4. Text entropy
    entropy = shannon_entropy(text)

    # 5. Weighted aggregation
    score = kl_weight * kl + proj_weight * proj_norm + ent_weight * entropy
    return float(score)


# ----------------------------------------------------------------------
# Helper / demo functions (unchanged semantics, but type‑annotated)
# ----------------------------------------------------------------------
def sample_actions(num: int = 3) -> List[MathAction]:
    """Generate a list of random MathAction objects for testing."""
    actions: List[MathAction] = []
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
    base_vec: Vector,
    counterfactuals: Iterable[MathCounterfactual],
    reference_vec: Vector,
    text: str,
) -> List[float]:
    """
    Evaluate a collection of counterfactual outcomes by feeding each
    ``outcome_value`` as a perturbed version of ``base_vec`` and returning the
    hybrid risk score.

    The perturbation simply adds the scalar ``outcome_value`` to every component
    of ``base_vec`` (clipped to the [0, 1] interval to keep distances bounded).
    """
    scores: List[float] = []
    base_arr = _to_numpy(base_vec)

    for cf in counterfactuals:
        perturbed = np.clip(base_arr + cf.outcome_value, 0.0, 1.0)
        score = hybrid_risk_score(perturbed, reference_vec, text)
        scores.append(score)

    return scores


# ----------------------------------------------------------------------
# Example usage (guarded by __main__) – not executed in import contexts.
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple sanity check
    a_vec = [0.2, 0.5, 0.7]
    b_vec = [0.3, 0.45, 0.65]
    txt = "The quick brown fox jumps over the lazy dog."

    print("Hybrid risk score:", hybrid_risk_score(a_vec, b_vec, txt))

    # Counterfactual demo
    cf_list = [
        MathCounterfactual(action_id="act_0", outcome_value=0.1),
        MathCounterfactual(action_id="act_1", outcome_value=-0.2),
    ]
    ref_vec = [0.4, 0.4, 0.4]
    print("Counterfactual scores:", evaluate_counterfactuals(a_vec, cf_list, ref_vec, txt))