# DARWIN HAMMER — match 3437, survivor 5
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hdc_se_hybrid_hybrid_hybrid_m1502_s1.py (gen6)
# parent_b: hybrid_variational_free_ene_hybrid_hybrid_label__m1556_s5.py (gen4)
# born: 2026-05-29T23:50:12Z

import numpy as np
import math
import hashlib
from typing import List, Dict, Tuple
from dataclasses import dataclass

Vector = np.ndarray  # bipolar hypervector of dtype int8


def _int_seed_from_hash(text: str) -> int:
    """Create a deterministic 64‑bit integer seed from a string."""
    digest = hashlib.md5(text.encode("utf-8")).digest()[:8]
    return int.from_bytes(digest, byteorder="big", signed=False)


def random_vector(dim: int = 10_000, seed: int | str | None = None) -> Vector:
    """
    Generate a bipolar random hypervector (elements are -1 or +1).

    Parameters
    ----------
    dim : int
        Dimensionality of the hypervector.
    seed : int | str | None
        Seed for reproducibility. ``str`` values are hashed to an integer.

    Returns
    -------
    Vector
        A ``np.int8`` array of shape ``(dim,)``.
    """
    if isinstance(seed, str):
        seed = _int_seed_from_hash(seed)
    rng = np.random.default_rng(seed)
    # 0 → -1, 1 → +1
    bits = rng.integers(0, 2, size=dim, dtype=np.int8)
    return np.where(bits == 0, -1, 1)


def symbol_vector(symbol: str, dim: int = 10_000) -> Vector:
    """
    Deterministically map a textual symbol to a bipolar hypervector.
    """
    return random_vector(dim, _int_seed_from_hash(symbol))


def bind(a: Vector, b: Vector) -> Vector:
    """
    Element‑wise binding (multiplication) of two hypervectors.
    """
    if a.shape != b.shape:
        raise ValueError("vectors must have equal shape")
    return a * b


def bundle(vectors: List[Vector]) -> Vector:
    """
    Superposition (addition) of hypervectors followed by binarization
    using the sign function.
    """
    if not vectors:
        raise ValueError("bundle requires at least one vector")
    stacked = np.stack(vectors, axis=0).astype(np.int32)
    summed = stacked.sum(axis=0)
    return np.where(summed >= 0, 1, -1).astype(np.int8)


def cosine_similarity(a: Vector, b: Vector, eps: float = 1e-12) -> float:
    """
    Cosine similarity robust to zero‑norm vectors.
    """
    if a.shape != b.shape:
        raise ValueError("vectors must have equal shape")
    dot = float(np.dot(a, b))
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    denom = max(norm_a * norm_b, eps)
    return dot / denom


def _safe_log(x: float, eps: float = 1e-12) -> float:
    """Log that never receives a non‑positive argument."""
    return math.log(max(x, eps))


@dataclass(frozen=True)
class LabelingFunctionResult:
    lf_name: str
    doc_id: str
    label: int  # 0 or 1


@dataclass(frozen=True)
class ProbabilisticLabel:
    doc_id: str
    label: int
    confidence: float  # in [0, 1]


def _encode_result_as_vector(result: LabelingFunctionResult, dim: int = 10_000) -> Vector:
    """
    Encode a labeling‑function result into a hypervector by binding the
    document identifier vector with the label vector.
    """
    doc_vec = symbol_vector(result.doc_id, dim)
    label_vec = symbol_vector(str(result.label), dim)
    return bind(doc_vec, label_vec)


def hybrid_precision(
    sigma_q: float,
    recovery_priority: float,
    entropy_modulation: float,
    similarity: float,
) -> float:
    """
    Compute a precision term that blends statistical uncertainty,
    algorithmic priors and the geometric similarity of hypervectors.
    The similarity term is scaled to [0, 1] and added linearly.
    """
    base = (1.0 / sigma_q ** 2) * (1.0 + recovery_priority) * (1.0 + entropy_modulation)
    # similarity ∈ [‑1, 1]; shift to [0, 1] for a non‑negative contribution
    sim_factor = 0.5 * (similarity + 1.0)
    return base * (1.0 + sim_factor)


def hybrid_free_energy(
    mu_q: float,
    sigma_q: float,
    mu_p: float,
    sigma_p: float,
    recovery_priority: float,
    entropy_modulation: float,
    similarity: float,
) -> float:
    """
    Variational free‑energy term enriched with hyperdimensional similarity.
    """
    # KL divergence between two univariate Gaussians
    kl = (mu_q - mu_p) ** 2 / (2.0 * sigma_p ** 2) + 0.5 * _safe_log(sigma_p ** 2 / sigma_q ** 2)

    # Surprise term: negative log‑likelihood of observing mu_q under N(mu_p, sigma_p)
    z = (mu_q - mu_p) / (sigma_p * math.sqrt(2.0))
    surprise = -_safe_log(0.5 * (1.0 + math.erf(z)))

    prec = hybrid_precision(sigma_q, recovery_priority, entropy_modulation, similarity)
    return kl - surprise / prec


def hybrid_belief_update(
    mu_q: float,
    sigma_q: float,
    mu_p: float,
    sigma_p: float,
    recovery_priority: float,
    entropy_modulation: float,
    similarity: float,
) -> Tuple[float, float]:
    """
    Posterior update for the variational parameters using the enriched precision.
    """
    prec = hybrid_precision(sigma_q, recovery_priority, entropy_modulation, similarity)
    mu_q_new = (prec * mu_q + (1.0 / sigma_p ** 2) * mu_p) / (prec + (1.0 / sigma_p ** 2))
    sigma_q_new = 1.0 / math.sqrt(prec + (1.0 / sigma_p ** 2))
    return mu_q_new, sigma_q_new


def hybrid_aggregate_labels(
    labeling_function_results: List[LabelingFunctionResult],
    recovery_priorities: Dict[str, float],
    entropy_modulations: Dict[str, float],
    dim: int = 10_000,
) -> List[ProbabilisticLabel]:
    """
    Fuse multiple labeling‑function outputs into probabilistic labels.
    The fusion leverages hyperdimensional similarity between each result
    and a global reference vector (the bundle of all results).
    """
    # Encode every result as a hypervector
    encoded = [_encode_result_as_vector(r, dim) for r in labeling_function_results]

    # Global reference: bundle of all encoded vectors (captures the consensus)
    reference = bundle(encoded)

    probabilistic_labels: List[ProbabilisticLabel] = []
    for result, vec in zip(labeling_function_results, encoded):
        # Retrieve algorithmic priors; default to neutral values if missing
        rp = recovery_priorities.get(result.lf_name, 0.0)
        em = entropy_modulations.get(result.lf_name, 0.0)

        # Geometric similarity to the consensus reference
        sim = cosine_similarity(vec, reference)

        # Compute a confidence via a logistic transform of free energy
        fe = hybrid_free_energy(
            mu_q=0.0,
            sigma_q=1.0,
            mu_p=0.0,
            sigma_p=1.0,
            recovery_priority=rp,
            entropy_modulation=em,
            similarity=sim,
        )
        confidence = 1.0 / (1.0 + math.exp(fe))  # sigmoid of negative free energy

        probabilistic_labels.append(
            ProbabilisticLabel(doc_id=result.doc_id, label=result.label, confidence=confidence)
        )
    return probabilistic_labels


if __name__ == "__main__":
    # Simple sanity check
    vec_a = random_vector()
    vec_b = random_vector()
    print("Cosine similarity (random):", cosine_similarity(vec_a, vec_b))

    # Create synthetic labeling‑function results
    results = [
        LabelingFunctionResult("lf_A", "doc_001", 1),
        LabelingFunctionResult("lf_B", "doc_001", 0),
        LabelingFunctionResult("lf_A", "doc_002", 0),
    ]

    rp = {"lf_A": 0.3, "lf_B": 0.7}
    em = {"lf_A": 0.1, "lf_B": 0.4}

    probs = hybrid_aggregate_labels(results, rp, em)
    for p in probs:
        print(f"Doc {p.doc_id} label {p.label} confidence {p.confidence:.4f}")