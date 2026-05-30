# DARWIN HAMMER — match 2508, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_regret_hybrid_hybrid_ternar_m241_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_geomet_m530_s2.py (gen5)
# born: 2026-05-29T23:42:41Z

import numpy as np
import hashlib
from dataclasses import dataclass
from typing import Iterable, List, Sequence


# ----------------------------------------------------------------------
# Core data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathAction:
    """Immutable description of a decision option."""
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0

    @property
    def regret_weight(self) -> float:
        """Regret‑weighted scalar used throughout the hybrid model."""
        return self.expected_value - self.cost - self.risk


@dataclass(frozen=True)
class MathCounterfactual:
    """Result of a counter‑factual evaluation for a single action."""
    action_id: str
    outcome_value: float
    probability: float = 1.0


# ----------------------------------------------------------------------
# Utility hashing / signature helpers
# ----------------------------------------------------------------------
def _hash(seed: int, token: str) -> int:
    """
    Deterministic 64‑bit hash based on a seed and a token string.
    The seed allows us to generate a family of independent hash functions.
    """
    h = hashlib.sha256(f"{seed}:{token}".encode("utf-8")).digest()
    # Take the first 8 bytes → 64‑bit unsigned integer
    return int.from_bytes(h[:8], byteorder="big", signed=False)


def signature(tokens: Iterable[str], k: int = 128) -> np.ndarray:
    """
    Min‑hash signature of a token set.
    Returns a 1‑D ``np.ndarray`` of length ``k`` containing the minimum hash
    value for each seed across all non‑empty tokens.
    """
    toks = [t for t in tokens if t]
    if k <= 0:
        raise ValueError("k must be a positive integer")
    if not toks:
        # All‑ones sentinel – useful for empty documents
        return np.full(k, (1 << 64) - 1, dtype=np.uint64)

    sig = np.empty(k, dtype=np.uint64)
    for i in range(k):
        # Vectorised min‑hash across tokens for seed i
        hashes = [_hash(i, t) for t in toks]
        sig[i] = min(hashes)
    return sig


def similarity(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    """
    Normalised Jaccard‑like similarity for two min‑hash signatures.
    ``sig_a`` and ``sig_b`` must have the same length.
    """
    if sig_a.shape != sig_b.shape:
        raise ValueError("signatures must have identical shape")
    if sig_a.size == 0:
        raise ValueError("signatures must not be empty")
    matches = np.count_nonzero(sig_a == sig_b)
    return matches / sig_a.size


# ----------------------------------------------------------------------
# Geometric‑algebra inspired action encoding
# ----------------------------------------------------------------------
def _action_multivector(action_id: str, dim: int = 64) -> np.ndarray:
    """
    Produce a binary multivector representation of an action.
    The i‑th component is the i‑th bit of the 64‑bit hash of ``action_id``.
    """
    h = _hash(0, action_id)  # seed 0 → deterministic base hash
    bits = np.unpackbits(
        np.array([h], dtype=">u8").view(np.uint8), bitorder="big"
    )
    # ``bits`` is length 64; pad/truncate to ``dim`` if needed
    if dim > 64:
        bits = np.concatenate([bits, np.zeros(dim - 64, dtype=np.uint8)])
    elif dim < 64:
        bits = bits[:dim]
    return bits.astype(np.float64)


def hybrid_action_space(actions: Sequence[MathAction], dim: int = 64) -> np.ndarray:
    """
    Encode a list of actions as a matrix of weighted multivectors.
    Each row corresponds to an action:
        (regret_weight) × (binary multivector)
    The result has shape ``(n_actions, dim)``.
    """
    if dim <= 0:
        raise ValueError("dim must be a positive integer")
    multivectors = np.vstack(
        [_action_multivector(a.id, dim) for a in actions]
    )
    weights = np.array([a.regret_weight for a in actions], dtype=np.float64)
    # Broadcast the scalar weight across the vector dimension
    return weights[:, np.newaxis] * multivectors


# ----------------------------------------------------------------------
# Counter‑factual generation with softmax probabilities
# ----------------------------------------------------------------------
def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    max_logit = np.max(logits)
    exps = np.exp(logits - max_logit)
    sum_exps = np.sum(exps)
    if sum_exps == 0.0:
        # Avoid division by zero – fall back to uniform distribution
        return np.full_like(logits, 1.0 / logits.size)
    return exps / sum_exps


def hybrid_counterfactuals(
    actions: Sequence[MathAction],
    outcomes: Sequence[float],
    dim: int = 64,
) -> List[MathCounterfactual]:
    """
    Produce counter‑factual probabilities for each ``action`` given a set of
    ``outcomes``.  The probability distribution is obtained by applying a
    softmax to the inner product between the weighted action multivectors and
    a shared outcome vector (here taken as the raw ``outcomes`` themselves).
    """
    if not actions:
        raise ValueError("actions list must not be empty")
    if not outcomes:
        raise ValueError("outcomes list must not be empty")

    # Encode actions once
    weighted_space = hybrid_action_space(actions, dim)          # (n, dim)

    # Build a simple outcome embedding – each outcome is broadcast over ``dim``
    outcome_matrix = np.tile(np.array(outcomes, dtype=np.float64)[:, np.newaxis], (1, dim))

    counterfactuals: List[MathCounterfactual] = []
    for outcome_idx, outcome_val in enumerate(outcomes):
        # Logits = ⟨ weighted_action , outcome_vector ⟩
        logits = np.dot(weighted_space, outcome_matrix[outcome_idx])
        probs = _softmax(logits)
        for action, prob in zip(actions, probs):
            counterfactuals.append(MathCounterfactual(action.id, outcome_val, prob))
    return counterfactuals


# ----------------------------------------------------------------------
# Deepened similarity: inner product of weighted multivector space with a signature
# ----------------------------------------------------------------------
def hybrid_similarity(
    signatures: np.ndarray,
    actions: Sequence[MathAction],
    dim: int = 64,
) -> float:
    """
    Compute a similarity score between a min‑hash signature and the
    regret‑weighted geometric representation of ``actions``.
    The procedure is:

    1. Encode actions as a weighted multivector matrix (shape ``(n, dim)``).
    2. Reduce the matrix to a single vector by summing over actions.
    3. Compute the normalised inner product with the signature (truncated/padded to ``dim``).

    The result lies in ``[-1, 1]`` after normalisation.
    """
    if signatures.ndim != 1:
        raise ValueError("signatures must be a 1‑D array")
    if dim <= 0:
        raise ValueError("dim must be a positive integer")

    # Ensure signature length matches ``dim`` (pad with zeros if shorter)
    if signatures.size < dim:
        sig_vec = np.pad(signatures.astype(np.float64), (0, dim - signatures.size))
    else:
        sig_vec = signatures[:dim].astype(np.float64)

    weighted_space = hybrid_action_space(actions, dim)          # (n, dim)
    aggregate_vec = weighted_space.sum(axis=0)                 # (dim,)

    # Normalise both vectors to unit length to bound the dot product
    norm_sig = np.linalg.norm(sig_vec)
    norm_agg = np.linalg.norm(aggregate_vec)
    if norm_sig == 0.0 or norm_agg == 0.0:
        return 0.0

    return float(np.dot(sig_vec, aggregate_vec) / (norm_sig * norm_agg))


# ----------------------------------------------------------------------
# Example usage (executed only when run as a script)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    actions = [
        MathAction("action1", 10.0, 2.0, 1.0),
        MathAction("action2", 20.0, 3.0, 2.0),
        MathAction("action3", 30.0, 4.0, 3.0),
    ]

    outcomes = [1.0, 2.0, 3.0]

    cf = hybrid_counterfactuals(actions, outcomes)
    for c in cf:
        print(c)

    sig = signature(["token1", "token2"], k=128)
    sim = hybrid_similarity(sig, actions)
    print(f"Hybrid similarity: {sim:.4f}")