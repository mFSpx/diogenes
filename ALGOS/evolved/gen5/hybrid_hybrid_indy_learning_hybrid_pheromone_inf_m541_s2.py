# DARWIN HAMMER — match 541, survivor 2
# gen: 5
# parent_a: hybrid_indy_learning_vector_hybrid_fold_change_d_m38_s1.py (gen4)
# parent_b: hybrid_pheromone_infotaxis_m3_s0.py (gen1)
# born: 2026-05-29T23:29:42Z

import hashlib
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------


def sha256_json(value: Any) -> str:
    """Deterministic hash of any JSON‑serialisable object."""
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
    ).hexdigest()


def tokenize(text: str) -> List[Dict[str, Any]]:
    """Very light tokenizer returning token text and character offsets."""
    word_re = re.compile(r"\S+")
    return [
        {"token": m.group(0), "start": m.start(), "end": m.end()}
        for m in word_re.finditer(text)
    ]


def chunk_text_tokens(
    text: str,
    *,
    max_tokens: int = 500,
    overlap_tokens: int = 0,
    source_ref: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """Split a token list into overlapping windows."""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError(
            "overlap_tokens must be >=0 and < max_tokens"
        )
    toks = tokenize(text)
    source_ref = dict(source_ref or {})
    if not toks:
        return []

    chunks: List[Dict[str, Any]] = []
    step = max_tokens - overlap_tokens
    for i in range(0, len(toks), step):
        chunk = toks[i : i + max_tokens]
        chunks.append({"tokens": chunk, "source_ref": source_ref})
    return chunks


# ----------------------------------------------------------------------
# Core mathematical building blocks
# ----------------------------------------------------------------------


def entropy(probs: np.ndarray, eps: float = 1e-12) -> float:
    """Shannon entropy, robust to zero probabilities."""
    probs = np.asarray(probs, dtype=float)
    total = probs.sum()
    if total <= 0:
        raise ValueError("positive probability mass required")
    probs = probs / total
    probs = np.clip(probs, eps, 1.0)
    return -float(np.sum(probs * np.log(probs)))


def log_count_statistics(tokens: List[Dict[str, Any]]) -> Counter:
    """Return raw token frequencies (log‑count bridge)."""
    return Counter(tok["token"] for tok in tokens)


# ----------------------------------------------------------------------
# Pheromone handling (decoupled from a specific DB driver)
# ----------------------------------------------------------------------


class Pheromone:
    """Simple container for a surface key and its signal strength."""

    __slots__ = ("surface_key", "signal_value")

    def __init__(self, surface_key: str, signal_value: float):
        self.surface_key = surface_key
        self.signal_value = signal_value


def fetch_pheromones(
    surface_key: str, limit: int, db_url: str | None = None
) -> List[Pheromone]:
    """
    Retrieve the most recent pheromone signals for a given surface.
    If a DB connection cannot be established (e.g. psycopg missing),
    fall back to a lightweight random mock – this keeps the module
    runnable in constrained environments while preserving the API.
    """
    try:
        import psycopg
        from psycopg.rows import dict_row

        with psycopg.connect(db_url, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT signal_value
                    FROM lucidota_runtime.surface_pheromone
                    WHERE surface_key = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (surface_key, limit),
                )
                rows = cur.fetchall()
                return [
                    Pheromone(surface_key, float(r["signal_value"])) for r in rows
                ]
    except Exception:
        # Graceful degradation: generate deterministic pseudo‑random values
        random.seed(hash(surface_key) % (2 ** 32))
        return [
            Pheromone(surface_key, random.random()) for _ in range(limit)
        ]


def pheromone_distribution(pheromones: List[Pheromone]) -> np.ndarray:
    """Convert raw pheromone signals into a proper probability distribution."""
    if not pheromones:
        raise ValueError("At least one pheromone required")
    signals = np.array([p.signal_value for p in pheromones], dtype=float)
    # Replace non‑positive signals with a small epsilon to avoid dead weight
    signals = np.where(signals > 0, signals, 1e-6)
    total = signals.sum()
    if total == 0:
        # Uniform fallback – mathematically safe
        return np.full_like(signals, 1.0 / len(signals))
    return signals / total


# ----------------------------------------------------------------------
# Fusion of deterministic (log‑count) and stochastic (pheromone) knowledge
# ----------------------------------------------------------------------


def joint_token_distribution(
    tokens: List[Dict[str, Any]], pheromones: List[Pheromone]
) -> Tuple[np.ndarray, List[str]]:
    """
    Produce a posterior distribution over the *unique* tokens observed in the
    current context by treating the log‑count frequencies as a Dirichlet prior
    and the pheromone distribution as a likelihood over the same index space.

    The returned tuple contains:
        - probs: a normalized probability vector aligned with `vocab`
        - vocab: list of token strings in the same order as `probs`
    """
    # 1️⃣ Token frequencies → prior (add‑one smoothing)
    token_counts = log_count_statistics(tokens)
    vocab = list(token_counts.keys())
    prior_counts = np.array([token_counts[t] for t in vocab], dtype=float) + 1.0

    # 2️⃣ Map pheromones onto the same vocab space.
    #    For simplicity we treat each pheromone as an independent “expert”
    #    that votes uniformly for all tokens, weighted by its signal.
    pheromone_weights = pheromone_distribution(pheromones)
    # Broadcast each pheromone weight across the vocab and sum.
    likelihood = np.sum(pheromone_weights[:, None], axis=0)  # shape (len(vocab),)

    # 3️⃣ Posterior ∝ prior * likelihood  (Bayes rule for categorical models)
    unnorm = prior_counts * likelihood
    total = unnorm.sum()
    if total == 0:
        # If everything collapsed, revert to a uniform prior.
        probs = np.full_like(unnorm, 1.0 / len(unnorm))
    else:
        probs = unnorm / total
    return probs, vocab


def expected_entropy_after_action(
    current_probs: np.ndarray, action_idx: int
) -> float:
    """
    Compute the expected entropy if we were to *observe* the chosen token
    (i.e. set its probability to 1 and the rest to 0). This is a deterministic
    reduction, so the expected entropy equals the entropy of the resulting
    delta distribution – which is zero.  However, in a bandit setting we often
    model a *noisy* observation; we therefore approximate the expected
    entropy by assuming the true token is drawn from the current distribution.
    """
    # Probability that the chosen token is the true one
    p_true = current_probs[action_idx]
    # If the token is correct, entropy becomes 0.
    # If it is incorrect, we remove its mass and renormalise.
    remaining = np.delete(current_probs, action_idx)
    remaining = remaining / (1.0 - p_true + 1e-12)
    entropy_if_wrong = entropy(remaining)
    return p_true * 0.0 + (1.0 - p_true) * entropy_if_wrong


def infotaxis_selection(tokens: List[Dict[str, Any]], pheromones: List[Pheromone]) -> str:
    """
    Choose the token that maximises expected information gain (i.e. minimises
    the expected entropy after the observation). This implements a deeper
    integration of the deterministic log‑count statistics with the stochastic
    pheromone signals.
    """
    probs, vocab = joint_token_distribution(tokens, pheromones)

    # Compute expected entropy for each candidate token
    expected_entropies = np.array(
        [expected_entropy_after_action(probs, i) for i in range(len(vocab))]
    )

    # Action with minimal expected entropy = maximal information gain
    best_idx = int(np.argmin(expected_entropies))
    return vocab[best_idx]


# ----------------------------------------------------------------------
# Demonstration / entry point
# ----------------------------------------------------------------------


def main() -> None:
    sample_text = (
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit, "
        "sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."
    )
    tokens = tokenize(sample_text)

    # In a real deployment `fetch_pheromones` would hit a PostgreSQL store.
    # Here we use a deterministic mock for reproducibility.
    pheromones = fetch_pheromones("example_surface", limit=7)

    best_token = infotaxis_selection(tokens, pheromones)
    print(f"Selected token: {best_token!r}")


if __name__ == "__main__":
    main()