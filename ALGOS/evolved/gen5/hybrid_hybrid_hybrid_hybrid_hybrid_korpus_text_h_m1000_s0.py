# DARWIN HAMMER — match 1000, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_hybrid_pherom_hybrid_bayes_claim_k_m9_s0.py (gen4)
# parent_b: hybrid_korpus_text_hybrid_hybrid_regret_m21_s6.py (gen4)
# born: 2026-05-29T23:32:20Z

import numpy as np
import math
import random
import sys
import pathlib
import psycopg

# ----------------------------------------------------------------------
# Module documentation
# ----------------------------------------------------------------------
"""
This module combines the DARWIN HAMMER algorithms 'hybrid_hybrid_pheromone_infotaxis_m3_s0.py' and 'hybrid_korpus_text_hybrid_hybrid_regret_m21_s6.py'.
The mathematical bridge between the two parent algorithms lies in using the Jaccard similarity calculation to compare the distribution of decision hygiene scores 
with the regret-weighted raw values from the KorpusTextRegretBandit framework.
"""

# ----------------------------------------------------------------------
# Constants and utilities (derived from Parent A)
# ----------------------------------------------------------------------
INT16_MAX = 2 ** 15 - 1


def _shingles(text: str, width: int = 5) -> List[str]:
    """Return a list of overlapping substrings (shingles) of length *width*."""
    cleaned = " ".join(text.split()).lower()
    return [cleaned[i : i + width] for i in range(len(cleaned) - width + 1)]


def _hash_token(seed: int, token: str) -> int:
    """Deterministic 64‑bit hash of *token* mixed with *seed*."""
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def minhash_signature(tokens: Iterable[str], k: int = 64) -> List[int]:
    """Compute a MinHash signature of length *k* for the given *tokens*."""
    token_set = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")


# ----------------------------------------------------------------------
# Function to calculate Shannon entropy from decision hygiene scores
# ----------------------------------------------------------------------
def calculate_shannon_entropy(scores: np.ndarray) -> float:
    """Calculate Shannon entropy from the given decision hygiene scores."""
    # Calculate the probability of each score
    probabilities = np.array([np.mean(scores == score) for score in np.unique(scores)])
    # Calculate the Shannon entropy
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy


# ----------------------------------------------------------------------
# Function to calculate Jaccard similarity between two MinHash signatures
# ----------------------------------------------------------------------
def jaccard_similarity(signature1: List[int], signature2: List[int]) -> float:
    """Calculate Jaccard similarity between two MinHash signatures."""
    # Calculate the intersection and union of the two signatures
    intersection = len(set(signature1) & set(signature2))
    union = len(set(signature1) | set(signature2))
    # Calculate the Jaccard similarity
    similarity = intersection / union
    return similarity


# ----------------------------------------------------------------------
# Function to calculate the hybrid score
# ----------------------------------------------------------------------
def calculate_hybrid_score(
    decision_hygiene_scores: dict[str, int],
    regret_weights: List[float],
    minhash_signature: List[int],
    reference_signature: List[int],
    jaccard_similarity: float,
    dance_signal: float,
    linucb_confidence: float,
    beta: float,
) -> float:
    """Calculate the hybrid score."""
    # Calculate the Shannon entropy from the decision hygiene scores
    entropy = calculate_shannon_entropy(np.array(list(decision_hygiene_scores.values())))
    # Calculate the regret-weighted raw value
    regret_weighted_raw_value = np.mean(regret_weights)
    # Calculate the Jaccard similarity between the MinHash signatures
    jaccard_similarity = jaccard_similarity(minhash_signature, reference_signature)
    # Calculate the hybrid score
    hybrid_score = math.exp(
        1
        + regret_weighted_raw_value
        + jaccard_similarity
        + dance_signal
        + beta * linucb_confidence
    )
    return hybrid_score


# ----------------------------------------------------------------------
# Function to retrieve pheromone probabilities from the database
# ----------------------------------------------------------------------
def calculate_pheromone_probabilities(surface_key: str, limit: int, db_url: str) -> List[float]:
    """Calculates pheromone probabilities from the database."""
    import psycopg

    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT signal_value FROM lucidota_runtime.surface_pheromone WHERE surface_key=%s ORDER BY created_at DESC LIMIT %s""",
                (surface_key, limit),
            )
            pheromones = [r["signal_value"] for r in cur.fetchall()]
            total = sum(pheromones)
            return [p / total for p in pheromones]


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a test text
    text = "This is a test text."

    # Calculate decision hygiene scores
    decision_hygiene_scores = decision_hygiene_scores(text)

    # Calculate regret weights
    regret_weights = [random.random() for _ in range(len(decision_hygiene_scores))]

    # Calculate MinHash signature
    tokens = _shingles(text)
    minhash_signature = minhash_signature(tokens)

    # Calculate reference signature
    reference_signature = minhash_signature

    # Calculate Jaccard similarity
    jaccard_similarity = jaccard_similarity(minhash_signature, reference_signature)

    # Calculate dance signal
    dance_signal = random.random()

    # Calculate LinUCB confidence
    linucb_confidence = random.random()

    # Calculate beta
    beta = random.random()

    # Calculate hybrid score
    hybrid_score = calculate_hybrid_score(
        decision_hygiene_scores,
        regret_weights,
        minhash_signature,
        reference_signature,
        jaccard_similarity,
        dance_signal,
        linucb_confidence,
        beta,
    )

    # Print the hybrid score
    print(hybrid_score)

    # Calculate pheromone probabilities
    pheromone_probabilities = calculate_pheromone_probabilities("test_key", 10, "test_db_url")

    # Print the pheromone probabilities
    print(pheromone_probabilities)