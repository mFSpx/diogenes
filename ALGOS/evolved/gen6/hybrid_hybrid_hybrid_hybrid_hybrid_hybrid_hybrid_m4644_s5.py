# DARWIN HAMMER — match 4644, survivor 5
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m607_s3.py (gen5)
# parent_b: hybrid_hybrid_hybrid_pherom_hybrid_ternary_route_m1532_s2.py (gen3)
# born: 2026-05-29T23:57:10Z

"""Hybrid Stylometry‑NLMS & Pheromone‑Entropy Ternary Routing Engine
===================================================================

Parent Algorithms
-----------------
* **Algorithm A** – *hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m607_s3.py*  
  Provides stylometric feature extraction and uses a “linguistic complexity” score (LC) to
  scale the Normalised Least‑Mean‑Squares (NLMS) weight update Δw for each endpoint.

* **Algorithm B** – *hybrid_hybrid_hybrid_pherom_hybrid_ternary_route_m1532_s2.py*  
  Tracks pheromone levels on a set of actions, computes Shannon entropy of the pheromone
  probability distribution and feeds that entropy into a ternary routing decision that also
  respects a minimum‑cost matrix.

Mathematical Bridge
-------------------
Both parents expose a **probabilistic weighting** that can be multiplied with a **scalar
modulation**:

* In A the scalar is the linguistic‑complexity factor `LC ∈ (0,∞)`.
* In B the probabilistic weighting is the pheromone‑derived probability vector `π`.

The fused algorithm therefore:

1. **Extracts stylometric features** → computes `LC`.
2. **Computes pheromone probabilities** `π = pheromones / Σ pheromones`.
3. **Evaluates Shannon entropy** `H = - Σ π_i log π_i`.
4. **Scales the NLMS learning step** by `LC·(1+β·H)·π_i` – the LC modulates the base
   NLMS step while the entropy‑adjusted pheromone probability biases the update toward
   more “informative” actions.
5. **Performs a ternary routing decision** using a cost matrix `C`.  The score for each
   of the three routes `r` is  

   `S_r = C_r - α·H·π_r`  

   The route with minimal `S_r` is selected, thus routing prefers low cost but also
   favours routes that carry higher entropy (i.e. higher exploration value).

The resulting system simultaneously adapts endpoint weights (NLMS) and routing choices
based on linguistic complexity, pheromone dynamics and information‑theoretic entropy.

"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Stylometric feature extraction (from Algorithm A)
# ----------------------------------------------------------------------
PUNCT = "!?;:,.—-()[]{}\"'`/\\|@#$%^&*+=~"
FUNCTION_CATS = {
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
    "conjunction": set(
        "and but or nor so yet because although while if when where whereas unless until".split()
    ),
    "negation": set(
        "no not never none neither cannot can't won't don't didn't isn't aren't was wasn't weren't".split()
    ),
    "quantifier": set(
        "all any both each few many more most much none several some such".split()
    ),
    "adverb_common": set(
        "very really just still already also even only then there here now often always sometimes".split()
    ),
}


def _tokenize(text: str) -> List[str]:
    """Very light tokenisation – split on whitespace and strip punctuation."""
    tokens = []
    for raw in (text or "").lower().split():
        token = raw.strip(PUNCT)
        if token:
            tokens.append(token)
    return tokens


def extract_stylometry(text: str) -> Dict[str, float]:
    """
    Compute a small set of stylometric statistics.
    Returns a dictionary with numeric features.
    """
    words = _tokenize(text)
    char_count = len(text)
    word_count = len(words)
    punct_count = sum(ch in PUNCT for ch in text)

    # Function‑category frequencies
    cat_counts = {cat: 0 for cat in FUNCTION_CATS}
    for w in words:
        for cat, vocab in FUNCTION_CATS.items():
            if w in vocab:
                cat_counts[cat] += 1

    # Normalise by word count to obtain densities
    features = {
        "word_count": word_count,
        "char_per_word": char_count / max(word_count, 1),
        "punct_density": punct_count / max(word_count, 1),
    }
    for cat, cnt in cat_counts.items():
        features[f"{cat}_freq"] = cnt / max(word_count, 1)

    return features


def linguistic_complexity(features: Dict[str, float]) -> float:
    """
    Combine stylometric features into a single scalar LC.
    Higher values indicate more complex / information‑rich text.
    The formula is a weighted sum; weights are chosen heuristically.
    """
    lc = 0.0
    lc += 0.4 * features["char_per_word"]
    lc += 1.2 * features["punct_density"]
    lc += 0.8 * features["pronoun_freq"]
    lc += 0.6 * features["article_freq"]
    lc += 0.5 * features["preposition_freq"]
    lc += 0.3 * features["auxiliary_freq"]
    lc += 0.2 * features["conjunction_freq"]
    lc += 0.7 * features["negation_freq"]
    lc += 0.5 * features["quantifier_freq"]
    lc += 0.4 * features["adverb_common_freq"]
    # Keep LC in a reasonable range
    return max(lc, 0.01)


# ----------------------------------------------------------------------
# Pheromone handling and entropy (from Algorithm B)
# ----------------------------------------------------------------------
def pheromone_probabilities(pheromones: np.ndarray) -> np.ndarray:
    """Normalise pheromone levels to a probability distribution."""
    total = np.sum(pheromones)
    if total == 0:
        # Avoid division by zero – uniform distribution
        return np.full_like(pheromones, 1.0 / pheromones.size, dtype=float)
    return pheromones / total


def shannon_entropy(probs: np.ndarray) -> float:
    """Classic Shannon entropy H = - Σ p log p (natural logarithm)."""
    # Guard against zero entries
    mask = probs > 0
    return -np.sum(probs[mask] * np.log(probs[mask]))


def update_pheromones(
    pheromones: np.ndarray,
    chosen_idx: int,
    reward: float,
    evaporation: float = 0.1,
) -> np.ndarray:
    """
    Simple pheromone update:
        π_i ← (1‑evap)·π_i + reward·δ_{i,chosen}
    """
    pheromones = pheromones * (1.0 - evaporation)
    pheromones[chosen_idx] += reward
    return pheromones


# ----------------------------------------------------------------------
# NLMS weight update (from Algorithm A) – now entropy‑aware
# ----------------------------------------------------------------------
def nlms_update(
    weights: np.ndarray,
    inputs: np.ndarray,
    desired: float,
    lc: float,
    prob: float,
    entropy: float,
    mu: float = 0.05,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """
    Perform a single NLMS weight update.
    The step size is scaled by:
        η = mu * lc * (1 + β·entropy) * prob
    where β is a modest coupling factor.
    """
    beta = 0.3  # coupling between entropy and learning rate
    eta = mu * lc * (1.0 + beta * entropy) * prob
    y = np.dot(weights, inputs)
    e = desired - y
    norm = np.dot(inputs, inputs) + epsilon
    delta_w = eta * e * inputs / norm
    return weights + delta_w


# ----------------------------------------------------------------------
# Ternary routing with minimum‑cost + entropy bias
# ----------------------------------------------------------------------
def ternary_route(
    cost_vector: np.ndarray,
    probs: np.ndarray,
    entropy: float,
    alpha: float = 0.5,
) -> int:
    """
    Choose one of three routes (indices 0,1,2) by minimising:

        score_i = cost_i - α·entropy·π_i

    Lower score favours low cost, but a high entropy boosts the
    attractiveness of routes with larger pheromone probability.
    """
    if cost_vector.size != 3 or probs.size != 3:
        raise ValueError("Both cost_vector and probs must have length 3.")
    scores = cost_vector - alpha * entropy * probs
    return int(np.argmin(scores))


# ----------------------------------------------------------------------
# High‑level hybrid step combining all pieces
# ----------------------------------------------------------------------
def hybrid_step(
    text: str,
    inputs: np.ndarray,
    desired: float,
    weights: np.ndarray,
    pheromones: np.ndarray,
    cost_matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Executes one iteration of the fused algorithm:

    1. Extract stylometry → LC.
    2. Compute pheromone probabilities → π and entropy H.
    3. NLMS weight update using LC, π_i and H.
    4. Ternary routing using cost matrix, π and H.
    5. Update pheromones based on routing outcome (reward = 1.0).

    Returns updated (weights, pheromones, chosen_route).
    """
    # 1. Stylometry → LC
    feats = extract_stylometry(text)
    lc = linguistic_complexity(feats)

    # 2. Pheromone distribution & entropy
    probs = pheromone_probabilities(pheromones)
    entropy = shannon_entropy(probs)

    # 3. NLMS update – we apply the update for each endpoint separately
    #    (here we treat each weight vector entry as an independent endpoint)
    updated_weights = np.empty_like(weights)
    for i in range(weights.size):
        updated_weights[i] = nlms_update(
            weights[i],
            inputs,
            desired,
            lc,
            prob=probs[i],
            entropy=entropy,
        )

    # 4. Ternary routing decision
    chosen_route = ternary_route(cost_matrix, probs, entropy)

    # 5. Pheromone reinforcement (simple reward = 1.0 for the selected route)
    pheromones = update_pheromones(pheromones, chosen_route, reward=1.0)

    return updated_weights, pheromones, chosen_route


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Sample text – deliberately mixed complexity
    sample_text = (
        "The quick brown fox jumps over the lazy dog, while the observer "
        "records every nuance of movement; however, no one can predict the "
        "next twist."
    )

    # NLMS setup: 3 endpoints (matching the ternary router size)
    np.random.seed(42)
    init_weights = np.random.randn(3, 5)  # 3 endpoints, 5‑dimensional input space
    input_vector = np.random.randn(5)
    desired_output = 0.0  # target for the NLMS error

    # Pheromone levels for the three routes
    pheromones = np.array([0.5, 0.3, 0.2])

    # Minimum‑cost matrix (one cost per route)
    costs = np.array([2.0, 1.5, 3.0])

    # Run a single hybrid iteration
    new_weights, new_pheromones, route = hybrid_step(
        sample_text,
        input_vector,
        desired_output,
        init_weights,
        pheromones,
        costs,
    )

    # Simple sanity prints (not required by the spec but useful for manual check)
    print("Chosen route :", route)
    print("Updated pheromones :", new_pheromones)
    print("Weight change norm :", np.linalg.norm(new_weights - init_weights))
    sys.exit(0)