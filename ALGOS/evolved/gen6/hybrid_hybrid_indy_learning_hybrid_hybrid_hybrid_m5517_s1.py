# DARWIN HAMMER — match 5517, survivor 1
# gen: 6
# parent_a: hybrid_indy_learning_vector_hybrid_fold_change_d_m38_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1635_s0.py (gen5)
# born: 2026-05-30T00:02:30Z

"""Hybrid Indy‑Learning Vector + Voronoi‑RBF Bandit
Parents:
- hybrid_indy_learning_vector_hybrid_fold_change_d_m38_s1.py
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1635_s0.py

Mathematical bridge
-------------------
Both parents expose a *bandit* decision core that relies on statistics
(total reward, count) per action.  The second parent augments the bandit
with a **Voronoi partition** in ℝⁿ, where each action owns a seed point and
contexts are assigned to the nearest seed using the Euclidean distance
`‖x‑cᵢ‖`.  The first parent creates high‑dimensional context vectors from
raw text by tokenisation, chunking and a deterministic hash‑based
bag‑of‑words embedding.  By treating those embeddings as points in the same
Euclidean space, the Voronoi geometry can be applied directly to the
outputs of the Indy‑learning pipeline.

The hybrid algorithm therefore:
1. Converts a text chunk into a fixed‑size vector `x ∈ ℝᴰ`.
2. Assigns `x` to the nearest Voronoi seed `cᵢ` → candidate action.
3. Computes a **radial‑basis‑function (RBF) surrogate** estimate of the
   expected reward from all previously observed contexts.
4. Forms a classic Upper‑Confidence‑Bound (UCB) score
   `Uᵢ = μ̂ᵢ + β·σᵢ` where `μ̂ᵢ` is the RBF estimate, `σᵢ` is the bandit
   confidence term, and `β` is a tunable exploration weight.
5. Updates the bandit statistics with a **fold‑change detection** term
   `Δ = log( (r_new+ε)/(r_avg+ε) )` that scales the confidence bound,
   allowing rapid adaptation when rewards shift sharply.

The code below implements this fused system in pure Python, using only
numpy and the standard library."""


import hashlib
import json
import math
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np

# ----------------------------------------------------------------------
# Global stores (shared by all functions)
# ----------------------------------------------------------------------
_POLICY: Dict[str, List[float]] = {}          # action_id → [total_reward, count]
_SEEDS: Dict[str, np.ndarray] = {}           # action_id → seed vector (Voronoi)
_KNOWN_CONTEXTS: List[np.ndarray] = []       # list of past context vectors
_KNOWN_REWARDS: List[float] = []             # parallel list of observed rewards
_DEFAULT_DIM = 128                           # dimensionality of text embeddings
_EPS = 1e-12                                 # numerical stability


# ----------------------------------------------------------------------
# Text processing (from Parent A)
# ----------------------------------------------------------------------
_WORD_RE = re.compile(r"\S+")


def tokenize(text: str) -> List[Dict[str, Any]]:
    """Return a list of token dicts with token string and character offsets."""
    return [{"token": m.group(0), "start": m.start(), "end": m.end()}
            for m in _WORD_RE.finditer(text)]


def chunk_text_tokens(text: str,
                      *,
                      max_tokens: int = 100,
                      overlap_tokens: int = 0) -> List[Dict[str, Any]]:
    """
    Split *text* into overlapping token chunks.
    Each chunk dict contains the raw string and its token list.
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if not (0 <= overlap_tokens < max_tokens):
        raise ValueError("overlap_tokens must be >=0 and < max_tokens")
    tokens = tokenize(text)
    chunks = []
    i = 0
    while i < len(tokens):
        chunk_toks = tokens[i:i + max_tokens]
        start = chunk_toks[0]["start"]
        end = chunk_toks[-1]["end"]
        chunk_str = text[start:end]
        chunks.append({
            "text": chunk_str,
            "tokens": [t["token"] for t in chunk_toks],
            "start": start,
            "end": end
        })
        i += max_tokens - overlap_tokens
    return chunks


def text_to_vector(chunk: Dict[str, Any],
                  dim: int = _DEFAULT_DIM) -> np.ndarray:
    """
    Deterministic hash‑based bag‑of‑words embedding.
    For each token we compute a 64‑bit hash, map it to an index in [0, dim)
    and increment that position.  The resulting count vector is L2‑normalised.
    """
    vec = np.zeros(dim, dtype=np.float64)
    for token in chunk["tokens"]:
        h = int(hashlib.sha256(token.encode()).hexdigest(), 16)
        idx = h % dim
        vec[idx] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


# ----------------------------------------------------------------------
# Voronoi & RBF surrogate (from Parent B)
# ----------------------------------------------------------------------
def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Standard Euclidean distance."""
    return float(np.linalg.norm(a - b))


def assign_voronoi(context_vec: np.ndarray) -> str:
    """
    Return the action_id whose seed is closest (Voronoi assignment).
    If no seeds exist yet, create a random one for a default action.
    """
    if not _SEEDS:
        # Initialise a single default action
        default_id = "action_0"
        _SEEDS[default_id] = np.random.randn(_DEFAULT_DIM)
        return default_id

    best_id = None
    best_dist = math.inf
    for aid, seed in _SEEDS.items():
        d = euclidean_distance(context_vec, seed)
        if d < best_dist:
            best_dist = d
            best_id = aid
    return best_id


def rbf_estimate(context_vec: np.ndarray,
                 sigma: float = 1.0) -> float:
    """
    Radial‑basis‑function surrogate estimate of expected reward.
    Uses a Gaussian kernel over all known contexts.
    If no data are available, returns 0.0.
    """
    if not _KNOWN_CONTEXTS:
        return 0.0
    diffs = np.stack(_KNOWN_CONTEXTS) - context_vec   # shape (N, D)
    d2 = np.sum(diffs ** 2, axis=1)                   # squared Euclidean distances
    kernels = np.exp(-d2 / (2 * sigma ** 2))
    weighted = np.dot(kernels, _KNOWN_REWARDS)
    norm = np.sum(kernels) + _EPS
    return float(weighted / norm)


# ----------------------------------------------------------------------
# Bandit core with fold‑change detection (fusion of both parents)
# ----------------------------------------------------------------------
def _policy_stats(action_id: str) -> Tuple[float, float]:
    """Return (total_reward, count) for *action_id*."""
    total, cnt = _POLICY.get(action_id, [0.0, 0.0])
    return total, cnt


def _average_reward(action_id: str) -> float:
    total, cnt = _policy_stats(action_id)
    return total / cnt if cnt > 0 else 0.0


def _confidence_bound(action_id: str,
                      total_counts: float,
                      beta: float = 1.0) -> float:
    """
    Classic UCB confidence term: sqrt( (2·log(total_counts)) / n )
    multiplied by an exploration weight *beta*.
    """
    _, cnt = _policy_stats(action_id)
    if cnt == 0:
        return float('inf')
    return beta * math.sqrt(2 * math.log(max(total_counts, 1.0)) / cnt)


def select_action(context_vec: np.ndarray,
                  beta: float = 1.0,
                  sigma: float = 1.0) -> str:
    """
    Hybrid selection:
    1. Voronoi assignment gives a candidate action.
    2. RBF surrogate provides a reward estimate μ̂.
    3. UCB score = μ̂ + confidence_bound.
    4. The action with the highest score is returned.
    """
    # Ensure at least one seed exists
    candidate = assign_voronoi(context_vec)

    # If the candidate is unknown, initialise its statistics and a seed
    if candidate not in _POLICY:
        _POLICY[candidate] = [0.0, 0.0]
        _SEEDS[candidate] = np.random.randn(_DEFAULT_DIM)

    # Compute scores for *all* actions (small number, so O(|A|) is fine)
    total_counts = sum(cnt for _, cnt in (_POLICY[a] for a in _POLICY))
    best_action = None
    best_score = -math.inf

    for aid in _POLICY.keys():
        # RBF estimate uses the global context pool, not per‑action
        mu = rbf_estimate(context_vec, sigma=sigma)
        conf = _confidence_bound(aid, total_counts, beta=beta)
        score = mu + conf
        if score > best_score:
            best_score = score
            best_action = aid

    # Fallback: if something went wrong, use the Voronoi candidate
    return best_action or candidate


def update_policy(action_id: str,
                  reward: float,
                  context_vec: np.ndarray,
                  alpha: float = 0.5) -> None:
    """
    Update bandit statistics with *reward* and record the context.
    A fold‑change term Δ = log( (r_new+ε)/(r_avg+ε) ) scales the confidence
    bound for future selections (implemented via a simple multiplicative
    factor *alpha* on the stored total reward).
    """
    total, cnt = _policy_stats(action_id)

    # Fold‑change detection
    prev_avg = total / cnt if cnt > 0 else 0.0
    new_avg = (total + reward) / (cnt + 1)
    delta = math.log((new_avg + _EPS) / (prev_avg + _EPS))

    # Apply a modest adaptation: boost total reward proportionally to Δ
    adapted_total = total + alpha * delta * (reward - prev_avg)

    _POLICY[action_id] = [adapted_total + reward, cnt + 1]

    # Record context for the surrogate model
    _KNOWN_CONTEXTS.append(context_vec.copy())
    _KNOWN_REWARDS.append(reward)


# ----------------------------------------------------------------------
# Helper to initialise a set of actions with random seeds
# ----------------------------------------------------------------------
def initialise_actions(action_ids: List[str]) -> None:
    """
    Populate _POLICY and _SEEDS for the supplied *action_ids*.
    Each seed is a random unit vector in ℝᴰ.
    """
    for aid in action_ids:
        if aid not in _POLICY:
            _POLICY[aid] = [0.0, 0.0]
        if aid not in _SEEDS:
            vec = np.random.randn(_DEFAULT_DIM)
            _SEEDS[aid] = vec / (np.linalg.norm(vec) + _EPS)


# ----------------------------------------------------------------------
# End‑to‑end demonstration function
# ----------------------------------------------------------------------
def process_and_act(text: str,
                    max_tokens: int = 100,
                    overlap: int = 0,
                    beta: float = 1.0,
                    sigma: float = 1.0) -> Tuple[str, float]:
    """
    1. Chunk the input *text*.
    2. Convert the first chunk to a vector.
    3. Select an action via the hybrid bandit.
    4. Simulate a stochastic reward (here: random in [0,1]).
    5. Update the policy.
    Returns the chosen action_id and the simulated reward.
    """
    chunks = chunk_text_tokens(text, max_tokens=max_tokens, overlap_tokens=overlap)
    if not chunks:
        raise ValueError("No tokens to process")
    vec = text_to_vector(chunks[0])
    action = select_action(vec, beta=beta, sigma=sigma)
    reward = random.random()  # placeholder for a real environment signal
    update_policy(action, reward, vec)
    return action, reward


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Initialise a small action set
    initialise_actions([f"action_{i}" for i in range(5)])

    sample_text = (
        "The quick brown fox jumps over the lazy dog. "
        "Artificial intelligence blends statistical learning with "
        "geometric reasoning to create adaptable agents."
    )

    for step in range(10):
        act, rew = process_and_act(sample_text, max_tokens=20, overlap=5)
        print(f"Step {step+1:02d}: selected {act!r}, reward={rew:.3f}")

    # Show final policy statistics
    print("\nFinal policy statistics:")
    for aid, (tot, cnt) in _POLICY.items():
        avg = tot / cnt if cnt else 0.0
        print(f"  {aid}: count={int(cnt)}, avg_reward={avg:.3f}")