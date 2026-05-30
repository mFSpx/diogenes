# DARWIN HAMMER — match 4206, survivor 4
# gen: 6
# parent_a: hybrid_hybrid_regret_engine_hybrid_hybrid_decisi_m82_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hdc_hy_hybrid_hybrid_fisher_m1920_s1.py (gen5)
# born: 2026-05-29T23:54:16Z

import re
import math
import random
import sys
import pathlib
import hashlib
from dataclasses import dataclass
from typing import List, Dict

import numpy as np

# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathAction:
    """An action with an expected value and optional cost/risk penalties."""
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True)
class MathCounterfactual:
    """A counterfactual adjustment for a specific action."""
    action_id: str
    outcome_value: float
    probability: float = 1.0


# ----------------------------------------------------------------------
# Regret‑weighted strategy
# ----------------------------------------------------------------------
def compute_regret_weighted_strategy(
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual],
) -> Dict[str, float]:
    """
    Returns a probability distribution over actions based on regret weighting.

    Regret for an action = (max expected value among all actions) - expected_value.
    The weight is exp(-regret) (higher expected value → lower regret → higher weight).
    Counterfactuals shift the expected value of the referenced action before regret
    is computed.
    """
    # Apply counterfactual adjustments
    adj_values: Dict[str, float] = {a.id: a.expected_value for a in actions}
    for cf in counterfactuals:
        if cf.action_id in adj_values:
            adj_values[cf.action_id] += cf.outcome_value * cf.probability

    # Compute regret and raw weights
    max_ev = max(adj_values.values())
    raw_weights = {}
    for aid, ev in adj_values.items():
        regret = max_ev - ev
        raw_weights[aid] = math.exp(-regret)

    # Normalise to a probability distribution
    total = sum(raw_weights.values())
    if total == 0:
        # fallback to uniform distribution
        return {aid: 1.0 / len(actions) for aid in raw_weights}
    return {aid: w / total for aid, w in raw_weights.items()}


# ----------------------------------------------------------------------
# High‑dimensional vector utilities
# ----------------------------------------------------------------------
Vector = List[int]


def random_vector(dim: int = 10000, seed: str | int | None = None) -> Vector:
    """Generates a random binary (+1 / -1) vector."""
    if isinstance(seed, str):
        seed_bytes = hashlib.sha256(seed.encode("utf-8")).digest()[:8]
        seed = int.from_bytes(seed_bytes, "big")
    rng = random.Random(seed)
    return [1 if rng.getrandbits(1) else -1 for _ in range(dim)]


def symbol_vector(symbol: str, dim: int = 10000) -> Vector:
    """Deterministically maps a symbol string to a binary vector."""
    return random_vector(dim, symbol)


def bind(a: Vector, b: Vector) -> Vector:
    """Element‑wise binding (Hadamard product) of two vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have equal length")
    return [x * y for x, y in zip(a, b)]


def bundle(vectors: List[Vector]) -> Vector:
    """Bundles a list of vectors by majority vote (sign of sum)."""
    if not vectors:
        raise ValueError("at least one vector is required")
    dim = len(vectors[0])
    if any(len(v) != dim for v in vectors):
        raise ValueError("all vectors must have equal length")
    sums = np.zeros(dim, dtype=int)
    for v in vectors:
        sums += np.array(v, dtype=int)
    return [1 if s >= 0 else -1 for s in sums]


def similarity(a: Vector, b: Vector) -> float:
    """Mean‑scaled dot product similarity (range [-1, 1])."""
    if len(a) != len(b):
        raise ValueError("vectors must have equal length")
    if not a:
        raise ValueError("vectors must not be empty")
    return sum(x * y for x, y in zip(a, b)) / len(a)


def fisher_score(theta: float, center: float = 0.0, width: float = 1.0, eps: float = 1e-12) -> float:
    """
    Logistic‑like Fisher transformation that maps any real theta to (0,1).

    The classic Fisher score: 1 / (1 + exp(-(theta-center)/width))
    """
    z = (theta - center) / max(width, eps)
    return 1.0 / (1.0 + math.exp(-z))


# ----------------------------------------------------------------------
# Decision‑hygiene cue handling
# ----------------------------------------------------------------------
EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)
PLANNING_RE = re.compile(
    r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
    re.I,
)


def extract_hygiene_symbols(text: str) -> List[str]:
    """
    Returns a list of matched cue symbols (lower‑cased) from the supplied text.
    Both evidence‑type and planning‑type cues are considered.
    """
    symbols = set()
    for match in EVIDENCE_RE.finditer(text):
        symbols.add(match.group(0).lower())
    for match in PLANNING_RE.finditer(text):
        symbols.add(match.group(0).lower())
    return list(symbols)


def build_hygiene_vector(text: str, dim: int = 10000) -> Vector:
    """
    Constructs a single hygiene vector by bundling the symbol vectors of all
    detected hygiene cues in the input text.
    """
    symbols = extract_hygiene_symbols(text)
    if not symbols:
        # No cues → use a neutral (all‑ones) vector
        return [1] * dim
    vectors = [symbol_vector(sym, dim) for sym in symbols]
    return bundle(vectors)


# ----------------------------------------------------------------------
# Hybrid scoring
# ----------------------------------------------------------------------
def hybrid_action_scores(
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual],
    hygiene_text: str,
    dim: int = 10000,
) -> Dict[str, float]:
    """
    Computes a hybrid score for each action:

        score(action) = regret_weight(action) * fisher_score( similarity(action_vec, hygiene_vec) )

    The regret weight is obtained from `compute_regret_weighted_strategy`.
    The similarity is the mean‑scaled dot product between the action's symbol vector
    and the bundled hygiene vector.
    The Fisher score maps similarity (≈[-1,1]) to a factor in (0,1).
    """
    # 1. Regret‑weighted probabilities
    regret_weights = compute_regret_weighted_strategy(actions, counterfactuals)

    # 2. Hygiene vector from the provided text
    h_vec = build_hygiene_vector(hygiene_text, dim)

    # 3. Action vectors and scores
    action_scores = {}
    for action in actions:
        a_vec = symbol_vector(action.id, dim)
        sim = similarity(a_vec, h_vec)
        hygiene_factor = fisher_score(sim)
        action_scores[action.id] = regret_weights[action.id] * hygiene_factor

    return action_scores


def select_best_action(
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual],
    hygiene_text: str,
    dim: int = 10000,
) -> MathAction:
    """
    Selects the action with the highest hybrid score.
    """
    action_scores = hybrid_action_scores(actions, counterfactuals, hygiene_text, dim)
    best_action_id = max(action_scores, key=action_scores.get)
    return next(a for a in actions if a.id == best_action_id)