# DARWIN HAMMER — match 3, survivor 5
# gen: 3
# parent_a: hybrid_regret_engine_hybrid_liquid_time_c_m13_s0.py (gen2)
# parent_b: hybrid_ternary_lens_router_hybrid_decision_hygi_m44_s2.py (gen2)
# born: 2026-05-29T23:25:20Z

"""Hybrid Regret‑Weighted Ternary‑Decision Analyzer (RW‑TD‑H).

This module fuses:

* **Parent A** – Regret‑Weighted Liquid‑Time‑Constant MinHash (RW‑LTC‑MH):
  - Generates MinHash signatures from token sets.
  - Computes a regret‑weighted probability distribution over actions.

* **Parent B** – Hybrid Ternary‑Decision Hygiene Analyzer (TD‑HA):
  - Produces deterministic ternary vectors from payload descriptors.
  - Computes Shannon‑entropy over a concatenated vector of ternary symbols
    and decision‑hygiene scores.

**Mathematical bridge**

Both parents ultimately yield *discrete probability‑mass samples*:

* RW‑LTC‑MH provides a probability vector `p` over actions (soft‑max of
  regret‑adjusted utilities).
* TD‑HA provides a ternary vector `t ∈ {‑1,0,+1}^D` that can be mapped to a
  categorical distribution by treating each symbol as a sample from a
  three‑symbol alphabet.

The hybrid algorithm maps the regret‑weighted probabilities `p` onto the
ternary alphabet by sign‑quantisation, concatenates the resulting symbolic
sequence with `t`, and evaluates the Shannon entropy of the combined
empirical distribution.  The entropy therefore reflects both *strategic
regret* and *payload‑level hygiene* in a single information‑theoretic
measure.  Additionally, the MinHash similarity between two token sets is
modulated by the regret weights, yielding a similarity score that is
aware of the underlying decision confidence.

The implementation below provides three core functions that realise this
fusion and a small smoke‑test when executed as a script.
"""

import hashlib
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Shared data structures (from Parent A)
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# Parent A utilities (MinHash & regret weighting)
# ----------------------------------------------------------------------
def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def signature(tokens: Iterable[str], k: int = 128) -> List[int]:
    """MinHash signature of a token collection."""
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [(1 << 64) - 1] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]


def similarity(sig_a: List[int], sig_b: List[int]) -> float:
    """Jaccard‑like similarity based on MinHash equality."""
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have equal length")
    if not sig_a:
        raise ValueError("signatures must not be empty")
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(x - np.max(x))
    return e / e.sum()


def compute_regret_weights(
    actions: List[MathAction], counterfactuals: List[MathCounterfactual]
) -> Tuple[np.ndarray, List[str]]:
    """
    Returns a probability vector `p` (regret‑weighted) and the aligned list of
    action identifiers.
    """
    if not actions:
        return np.array([]), []
    # Map counterfactual outcomes to a dict for quick lookup
    cf_dict = {c.action_id: c.outcome_value * c.probability for c in counterfactuals}
    # Regret‑adjusted utility for each action
    utilities = []
    ids = []
    for a in actions:
        cf_val = cf_dict.get(a.id, 0.0)
        # Simple regret model: expected value minus cost and risk, plus counterfactual term
        util = a.expected_value - a.cost - a.risk + cf_val
        utilities.append(util)
        ids.append(a.id)
    probs = _softmax(np.array(utilities, dtype=float))
    return probs, ids


# ----------------------------------------------------------------------
# Parent B utilities (ternary vector & decision hygiene)
# ----------------------------------------------------------------------
TERNARY_DIMS = 12  # fixed dimensionality used by the original parent


def _int_from_hash(data: bytes) -> int:
    """Convert a SHA‑256 digest to a 256‑bit integer."""
    return int.from_bytes(hashlib.sha256(data).digest(), "big")


def ternary_vector(
    raw_command: str, normalized_intent: str, context: dict[str, Any], dims: int = TERNARY_DIMS
) -> np.ndarray:
    """
    Deterministic ternary vector (values ∈ {‑1,0,+1}) derived from a payload.
    The hash of the JSON‑encoded payload is sliced into `dims` 2‑bit chunks:
    00 → -1, 01 → 0, 10 → +1, 11 → -1 (wrap‑around).
    """
    payload = {
        "raw_command": raw_command,
        "normalized_intent": normalized_intent,
        "context": context,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    h = _int_from_hash(encoded)
    bits = h.bit_length()
    # Ensure enough bits; repeat the hash if necessary
    while bits < dims * 2:
        h = (h << 256) + _int_from_hash(encoded + b"pad")
        bits = h.bit_length()
    vec = np.empty(dims, dtype=int)
    for i in range(dims):
        chunk = (h >> (2 * i)) & 0b11
        if chunk == 0b00:
            vec[i] = -1
        elif chunk == 0b01:
            vec[i] = 0
        else:  # 0b10 or 0b11
            vec[i] = +1
    return vec


def shannon_entropy(symbols: np.ndarray) -> float:
    """
    Empirical Shannon entropy of a 1‑D integer array.
    Symbols are treated as categorical outcomes.
    """
    if symbols.ndim != 1:
        raise ValueError("symbols must be a 1‑D array")
    values, counts = np.unique(symbols, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs + 1e-12))


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def hybrid_probability_vector(
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual],
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute the regret‑weighted probability vector and map it onto the ternary
    alphabet {‑1,0,+1} by sign quantisation.
    Returns the ternary‑mapped vector and the ordered action ids.
    """
    probs, ids = compute_regret_weights(actions, counterfactuals)
    # Sign quantisation: prob > 1/len -> +1, prob < 1/len -> -1, else 0
    threshold = 1.0 / max(len(probs), 1)
    ternary_mapped = np.where(probs > threshold, 1, np.where(probs < threshold, -1, 0))
    return ternary_mapped.astype(int), ids


def hybrid_entropy(
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual],
    raw_command: str,
    normalized_intent: str,
    context: dict[str, Any],
) -> float:
    """
    Combine the ternary vector from the payload with the ternary‑mapped
    regret probabilities, then compute Shannon entropy of the concatenated
    symbol sequence.
    """
    # Ternary payload
    t_vec = ternary_vector(raw_command, normalized_intent, context)
    # Ternary‑mapped regret probabilities (length = number of actions)
    r_vec, _ = hybrid_probability_vector(actions, counterfactuals)
    # Concatenate; if lengths differ, pad the shorter one with zeros
    max_len = max(len(t_vec), len(r_vec))
    t_pad = np.pad(t_vec, (0, max_len - len(t_vec)), constant_values=0)
    r_pad = np.pad(r_vec, (0, max_len - len(r_vec)), constant_values=0)
    combined = np.concatenate([t_pad, r_pad])
    return shannon_entropy(combined)


def regret_weighted_similarity(
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual],
    tokens_a: Iterable[str],
    tokens_b: Iterable[str],
) -> float:
    """
    Compute MinHash similarity between two token sets and weight it by the
    KL‑divergence between the regret‑weighted distribution and a uniform prior.
    The result lies in [0,1]; higher values indicate both high token similarity
    and low regret (i.e., a confident decision).
    """
    sig_a = signature(tokens_a)
    sig_b = signature(tokens_b)
    base_sim = similarity(sig_a, sig_b)

    probs, _ = compute_regret_weights(actions, counterfactuals)
    if probs.size == 0:
        return base_sim  # no regret information to modulate

    uniform = np.full_like(probs, 1.0 / probs.size)
    # KL divergence D_KL(p || u)
    kl = np.sum(probs * np.log2((probs + 1e-12) / uniform))
    # Map KL to a confidence factor in (0,1]; larger KL → lower confidence
    confidence = 1.0 / (1.0 + kl)
    return base_sim * confidence


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a tiny action set
    actions = [
        MathAction(id="A", expected_value=10.0, cost=2.0, risk=1.0),
        MathAction(id="B", expected_value=8.0, cost=1.5, risk=0.5),
        MathAction(id="C", expected_value=5.0, cost=0.5, risk=2.0),
    ]

    counterfactuals = [
        MathCounterfactual(action_id="A", outcome_value=9.0, probability=0.9),
        MathCounterfactual(action_id="B", outcome_value=7.5, probability=0.8),
        MathCounterfactual(action_id="C", outcome_value=4.0, probability=0.6),
    ]

    # Payload example
    raw_cmd = "deploy --env prod"
    norm_intent = "deployment"
    ctx = {"user": "alice", "timestamp": "2026-05-29T23:00:00Z"}

    # Token sets for similarity
    tokens1 = ["deploy", "prod", "service"]
    tokens2 = ["deploy", "staging", "service"]

    # Run hybrid functions
    prob_vec, ids = compute_regret_weights(actions, counterfactuals)
    print("Regret‑weighted probabilities:", dict(zip(ids, prob_vec.round(4))))

    ent = hybrid_entropy(actions, counterfactuals, raw_cmd, norm_intent, ctx)
    print("Hybrid entropy (payload + regret):", round(ent, 4))

    sim = regret_weighted_similarity(actions, counterfactuals, tokens1, tokens2)
    print("Regret‑weighted MinHash similarity:", round(sim, 4))

    # Ensure no exception occurs
    sys.exit(0)