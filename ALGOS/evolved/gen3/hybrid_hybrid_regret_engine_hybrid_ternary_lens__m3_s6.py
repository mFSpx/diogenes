# DARWIN HAMMER — match 3, survivor 6
# gen: 3
# parent_a: hybrid_regret_engine_hybrid_liquid_time_c_m13_s0.py (gen2)
# parent_b: hybrid_ternary_lens_router_hybrid_decision_hygi_m44_s2.py (gen2)
# born: 2026-05-29T23:25:20Z

"""Hybrid Regret‑Weighted Liquid‑Time‑Constant MinHash & Ternary Decision‑Hygiene Analyzer.

Parent A: *hybrid_regret_engine_hybrid_liquid_time_c_m13_s0.py* – provides a
regret‑weighted strategy whose hidden state is projected via a MinHash
signature.  
Parent B: *hybrid_ternary_lens_router_hybrid_decision_hygi_m44_s2.py* – supplies
a deterministic ternary vector derived from a payload hash together with a
Shannon‑entropy based decision‑hygiene score.

**Mathematical bridge**  
Both parents emit discrete integer sequences:
* the MinHash signature `σ ∈ ℕ^k` (parent A) and
* the ternary vector `τ ∈ {‑1,0,+1}^d` (parent B).

We map the similarity `s = Sim(σ, σ_ref)` between the current MinHash
signature and a reference signature to a ternary token `τ_s ∈ {‑1,0,+1}`
by thresholding (`s > 2/3 → +1`, `s < 1/3 → –1`, otherwise 0).  The hybrid
state is the concatenation  


h = [τ_s] ⊕ τ                # length d+1


From `h` we build an empirical distribution, compute its Shannon entropy,
and finally modulate the regret‑weighted action probabilities by a factor
derived from that entropy (higher entropy → more exploration).  The result
is a single unified system that respects the governing equations of both
parents while providing a mathematically coherent fusion."""

import hashlib
import json
import math
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Shared utilities (Parent A)
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


def _hash(seed: int, token: str) -> int:
    """Deterministic 64‑bit integer hash."""
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def signature(tokens: Iterable[str], k: int = 128) -> List[int]:
    """MinHash signature of a token set."""
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [((1 << 64) - 1)] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]


def similarity(sig_a: List[int], sig_b: List[int]) -> float:
    """Jaccard‑like similarity between two MinHash signatures."""
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have equal length")
    if not sig_a:
        raise ValueError("signatures must not be empty")
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)


def compute_regret_weighted_strategy(
    actions: List[MathAction], counterfactuals: List[MathCounterfactual]
) -> dict[str, float]:
    """
    Classic regret‑matching: for each action a,
        regret_a = Σ_c (outcome_a - expected_a) * prob_c
    The probability is proportional to exp(regret_a) (soft‑max).
    """
    if not actions:
        return {}
    # Map action id → expected value
    exp_map = {a.id: a.expected_value for a in actions}
    # Initialise regrets
    regrets = {a.id: 0.0 for a in actions}
    for cf in counterfactuals:
        # Only consider counterfactuals that refer to known actions
        if cf.action_id not in exp_map:
            continue
        diff = cf.outcome_value - exp_map[cf.action_id]
        regrets[cf.action_id] += diff * cf.probability

    # Soft‑max over regrets (temperature = 1.0)
    max_r = max(regrets.values())
    exp_vals = {aid: math.exp(r - max_r) for aid, r in regrets.items()}
    z = sum(exp_vals.values())
    probs = {aid: v / z for aid, v in exp_vals.items()}
    return probs


# ----------------------------------------------------------------------
# Parent B utilities
# ----------------------------------------------------------------------


TERNARY_DIMS = 12  # length of the deterministic ternary vector


def utc_now() -> str:
    """Current UTC timestamp in ISO‑8601 without microseconds."""
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def payload_hash(raw_command: str, normalized_intent: str, context: dict[str, Any]) -> str:
    """Deterministic SHA‑256 over a JSON envelope."""
    payload = {
        "raw_command": raw_command,
        "normalized_intent": normalized_intent,
        "context": context,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def ternary_vector(raw_command: str, normalized_intent: str, context: dict[str, Any]) -> List[int]:
    """
    Produce a deterministic ternary vector of length TERNARY_DIMS.
    The SHA‑256 hash is interpreted as a binary stream; every two bits
    map to -1, 0, +1 (00→‑1, 01→0, 10→+1, 11→‑1 to keep balance).
    """
    h = payload_hash(raw_command, normalized_intent, context)
    # Convert hex to binary string
    bin_str = bin(int(h, 16))[2:].zfill(256)
    vec = []
    for i in range(TERNARY_DIMS):
        bits = bin_str[2 * i : 2 * i + 2]
        if bits == "00" or bits == "11":
            val = -1
        elif bits == "01":
            val = 0
        else:  # "10"
            val = 1
        vec.append(val)
    return vec


def shannon_entropy(values: Iterable[int]) -> float:
    """Shannon entropy of the empirical distribution of integer values."""
    vals = list(values)
    if not vals:
        return 0.0
    counts = {}
    for v in vals:
        counts[v] = counts.get(v, 0) + 1
    total = len(vals)
    entropy = 0.0
    for c in counts.values():
        p = c / total
        entropy -= p * math.log2(p)
    return entropy


# ----------------------------------------------------------------------
# Hybrid core (mathematical bridge)
# ----------------------------------------------------------------------


def similarity_to_ternary(s: float) -> int:
    """
    Map a MinHash similarity (0‑1) to a ternary token:
        s > 2/3 → +1
        s < 1/3 → -1
        otherwise → 0
    """
    if s > 2.0 / 3.0:
        return 1
    if s < 1.0 / 3.0:
        return -1
    return 0


def hybrid_state(
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual],
    reference_tokens: Iterable[str],
    current_tokens: Iterable[str],
    raw_command: str,
    normalized_intent: str,
    context: dict[str, Any],
) -> dict[str, Any]:
    """
    Build the fused representation and return a dictionary containing:
        - regret_weights: soft‑max probabilities from the regret engine
        - minhash_sig: signature of current_tokens
        - similarity: similarity to reference signature
        - ternary_token: ternary value derived from similarity
        - ternary_vec: deterministic ternary payload vector
        - hybrid_vec: concatenation of ternary_token and ternary_vec
        - entropy: Shannon entropy of hybrid_vec
        - adjusted_weights: regret_weights re‑scaled by (1 + entropy)
    """
    # 1️⃣ Regret‑weighted probabilities
    regret_weights = compute_regret_weighted_strategy(actions, counterfactuals)

    # 2️⃣ MinHash signatures and similarity
    sig_ref = signature(reference_tokens)
    sig_cur = signature(current_tokens)
    s = similarity(sig_ref, sig_cur)

    # 3️⃣ Map similarity → ternary token
    ternary_token = similarity_to_ternary(s)

    # 4️⃣ Deterministic ternary payload vector
    tern_vec = ternary_vector(raw_command, normalized_intent, context)

    # 5️⃣ Hybrid vector (length = TERNARY_DIMS + 1)
    hybrid_vec = [ternary_token] + tern_vec

    # 6️⃣ Entropy over the hybrid vector
    ent = shannon_entropy(hybrid_vec)

    # 7️⃣ Adjust regret probabilities by entropy (more uncertainty → flatter distribution)
    factor = 1.0 + ent  # simple linear scaling
    adj_weights = {aid: prob * factor for aid, prob in regret_weights.items()}
    # re‑normalize after scaling
    z = sum(adj_weights.values())
    adj_weights = {aid: p / z for aid, p in adj_weights.items()}

    return {
        "regret_weights": regret_weights,
        "minhash_sig": sig_cur,
        "similarity": s,
        "ternary_token": ternary_token,
        "ternary_vec": tern_vec,
        "hybrid_vec": hybrid_vec,
        "entropy": ent,
        "adjusted_weights": adj_weights,
    }


def decision_hygiene_score(actions: List[MathAction]) -> float:
    """
    Simple hygiene score: mean of (expected_value – cost – risk).
    Higher scores indicate cleaner decisions.
    """
    if not actions:
        return 0.0
    vals = [a.expected_value - a.cost - a.risk for a in actions]
    return sum(vals) / len(vals)


def hybrid_decision_pipeline(
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual],
    reference_tokens: Iterable[str],
    current_tokens: Iterable[str],
    raw_command: str,
    normalized_intent: str,
    context: dict[str, Any],
) -> dict[str, Any]:
    """
    End‑to‑end hybrid pipeline:
        1. Build hybrid state (see ``hybrid_state``).
        2. Compute a decision‑hygiene score.
        3. Mix the hygiene score with the entropy‑adjusted regret weights
           to obtain a final decision vector.
    Returns a dictionary with the intermediate and final results.
    """
    state = hybrid_state(
        actions,
        counterfactuals,
        reference_tokens,
        current_tokens,
        raw_command,
        normalized_intent,
        context,
    )

    hygiene = decision_hygiene_score(actions)

    # Linear mixing: final_score = λ * hygiene + (1‑λ) * entropy
    λ = 0.6
    final_metric = λ * hygiene + (1 - λ) * state["entropy"]

    # Produce final decision distribution by blending adjusted_weights with
    # the normalized final_metric (treated as a scalar bias toward the highest
    # probability action).
    max_aid = max(state["adjusted_weights"], key=state["adjusted_weights"].get)
    final_dist = {}
    for aid, prob in state["adjusted_weights"].items():
        bias = final_metric if aid == max_aid else 0.0
        final_dist[aid] = prob + bias
    # Re‑normalize
    total = sum(final_dist.values())
    final_dist = {aid: p / total for aid, p in final_dist.items()}

    return {
        "state": state,
        "hygiene_score": hygiene,
        "final_metric": final_metric,
        "final_distribution": final_dist,
    }


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Dummy data
    actions = [
        MathAction(id="A", expected_value=10.0, cost=2.0, risk=1.0),
        MathAction(id="B", expected_value=7.0, cost=1.0, risk=0.5),
        MathAction(id="C", expected_value=5.0, cost=0.5, risk=0.2),
    ]

    counterfactuals = [
        MathCounterfactual(action_id="A", outcome_value=12.0, probability=0.8),
        MathCounterfactual(action_id="B", outcome_value=6.0, probability=0.6),
        MathCounterfactual(action_id="C", outcome_value=4.0, probability=0.9),
    ]

    reference_tokens = ["alpha", "beta", "gamma"]
    current_tokens = ["delta", "epsilon", "zeta"]

    raw_cmd = "launch sequence"
    norm_intent = "execute"
    ctx = {"user": "tester", "session": 42}

    result = hybrid_decision_pipeline(
        actions,
        counterfactuals,
        reference_tokens,
        current_tokens,
        raw_cmd,
        norm_intent,
        ctx,
    )

    print("Hybrid state summary:")
    for key, val in result["state"].items():
        if key in ("minhash_sig", "ternary_vec", "hybrid_vec"):
            print(f"  {key}: {val[:8]}{'...' if len(val) > 8 else ''}")
        else:
            print(f"  {key}: {val}")

    print(f"\nDecision hygiene score: {result['hygiene_score']:.4f}")
    print(f"Final blended metric:   {result['final_metric']:.4f}")
    print("Final action distribution:")
    for aid, prob in result["final_distribution"].items():
        print(f"  {aid}: {prob:.4f}")

    sys.exit(0)