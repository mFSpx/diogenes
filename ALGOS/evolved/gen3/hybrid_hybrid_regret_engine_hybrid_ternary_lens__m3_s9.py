# DARWIN HAMMER — match 3, survivor 9
# gen: 3
# parent_a: hybrid_regret_engine_hybrid_liquid_time_c_m13_s0.py (gen2)
# parent_b: hybrid_ternary_lens_router_hybrid_decision_hygi_m44_s2.py (gen2)
# born: 2026-05-29T23:25:20Z

import hashlib
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, List, Tuple, Dict

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


def _softmax(values: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Temperature‑scaled soft‑max."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = values / temperature
    max_val = np.max(scaled)
    exp_vals = np.exp(scaled - max_val)
    return exp_vals / exp_vals.sum()


def compute_regret_weighted_strategy(
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual],
    temperature: float = 1.0,
) -> Dict[str, float]:
    """
    Regret‑matching with temperature‑scaled soft‑max.
    """
    if not actions:
        return {}
    exp_map = {a.id: a.expected_value for a in actions}
    regrets = {a.id: 0.0 for a in actions}
    for cf in counterfactuals:
        if cf.action_id not in exp_map:
            continue
        diff = cf.outcome_value - exp_map[cf.action_id]
        regrets[cf.action_id] += diff * cf.probability
    regrets_arr = np.array([regrets[a.id] for a in actions])
    probs_arr = _softmax(regrets_arr, temperature)
    return {a.id: float(p) for a, p in zip(actions, probs_arr)}


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


def _bits_to_ternary(bits: str) -> int:
    """
    Balanced mapping of two bits to {-1, 0, +1}.
    00 → -1
    01 → 0
    10 → +1
    11 → 0   (second zero to keep exact balance)
    """
    if bits == "00":
        return -1
    if bits == "01" or bits == "11":
        return 0
    # bits == "10"
    return 1


def ternary_vector(raw_command: str, normalized_intent: str, context: dict[str, Any]) -> List[int]:
    """
    Deterministic ternary vector of length TERNARY_DIMS using a balanced
    two‑bit → ternary mapping.
    """
    h = payload_hash(raw_command, normalized_intent, context)
    bin_str = bin(int(h, 16))[2:].zfill(256)
    vec = [_bits_to_ternary(bin_str[2 * i : 2 * i + 2]) for i in range(TERNARY_DIMS)]
    return vec


def shannon_entropy_counts(counts: Dict[int, int]) -> float:
    """Shannon entropy from a histogram (counts)."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        p = c / total
        ent -= p * math.log2(p)
    return ent


def ternary_entropy(vec: Iterable[int]) -> float:
    """Entropy of a ternary vector based on value frequencies."""
    counts: Dict[int, int] = {}
    for v in vec:
        counts[v] = counts.get(v, 0) + 1
    return shannon_entropy_counts(counts)


# ----------------------------------------------------------------------
# Hybrid core (mathematical bridge) – deeper integration
# ----------------------------------------------------------------------


def similarity_to_continuous(s: float) -> float:
    """
    Smooth mapping of MinHash similarity (0‑1) to a continuous token in [-1, +1].
    Uses a scaled sigmoid centred at 0.5 for gradual transition.
    """
    # sigmoid centred at 0.5, steepness = 8
    steep = 8.0
    sig = 1 / (1 + math.exp(-steep * (s - 0.5)))
    return 2 * sig - 1  # map to [-1, +1]


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
    Build a fused representation with a deeper mathematical coupling:

    1. Compute regret‑weighted probabilities (soft‑max) at a base temperature.
    2. Derive MinHash signatures and a smooth similarity token.
    3. Produce a deterministic ternary payload vector.
    4. Form a hybrid vector consisting of the continuous similarity token
       followed by the ternary payload.
    5. Compute entropy of the ternary part (ignoring the continuous token).
    6. Use the entropy to adapt the temperature of the regret‑soft‑max:
       higher entropy → higher temperature → more exploration.
    7. Return all intermediate artefacts and the final adjusted probabilities.
    """
    # ---------- 1. Base regret probabilities ----------
    base_temperature = 1.0
    base_regret_weights = compute_regret_weighted_strategy(
        actions, counterfactuals, temperature=base_temperature
    )

    # ---------- 2. MinHash similarity ----------
    sig_ref = signature(reference_tokens)
    sig_cur = signature(current_tokens)
    sim = similarity(sig_ref, sig_cur)

    # Continuous token in [-1, +1]
    cont_token = similarity_to_continuous(sim)

    # ---------- 3. Deterministic ternary payload ----------
    tern_vec = ternary_vector(raw_command, normalized_intent, context)

    # ---------- 4. Hybrid vector ----------
    hybrid_vec = [cont_token] + tern_vec  # length = TERNARY_DIMS + 1

    # ---------- 5. Entropy of ternary component ----------
    tern_entropy = ternary_entropy(tern_vec)

    # ---------- 6. Entropy‑driven temperature ----------
    # Scale entropy to a reasonable temperature factor.
    # Entropy of a perfectly balanced ternary vector of length L is log2(3) ≈ 1.585.
    # We map [0, log2(3)] → [0.5, 2.0] and add 1.0 as base.
    max_ent = math.log2(3)
    temp_factor = 0.5 + 1.5 * (tern_entropy / max_ent)  # in [0.5, 2.0]
    adjusted_temperature = base_temperature * temp_factor

    # ---------- 7. Adjusted regret probabilities ----------
    adjusted_weights = compute_regret_weighted_strategy(
        actions, counterfactuals, temperature=adjusted_temperature
    )

    return {
        "base_regret_weights": base_regret_weights,
        "adjusted_regret_weights": adjusted_weights,
        "minhash_signature_current": sig_cur,
        "minhash_signature_reference": sig_ref,
        "similarity": sim,
        "continuous_similarity_token": cont_token,
        "ternary_vector": tern_vec,
        "hybrid_vector": hybrid_vec,
        "ternary_entropy": tern_entropy,
        "adjusted_temperature": adjusted_temperature,
    }